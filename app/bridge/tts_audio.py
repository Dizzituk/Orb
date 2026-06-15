# FILE: app/bridge/tts_audio.py
# Purpose: Serve cached per-message TTS audio to the phone (HTTP Range) + ensure-synthesised endpoint.
# Called-by: main.py router registration; AstraBridge replay/download-ahead path
# Depends-on: app.bridge.tts_cache, app.bridge.chat_and_speak (sentence synthesis), app.memory.models
# Last-renovated: 2026-06-11
"""
Message-keyed TTS audio endpoints for the Bridge.

GET  /bridge/tts/audio/{message_id}
    Serves the complete cached MP3 for an assistant message. Supports HTTP
    Range (single range) so the phone can resume an interrupted download
    after a dead zone without re-synthesising anything.
    200 = full body, 206 = partial, 202 = synthesis still in progress
    (retry shortly), 404 = no audio cached (call /ensure), 416 = bad range.

POST /bridge/tts/audio/{message_id}/ensure
    Guarantees audio exists for a message: cache hit -> immediate "ready";
    synthesis pending -> "synthesising"; otherwise synthesises the message
    content sentence-by-sentence into the cache (the LAST-RESORT path for
    messages that never had audio). Never re-runs synthesis for cached audio.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.db import get_db
from sqlalchemy.orm import Session

from . import tts_cache
from .schemas import require_bridge_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/bridge/tts/audio", tags=["Bridge TTS"])

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
_CHUNK = 64 * 1024

# message_ids with an ensure-synthesis currently running in this process
_ensure_inflight: set[int] = set()
_ensure_lock = threading.Lock()


def _file_slice(path: Path, start: int, end: int):
    """Yield bytes [start, end] inclusive from path in chunks."""
    remaining = end - start + 1
    with open(path, "rb") as fh:
        fh.seek(start)
        while remaining > 0:
            chunk = fh.read(min(_CHUNK, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@router.get("/{message_id}")
async def get_tts_audio(
    message_id: int,
    request: Request,
    _auth: bool = Depends(require_bridge_auth),
):
    """Serve cached audio for a message, honouring a single-range Range header."""
    path = tts_cache.get_cached_path(message_id)
    if path is None:
        if tts_cache.is_pending(message_id):
            return JSONResponse({"state": "synthesising"}, status_code=202)
        raise HTTPException(404, "No cached audio for this message")

    tts_cache.touch(message_id)
    size = path.stat().st_size
    base_headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store",
        "X-Message-Id": str(message_id),
    }

    range_header = request.headers.get("range", "").strip()
    if not range_header:
        return StreamingResponse(
            _file_slice(path, 0, size - 1),
            media_type="audio/mpeg",
            headers={**base_headers, "Content-Length": str(size)},
        )

    m = _RANGE_RE.match(range_header)
    if not m or (not m.group(1) and not m.group(2)):
        raise HTTPException(416, "Malformed Range header")
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:  # suffix form bytes=-N
        suffix = int(m.group(2))
        start = max(0, size - suffix)
        end = size - 1
    end = min(end, size - 1)
    if start >= size or start > end:
        return JSONResponse(
            {"detail": "Range not satisfiable"},
            status_code=416,
            headers={"Content-Range": f"bytes */{size}"},
        )

    return StreamingResponse(
        _file_slice(path, start, end),
        status_code=206,
        media_type="audio/mpeg",
        headers={
            **base_headers,
            "Content-Length": str(end - start + 1),
            "Content-Range": f"bytes {start}-{end}/{size}",
        },
    )


@router.post("/{message_id}/ensure")
async def ensure_tts_audio(
    message_id: int,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Make sure audio exists for an assistant message (synthesis = last resort)."""
    cached = tts_cache.get_cached_path(message_id)
    if cached is not None:
        tts_cache.touch(message_id)
        return {"state": "ready", "bytes": cached.stat().st_size, "synthesised": False}
    if tts_cache.is_pending(message_id):
        return JSONResponse({"state": "synthesising"}, status_code=202)

    with _ensure_lock:
        if message_id in _ensure_inflight:
            return JSONResponse({"state": "synthesising"}, status_code=202)
        _ensure_inflight.add(message_id)

    try:
        from app.memory.models import Message
        msg = db.query(Message).filter(Message.id == message_id).first()
        if msg is None or msg.role != "assistant" or not (msg.content or "").strip():
            raise HTTPException(404, "Message not found or has no speakable content")

        from app.bridge.artifacts import extract_artifacts, strip_artifacts
        text = msg.content
        if extract_artifacts(text):
            text = strip_artifacts(text)

        from .chat_and_speak import _split_into_sentences, _synthesise_sentence
        sentences = _split_into_sentences(text)
        writer = tts_cache.open_writer(message_id)
        writer.start_header()  # placeholder duration header (2026-06-13)
        try:
            for sentence in sentences:
                writer.add_chunk(await _synthesise_sentence(sentence))
        except Exception as e:
            writer.discard()
            logger.error("[tts_audio] ensure synthesis failed for %s: %s", message_id, e)
            raise HTTPException(502, f"TTS synthesis failed: {e}")
        final = writer.finalize()
        if final is None:
            raise HTTPException(500, "Could not finalize cached audio")
        logger.info("[tts_audio] ensure synthesised message %s (%d sentences)",
                    message_id, len(sentences))
        return {"state": "ready", "bytes": final.stat().st_size, "synthesised": True}
    finally:
        with _ensure_lock:
            _ensure_inflight.discard(message_id)
