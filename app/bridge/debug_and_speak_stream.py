# FILE: app/bridge/debug_and_speak_stream.py
# Purpose: Build the /bridge/debug-and-speak audio StreamingResponse. Starts a
#          registry-backed BACKGROUND run (survives a phone disconnect, unlike
#          chat-and-speak's inline generator) that drives stream_debug_locked
#          -- the SAME debug brain the desktop uses, called unmodified -- maps
#          its SSE chunks to speech via debug_speech.SpeechAccumulator,
#          synthesises each sentence, tees into tts_cache/{run_id}.mp3 (the
#          SAME file the existing GET /bridge/tts/audio/{run_id} Range-resume
#          endpoint already knows how to serve, unmodified), and relays the
#          live bytes to whichever HTTP request is currently attached.
# Called-by: app.bridge.router (POST /bridge/debug-and-speak)
# Depends-on: app.bridge.debug_run_registry, app.bridge.debug_speech,
#             app.bridge.tts_cache, app.bridge.chat_and_speak (_synthesise_sentence),
#             app.debug.debug_lock_stream (stream_debug_locked)
# Last-renovated: 2026-07-01
from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote as url_quote

from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import debug_run_registry, tts_cache
from .chat_and_speak import _synthesise_sentence
from .debug_speech import SpeechAccumulator

logger = logging.getLogger(__name__)

_SENTINEL = object()


def _headers(run_id: int, project_id: int, project_name: str) -> dict:
    return {
        "X-Project-Id": str(project_id),
        "X-Project-Name": url_quote(project_name or ""),
        # X-Message-Id reuses the EXISTING GET /bridge/tts/audio/{id} Range-resume
        # endpoint unmodified; X-Run-Id is the explicit alias the cancel button
        # (Phase C) reads. Same value, both keyed the same way (see
        # debug_run_registry.new_run_id -- an int, not a uuid string, precisely
        # so this cache-key reuse works without touching tts_cache.py/tts_audio.py).
        "X-Message-Id": str(run_id),
        "X-Run-Id": str(run_id),
        # Unlike chat-and-speak, the reply text isn't known upfront -- it's a
        # live narration. No on-screen transcript for this MVP (voice-first,
        # per the plan's framing); left empty rather than omitted so phone
        # header-parsing code that expects the key to exist doesn't choke.
        "X-Full-Text": "",
        "X-Directives": url_quote("[]"),
        "X-Artifacts": url_quote("[]"),
        "Transfer-Encoding": "chunked",
    }


def _iter_sse_events(raw: bytes):
    """Parse 'data: {json}\\n\\n' SSE frame(s) out of a raw chunk from
    stream_debug_locked. In practice each chunk is exactly one frame (one
    _sse() call = one yield), but this splits defensively in case two ever
    arrive concatenated."""
    for part in raw.decode("utf-8", errors="replace").split("\n\n"):
        part = part.strip()
        if not part.startswith("data:"):
            continue
        try:
            yield json.loads(part[len("data:"):].strip())
        except Exception:
            continue


async def _run_debug_narration(
    *, project_id: int, message: str, panel_history: list,
    debug_project_id, reasoning_dial,
    writer: "tts_cache.TtsCacheWriter", live_queue: "asyncio.Queue",
) -> None:
    """Background task body (runs detached from the HTTP request — see
    debug_run_registry.launch). Opens its OWN db session: the request's
    Depends(get_db) session is closed when the endpoint returns, which
    happens as soon as the StreamingResponse is constructed, long before this
    finishes on the road."""
    from app.db import get_db_session
    from app.debug.debug_lock_stream import stream_debug_locked

    db = get_db_session()
    accumulator = SpeechAccumulator()

    async def _speak(sentences: list) -> None:
        for sentence in sentences:
            try:
                mp3_bytes = await _synthesise_sentence(sentence)
            except Exception as e:
                logger.error("[debug_and_speak] TTS failed for %r: %s", sentence[:60], e)
                continue
            writer.add_chunk(mp3_bytes)
            live_queue.put_nowait(mp3_bytes)

    try:
        header = writer.start_header()
        if header:
            live_queue.put_nowait(header)

        async for raw in stream_debug_locked(
            db=db, project_id=project_id, message=message,
            panel_history=panel_history, debug_project_id=debug_project_id,
            reasoning_dial=reasoning_dial,
        ):
            for event in _iter_sse_events(raw):
                await _speak(accumulator.feed(event))
        await _speak(accumulator.flush())
    except asyncio.CancelledError:
        # Hard-cancel backstop (registry.cancel): whatever synthesised so far
        # stays cached. No "leave a half-written tree" concern here -- this
        # only ever writes to the sandboxed/host-only debug tools and the
        # tts_cache file, both already safe to abandon mid-write.
        raise
    finally:
        writer.finalize()
        live_queue.put_nowait(_SENTINEL)
        db.close()


async def run_debug_and_speak(req, db: Session):
    """Full handler body for POST /bridge/debug-and-speak."""
    from app.memory.service import list_messages
    from .router import _resolve_or_create_project

    project = _resolve_or_create_project(req, db)
    panel_history = [
        {"role": m.role, "content": m.content}
        for m in list_messages(db, project.id, limit=20)
        if m.role in ("user", "assistant")
    ]

    state = debug_run_registry.reserve(project.id)
    writer = tts_cache.open_writer(state.run_id)
    state.cache_writer = writer
    live_queue: "asyncio.Queue" = asyncio.Queue()

    debug_project_id = req.debug_project_id
    reasoning_dial = req.reasoning_dial
    message = req.message
    project_id = project.id

    async def _coro() -> None:
        await _run_debug_narration(
            project_id=project_id, message=message, panel_history=panel_history,
            debug_project_id=debug_project_id, reasoning_dial=reasoning_dial,
            writer=writer, live_queue=live_queue,
        )

    debug_run_registry.launch(state, _coro)
    logger.info("[debug_and_speak] run=%s project=%s started", state.run_id, project_id)

    async def _generate():
        # The background task is fully independent of this generator (see
        # debug_run_registry.launch): if the phone drops here, this generator
        # just stops being iterated -- live_queue.put_nowait() never blocks
        # (unbounded queue), so the producer is never affected by a gone
        # consumer. Resume happens via the existing GET /bridge/tts/audio/{id}
        # Range endpoint reading straight off `writer`'s file on disk.
        while True:
            item = await live_queue.get()
            if item is _SENTINEL:
                break
            yield item

    return StreamingResponse(
        _generate(),
        media_type="audio/mpeg",
        headers=_headers(state.run_id, project_id, project.name),
    )
