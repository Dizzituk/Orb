# FILE: app/bridge/chat_speak_stream.py
# Purpose: Build the chat-and-speak audio StreamingResponse — tee to tts_cache, finish synthesis on disconnect, idempotent replay.
# Called-by: app.bridge.router.bridge_chat_and_speak
# Depends-on: app.bridge.tts_cache, app.bridge.chat_and_speak (sentence synthesis), app.memory.models
# Last-renovated: 2026-06-11
"""
Streaming plumbing for /bridge/chat-and-speak.

build_audio_response(): streams sentence-MP3 chunks to the phone while
TEEING every chunk into tts_cache/{message_id}.mp3. If the phone drops
mid-stream (dead zone), the remaining sentences keep synthesising into the
cache via a background task, so the phone can resume the SAME audio later
through GET /bridge/tts/audio/{message_id} with a Range header — nothing is
ever synthesised twice.

replay_cached_reply(): when a retried request carries an X-Idempotency-Key
we have already served, re-serve the original assistant message (text via
headers, audio from cache) WITHOUT running the LLM or TTS again. This kills
the "retry creates a second reply" half of the replay-ghost bug.
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote as url_quote

from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from . import tts_cache
from .chat_and_speak import _split_into_sentences, _synthesise_sentence

logger = logging.getLogger(__name__)


def _speak_headers(
    message_id: int,
    project_id: int,
    project_name: str,
    display_text: str,
    attachments_payload: str,
    cached: bool = False,
    directives_json: str = "[]",
) -> dict:
    headers = {
        "X-Project-Id": str(project_id),
        "X-Project-Name": url_quote(project_name or ""),
        "X-Full-Text": url_quote(display_text[:8000]),
        "X-Message-Id": str(message_id),
        "X-Artifacts": url_quote(attachments_payload),
        "Transfer-Encoding": "chunked",
    }
    if directives_json and directives_json != "[]":
        headers["X-Directives"] = url_quote(directives_json)
    if cached:
        headers["X-Audio-Cached"] = "1"
    return headers


async def _finish_synthesis_to_cache(writer: tts_cache.TtsCacheWriter, remaining: list[str]) -> None:
    """Background continuation after a client disconnect mid-stream."""
    try:
        for sentence in remaining:
            writer.add_chunk(await _synthesise_sentence(sentence))
        writer.finalize()
        logger.info("[chat_speak] finished caching %d remaining sentence(s) for message %s after disconnect",
                    len(remaining), writer.message_id)
    except Exception as e:
        writer.abort()
        logger.error("[chat_speak] background synthesis for %s failed: %s", writer.message_id, e)


def build_audio_response(
    message_id: int,
    display_text: str,
    project_id: int,
    project_name: str,
    attachments_payload: str,
    directives_json: str = "[]",
) -> StreamingResponse:
    """Sentence-synthesise display_text, streaming to the phone + caching on disk."""
    sentences = _split_into_sentences(display_text)
    writer = tts_cache.open_writer(message_id)
    state = {"idx": 0}

    async def generate_audio():
        try:
            # Placeholder duration header (2026-06-13): the same 192 bytes go
            # to the live stream AND the cache file so Range offsets stay
            # aligned; finalize() patches real totals into the disk copy.
            header = writer.start_header()
            if header:
                yield header
            while state["idx"] < len(sentences):
                sentence = sentences[state["idx"]]
                try:
                    mp3_bytes = await _synthesise_sentence(sentence)
                except Exception as e:
                    logger.error("[chat_speak] TTS failed for sentence: %s", e)
                    state["idx"] += 1
                    continue
                writer.add_chunk(mp3_bytes)
                state["idx"] += 1
                yield mp3_bytes
            writer.finalize()
        except (GeneratorExit, asyncio.CancelledError):
            remaining = sentences[state["idx"]:]
            if remaining:
                logger.info("[chat_speak] client gone after %d/%d sentences for message %s — continuing to cache",
                            state["idx"], len(sentences), message_id)
                try:
                    asyncio.get_running_loop().create_task(
                        _finish_synthesis_to_cache(writer, remaining)
                    )
                except RuntimeError:
                    writer.abort()
            else:
                writer.finalize()
            raise

    return StreamingResponse(
        generate_audio(),
        media_type="audio/mpeg",
        headers=_speak_headers(message_id, project_id, project_name,
                               display_text, attachments_payload,
                               directives_json=directives_json),
    )


async def replay_cached_reply(message_id: int, db: Session, process_artifacts) -> StreamingResponse | None:
    """Re-serve a previously generated reply for a retried idempotency key.

    Returns None when the message no longer exists (caller falls through to
    the normal flow). Audio comes from the cache when present; if the cache
    was evicted, the audio is re-synthesised into the cache once — the LLM
    is never re-run either way.
    """
    from app.memory.models import Message
    msg = db.query(Message).filter(Message.id == int(message_id)).first()
    if msg is None or msg.role != "assistant":
        return None

    # A disconnect-continuation may still be synthesising this very reply
    # (2026-06-13): wait briefly for it to finalize instead of opening a
    # second writer on the same .part (which would truncate it mid-write —
    # sentences synthesise in ~1-2 s each, so the wait is short in practice).
    if tts_cache.get_cached_path(message_id) is None and tts_cache.is_pending(message_id):
        for _ in range(24):
            await asyncio.sleep(0.5)
            if tts_cache.get_cached_path(message_id) is not None:
                break

    from app.memory.models import Project
    project = db.query(Project).filter(Project.id == msg.project_id).first()
    project_name = project.name if project else ""

    from .directives import extract_directives, strip_directives, directives_payload
    display_text, attachments = process_artifacts(msg.content or "")
    directives = extract_directives(display_text)
    if directives:
        display_text = strip_directives(display_text)
    directives_json = directives_payload(directives)
    attachments_payload = json.dumps([a.model_dump() for a in attachments])

    cached = tts_cache.get_cached_path(message_id)
    if cached is not None:
        tts_cache.touch(message_id)
        logger.info("[chat_speak] idempotent replay of message %s from cache (%d KB)",
                    message_id, cached.stat().st_size // 1024)

        def stream_file():
            with open(cached, "rb") as fh:
                while True:
                    chunk = fh.read(64 * 1024)
                    if not chunk:
                        break
                    yield chunk

        return StreamingResponse(
            stream_file(),
            media_type="audio/mpeg",
            headers=_speak_headers(message_id, msg.project_id, project_name,
                                   display_text, attachments_payload, cached=True,
                                   directives_json=directives_json),
        )

    logger.info("[chat_speak] idempotent replay of message %s (no cached audio — re-synthesising once)",
                message_id)
    return build_audio_response(
        message_id=message_id,
        display_text=display_text,
        project_id=msg.project_id,
        project_name=project_name,
        attachments_payload=attachments_payload,
        directives_json=directives_json,
    )
