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
from .markdown_sanitize import for_display, for_speech

logger = logging.getLogger(__name__)


def _speak_headers(
    message_id: int,
    project_id: int,
    project_name: str,
    display_text: str,
    attachments_payload: str,
    cached: bool = False,
    directives_json: str = "[]",
    provider: str | None = None,
    model: str | None = None,
) -> dict:
    headers = {
        "X-Project-Id": str(project_id),
        "X-Project-Name": url_quote(project_name or ""),
        # The phone renders X-Full-Text raw (no markdown engine), so the
        # bubble gets the sanitised form; the DB row keeps the raw text.
        "X-Full-Text": url_quote(for_display(display_text)[:8000]),
        "X-Message-Id": str(message_id),
        "X-Artifacts": url_quote(attachments_payload),
        "Transfer-Encoding": "chunked",
    }
    if directives_json and directives_json != "[]":
        headers["X-Directives"] = url_quote(directives_json)
    if cached:
        headers["X-Audio-Cached"] = "1"
    # Model badge parity (2026-07-03): same provenance the desktop chips
    # show. Additive — a phone build without the chip UI ignores them.
    if provider:
        headers["X-Provider"] = url_quote(str(provider))
    if model:
        headers["X-Model"] = url_quote(str(model))
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
    provider: str | None = None,
    model: str | None = None,
) -> StreamingResponse:
    """Sentence-synthesise display_text, streaming to the phone + caching on disk.

    display_text arrives with artifact markers + directives already stripped
    but markdown intact: the TTS input is for_speech() (no spoken "asterisk"),
    the X-Full-Text header is for_display() (no raw ** on the phone). Both are
    surface-side only — the stored message row keeps the raw text.
    """
    sentences = _split_into_sentences(for_speech(display_text))
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
                               directives_json=directives_json,
                               provider=provider, model=model),
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
        # Wait generously (2026-06-16): an in-flight coalesced retry can wake the
        # instant the owner opens this .part, so the final file may be a whole
        # reply's synthesis away. Polling is cheap; the cap still falls through
        # to a single re-synthesis if the owner never finalises.
        for _ in range(120):
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
                                   directives_json=directives_json,
                                   provider=getattr(msg, "provider", None),
                                   model=getattr(msg, "model", None)),
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
        provider=getattr(msg, "provider", None),
        model=getattr(msg, "model", None),
    )


async def run_chat_and_speak(req, request, db):
    """Full handler body for POST /bridge/chat-and-speak.

    Lives here (not inline in the >30 KB router) so the in-flight idempotency
    claim taken in inflight_idem.begin() is released in a finally on EVERY exit
    — including an exception mid-generation, which is exactly when a weak link
    makes the owner's LLM call most likely to fail. Without that guaranteed
    release a coalesced retry would block the whole wait window, then regenerate.
    """
    from app.memory.service import create_message
    from app.memory.schemas import MessageCreate
    from app.memory._service_utils_2 import list_messages
    from . import tts_cache, inflight_idem
    from .router import (
        _resolve_or_create_project, _run_translation, _run_web_search, _process_artifacts,
    )

    # Idempotency front door (2026-06-16): a completed key replays its reply; an
    # in-flight key (a retried POST racing a slow grounded turn) waits for and
    # replays that one reply instead of a duplicate LLM+TTS run. None => we own
    # the key and MUST release it (the finally below) once done.
    idempotency_key = request.headers.get("X-Idempotency-Key")
    coalesced = await inflight_idem.begin(idempotency_key, db, _process_artifacts)
    if coalesced is not None:
        return coalesced

    try:
        project = _resolve_or_create_project(req, db)
        # Phase 3: scan for identity facts (DOB, birthplace, location, etc.)
        # Fire-and-forget — writes to Tier 1 identity store if matched.
        from app.bridge.identity_hook import capture_from_bridge_message
        capture_from_bridge_message(req.message, source_label=f"bridge:{project.id}")

        # Phase 7: capture fragment for long-term associative memory.
        # Every message embedded + clustered into themes; decays over time.
        from app.bridge.identity_hook import capture_fragment_from_bridge
        capture_fragment_from_bridge(req.message, project_id=project.id)

        history_before_save = list_messages(db, project.id, limit=100)
        existing_user_message = next(
            (
                message for message in reversed(history_before_save)
                if message.role == "user" and message.content == req.message
            ),
            None,
        ) if idempotency_key else None

        if existing_user_message is None:
            create_message(db, MessageCreate(
                project_id=project.id, role="user",
                content=req.message, provider="bridge", model="phone-input",
            ))

        history = list_messages(db, project.id, limit=20)
        history_messages = [
            {"role": m.role, "content": m.content}
            for m in history if m.role in ("user", "assistant")
        ]

        # Build-command parity (Phase 1): handle "build the apk" + voice confirmation
        # here, before translation, so the phone triggers the same host build flow as
        # the desktop. The build runs async; completion arrives via /missed-replies.
        # The acknowledgement is spoken via the normal sentence-TTS streaming path.
        from app.bridge.build_actions import maybe_handle_build_turn
        build_turn = maybe_handle_build_turn(req.message, project.id, db)
        if build_turn is not None:
            bt_msg = create_message(db, MessageCreate(
                project_id=project.id, role="assistant",
                content=build_turn.reply, provider="bridge", model=build_turn.model,
            ))
            bt_response = build_audio_response(
                message_id=bt_msg.id,
                display_text=build_turn.reply,
                project_id=project.id,
                project_name=project.name,
                attachments_payload="[]",
                provider="bridge",
                model=build_turn.model,
            )
            # idem map AFTER the .part writer is open; the finally wakes waiters.
            if idempotency_key:
                tts_cache.idem_put(idempotency_key, bt_msg.id)
            return bt_response

        from app.bridge.capability_honesty import (
            is_unsupported_on_bridge, get_unsupported_message,
        )

        domain_context, translation_result, domain_info = _run_translation(req.message, db, project_id=project.id)

        resolved_intent = (
            translation_result.resolved_intent.value
            if translation_result and translation_result.resolved_intent
            else None
        )

        web_search_context, search_executed, search_succeeded, honest_early_reply = (
            await _run_web_search(resolved_intent, translation_result, req.message)
        )

        from app.bridge.capability_layer import run_astra_chat

        model_source = None  # v2026-06-24: routing provenance for sticky restore
        # Single writer for the capability gate: full_text set here is
        # persisted once by the generic create_message below. (An earlier,
        # second create_message used to run too — two identical bubbles per
        # gated request on the phone, seen live 2026-07-03. Removed.)
        if is_unsupported_on_bridge(resolved_intent):
            logger.info("[bridge] chat-and-speak: blocked unsupported %s", resolved_intent)
            full_text = get_unsupported_message(resolved_intent)
            provider = "bridge"
            model = "capability-gate"
        elif honest_early_reply:
            full_text = honest_early_reply
            provider = "bridge"
            model = "search-gate"
        else:
            # Augment with attachment vision/document text; req.message stays
            # untouched so identity_hook, translation, and the DB row see the
            # user's actual words. Shared with bridge_chat (see augment_user_message).
            from app.bridge.attachment_describe import augment_user_message
            llm_input_message = augment_user_message(req, history_messages)

            result = await run_astra_chat(
                message=llm_input_message,
                project_id=project.id,
                history=history_messages,
                db=db,
                source="bridge-tts",
                domain_context=domain_context,
                translation_result=translation_result,
                web_search_context=web_search_context,
                search_executed=search_executed,
                search_succeeded=search_succeeded,
                raw_message=req.message,
                client_request_id=idempotency_key,  # J2: per-turn image cost guard
            )
            full_text = result["reply"]
            provider = result["provider"]
            model = result["model"]
            model_source = result.get("model_source")

        project_id = project.id
        project_name = project.name

        assistant_message = create_message(db, MessageCreate(
            project_id=project_id, role="assistant",
            content=full_text, provider=provider, model=model,
            model_source=model_source,
        ))

        # v2026-06-10: session + summary tracking (see bridge_chat above).
        try:
            from app.memory.integration import record_session_activity
            record_session_activity(
                project_id=project_id, provider=provider,
                model=model, db_session=db,
            )
        except Exception as _sess_err:
            logger.debug("[bridge] session tracking failed: %s", _sess_err)

        # Process artifact markers. The DB row keeps the raw full_text
        # (with markers) so history reload re-emits the chips; the streaming
        # response uses display_text for the X-Full-Text header and TTS so
        # the marker syntax never reaches the user. attachments_payload is
        # JSON-serialised into the X-Artifacts header for the phone to parse.
        import json
        display_text, attachments = _process_artifacts(full_text)
        attachments_payload = json.dumps([a.model_dump() for a in attachments])

        # Phone-action directives ([[astra:...]] — see directives.py): stripped
        # before the text reaches the screen or the voice, sent via X-Directives.
        from .directives import extract_directives, strip_directives, directives_payload
        directives = extract_directives(display_text)
        if directives:
            display_text = strip_directives(display_text)

        # Stream sentence-MP3s to the phone while teeing them into the
        # per-message tts_cache; survives client disconnects (see
        # build_audio_response for the full contract).
        response = build_audio_response(
            message_id=assistant_message.id,
            display_text=display_text,
            project_id=project_id,
            project_name=project_name,
            attachments_payload=attachments_payload,
            directives_json=directives_payload(directives),
            provider=provider,
            model=model,
        )
        # Map the key AFTER the .part writer is open; the finally wakes waiters,
        # so a woken waiter resumes the audio instead of racing a 2nd synthesis.
        if idempotency_key:
            tts_cache.idem_put(idempotency_key, assistant_message.id)
        return response
    finally:
        inflight_idem.complete(idempotency_key)
