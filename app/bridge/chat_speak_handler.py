# FILE: app/bridge/chat_speak_handler.py
# Purpose: /bridge/chat-and-speak turn logic — idempotent replay, in-flight
#          guard, generation + sentence-TTS streaming. Extracted from router.py
#          (2026-06-16) to stay under the 30KB file cap and give the in-flight
#          guard a clean wrap boundary.
# Called-by: app.bridge.router.bridge_chat_and_speak (thin delegator)
# Depends-on: app.bridge.chat_turn_guard, app.bridge.chat_speak_stream,
#             app.bridge.tts_cache, app.bridge.capability_layer, app.bridge.router (helpers, lazy)
# Last-renovated: 2026-06-16 (JOB-3: double-news-update fix)
"""
Turn handler for POST /bridge/chat-and-speak.

run_chat_and_speak() is the single entry the router delegates to. It layers two
dedupe stages over the generation body, both keyed on the client's
X-Idempotency-Key:

  1. Post-completion replay (pre-existing): a key we have ALREADY answered
     re-serves the original reply (text headers + cached audio) — no LLM run.
  2. In-flight guard (JOB-3, 2026-06-16): a same-key retry that arrives WHILE
     the first turn is still generating now JOINS it (waits, then replays)
     instead of starting a second run_astra_chat. This is the fix for the
     "one news request -> two news updates" bug: the phone's Phase-A POST loop
     re-POSTs the same key when the first response header is slow (a grounded
     reply emits none until grounding + the web_search tool loop finish), and
     stage 1 alone could not catch it because idem_put only runs AFTER
     generation completes. See app.bridge.chat_turn_guard.

Generation itself (_generate_chat_and_speak) is a faithful move of the former
router body — same translation, capability-honesty gate, web search, run_astra_chat,
assistant-message persistence, idem_put, artifact/directive processing and
sentence-TTS streaming response. The shared helpers (_resolve_or_create_project,
_run_translation, _run_web_search) still live in router.py and are imported
lazily here to avoid an import cycle.
"""
from __future__ import annotations

import json
import logging

from sqlalchemy.orm import Session
from fastapi import Request

from .schemas import BridgeChatRequest
from . import tts_cache
from . import chat_turn_guard
from .chat_speak_stream import build_audio_response, replay_cached_reply

logger = logging.getLogger(__name__)


async def _try_replay(idempotency_key: str, db: Session, process_artifacts):
    """Return a replay StreamingResponse for an already-answered key, else None.

    Wraps the idem_get -> replay_cached_reply pair so both the post-completion
    check and the post-join check share one implementation.
    """
    replayed_id = tts_cache.idem_get(idempotency_key)
    if replayed_id is None:
        return None
    replayed = await replay_cached_reply(replayed_id, db, process_artifacts)
    if replayed is not None:
        logger.info("[bridge] chat-and-speak idempotent replay (key=%s -> msg %s)",
                    idempotency_key[:12], replayed_id)
    return replayed


async def run_chat_and_speak(
    req: BridgeChatRequest,
    request: Request,
    db: Session,
    process_artifacts,
):
    """Entry point delegated to from the /bridge/chat-and-speak endpoint."""
    idempotency_key = request.headers.get("X-Idempotency-Key")

    # Stage 1: a key we've already answered re-serves the original reply —
    # no second LLM run, no duplicate rows, no replay-ghost. Before side effects.
    if idempotency_key:
        replayed = await _try_replay(idempotency_key, db, process_artifacts)
        if replayed is not None:
            return replayed

    # Stage 2 (JOB-3): in-flight guard. If a turn for this key is still
    # generating, JOIN it (wait, then replay) instead of starting a second one.
    if idempotency_key and chat_turn_guard.enabled():
        is_owner = await chat_turn_guard.begin(idempotency_key)
        if not is_owner:
            # The owning turn finished while we waited; it idem_put before
            # releasing, so the original reply is now replayable.
            replayed = await _try_replay(idempotency_key, db, process_artifacts)
            if replayed is not None:
                logger.info(
                    "[bridge] chat-and-speak joined in-flight turn (key=%s) -> replay",
                    idempotency_key[:12],
                )
                return replayed
            # Degraded fallback (rare): owner produced nothing replayable
            # (e.g. timed out / errored). Generate once rather than hang.
            logger.warning(
                "[bridge] joined in-flight turn (key=%s) but nothing to replay; "
                "generating", idempotency_key[:12],
            )
            return await _generate_chat_and_speak(req, db, process_artifacts, idempotency_key)
        try:
            return await _generate_chat_and_speak(req, db, process_artifacts, idempotency_key)
        finally:
            chat_turn_guard.end(idempotency_key)

    # No idempotency key, or guard disabled: generate as a standalone turn.
    return await _generate_chat_and_speak(req, db, process_artifacts, idempotency_key)


async def _generate_chat_and_speak(
    req: BridgeChatRequest,
    db: Session,
    process_artifacts,
    idempotency_key,
):
    """Generate one chat-and-speak reply and return the streaming audio response.

    Faithful move of the former router.bridge_chat_and_speak body (everything
    after the stage-1 replay check). Behaviour is unchanged.
    """
    from app.memory.service import create_message
    from app.memory.schemas import MessageCreate
    from app.memory._service_utils_2 import list_messages
    # Shared helpers still live in router.py; lazy import dodges the cycle.
    from app.bridge.router import (
        _resolve_or_create_project, _run_translation, _run_web_search,
    )

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
        if idempotency_key:
            tts_cache.idem_put(idempotency_key, bt_msg.id)
        return build_audio_response(
            message_id=bt_msg.id,
            display_text=build_turn.reply,
            project_id=project.id,
            project_name=project.name,
            attachments_payload="[]",
        )

    from app.bridge.capability_honesty import (
        is_unsupported_on_bridge, get_unsupported_message,
    )

    domain_context, translation_result, domain_info = _run_translation(req.message, db)

    resolved_intent = (
        translation_result.resolved_intent.value
        if translation_result and translation_result.resolved_intent
        else None
    )

    if is_unsupported_on_bridge(resolved_intent):
        honest_reply = get_unsupported_message(resolved_intent)
        logger.info("[bridge] chat-and-speak: blocked unsupported %s", resolved_intent)
        create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=honest_reply, provider="bridge", model="capability-gate",
        ))

    web_search_context, search_executed, search_succeeded, honest_early_reply = (
        await _run_web_search(resolved_intent, translation_result, req.message)
    )

    from app.bridge.capability_layer import run_astra_chat

    if is_unsupported_on_bridge(resolved_intent):
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

    project_id = project.id
    project_name = project.name

    assistant_message = create_message(db, MessageCreate(
        project_id=project_id, role="assistant",
        content=full_text, provider=provider, model=model,
    ))

    # Map the client's idempotency key to this reply so a retried request
    # (OkHttp resend after connection loss) replays instead of re-running.
    if idempotency_key:
        tts_cache.idem_put(idempotency_key, assistant_message.id)

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
    display_text, attachments = process_artifacts(full_text)
    attachments_payload = json.dumps([a.model_dump() for a in attachments])

    # Phone-action directives ([[astra:...]] — see directives.py): stripped
    # before the text reaches the screen or the voice, sent via X-Directives.
    from .directives import extract_directives, strip_directives, directives_payload
    directives = extract_directives(display_text)
    if directives:
        display_text = strip_directives(display_text)

    # Stream sentence-MP3s to the phone while teeing them into the
    # per-message tts_cache; survives client disconnects (see
    # chat_speak_stream.build_audio_response for the full contract).
    return build_audio_response(
        message_id=assistant_message.id,
        display_text=display_text,
        project_id=project_id,
        project_name=project_name,
        attachments_payload=attachments_payload,
        directives_json=directives_payload(directives),
    )


__all__ = ["run_chat_and_speak"]
