# FILE: app/bridge/chat_endpoints.py
# Purpose: Bridge /chat generation pipeline body (run_bridge_chat).
# Called-by: app.bridge.router.bridge_chat (lazy delegate)
# Depends-on: app.bridge.chat_helpers, app.bridge.schemas, app.bridge.llm_helpers
# Last-renovated: 2026-06-21
"""The /chat generation pipeline for the Astra Bridge API.

Split out of router.py (batch 3, 2026-06-21). The @router.post('/chat')
decorator stays in router.py and lazily delegates here (same pattern as
bridge_chat_and_speak -> chat_speak_stream.run_chat_and_speak), so there is
no router<->chat_endpoints module-load cycle.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from .schemas import BridgeChatRequest, BridgeChatResponse
from .llm_helpers import push_desktop_navigation
from .markdown_sanitize import for_display
from .chat_helpers import (
    _process_artifacts,
    _resolve_or_create_project,
    _run_translation,
    _run_web_search,
)

logger = logging.getLogger(__name__)


async def run_bridge_chat(req: BridgeChatRequest, db: Session) -> BridgeChatResponse:
    from app.memory.service import create_message
    from app.memory.schemas import MessageCreate
    from app.memory._service_utils_2 import list_messages

    project = _resolve_or_create_project(req, db)
    # Phase 3: scan for identity facts (DOB, birthplace, location, etc.)
    # Fire-and-forget — writes to Tier 1 identity store if matched.
    from app.bridge.identity_hook import capture_from_bridge_message
    capture_from_bridge_message(req.message, source_label=f"bridge:{project.id}")

    # Phase 7: capture fragment for long-term associative memory.
    # Every message embedded + clustered into themes; decays over time.
    from app.bridge.identity_hook import capture_fragment_from_bridge
    capture_fragment_from_bridge(req.message, project_id=project.id)

    create_message(db, MessageCreate(
        project_id=project.id, role="user",
        content=req.message, provider="bridge", model="phone-input",
    ))

    history = list_messages(db, project.id, limit=20)
    history_messages = [
        {"role": m.role, "content": m.content}
        for m in history if m.role in ("user", "assistant")
    ]

    # Build-command parity (Phase 1): "build the apk" and its "yes"/"no" voice
    # confirmation are handled here, before translation, so the phone triggers
    # the same host build flow the desktop uses. The build runs async; the
    # completion is delivered (and spoken) via the /bridge/missed-replies poller.
    from app.bridge.build_actions import maybe_handle_build_turn
    build_turn = maybe_handle_build_turn(req.message, project.id, db)
    if build_turn is not None:
        bt_message = create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=build_turn.reply, provider="bridge", model=build_turn.model,
        ))
        return BridgeChatResponse(
            reply=build_turn.reply, project_id=project.id,
            project_name=project.name, domain="",
            message_id=bt_message.id,
            provider="bridge", model=build_turn.model,
        )

    domain_context, translation_result, domain_info = _run_translation(req.message, db, project_id=project.id)

    from app.bridge.capability_honesty import (
        is_unsupported_on_bridge, get_unsupported_message,
    )

    resolved_intent = (
        translation_result.resolved_intent.value
        if translation_result and translation_result.resolved_intent
        else None
    )

    if is_unsupported_on_bridge(resolved_intent):
        reply = get_unsupported_message(resolved_intent)
        logger.info("[bridge] Blocked unsupported intent: %s", resolved_intent)
        gate_message = create_message(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=reply, provider="bridge", model="capability-gate",
        ))
        return BridgeChatResponse(
            reply=reply, project_id=project.id, project_name=project.name,
            domain=domain_info.get("domain", "") if domain_info else "",
            message_id=gate_message.id,
            provider="bridge", model="capability-gate",
        )

    web_search_context, search_executed, search_succeeded, early_reply = (
        await _run_web_search(resolved_intent, translation_result, req.message)
    )

    if early_reply:
        from app.memory.service import create_message as _cm
        early_message = _cm(db, MessageCreate(
            project_id=project.id, role="assistant",
            content=early_reply, provider="bridge", model="search-gate",
        ))
        return BridgeChatResponse(
            reply=early_reply, project_id=project.id, project_name=project.name,
            domain=domain_info.get("domain", "") if domain_info else "",
            message_id=early_message.id,
            provider="bridge", model="search-gate",
        )

    from app.bridge.capability_layer import run_astra_chat
    from app.bridge.attachment_describe import augment_user_message

    # Same attachment handling as bridge_chat_and_speak: prepend vision /
    # document text to the LLM-input message, keep req.message clean for
    # the DB record and identity hook. No-op when attachment_ids is empty.
    llm_input_message = augment_user_message(req, history_messages)

    result = await run_astra_chat(
        message=llm_input_message,
        project_id=project.id,
        history=history_messages,
        db=db,
        source="bridge",
        domain_context=domain_context,
        translation_result=translation_result,
        web_search_context=web_search_context,
        search_executed=search_executed,
        search_succeeded=search_succeeded,
        raw_message=req.message,
    )

    reply = result["reply"]
    provider = result["provider"]
    model = result["model"]

    assistant_message = create_message(db, MessageCreate(
        project_id=project.id, role="assistant",
        content=reply, provider=provider, model=model,
    ))

    # v2026-06-10: session + summary tracking for bridge conversations.
    # The hook previously only ran on the desktop stream path, so phone
    # chats never got conversation_sessions rows or rolling summaries.
    try:
        from app.memory.integration import record_session_activity
        record_session_activity(
            project_id=project.id, provider=provider,
            model=model, db_session=db,
        )
    except Exception as _sess_err:
        logger.debug("[bridge] session tracking failed: %s", _sess_err)

    _detected_domain = domain_info.get("domain") if domain_info else None
    if _detected_domain:
        push_desktop_navigation(_detected_domain)

    # Process artifact markers: strip from reply for clean display, build
    # BridgeArtifactRef list for the response. The DB still holds the raw
    # reply (with markers) so history reload can re-emit the same chips.
    display_reply, attachments = _process_artifacts(reply)

    # Phone-action directives ([[astra:...]] markers — see directives.py):
    # stripped from the visible reply, forwarded for the app to execute.
    from .directives import extract_directives, strip_directives
    directives = extract_directives(display_reply)
    if directives:
        display_reply = strip_directives(display_reply)

    # for_display: the phone renders this string raw (no markdown engine).
    # The row stored above keeps the raw reply — sanitise on the way out only.
    return BridgeChatResponse(
        reply=for_display(display_reply), project_id=project.id,
        project_name=project.name, domain=_detected_domain,
        attachments=attachments,
        directives=directives,
        message_id=assistant_message.id,
        provider=provider, model=model,
    )
