# FILE: app/bridge/chat_helpers.py
# Purpose: Shared bridge chat-pipeline helpers (artifact strip, project resolve, translation, web search).
# Called-by: app.bridge.router (shim re-export), app.bridge.chat_speak_stream, app.bridge.missed_replies
# Depends-on: app.bridge.schemas (+ lazy app.memory/app.translation/app.llm.web_search inside the fns)
# Last-renovated: 2026-06-21
"""Shared chat-pipeline helpers for the Astra Bridge API.

Split out of router.py (batch 3, 2026-06-21) as a pure LEAF so the bridge
router, chat_speak_stream and missed_replies can all share them without the
helpers living inside an endpoint module. Re-exported by app.bridge.router.
"""
from __future__ import annotations

import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session

from .schemas import BridgeChatRequest, BridgeArtifactRef

logger = logging.getLogger(__name__)


def _process_artifacts(content: str) -> tuple[str, list[BridgeArtifactRef]]:
    """Extract artifact markers from message content.

    Returns (stripped_content, refs). The stripped content has the markers
    removed so the phone displays clean text in the chat bubble and TTS
    speaks clean audio. The refs are enriched with size + mime via stat
    on the actual file and given absolute /bridge/artifacts URLs so the
    phone can fetch them via the auth-gated endpoint.

    Empty refs list when the content has no markers, which is the
    overwhelmingly common case. Both outputs are safe to pass to clients
    even when no artifacts exist.
    """
    from app.bridge.artifacts import (
        extract_artifacts, strip_artifacts, enrich_artifact_ref,
    )
    refs = extract_artifacts(content)
    if not refs:
        return content, []
    stripped = strip_artifacts(content)
    out: list[BridgeArtifactRef] = []
    for r in refs:
        enriched = enrich_artifact_ref(r)
        out.append(BridgeArtifactRef(
            kind=enriched.kind,
            filename=enriched.filename,
            size_bytes=enriched.size_bytes,
            mime_type=enriched.mime_type,
            url=f"/bridge/artifacts/{enriched.kind}/{enriched.filename}",
        ))
    return stripped, out


def _resolve_or_create_project(req: BridgeChatRequest, db: Session):
    from app.memory.service import get_project
    from app.memory.schemas import ProjectCreate
    from app.memory._service_utils_2 import create_project, get_project_by_name

    project = None
    if req.project_id:
        project = get_project(db, req.project_id)
        if not project:
            raise HTTPException(404, f"Project {req.project_id} not found")

    if not project:
        name_preview = req.message[:50].strip()
        if len(req.message) > 50:
            name_preview += "..."
        existing = get_project_by_name(db, name_preview)
        if existing:
            project = existing
            logger.debug("[bridge] Reusing existing project %d: %s", project.id, project.name)
        else:
            project = create_project(db, ProjectCreate(
                name=name_preview,
                description="Chat from Astra Bridge (phone)",
                type="bridge",
            ))
            logger.info("[bridge] Created project %d: %s", project.id, project.name)
    return project


def _run_translation(message: str, db: Session, project_id: int | None = None) -> tuple:
    domain_context = ""
    translation_result = None
    domain_info = None
    try:
        from app.translation import translate_message_sync
        from app.translation.modes import UIContext
        bridge_ctx = UIContext(in_job_config=True)
        translation_result = translate_message_sync(message, ui_context=bridge_ctx)
        if (translation_result
                and translation_result.resolved_intent
                and translation_result.resolved_intent.value.startswith("DOMAIN_")):
            from app.llm.translation_routing import intent_to_routing_info
            domain_info = intent_to_routing_info(translation_result.resolved_intent)
            if domain_info and domain_info.get("type") == "domain_chat":
                from app.llm.routing.domain_context import get_domain_context
                domain_context = get_domain_context(domain_info["domain"], db)
                logger.info("[bridge] Domain detected: %s (%d chars context)",
                           domain_info["domain"], len(domain_context))
    except Exception as e:
        logger.info("[bridge] Translation layer error: %s", e)

    # Record the classifier decision so the chat LLM can accurately answer
    # "why did you search the web?" on the PHONE path too (2026-06-24). The read
    # side (build_decisions_block via build_full_context) was already shared; only
    # this WRITE was desktop-only (stream_router). Mirrors that call, keyed on
    # str(project_id) which is what the read looks up. Best-effort: a failure here
    # never breaks a reply, and project_id=None (legacy callers) simply skips it.
    if project_id is not None and translation_result is not None and translation_result.resolved_intent:
        try:
            from app.translation.recent_decisions import (
                record_decision, STATUS_PENDING, STATUS_AUTO,
            )
            _ctx = translation_result.extracted_context or {}
            _rd_gate = getattr(translation_result, "confirmation_gate", None)
            _rd_pending = bool(
                _rd_gate is not None
                and getattr(_rd_gate, "requires_confirmation", False)
                and not getattr(_rd_gate, "passed", False)
            )
            record_decision(
                conversation_id=str(project_id),
                intent=translation_result.resolved_intent.value,
                rule_name=_ctx.get("_classifier_rule") or "unknown",
                reason=_ctx.get("_classifier_reason") or "",
                message_excerpt=message,
                confidence=translation_result.intent_confidence,
                status=STATUS_PENDING if _rd_pending else STATUS_AUTO,
            )
        except Exception as _rd_err:
            logger.debug("[bridge] Failed to record classifier decision: %s", _rd_err)

    return domain_context, translation_result, domain_info


async def _run_web_search(
    resolved_intent: str | None,
    translation_result,
    message: str,
) -> tuple:
    from app.bridge.capability_honesty import get_search_failed_message

    web_search_context = ""
    search_executed = False
    search_succeeded = False
    early_reply = None

    if resolved_intent not in ("WEB_SEARCH", "DEEP_RESEARCH"):
        return web_search_context, search_executed, search_succeeded, early_reply

    search_executed = True
    try:
        from app.llm.web_search import search_and_answer, WebSearchRequest
        search_query = (
            (translation_result.extracted_context or {}).get("extracted_query", "")
            or message
        )
        logger.info("[bridge] Web search triggered: %s", search_query[:80])
        search_result = await search_and_answer(
            WebSearchRequest(query=search_query, max_results=5)
        )
        if search_result and search_result.ok:
            search_succeeded = True
            sources_text = "\n".join(
                f"- [{s.title}]({s.url}): {s.snippet}"
                for s in search_result.sources[:5]
            )
            web_search_context = (
                f"## Web Search Results for: {search_result.query}\n\n"
                f"{search_result.answer}\n\n"
                f"Sources:\n{sources_text}\n"
            )
            logger.info("[bridge] Web search: %d sources, %d chars",
                       len(search_result.sources), len(web_search_context))
        else:
            logger.info("[bridge] Web search returned no results")
            early_reply = get_search_failed_message()
    except Exception as ws_err:
        logger.info("[bridge] Web search failed: %s", ws_err)
        early_reply = get_search_failed_message()

    return web_search_context, search_executed, search_succeeded, early_reply
