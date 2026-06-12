# FILE: app/llm/stream_router.py
# Purpose: Streaming endpoints for real-time LLM responses.
# Called-by: main, tests.test_stream_router
# Depends-on: app.auth, app.auth.middleware, app.cloud.build_and_deploy, app.db (+24 more)
# Last-renovated: 2026-06-11
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

v5.5 (2026-03-23): ASTRA commands bypass debug lock
    - Explicit "Astra, command:" messages now punch through debug_locked routing
    - Fixes architecture scan, pipeline, sandbox commands being swallowed by debug chat
    - Uses COMMAND_WAKE_PATTERN from translation/modes.py for detection

v5.1 (2026-01-31): CRITICAL FIX - Explicit commands bypass flow state interception
    - _handle_flow_state_routing now checks for explicit command intents
    - RUN_CRITICAL_PIPELINE_FOR_JOB, OVERWATCHER_EXECUTE_CHANGES, etc. no longer
      get intercepted by SPEC_GATE_QUESTIONS flow state
    - Flow state interception only applies to regular chat messages

v5.0 (2026-01-20): MAJOR REFACTOR - Modularized into routing/ subpackage
    - handler_registry.py: Centralized handler imports & availability
    - command_dispatch.py: Intent → handler dispatch
    - chat_routing.py: Chat mode, normal routing, legacy triggers
    - prompt_builders.py: System prompt construction
    - rag_fallback.py: Architecture query detection

v4.14 (2026-01): Added LATEST_ARCHITECTURE_MAP and LATEST_CODEBASE_REPORT_FULL routing
v4.13 (2026-01): Added CODEBASE_REPORT command routing for hygiene/bloat/drift reports
v4.12 (2026-01): RAG fallback in _handle_normal_routing for architecture queries
v4.11 (2026-01): Split architecture map: ALL CAPS → full scan, lowercase → DB only
v4.10 (2026-01): Removed host filesystem scan (sandbox only), cleaned up routing
v4.9 (2026-01): Added ASTRA capability layer injection to system prompts
v4.8 (2026-01): Chat mode uses stage_models, added debug logging throughout
v4.7 (2026-01): Added stage tracing, routing failure visibility, model audit
v4.6 (2026-01): Refactored into modules (stream_handlers, translation_routing, legacy_triggers)
v4.5 (2026-01): DB-backed validated spec lookup (survives restarts)
v4.4 (2026-01): Added SPEC_VALIDATED flow state check for Critical Pipeline routing
v4.3 (2026-01): Added Spec Gate, Critical Pipeline, Overwatcher stream handlers
v4.2 (2026-01): Added Weaver stream handler for spec building
v4.1 (2026-01): CRITICAL FIX - CHAT mode returns early, bypasses job classification
v4.0 (2026-01): ASTRA Translation Layer integration - prevents misfires
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult

from app.llm.audit_logger import get_audit_logger
from app.llm.stream_utils import make_session_id

# =============================================================================
# MODULAR IMPORTS (v5.0)
# =============================================================================

from app.llm.translation_routing import (
    TRANSLATION_LAYER_AVAILABLE,
    TranslationMode,
    CanonicalIntent,
    route_via_translation_layer,
    _get_spec_gate_config,
    _get_critical_pipeline_config,
    _get_weaver_config,
)

# v5.5: ASTRA command wake pattern — used to bypass debug lock for explicit commands
from app.translation.modes import COMMAND_WAKE_PATTERN as _ASTRA_COMMAND_PATTERN

# Handler registry - centralized imports and availability flags
from app.llm.routing.handler_registry import (
    _WEAVER_AVAILABLE,
    _SPEC_GATE_STREAM_AVAILABLE,
    _CRITICAL_PIPELINE_AVAILABLE,
    _FLOW_STATE_AVAILABLE,
    _SPEC_SERVICE_AVAILABLE,
    _STAGE_TRACE_AVAILABLE,
    # Flow state functions
    get_active_flow,
    SpecFlowStage,
    check_weaver_answer_keywords,
    capture_weaver_answers,
    # Spec service
    get_latest_validated_spec,
    # Handlers
    generate_weaver_stream,
    generate_spec_gate_stream,
    generate_critical_pipeline_stream,
    generate_feedback_stream,
    generate_confirmation_stream,
    # Stage trace
    get_env_model_audit,
)

# Command dispatch
from app.llm.routing.command_dispatch import (
    handle_command_execution,
    create_stage_trace,
    log_routing_failure,
)

# Chat and normal routing
from app.llm.routing.chat_routing import (
    handle_chat_mode,
    handle_normal_routing,
    handle_legacy_triggers,
)
from app.llm._stream_router_utils import StreamRequest, _EXPLICIT_COMMAND_INTENTS, _WEAVER_EXIT_INTENTS, _create_handler_unavailable_response, _create_no_spec_error_response, _handle_blocked_command, _handle_db_spec_routing, _handle_flow_state_routing

# v5.4: Memory integration hooks (confidence learning, preference capture, context extraction)
from app.memory.integration import on_intent_confirmed, after_user_message


router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST MODEL
# =============================================================================


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@router.post("/chat")
async def stream_chat(
    req: StreamRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """Main streaming chat endpoint with translation layer routing."""
    
    # Validate project
    from app.memory.service import get_project
    project = get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # v1.0 (2026-05-24): Working-set context var.  Sets the current
    # project_id (and an informational model hint) so tool executions
    # in this turn auto-register their touched files.  Cleared
    # implicitly when the request handler returns (each FastAPI
    # request gets its own contextvars scope).
    try:
        from app.memory.working_set import set_current as _ws_set_current
        _model_hint = (getattr(req, "model", None) or "")
        _ws_set_current(req.project_id, _model_hint)
    except Exception as _ws_err:
        logger.debug("[stream_router] working-set context bind failed: %s", _ws_err)

    # v1.1 (2026-05-24): Bootstrap — if the user's message references a
    # known file ("the HTML", "the dashboard"), auto-discover the file
    # in the project's canonical folder and register it into the working
    # set BEFORE prompt building.  Strict rules: exactly-one match
    # only, never auto-pick on ambiguity.
    try:
        from app.memory._working_set_bootstrap import bootstrap_from_message
        _bs_registered = bootstrap_from_message(
            project_id=req.project_id,
            project_name=getattr(project, "name", None),
            message=req.message,
        )
        if _bs_registered:
            logger.info(
                "[stream_router] Working-set bootstrap registered %d file(s)",
                _bs_registered,
            )
    except Exception as _bs_err:
        logger.debug("[stream_router] working-set bootstrap failed: %s", _bs_err)

    # =========================================================================
    # v5.5: ASTRA COMMAND BYPASS — explicit commands punch through debug lock
    # "Astra, command: CREATE ARCHITECTURE MAP" must reach the translation
    # layer even when the sidebar is debug-locked, otherwise all local tool
    # routing (architecture scan, pipeline, sandbox, etc.) is unreachable
    # from the Debug tab.
    # =========================================================================
    _is_explicit_astra_command = bool(_ASTRA_COMMAND_PATTERN.match(req.message))
    # Also catch implicit command patterns (e.g. bare "CREATE ARCHITECTURE MAP",
    # "Astra, CREATE ARCHITECTURE MAP") — these also need translation layer routing
    if not _is_explicit_astra_command:
        from app.translation.modes import IMPLICIT_COMMAND_PATTERNS
        import re as _re
        _msg_stripped = req.message.strip()
        _is_explicit_astra_command = any(
            _re.match(p, _msg_stripped) for p in IMPLICIT_COMMAND_PATTERNS
        )
    if _is_explicit_astra_command:
        logger.info("[stream_router] Explicit ASTRA command detected — bypassing debug lock")

    # =========================================================================
    # v2.1 DEBUG LOCK: Force Gemini + tools + RAG when sidebar is debug-locked
    # =========================================================================
    if req.debug_locked and not _is_explicit_astra_command:
        logger.info("[stream_router] Debug lock active — routing to debug chat with Gemini + tools")
        from app.debug.debug_chat import stream_debug_locked
        return StreamingResponse(
            stream_debug_locked(
                db=db,
                project_id=req.project_id,
                message=req.message,
                panel_history=req.panel_history or [],
                provider=req.provider or "openai",
                model=req.model or "gpt-5.4",
                debug_project_id=req.debug_project_id,
                video_file_uri=req.video_file_uri,
                video_mime_type=req.video_mime_type,
                video_local_path=req.video_local_path,
                file_upload_uri=req.file_upload_uri,
                file_upload_mime=req.file_upload_mime,
                file_upload_name=req.file_upload_name,
                file_upload_local_path=req.file_upload_local_path,
                file_upload_gemini_name=req.file_upload_gemini_name,
                documents=req.documents,
            ),
            media_type="text/event-stream",
        )
    # =========================================================================
    # v6.0 (2026-05-24): IMAGE-BEARING TURN INTERCEPT
    # When a chat request carries a freshly-uploaded image (Gemini Files API
    # ref present on the request), bypass MODEL_SELECT and route the entire
    # turn through Gemini multimodal + tools. Avoids the vision-to-text-to-GPT
    # hop, gives Gemini the ability to act on what it sees (file edits, tool
    # calls) in a single pass. Text-only turns continue to existing routing.
    # =========================================================================
    from app.llm.routing.image_chat_routing import has_image_attachment, stream_image_chat
    if has_image_attachment(req):
        logger.info("[stream_router] Image attachment detected — routing to Gemini multimodal+tools")
        return StreamingResponse(
            stream_image_chat(req=req, db=db),
            media_type="text/event-stream",
        )


    # =========================================================================
    # v11.2: BUILD-DEPLOY INTERCEPT — now goes through confirmation gate.
    # The gate learns over time: after 5 approvals with 0 rejections,
    # it auto-approves and stops asking. Rejections teach it to be cautious.
    # =========================================================================
    from app.llm.routing.chat_routing import _detect_build_deploy_intent
    if _detect_build_deploy_intent(req.message):
        logger.info("[stream_router] Build-deploy intent detected — checking confirmation gate")

        # Check if user already confirmed this action
        if req.confirmed_intent == "BUILD_AND_DEPLOY":
            logger.info("[stream_router] Build-deploy confirmed — executing")
            # Log the approval for learning
            try:
                from app.llm.routing.confirmation_gate import process_confirmation_response, _make_pattern_key
                pattern_key = _make_pattern_key("intent_routing", "BUILD_AND_DEPLOY", req.message)
                process_confirmation_response(
                    pattern_key=pattern_key,
                    gate_type="intent_routing",
                    proposed_action="BUILD_AND_DEPLOY",
                    approved=True,
                    original_message=req.message,
                    confidence=0.8,
                )
            except Exception as _log_err:
                logger.warning("[stream_router] Failed to log build-deploy approval: %s", _log_err)

            from app.cloud.build_and_deploy import build_and_deploy

            async def _build_deploy_stream():
                import json as _json
                _sse = lambda obj: f"data: {_json.dumps(obj)}\n\n"
                yield _sse({"type": "metadata", "provider": "local", "model": "build-deploy"})
                yield _sse({"type": "token", "content": "Building APK...\n"})
                result = await build_and_deploy(text=req.message)
                if result.get("build", {}).get("success"):
                    _size_kb = result["build"].get("apk_size", 0) // 1024
                    yield _sse({"type": "token", "content": f"Build successful ({_size_kb}KB).\n"})
                if result.get("cloud", {}).get("success"):
                    _prov = result["cloud"].get("provider", "cloud")
                    yield _sse({"type": "token", "content": f"Uploaded to {_prov}.\n\n"})
                elif result.get("upload_failed"):
                    yield _sse({"type": "token", "content": "Cloud upload failed. APK ready locally.\n\n"})
                msg = result.get("message", result.get("error", "Unknown result"))
                yield _sse({"type": "token", "content": msg + "\n"})
                yield _sse({"type": "done", "provider": "local", "model": "build-deploy", "total_length": 0})

            return StreamingResponse(
                _build_deploy_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # Not yet confirmed — run through the confirmation gate
        try:
            from app.llm.routing.confirmation_gate import (
                should_confirm_intent_routing,
                format_confirmation_sse,
                ConfirmationRequest,
                _make_pattern_key,
                _load_pattern_history,
                CONFIDENCE_AUTO_APPROVE,
            )
            pattern_key = _make_pattern_key("intent_routing", "BUILD_AND_DEPLOY", req.message)
            history = _load_pattern_history(pattern_key)

            # Auto-approve if learned (5+ approvals, 0 rejections)
            if history.should_auto_approve:
                logger.info("[stream_router] Build-deploy auto-approved (history: %d/%d)",
                           history.approvals, history.total_asks)
            else:
                # Ask for confirmation
                confirm_req = ConfirmationRequest(
                    gate_type="intent_routing",
                    description="Build APK and upload to cloud?",
                    detail="I detected a build-and-deploy intent. This will compile the APK and upload it.",
                    original_message=req.message,
                    proposed_action="BUILD_AND_DEPLOY",
                    proposed_intent="BUILD_AND_DEPLOY",
                    confidence=0.8,
                    pattern_key=pattern_key,
                )

                async def _confirm_stream():
                    import json as _json
                    yield format_confirmation_sse(confirm_req)
                    yield f"data: {_json.dumps({'type': 'done', 'provider': 'local', 'model': 'confirmation_gate', 'total_length': 0})}\n\n"

                return StreamingResponse(
                    _confirm_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
        except ImportError:
            logger.warning("[stream_router] Confirmation gate not available, executing build-deploy directly")

        # Fallback: if confirmation gate import failed, execute directly
        from app.cloud.build_and_deploy import build_and_deploy

        async def _build_deploy_stream_fallback():
            import json as _json
            _sse = lambda obj: f"data: {_json.dumps(obj)}\n\n"
            yield _sse({"type": "metadata", "provider": "local", "model": "build-deploy"})
            yield _sse({"type": "token", "content": "Building APK...\n"})
            result = await build_and_deploy(text=req.message)
            msg = result.get("message", result.get("error", "Unknown result"))
            yield _sse({"type": "token", "content": msg + "\n"})
            yield _sse({"type": "done", "provider": "local", "model": "build-deploy", "total_length": 0})

        return StreamingResponse(
            _build_deploy_stream_fallback(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Initialize trace
    audit_logger = get_audit_logger()
    trace = audit_logger.start_trace(
        session_id=make_session_id(auth),
        project_id=req.project_id,
    )
    
    user_id = str(auth.user_id) if hasattr(auth, 'user_id') else "default"
    conversation_id = f"{req.project_id}-{make_session_id(auth)}"
    
    stage_trace = None
    
    # =========================================================================
    # CONFIRMED INTENT BYPASS (v1.9)
    # When the frontend sends confirmed_intent, skip translation entirely
    # and dispatch directly to the command handler.
    # =========================================================================
    if req.confirmed_intent:
        # v12.0: BUILD_AND_DEPLOY confirmed — redirect to the build-deploy intercept
        if req.confirmed_intent == "BUILD_AND_DEPLOY":
            # This will be caught by the build-deploy intercept section above
            # which checks req.confirmed_intent == "BUILD_AND_DEPLOY"
            # But since we're past that section, handle it here directly
            logger.info("[stream_router] BUILD_AND_DEPLOY confirmed via intent bypass")
            try:
                from app.llm.routing.confirmation_gate import process_confirmation_response, _make_pattern_key
                pattern_key = _make_pattern_key("intent_routing", "BUILD_AND_DEPLOY", req.message)
                process_confirmation_response(
                    pattern_key=pattern_key, gate_type="intent_routing",
                    proposed_action="BUILD_AND_DEPLOY", approved=True,
                    original_message=req.message, confidence=0.8,
                )
            except Exception:
                pass
            # v12.1: Recover the original message from panel history.
            # req.message is "Confirmed: ..." which has no project keywords.
            # The original build request is in the conversation history.
            _bd_text = req.message
            if hasattr(req, "panel_history") and req.panel_history:
                for _hist_msg in reversed(req.panel_history):
                    if _hist_msg.get("role") == "user" and "bridge" in _hist_msg.get("content", "").lower():
                        _bd_text = _hist_msg["content"]
                        break
                    if _hist_msg.get("role") == "user" and "build" in _hist_msg.get("content", "").lower():
                        _bd_text = _hist_msg["content"]
                        break
            logger.info("[stream_router] BUILD_AND_DEPLOY resolved text: %s", _bd_text[:80])
            from app.cloud.build_and_deploy import build_and_deploy
            async def _bd_confirmed():
                import json as _json
                _sse = lambda obj: f"data: {_json.dumps(obj)}\n\n"
                yield _sse({"type": "metadata", "provider": "local", "model": "build-deploy"})
                yield _sse({"type": "token", "content": "Building APK...\n"})
                result = await build_and_deploy(text=_bd_text)
                if result.get("build", {}).get("success"):
                    _sz = result["build"].get("apk_size", 0) // 1024
                    yield _sse({"type": "token", "content": f"Build successful ({_sz}KB).\n"})
                if result.get("cloud", {}).get("success"):
                    _pv = result["cloud"].get("provider", "cloud")
                    yield _sse({"type": "token", "content": f"Uploaded to {_pv}.\n\n"})
                elif result.get("upload_failed"):
                    yield _sse({"type": "token", "content": "Cloud upload failed. APK ready locally.\n\n"})
                msg = result.get("message", result.get("error", "Unknown result"))
                yield _sse({"type": "token", "content": msg + "\n"})
                yield _sse({"type": "done", "provider": "local", "model": "build-deploy", "total_length": 0})
            return StreamingResponse(
                _bd_confirmed(), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        # v2.2: MODEL_ESCALATION confirmed — route original message with upgraded model
        if req.confirmed_intent.startswith("MODEL_ESCALATION:"):
            action = req.confirmed_intent.split(":", 1)[1]  # e.g. "lookup_to_reasoning"
            logger.info(f"[stream_router] Model escalation confirmed: {action}")
            # Log the approval for auto-approve learning
            try:
                from app.llm.routing.confirmation_gate import process_confirmation_response, _make_pattern_key
                pattern_key = _make_pattern_key("model_escalation", action, req.message)
                process_confirmation_response(
                    pattern_key=pattern_key,
                    gate_type="model_escalation",
                    proposed_action=action,
                    approved=True,
                    original_message=req.message,
                    confidence=0.0,
                )
            except Exception as e:
                logger.warning(f"[stream_router] Failed to log escalation decision: {e}")
            # Route as chat with the escalated model
            # v3.2: Read from env instead of hardcoding OpenAI
            import os as _os
            tier_map = {
                "lookup_to_deep": (
                    _os.getenv("CHAT_DEEP_PROVIDER", "anthropic"),
                    _os.getenv("CHAT_DEEP_MODEL", "claude-opus-4-6"),
                ),
                "lookup_to_reasoning": (
                    _os.getenv("CHAT_PROVIDER", "google"),
                    _os.getenv("CHAT_MODEL", "gemini-2.5-flash"),
                ),
                "lookup_to_multimodal": ("google", "gemini-3.1-pro-preview"),
            }
            provider, model = tier_map.get(action, (None, None))
            if provider and model:
                req.provider = provider
                req.model = model
                from app.memory.service import get_project
                project = get_project(db, req.project_id)
                return handle_chat_mode(req, project, db, trace)
            # Fallback if unknown action
            logger.warning(f"[stream_router] Unknown escalation action: {action}")

        try:
            direct_intent = CanonicalIntent(req.confirmed_intent)
            logger.info(f"[stream_router] Confirmed intent bypass: {direct_intent.value}")
            stage_trace = create_stage_trace(direct_intent.value, req.project_id)
            
            from app.translation.schemas import (
                TranslationResult, ConfirmationGateResult, LatencyTier,
            )
            # v2.2: Thread extracted_query from frontend confirm roundtrip
            _extracted_ctx = {}
            if req.extracted_query:
                _extracted_ctx['extracted_query'] = req.extracted_query
            translation_result = TranslationResult(
                original_text=req.message,
                mode=TranslationMode.COMMAND_CAPABLE,
                resolved_intent=direct_intent,
                intent_confidence=1.0,
                latency_tier=LatencyTier.TIER_0_RULES,
                extracted_context=_extracted_ctx,
                confirmation_gate=ConfirmationGateResult(
                    gate_name="confirmation",
                    passed=True,
                    requires_confirmation=False,
                    awaiting_confirmation=False,
                ),
            )
            
            # v5.4: Record confirmation in confidence learning
            on_intent_confirmed(req.message, direct_intent.value, user_id)
            
            sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            response = handle_command_execution(
                req, translation_result, db, trace, conversation_id, stage_trace
            )
            if response:
                return response
            logger.warning(f"[stream_router] Confirmed intent {direct_intent.value} had no handler, falling through")
        except (ValueError, KeyError) as e:
            logger.warning(f"[stream_router] Invalid confirmed_intent '{req.confirmed_intent}': {e}")

    # =========================================================================
    # TRANSLATION LAYER ROUTING
    # =========================================================================
    
    translation_result = route_via_translation_layer(
        message=req.message,
        user_id=user_id,
        conversation_id=conversation_id,
    )

    # Record classifier decision for chat-LLM context. See
    # app/translation/recent_decisions.py — the chat LLM uses this to
    # accurately answer "why did you think that was X?" instead of
    # confabulating. Status reflects gate outcome: PENDING if user
    # confirmation is awaited, AUTO if the intent will execute directly.
    if translation_result is not None and translation_result.resolved_intent:
        try:
            from app.translation.recent_decisions import (
                record_decision, STATUS_PENDING, STATUS_AUTO,
            )
            _rd_rule = translation_result.extracted_context.get(
                "_classifier_rule"
            ) or "unknown"
            _rd_reason = translation_result.extracted_context.get(
                "_classifier_reason"
            ) or ""
            _rd_gate = translation_result.confirmation_gate
            _rd_pending = bool(
                _rd_gate is not None
                and _rd_gate.requires_confirmation
                and not _rd_gate.passed
            )
            record_decision(
                conversation_id=str(req.project_id),
                intent=translation_result.resolved_intent.value,
                rule_name=_rd_rule,
                reason=_rd_reason,
                message_excerpt=req.message,
                confidence=translation_result.intent_confidence,
                status=STATUS_PENDING if _rd_pending else STATUS_AUTO,
            )
        except Exception as _rd_err:
            logger.debug(
                "[stream_router] Failed to record classifier decision: %s",
                _rd_err,
            )

    if translation_result is not None:
        
        # Create stage trace for COMMAND mode
        if translation_result.mode == TranslationMode.COMMAND_CAPABLE:
            command_type = translation_result.resolved_intent.value if translation_result.resolved_intent else "unknown"
            stage_trace = create_stage_trace(command_type, req.project_id, req.continue_job_id)
            
            if _STAGE_TRACE_AVAILABLE:
                env_audit = get_env_model_audit()
                print(f"[MODEL_ENV_AUDIT] {env_audit}")
        
        # =================================================================
        # FLOW STATE: Route clarifications to Spec Gate
        # v5.1: Pass translation_result so explicit commands can bypass
        # =================================================================
        response = _handle_flow_state_routing(req, db, trace, conversation_id, stage_trace, translation_result)
        if response:
            return response
        
        # =================================================================
        # WEAVER AUTO-REWEAVE (v5.2)
        # If user replies during active weaver flow, auto-route to UPDATE
        # =================================================================
        from app.llm.routing.weaver_reweave_handler import handle_weaver_design_questions
        response = handle_weaver_design_questions(req, db, trace, stage_trace, translation_result)
        if response:
            return response
        
        # =================================================================
        # DB-BACKED SPEC: Route "critical pipeline" to Critical Pipeline
        # v5.3: Pass translation_result to prevent chat-mode false positives
        # =================================================================
        response = _handle_db_spec_routing(req, db, trace, conversation_id, stage_trace, translation_result)
        if response:
            return response
        
        # =================================================================
        # v3.0 DOMAIN CHAT: Route to domain-aware LLM with real data
        # Emits domain_navigate SSE so frontend auto-switches tabs.
        # =================================================================
        if (translation_result.resolved_intent
                and translation_result.resolved_intent.value.startswith("DOMAIN_")):
            from app.llm.translation_routing import intent_to_routing_info
            _domain_info = intent_to_routing_info(translation_result.resolved_intent)
            if _domain_info and _domain_info.get("type") == "domain_chat":
                _domain_name = _domain_info["domain"]
                logger.info("[stream_router] Domain chat: %s", _domain_name)

                # Lifestyle quick-commands (2026-06-11): deterministic voice
                # actions — "copy yesterday's food into today", "I'm eating
                # the same as yesterday" — execute directly against the
                # nutrition service. No LLM round-trip, no misrouting; the
                # diary updates instantly and ASTRA confirms in one line.
                if _domain_name == "lifestyle":
                    _quick_reply = None
                    try:
                        from app.lifestyle.nutrition_copy import try_quick_nutrition_command
                        _quick_reply = try_quick_nutrition_command(req.message, db)
                    except Exception as _qn_err:
                        logger.warning("[stream_router] lifestyle quick-command failed: %s", _qn_err)
                    if _quick_reply is None:
                        # Energy debrief (2026-06-11): "heavy day today" etc. —
                        # deterministic, logs effort + reprices the day's burn.
                        try:
                            from app.lifestyle.energy import try_quick_debrief
                            _quick_reply = try_quick_debrief(db, req.message)
                        except Exception as _qd_err:
                            logger.warning("[stream_router] energy debrief failed: %s", _qd_err)
                    if _quick_reply:
                        after_user_message(
                            req.message,
                            project_id=str(req.project_id),
                            user_id=user_id,
                            provider=getattr(req, 'provider', None),
                            model=getattr(req, 'model', None),
                            db_session=db,
                        )

                        async def _quick_stream(_text=_quick_reply):
                            import json as _json
                            _sse = lambda obj: f"data: {_json.dumps(obj)}\n\n"
                            yield _sse({"type": "domain_navigate", "domain": "lifestyle", "job_type": "health_fitness"})
                            yield _sse({"type": "metadata", "provider": "local", "model": "lifestyle-quick-command"})
                            yield _sse({"type": "token", "content": _text})
                            yield _sse({"type": "done", "provider": "local", "model": "lifestyle-quick-command", "total_length": len(_text)})

                        return StreamingResponse(
                            _quick_stream(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )

                try:
                    from app.llm.routing.domain_context import get_domain_context
                    _domain_ctx = get_domain_context(_domain_name, db)
                except Exception as _dc_err:
                    logger.warning("[stream_router] Domain context failed: %s", _dc_err)
                    _domain_ctx = f"[{_domain_name} data unavailable]"

                # Inject domain context into the request as extra system context
                if not req.panel_history:
                    req.panel_history = []
                req.panel_history = [
                    {"role": "user", "content": f"[ASTRA domain context for {_domain_name}]\n{_domain_ctx}"},
                    {"role": "assistant", "content": f"I have the latest {_domain_name} data. What would you like to know?"},
                ] + req.panel_history

                after_user_message(
                    req.message,
                    project_id=str(req.project_id),
                    user_id=user_id,
                    provider=getattr(req, 'provider', None),
                    model=getattr(req, 'model', None),
                    db_session=db,
                )

                # Map domain to frontend job_type for tab navigation
                _DOMAIN_TO_JOB = {
                    "finance": "accounts",
                    "investments": "investments",
                    "content": "content",
                    "social": "social_media",
                    "lifestyle": "health_fitness",
                    "debug": "debug",
                    "education": "education",
                    "builds": "project_builds",
                }
                _job_type = _DOMAIN_TO_JOB.get(_domain_name, _domain_name)

                # Get the chat response (StreamingResponse)
                _chat_response = handle_chat_mode(req, project, db, trace)

                # Wrap it: emit domain_navigate first, then stream the chat
                async def _domain_stream(_chat_resp=_chat_response, _jt=_job_type, _dn=_domain_name):
                    # Emit navigation event so frontend switches tab
                    import json as _json
                    yield f"data: {_json.dumps({'type': 'domain_navigate', 'domain': _dn, 'job_type': _jt})}\n\n"
                    # Then stream the normal chat response
                    async for chunk in _chat_resp.body_iterator:
                        yield chunk

                return StreamingResponse(
                    _domain_stream(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

        # =================================================================
        # CHAT MODE: Return early with lightweight model
        # =================================================================
        if translation_result.mode == TranslationMode.CHAT:
            logger.info("[translation] CHAT MODE - bypassing job classification")
            # v5.4: Capture preferences and context from chat messages
            # v10.0: Pass db_session for conversation session management
            after_user_message(
                req.message,
                project_id=str(req.project_id),
                user_id=user_id,
                provider=getattr(req, 'provider', None),
                model=getattr(req, 'model', None),
                db_session=db,
            )
            return handle_chat_mode(req, project, db, trace)
        
        # =================================================================
        # FEEDBACK MODE
        # =================================================================
        if translation_result.mode == TranslationMode.FEEDBACK:
            return StreamingResponse(
                generate_feedback_stream(
                    project_id=req.project_id,
                    message=req.message,
                    translation_result=translation_result,
                    db=db,
                    trace=trace,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # =================================================================
        # COMMAND MODE
        # =================================================================
        if translation_result.mode == TranslationMode.COMMAND_CAPABLE:
            # v5.3: Save user command message to history for cross-model context
            try:
                from app.memory import service as _mem_svc, schemas as _mem_schemas
                # v7.1: Inline document content before persisting
                try:
                    from app.llm.routing.chat_routing import _resolve_message_with_documents
                    _resolved = _resolve_message_with_documents(req)
                except Exception:
                    _resolved = req.message
                _mem_svc.create_message(
                    db,
                    _mem_schemas.MessageCreate(
                        project_id=req.project_id,
                        role="user",
                        content=_resolved,
                        provider="system",
                    ),
                )
            except Exception:
                pass  # Non-fatal — don't block command execution

            # Phase 7 fix: CHAT_ONLY command path also needs memory capture,
            # otherwise identity facts and fragments never get recorded for
            # conversational messages that arrive via command mode (e.g. STT).
            try:
                after_user_message(
                    req.message,
                    project_id=str(req.project_id),
                    user_id=user_id,
                    provider=getattr(req, 'provider', None),
                    model=getattr(req, 'model', None),
                    db_session=db,
                )
            except Exception as _aum_err:
                logger.debug("[integration] command-mode after_user_message failed: %s", _aum_err)
            
            # Awaiting confirmation
            if (translation_result.confirmation_gate and 
                translation_result.confirmation_gate.awaiting_confirmation):
                return StreamingResponse(
                    generate_confirmation_stream(
                        project_id=req.project_id,
                        message=req.message,
                        translation_result=translation_result,
                        db=db,
                        trace=trace,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            
            # Execute approved commands
            if translation_result.should_execute:
                # v2.1: Confirmation gate — check if this routing decision needs user approval
                try:
                    from app.llm.routing.confirmation_gate import (
                        should_confirm_intent_routing,
                        format_confirmation_sse,
                    )
                    intent_str = translation_result.resolved_intent.value if translation_result.resolved_intent else None
                    if intent_str:
                        confirm_req = should_confirm_intent_routing(
                            intent=intent_str,
                            confidence=translation_result.intent_confidence,
                            message=req.message,
                        )
                        if confirm_req:
                            # v2.2: Thread extracted_query so frontend can send it back
                            _eq = translation_result.extracted_context.get('extracted_query')
                            # Emit confirmation event and wait for user response
                            async def _confirm_stream(_eq=_eq):
                                yield format_confirmation_sse(confirm_req, extracted_query=_eq)
                                import json as _json
                                yield f"data: {_json.dumps({'type': 'done', 'provider': 'local', 'model': 'confirmation_gate', 'total_length': 0})}\n\n"
                            return StreamingResponse(
                                _confirm_stream(),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                            )
                except ImportError:
                    pass  # Gate not available, proceed normally

                # v5.4: Record successful intent resolution in confidence learning
                intent_val = translation_result.resolved_intent.value if translation_result.resolved_intent else None
                if intent_val:
                    on_intent_confirmed(req.message, intent_val, user_id)
                # v10.0: Pass db_session for conversation session management
                after_user_message(
                    req.message,
                    project_id=str(req.project_id),
                    user_id=user_id,
                    provider=getattr(req, 'provider', None),
                    model=getattr(req, 'model', None),
                    db_session=db,
                )
                
                response = handle_command_execution(
                    req, translation_result, db, trace, conversation_id, stage_trace
                )
                if response:
                    return response
            
            # Command blocked - handle high-stakes commands specially
            if not translation_result.should_execute:
                response = _handle_blocked_command(translation_result, stage_trace)
                if response:
                    return response
    
    # =========================================================================
    # LEGACY FALLBACK (translation layer unavailable)
    # =========================================================================
    
    if translation_result is None:
        response = handle_legacy_triggers(req, db, trace)
        if response:
            return response
    
    # =========================================================================
    # PROJECT STALENESS CHECK (v11.0)
    # =========================================================================
    try:
        from app.project_registry.chat_injection import get_staleness_sse_events
        _staleness_events = get_staleness_sse_events(db, req.message)
    except Exception:
        _staleness_events = None

    # =========================================================================
    # v16.0 (2026-05-01): Image generation no longer intercepted here.
    # Image gen is handled inside handle_chat_mode — the chat LLM emits an
    # [IMAGE_PROMPT]: marker that gets pulled out by image_extractor and
    # fired straight at gpt-image-2. This block previously bypassed the
    # chat LLM entirely (sending the raw user message to a fresh Gemini
    # synth that lost all conversation context). Removed.
    # =========================================================================

    # =========================================================================
    # NORMAL ROUTING
    # =========================================================================
    
    _normal_response = handle_normal_routing(req, project, db, trace)
    
    # v11.0: If staleness detected, wrap the response to prepend warning
    if _staleness_events and _normal_response:
        from app.project_registry.chat_injection import inject_staleness_into_stream
        original_gen = _normal_response.body_iterator
        wrapped_gen = inject_staleness_into_stream(original_gen, db, req.message)
        return StreamingResponse(
            wrapped_gen,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    return _normal_response


# =============================================================================
# REJECT INTENT ENDPOINT (v2.2)
# =============================================================================

class RejectIntentRequest(BaseModel):
    project_id: int
    intent: str
    original_message: str = ""


@router.post("/reject-intent")
async def reject_intent(
    req: RejectIntentRequest,
    auth: AuthResult = Depends(require_auth),
):
    """Log a rejected confirmation for confidence learning.

    When the user clicks 'Not what I meant' on a confirmation gate,
    the frontend calls this endpoint to record the negative signal.
    This feeds into the graduated confidence system.
    """
    from app.llm.routing.reject_intent_handler import handle_reject_intent
    return await handle_reject_intent(req)

