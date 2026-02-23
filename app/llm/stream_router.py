# FILE: app/llm/stream_router.py
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

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

import json
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
        try:
            direct_intent = CanonicalIntent(req.confirmed_intent)
            logger.info(f"[stream_router] Confirmed intent bypass: {direct_intent.value}")
            stage_trace = create_stage_trace(direct_intent.value, req.project_id)
            
            from app.translation.schemas import (
                TranslationResult, ConfirmationGateResult, LatencyTier,
            )
            translation_result = TranslationResult(
                original_text=req.message,
                mode=TranslationMode.COMMAND_CAPABLE,
                resolved_intent=direct_intent,
                intent_confidence=1.0,
                latency_tier=LatencyTier.TIER_0_RULES,
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
        response = _handle_weaver_design_questions(req, db, trace, stage_trace, translation_result)
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
        # CHAT MODE: Return early with lightweight model
        # =================================================================
        if translation_result.mode == TranslationMode.CHAT:
            logger.info("[translation] CHAT MODE - bypassing job classification")
            # v5.4: Capture preferences and context from chat messages
            after_user_message(req.message, project_id=str(req.project_id), user_id=user_id)
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
                _mem_svc.create_message(
                    db,
                    _mem_schemas.MessageCreate(
                        project_id=req.project_id,
                        role="user",
                        content=req.message,
                        provider="system",
                    ),
                )
            except Exception:
                pass  # Non-fatal — don't block command execution
            
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
                # v5.4: Record successful intent resolution in confidence learning
                intent_val = translation_result.resolved_intent.value if translation_result.resolved_intent else None
                if intent_val:
                    on_intent_confirmed(req.message, intent_val, user_id)
                after_user_message(req.message, project_id=str(req.project_id), user_id=user_id)
                
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
    # NORMAL ROUTING
    # =========================================================================
    
    return handle_normal_routing(req, project, db, trace)


# =============================================================================
# INTERNAL ROUTING HELPERS
# =============================================================================

# v5.1: Explicit command intents that should NOT be intercepted by flow state


# v5.2: Intents that indicate the user wants to LEAVE the weaver flow
# If the user says one of these, don't auto-reweave — let it through.


def _handle_weaver_design_questions(req, db, trace, stage_trace, translation_result=None):
    """Handle auto-reweave: route user replies back to Weaver UPDATE.
    
    v5.2 (2026-02-04): AUTO-REWEAVE
    When Weaver finishes, flow enters AWAITING_SPEC_GATE_CONFIRM.
    If the user replies with anything that ISN'T an explicit command
    (like 'send to spec gate' or 'run critical pipeline'), we assume
    they're adding more requirements or answering questions, so we
    auto-route back to Weaver UPDATE mode.
    
    This creates a natural loop:
      Weaver outputs (with or without questions)
      → User replies (answers, additions, refinements)
      → Auto-triggers Weaver UPDATE
      → Repeat until user says 'send to spec gate'
    
    Previous behaviour (broken): Required keyword detection from
    hardcoded WEAVER_DESIGN_QUESTIONS stage, which was never set
    after v4.0 removed the slot/question infrastructure.
    """
    if not _FLOW_STATE_AVAILABLE or not get_active_flow:
        return None
    
    active_flow = get_active_flow(req.project_id)
    if not active_flow:
        return None
    
    # Only intercept in weaver-active stages
    if active_flow.stage not in (
        SpecFlowStage.WEAVER_DESIGN_QUESTIONS,
        SpecFlowStage.AWAITING_SPEC_GATE_CONFIRM,
    ):
        return None
    
    # v5.2: If translation resolved to a weaver-exit intent, DON'T intercept
    # Let the user proceed to spec gate / critical pipeline / etc.
    if translation_result is not None:
        intent = translation_result.resolved_intent
        if intent and intent in _WEAVER_EXIT_INTENTS:
            logger.info(
                "[weaver_reweave] User issued exit intent '%s' — leaving weaver flow",
                intent.value
            )
            print(f"[WEAVER_REWEAVE] Exit intent '{intent.value}' — NOT auto-reweaving")
            return None
        # Also skip if it's any explicit command (architecture, sandbox, etc.)
        if intent and intent in _EXPLICIT_COMMAND_INTENTS:
            logger.info(
                "[weaver_reweave] Explicit command '%s' — bypassing auto-reweave",
                intent.value
            )
            print(f"[WEAVER_REWEAVE] Explicit command '{intent.value}' — NOT auto-reweaving")
            return None
    
    # Auto-route to Weaver UPDATE
    if _WEAVER_AVAILABLE:
        weaver_provider, weaver_model = _get_weaver_config()
        if stage_trace:
            stage_trace.enter_stage("weaver_auto_reweave", provider=weaver_provider, model=weaver_model)
        
        logger.info("[weaver_reweave] Auto-routing to Weaver UPDATE (flow stage: %s)", active_flow.stage.value)
        print(f"[WEAVER_REWEAVE] Auto-routing to Weaver UPDATE (user replied in active weaver flow)")
        
        # v4.1.0: Pass req.message as pending_user_message to fix race condition.
        # The user's reply may not be in the DB yet when Weaver reads messages,
        # causing hash-based dedup to see "nothing new". This ensures the reply
        # is always visible to the weaver regardless of persistence timing.
        return StreamingResponse(
            generate_weaver_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
                trace=trace,
                conversation_id=str(req.project_id),
                is_continuation=True,
                captured_answers=None,
                pending_user_message=req.message,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        log_routing_failure(
            stage_trace,
            "Weaver handler not available for auto-reweave routing",
            "generate_weaver_stream",
            "falling through to normal routing"
        )
    
    return None
