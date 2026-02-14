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


router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)


# =============================================================================
# REQUEST MODEL
# =============================================================================

class StreamRequest(BaseModel):
    project_id: int
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    job_type: Optional[str] = None
    include_history: bool = True
    history_limit: int = 20
    use_semantic_search: bool = True
    enable_reasoning: bool = False
    continue_job_id: Optional[str] = None
    job_state: Optional[str] = None


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
_EXPLICIT_COMMAND_INTENTS = {
    CanonicalIntent.RUN_PIPELINE,  # v5.4: unified pipeline
    CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,  # v5.4: deprecated alias
    CanonicalIntent.RUN_SEGMENT_LOOP,  # v5.4: deprecated alias
    CanonicalIntent.IMPLEMENT_SEGMENTS,  # v5.13: phase 2 execution
    CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
    CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES,
    CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY,
    CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY,
    CanonicalIntent.START_SANDBOX_ZOMBIE_SELF,
    CanonicalIntent.SCAN_SANDBOX_STRUCTURE,
    CanonicalIntent.RAG_CODEBASE_QUERY,
    CanonicalIntent.EMBEDDING_STATUS,
    CanonicalIntent.GENERATE_EMBEDDINGS,
    CanonicalIntent.FILESYSTEM_QUERY,
    CanonicalIntent.CODEBASE_REPORT,
    CanonicalIntent.LATEST_ARCHITECTURE_MAP,
    CanonicalIntent.LATEST_CODEBASE_REPORT_FULL,
}


def _handle_flow_state_routing(req, db, trace, conversation_id, stage_trace, translation_result=None):
    """Handle flow state routing (Spec Gate clarifications).
    
    v5.1: Now checks for explicit command intents and skips flow state interception
    for commands like RUN_CRITICAL_PIPELINE_FOR_JOB, OVERWATCHER_EXECUTE_CHANGES, etc.
    This prevents explicit commands from being incorrectly routed to spec_gate_clarification.
    """
    if not _FLOW_STATE_AVAILABLE or not get_active_flow:
        return None
    
    # v5.1: Check if this is an explicit command that should bypass flow state
    if translation_result is not None:
        intent = translation_result.resolved_intent
        if intent and intent in _EXPLICIT_COMMAND_INTENTS:
            logger.info(
                "[flow_state] v5.1 EXPLICIT COMMAND BYPASS: intent=%s skips flow state interception",
                intent.value
            )
            print(f"[FLOW_STATE_BYPASS] Explicit command '{intent.value}' bypasses SPEC_GATE_QUESTIONS interception")
            return None
    
    active_flow = get_active_flow(req.project_id)
    if not active_flow or active_flow.stage != SpecFlowStage.SPEC_GATE_QUESTIONS:
        return None
    
    logger.info(f"[flow_state] Routing to Spec Gate (round {active_flow.clarification_round + 1})")
    
    if _SPEC_GATE_STREAM_AVAILABLE:
        if stage_trace:
            spec_provider, spec_model = _get_spec_gate_config()
            stage_trace.enter_stage("spec_gate_clarification", provider=spec_provider, model=spec_model)
        
        return StreamingResponse(
            generate_spec_gate_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
                trace=trace,
                conversation_id=conversation_id,
                is_clarification_response=True,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        log_routing_failure(
            stage_trace,
            "Spec Gate stream handler not available for clarification routing",
            "generate_spec_gate_stream",
            "falling through to normal routing"
        )
    
    return None


# v5.2: Intents that indicate the user wants to LEAVE the weaver flow
# If the user says one of these, don't auto-reweave — let it through.
_WEAVER_EXIT_INTENTS = {
    CanonicalIntent.SEND_TO_SPEC_GATE,
    CanonicalIntent.RUN_PIPELINE,  # v5.4: unified pipeline
    CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,  # v5.4: deprecated alias
    CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
}


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


def _handle_db_spec_routing(req, db, trace, conversation_id, stage_trace, translation_result=None):
    """Handle DB-backed spec routing for critical pipeline.
    
    v5.3: Only triggers for COMMAND_CAPABLE mode to prevent false positives
    when user mentions 'critical pipeline' in conversational context.
    """
    if not _SPEC_SERVICE_AVAILABLE or not get_latest_validated_spec:
        return None
    
    # v5.3: MUST be in command mode — never trigger from chat/conversational text
    if translation_result is None or translation_result.mode != TranslationMode.COMMAND_CAPABLE:
        return None
    
    msg_lower = req.message.lower()
    if "critical pipeline" not in msg_lower and "run pipeline" not in msg_lower:
        return None
    
    validated_spec = get_latest_validated_spec(db, req.project_id)
    
    if validated_spec:
        logger.info(f"[db_spec] Found validated spec: {validated_spec.spec_id}")
        
        if _CRITICAL_PIPELINE_AVAILABLE:
            if stage_trace:
                crit_provider, crit_model = _get_critical_pipeline_config()
                stage_trace.enter_stage(
                    "critical_pipeline",
                    provider=crit_provider,
                    model=crit_model,
                    spec_id=str(validated_spec.spec_id),
                )
            
            return StreamingResponse(
                generate_critical_pipeline_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=conversation_id,
                    spec_id=str(validated_spec.spec_id),
                    spec_hash=validated_spec.spec_hash,
                    job_id=getattr(validated_spec, 'job_id', None),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            log_routing_failure(
                stage_trace,
                "Critical Pipeline handler not available (import failed)",
                "generate_critical_pipeline_stream",
                "falling through - command will NOT execute"
            )
            
            return _create_handler_unavailable_response(
                "Critical Pipeline Handler",
                stage_trace,
            )
    else:
        logger.warning(f"[db_spec] No validated spec found for project {req.project_id}")
        if stage_trace:
            stage_trace.record_routing_failure(
                "No validated spec in DB - cannot run critical pipeline",
                "critical_pipeline",
                "user needs to complete spec validation first"
            )
        
        return _create_no_spec_error_response(req.project_id, stage_trace)
    
    return None


def _handle_blocked_command(translation_result, stage_trace):
    """Handle blocked high-stakes commands."""
    logger.debug(f"[translation] Command blocked: {translation_result.execution_blocked_reason}")
    
    if stage_trace:
        stage_trace.record_routing_failure(
            f"Command blocked: {translation_result.execution_blocked_reason}",
            translation_result.resolved_intent.value if translation_result.resolved_intent else "unknown"
        )
    
    intent = translation_result.resolved_intent
    
    # Only return hard error for high-stakes commands
    if intent not in (
        CanonicalIntent.RUN_PIPELINE,
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
        CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
        CanonicalIntent.SEND_TO_SPEC_GATE,
        CanonicalIntent.IMPLEMENT_SEGMENTS,
    ):
        return None
    
    blocked_reason = translation_result.execution_blocked_reason or "Unknown blocking reason"
    
    if "spec_id" in blocked_reason.lower() or "validated spec" in blocked_reason.lower():
        error_msg = (
            f"⚠️ **Command Blocked: {intent.value}**\n\n"
            f"Reason: {blocked_reason}\n\n"
            f"**What to do next:**\n"
            f"1. Run `send to spec gate` to validate your spec\n"
            f"2. Answer any clarification questions\n"
            f"3. Once validated, retry `run critical pipeline`"
        )
    else:
        error_msg = (
            f"⚠️ **Command Blocked: {intent.value}**\n\n"
            f"Reason: {blocked_reason}\n\n"
            f"Please resolve the blocking condition and retry."
        )
    
    async def _blocked_error_stream():
        yield "data: " + json.dumps({'type': 'token', 'content': error_msg}) + "\n\n"
        yield "data: " + json.dumps({
            'type': 'command_blocked',
            'intent': intent.value,
            'reason': blocked_reason,
        }) + "\n\n"
        yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"
    
    if stage_trace:
        stage_trace.finish(success=False, outcome="command_blocked", error=blocked_reason)
    
    return StreamingResponse(
        _blocked_error_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _create_handler_unavailable_response(handler_name: str, stage_trace):
    """Create error response for unavailable handler."""
    from app.llm.routing.handler_registry import log_handler_availability
    log_handler_availability()
    
    error_msg = (
        f"⚠️ **{handler_name} Not Available**\n\n"
        f"The {handler_name.lower()} module failed to import. "
        f"Check server logs for `ImportError` details.\n\n"
        f"This is an internal configuration issue."
    )
    
    async def _handler_missing_stream():
        yield "data: " + json.dumps({'type': 'error', 'error': f'{handler_name} not available'}) + "\n\n"
        yield "data: " + json.dumps({'type': 'token', 'content': error_msg}) + "\n\n"
        yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"
    
    if stage_trace:
        stage_trace.finish(success=False, outcome="handler_unavailable")
    
    return StreamingResponse(
        _handler_missing_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _create_no_spec_error_response(project_id: int, stage_trace):
    """Create error response for missing validated spec."""
    error_msg = (
        "⚠️ **Cannot Run Critical Pipeline**\n\n"
        "**Reason:** No validated specification found for this project.\n\n"
        "**What to do:**\n"
        "1. First, describe what you want to build (ramble)\n"
        "2. Say `how does that look all together` to build a spec\n"
        "3. Say `send to spec gate` to validate the spec\n"
        "4. Answer any clarification questions\n"
        "5. Once validated, retry `run critical pipeline`\n\n"
        "*The spec must be validated and persisted to DB before the critical pipeline can execute.*"
    )
    
    async def _no_spec_error_stream():
        yield "data: " + json.dumps({'type': 'token', 'content': error_msg}) + "\n\n"
        yield "data: " + json.dumps({
            'type': 'command_blocked',
            'intent': 'RUN_CRITICAL_PIPELINE_FOR_JOB',
            'reason': 'no_validated_spec',
            'project_id': project_id,
        }) + "\n\n"
        yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"
    
    if stage_trace:
        stage_trace.finish(success=False, outcome="no_validated_spec")
    
    return StreamingResponse(
        _no_spec_error_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
