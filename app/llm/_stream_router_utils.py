from __future__ import annotations
import json
import logging
from app.llm.routing.command_dispatch import log_routing_failure
from app.llm.routing.handler_registry import SpecFlowStage, _CRITICAL_PIPELINE_AVAILABLE, _FLOW_STATE_AVAILABLE, _SPEC_GATE_STREAM_AVAILABLE, _SPEC_SERVICE_AVAILABLE, generate_critical_pipeline_stream, generate_spec_gate_stream, get_active_flow, get_latest_validated_spec
from app.llm.translation_routing import CanonicalIntent, TranslationMode, _get_critical_pipeline_config, _get_spec_gate_config
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


class UIContext(BaseModel):
    """Context about which tab/view the user is currently viewing."""
    view_type: Optional[str] = None       # 'chat' | 'settings' | 'job'
    job_type: Optional[str] = None        # e.g. 'content', 'debug', 'project_builds'
    workspace_id: Optional[str] = None    # e.g. content project UUID
    label: Optional[str] = None           # human-readable label


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
    confirmed_intent: Optional[str] = None  # v1.9: bypass translation for confirmed commands
    extracted_query: Optional[str] = None     # v2.2: carry extracted search query through confirmation roundtrip
    ui_context: Optional[UIContext] = None    # v6.0: Universal Chat Panel — which tab/view the user is in
    panel_history: Optional[list] = None         # v6.0: Chat panel's local conversation history [{role, content}]
    debug_locked: bool = False                      # v2.1: Debug context lock — force Gemini + tools + RAG
    debug_project_id: Optional[str] = None            # v2.1: Active debug project ID for context pre-load
    enable_tools: bool = False                      # v2.1: Enable tool execution (sandbox read/write/shell)
    video_file_uri: Optional[str] = None             # v2.2: Screen recording — Gemini Files API URI
    video_mime_type: Optional[str] = None             # v2.2: Screen recording — MIME type
    video_local_path: Optional[str] = None            # v2.2: Screen recording — local file for cleanup

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

_WEAVER_EXIT_INTENTS = {
    CanonicalIntent.SEND_TO_SPEC_GATE,
    CanonicalIntent.RUN_PIPELINE,  # v5.4: unified pipeline
    CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,  # v5.4: deprecated alias
    CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
}

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
