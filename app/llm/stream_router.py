# FILE: app/llm/stream_router.py
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

v4.10 (2026-01): Removed host filesystem scan (sandbox only), cleaned up routing
v4.9 (2026-01): Added ASTRA capability layer injection to system prompts
v4.8 (2026-01): Chat mode uses stage_models, added debug logging throughout
v4.7 (2026-01): Added stage tracing, routing failure visibility, model audit
v4.6 (2026-01): Refactored into modules (stream_handlers, translation_routing, legacy_triggers)
v4.5 (2026-01): DB-backed validated spec lookup (survives restarts)
v4.4 (2026-01): Added SPEC_VALIDATED flow state check for Critical Pipeline routing
v4.3 (2026-01): Added Spec Gate, Critical Pipeline, Overwatcher stream handlers
v4.2 (2026-01): Added Weaver stream handler for spec buildingF
v4.1 (2026-01): CRITICAL FIX - CHAT mode returns early, bypasses job classification
v4.0 (2026-01): ASTRA Translation Layer integration - prevents misfires
"""

import json
import logging
import os
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service

from app.llm.audit_logger import get_audit_logger
from app.llm.router import is_high_stakes_job, is_opus_model
from .streaming import get_available_streaming_provider

from .stream_utils import (
    DEFAULT_MODELS,
    make_session_id,
    build_context_block,
    build_document_context,
    get_semantic_context,
    classify_job_type,
    select_provider_for_job_type,
)

# =============================================================================
# MODULAR IMPORTS (v4.6)
# =============================================================================

from .stream_handlers import (
    generate_sse_stream,
    generate_sandbox_stream,
    generate_introspection_stream,
    generate_feedback_stream,
    generate_confirmation_stream,
)

from .translation_routing import (
    TRANSLATION_LAYER_AVAILABLE,
    TranslationMode,
    CanonicalIntent,
    route_via_translation_layer,
    intent_to_routing_info,
    _get_spec_gate_config,
    _get_critical_pipeline_config,
)

from .legacy_triggers import (
    is_zobie_map_trigger,
    is_archmap_trigger,
    is_update_arch_trigger,
    is_introspection_trigger,
    is_sandbox_trigger,
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
)

from .high_stakes_stream import generate_high_stakes_critique_stream

# v4.9: ASTRA Capability Layer
try:
    from app.capabilities import get_capability_context
    _CAPABILITIES_AVAILABLE = True
    print("[stream_router] ASTRA capability layer loaded successfully")
except ImportError:
    _CAPABILITIES_AVAILABLE = False
    get_capability_context = None

# v4.7: Stage tracing
try:
    from .stage_trace import StageTrace, log_model_resolution, get_env_model_audit
    _STAGE_TRACE_AVAILABLE = True
except ImportError:
    _STAGE_TRACE_AVAILABLE = False
    StageTrace = None

# =============================================================================
# OPTIONAL STREAM HANDLERS
# =============================================================================

try:
    from app.llm.weaver_stream import generate_weaver_stream
    _WEAVER_AVAILABLE = True
except ImportError as e:
    import traceback
    print(f"[WEAVER_IMPORT_ERROR] Failed to import weaver_stream: {e}")
    traceback.print_exc()
    _WEAVER_AVAILABLE = False
except Exception as e:
    import traceback
    print(f"[WEAVER_IMPORT_ERROR] Unexpected error importing weaver_stream: {e}")
    traceback.print_exc()
    _WEAVER_AVAILABLE = False

try:
    from app.llm.spec_gate_stream import generate_spec_gate_stream
    _SPEC_GATE_STREAM_AVAILABLE = True
except ImportError:
    _SPEC_GATE_STREAM_AVAILABLE = False

try:
    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    _CRITICAL_PIPELINE_AVAILABLE = True
except ImportError:
    _CRITICAL_PIPELINE_AVAILABLE = False

try:
    from app.llm.overwatcher_stream import generate_overwatcher_stream
    _OVERWATCHER_AVAILABLE = True
except ImportError:
    _OVERWATCHER_AVAILABLE = False

try:
    from app.llm.spec_flow_state import get_active_flow, SpecFlowStage
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    _FLOW_STATE_AVAILABLE = False
    get_active_flow = None

try:
    from app.specs.service import get_latest_validated_spec
    _SPEC_SERVICE_AVAILABLE = True
except ImportError:
    _SPEC_SERVICE_AVAILABLE = False
    get_latest_validated_spec = None

try:
    from app.llm.local_tools.zobie_tools import (
        generate_local_architecture_map_stream,
        generate_local_zobie_map_stream,
        generate_update_architecture_stream,
        generate_sandbox_structure_scan_stream,
    )
    _LOCAL_TOOLS_AVAILABLE = True
    print("[stream_router] Local tools loaded successfully")
except Exception as e:
    import traceback
    _LOCAL_TOOLS_AVAILABLE = False
    print(f"[LOCAL_TOOLS_IMPORT_ERROR] Failed to import local tools: {e}")
    traceback.print_exc()

try:
    from app.sandbox import handle_sandbox_prompt
    _SANDBOX_AVAILABLE = True
except ImportError:
    _SANDBOX_AVAILABLE = False

try:
    from app.introspection.chat_integration import detect_log_intent
    _INTROSPECTION_AVAILABLE = True
except ImportError:
    _INTROSPECTION_AVAILABLE = False


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
# STAGE TRACE HELPERS (v4.7)
# =============================================================================

def _create_stage_trace(command_type: str, project_id: int, job_id: Optional[str] = None) -> Optional["StageTrace"]:
    """Create a stage trace if available."""
    if _STAGE_TRACE_AVAILABLE and StageTrace:
        return StageTrace.start(command_type, project_id=project_id, job_id=job_id)
    return None


def _log_routing_failure(
    stage_trace: Optional["StageTrace"],
    reason: str,
    expected_handler: str,
    fallback_action: Optional[str] = None,
) -> None:
    """Log a routing failure."""
    # Always log to console for visibility
    print(f"[ROUTING_FAILURE] {reason}")
    print(f"[ROUTING_FAILURE]   expected_handler={expected_handler}")
    if fallback_action:
        print(f"[ROUTING_FAILURE]   fallback={fallback_action}")
    
    logger.warning(f"[stream_router] ROUTING_FAILURE: {reason} (handler={expected_handler})")
    
    if stage_trace:
        stage_trace.record_routing_failure(reason, expected_handler, fallback_action)


def _log_handler_availability() -> None:
    """Log availability of all handlers for debugging."""
    print(f"[HANDLER_STATUS] Weaver: {_WEAVER_AVAILABLE}")
    print(f"[HANDLER_STATUS] SpecGateStream: {_SPEC_GATE_STREAM_AVAILABLE}")
    print(f"[HANDLER_STATUS] CriticalPipeline: {_CRITICAL_PIPELINE_AVAILABLE}")
    print(f"[HANDLER_STATUS] Overwatcher: {_OVERWATCHER_AVAILABLE}")
    print(f"[HANDLER_STATUS] FlowState: {_FLOW_STATE_AVAILABLE}")
    print(f"[HANDLER_STATUS] SpecService: {_SPEC_SERVICE_AVAILABLE}")


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
    
    # v4.7: Stage trace for commands
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
        
        # v4.7: Create stage trace for COMMAND mode
        if translation_result.mode == TranslationMode.COMMAND_CAPABLE:
            command_type = translation_result.resolved_intent.value if translation_result.resolved_intent else "unknown"
            stage_trace = _create_stage_trace(command_type, req.project_id, req.continue_job_id)
            
            # Log env var state for audit
            if _STAGE_TRACE_AVAILABLE:
                env_audit = get_env_model_audit()
                print(f"[MODEL_ENV_AUDIT] {env_audit}")
        
        # =====================================================================
        # FLOW STATE: Route clarifications to Spec Gate
        # =====================================================================
        if _FLOW_STATE_AVAILABLE and get_active_flow:
            active_flow = get_active_flow(req.project_id)
            if active_flow and active_flow.stage == SpecFlowStage.SPEC_GATE_QUESTIONS:
                logger.info(f"[flow_state] Routing to Spec Gate (round {active_flow.clarification_round + 1})")
                if _SPEC_GATE_STREAM_AVAILABLE:
                    if stage_trace:
                        spec_provider, spec_model = _get_spec_gate_config()
                        stage_trace.enter_stage("spec_gate_clarification", provider=spec_provider, model=spec_model)
                    return StreamingResponse(
                        generate_spec_gate_stream(
                            project_id=req.project_id, message=req.message, db=db, trace=trace,
                            conversation_id=conversation_id, is_clarification_response=True
                        ),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
                else:
                    _log_routing_failure(
                        stage_trace,
                        "Spec Gate stream handler not available for clarification routing",
                        "generate_spec_gate_stream",
                        "falling through to normal routing"
                    )
        
        # =====================================================================
        # DB-BACKED SPEC: Route "critical pipeline" to Critical Pipeline
        # =====================================================================
        if _SPEC_SERVICE_AVAILABLE and get_latest_validated_spec:
            msg_lower = req.message.lower()
            if "critical pipeline" in msg_lower or "run pipeline" in msg_lower:
                validated_spec = get_latest_validated_spec(db, req.project_id)
                if validated_spec:
                    logger.info(f"[db_spec] Found validated spec: {validated_spec.spec_id}")
                    
                    if _CRITICAL_PIPELINE_AVAILABLE:
                        if stage_trace:
                            crit_provider, crit_model = _get_critical_pipeline_config()
                            stage_trace.enter_stage("critical_pipeline", provider=crit_provider, model=crit_model,
                                                   spec_id=str(validated_spec.spec_id))
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
                        # v4.7: CRITICAL - Log this failure explicitly
                        _log_routing_failure(
                            stage_trace,
                            "Critical Pipeline handler not available (import failed)",
                            "generate_critical_pipeline_stream",
                            "falling through - command will NOT execute"
                        )
                        _log_handler_availability()
                        
                        # v4.7: Return hard error instead of falling through
                        error_msg = (
                            "⚠️ **Critical Pipeline Handler Not Available**\n\n"
                            "The critical pipeline module failed to import. "
                            "Check server logs for `ImportError` details.\n\n"
                            "This is an internal configuration issue."
                        )
                        
                        async def _handler_missing_stream():
                            yield "data: " + json.dumps({'type': 'error', 'error': 'Critical pipeline handler not available'}) + "\n\n"
                            yield "data: " + json.dumps({'type': 'token', 'content': error_msg}) + "\n\n"
                            yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"
                        
                        if stage_trace:
                            stage_trace.finish(success=False, outcome="handler_unavailable")
                        
                        return StreamingResponse(
                            _handler_missing_stream(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
                else:
                    logger.warning(f"[db_spec] No validated spec found for project {req.project_id}")
                    if stage_trace:
                        stage_trace.record_routing_failure(
                            "No validated spec in DB - cannot run critical pipeline",
                            "critical_pipeline",
                            "user needs to complete spec validation first"
                        )
                    
                    # v4.7: Return actionable error instead of falling through to chat
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
                            'project_id': req.project_id,
                        }) + "\n\n"
                        yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"
                    
                    if stage_trace:
                        stage_trace.finish(success=False, outcome="no_validated_spec")
                    
                    return StreamingResponse(
                        _no_spec_error_stream(),
                        media_type="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                    )
        
        # =====================================================================
        # CHAT MODE: Return early with lightweight model
        # =====================================================================
        if translation_result.mode == TranslationMode.CHAT:
            logger.info("[translation] CHAT MODE - bypassing job classification")
            return _handle_chat_mode(req, project, db, trace)
        
        # =====================================================================
        # FEEDBACK MODE
        # =====================================================================
        if translation_result.mode == TranslationMode.FEEDBACK:
            return StreamingResponse(
                generate_feedback_stream(
                    project_id=req.project_id, message=req.message,
                    translation_result=translation_result, db=db, trace=trace
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # =====================================================================
        # COMMAND MODE
        # =====================================================================
        if translation_result.mode == TranslationMode.COMMAND_CAPABLE:
            
            # Awaiting confirmation
            if (translation_result.confirmation_gate and 
                translation_result.confirmation_gate.awaiting_confirmation):
                return StreamingResponse(
                    generate_confirmation_stream(
                        project_id=req.project_id, message=req.message,
                        translation_result=translation_result, db=db, trace=trace
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            
            # Execute approved commands
            if translation_result.should_execute:
                response = _handle_command_execution(
                    req, translation_result, db, trace, conversation_id, stage_trace
                )
                if response:
                    return response
            
            # Command blocked - fall through
            if not translation_result.should_execute:
                logger.debug(f"[translation] Command blocked: {translation_result.execution_blocked_reason}")
                if stage_trace:
                    stage_trace.record_routing_failure(
                        f"Command blocked: {translation_result.execution_blocked_reason}",
                        translation_result.resolved_intent.value if translation_result.resolved_intent else "unknown"
                    )
                
                # v4.7: CRITICAL - Don't fall through to chat for high-stakes commands
                # Return a hard error so user knows what went wrong
                intent = translation_result.resolved_intent
                if intent in (
                    CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB,
                    CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES,
                    CanonicalIntent.SEND_TO_SPEC_GATE,
                ):
                    blocked_reason = translation_result.execution_blocked_reason or "Unknown blocking reason"
                    
                    # Build actionable error message
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
    
    # =========================================================================
    # LEGACY FALLBACK (translation layer unavailable)
    # =========================================================================
    
    if translation_result is None:
        response = _handle_legacy_triggers(req, db, trace)
        if response:
            return response
    
    # =========================================================================
    # NORMAL ROUTING
    # =========================================================================
    
    return _handle_normal_routing(req, project, db, trace)


# =============================================================================
# ROUTING HELPERS
# =============================================================================

def _handle_chat_mode(req: StreamRequest, project, db: Session, trace):
    """Handle CHAT mode - lightweight model, no commands."""
    # v4.8: Debug logging for chat mode
    print(f"[CHAT_MODE] Handling chat for project={req.project_id}, message={req.message[:50]}...")
    
    context_block = build_context_block(db, req.project_id)
    semantic_context = get_semantic_context(db, req.project_id, req.message) if req.use_semantic_search else ""
    doc_context = build_document_context(db, req.project_id)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===" + doc_context
    
    # v4.8: Use centralized stage_models for provider/model
    try:
        from .stage_models import get_chat_config
        chat_config = get_chat_config()
        provider = chat_config.provider
        model = chat_config.model
        print(f"[CHAT_MODE] Using stage_models: provider={provider}, model={model}")
    except ImportError:
        # Fallback if stage_models not available
        provider = "openai"
        model = DEFAULT_MODELS.get("openai", "gpt-4.1-mini")
        print(f"[CHAT_MODE] stage_models unavailable, using fallback: provider={provider}, model={model}")
    
    # Check provider availability
    available = get_available_streaming_provider()
    print(f"[CHAT_MODE] Available streaming provider: {available}")
    
    if provider not in ("openai", "anthropic", "google", "gemini"):
        print(f"[CHAT_MODE] WARNING: Unknown provider '{provider}', defaulting to available: {available}")
        provider = available or "openai"
    
    from .streaming import get_available_streaming_providers
    providers_available = get_available_streaming_providers()
    print(f"[CHAT_MODE] Provider availability: {providers_available}")
    
    if not providers_available.get(provider, False) and not providers_available.get("gemini" if provider == "google" else provider, False):
        print(f"[CHAT_MODE] Provider {provider} not available, falling back to {available}")
        if available:
            provider = available
            # Keep the model from stage_models, streaming.py will handle provider-specific defaults if needed
    
    messages: List[dict] = []
    if req.include_history:
        history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        messages = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append({"role": "user", "content": req.message})
    
    # v4.9: Use centralized _build_system_prompt (includes capability layer)
    system_prompt = _build_system_prompt(project, full_context)
    
    # v4.8: Final logging before stream
    print(f"[CHAT_MODE] Calling generate_sse_stream: provider={provider}, model={model}, messages={len(messages)}")
    
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id, message=req.message, provider=provider, model=model,
            system_prompt=system_prompt, messages=messages, db=db, trace=trace,
            enable_reasoning=req.enable_reasoning,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _handle_command_execution(req, translation_result, db, trace, conversation_id, stage_trace=None):
    """Handle approved command execution."""
    intent = translation_result.resolved_intent
    routing_info = intent_to_routing_info(intent)
    
    if not routing_info:
        _log_routing_failure(
            stage_trace,
            f"No routing info for intent: {intent.value if intent else 'None'}",
            "intent_to_routing_info",
            "falling through to normal routing"
        )
        return None
    
    # v4.7: Log the resolved routing
    print(f"[COMMAND_ROUTE] Intent: {intent.value if intent else 'None'}")
    print(f"[COMMAND_ROUTE] Routing: type={routing_info['type']}, provider={routing_info['provider']}, model={routing_info['model']}")
    print(f"[COMMAND_ROUTE] Reason: {routing_info['reason']}")
    
    if trace:
        trace.log_request_start(
            job_type=req.job_type or "", resolved_job_type=routing_info["type"],
            provider=routing_info["provider"], model=routing_info["model"],
            reason=routing_info["reason"], frontier_override=False,
            file_map_injected=False, attachments=None
        )
    
    # Sandbox
    if intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF and _SANDBOX_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("sandbox", provider="local", model="sandbox_manager")
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Update architecture
    if intent == CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY and _LOCAL_TOOLS_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("update_architecture", provider="local", model="architecture_scanner")
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
        # Sandbox structure scan (read-only)
    if intent == CanonicalIntent.SCAN_SANDBOX_STRUCTURE and _LOCAL_TOOLS_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage(
                "sandbox_structure_scan",
                provider="local",
                model="sandbox_structure_scanner",
            )
        return StreamingResponse(
            generate_sandbox_structure_scan_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
                trace=trace,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Architecture map
    if intent in (CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES, CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY):
        if _LOCAL_TOOLS_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL)
            return StreamingResponse(
                generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
    
    # Weaver
    if intent == CanonicalIntent.WEAVER_BUILD_SPEC:
        if _WEAVER_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("weaver", provider=routing_info["provider"], model=routing_info["model"])
            return StreamingResponse(
                generate_weaver_stream(
                    project_id=req.project_id, message=req.message, db=db, trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            _log_routing_failure(stage_trace, "Weaver handler not available", "generate_weaver_stream")
    
    # Spec Gate
    if intent == CanonicalIntent.SEND_TO_SPEC_GATE:
        if _SPEC_GATE_STREAM_AVAILABLE:
            spec_provider, spec_model = _get_spec_gate_config()
            print(f"[SPEC_GATE_ROUTE] Using provider={spec_provider}, model={spec_model}")
            if stage_trace:
                stage_trace.enter_stage("spec_gate", provider=spec_provider, model=spec_model)
            return StreamingResponse(
                generate_spec_gate_stream(
                    project_id=req.project_id, message=req.message, db=db, trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            _log_routing_failure(stage_trace, "Spec Gate stream handler not available", "generate_spec_gate_stream")
    
    # Critical Pipeline
    if intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB:
        if _CRITICAL_PIPELINE_AVAILABLE:
            crit_provider, crit_model = _get_critical_pipeline_config()
            print(f"[CRITICAL_PIPELINE_ROUTE] Using provider={crit_provider}, model={crit_model}")
            if stage_trace:
                stage_trace.enter_stage("critical_pipeline", provider=crit_provider, model=crit_model)
            return StreamingResponse(
                generate_critical_pipeline_stream(
                    project_id=req.project_id, message=req.message, db=db, trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            # v4.7: CRITICAL - This is likely the bug - handler not available
            _log_routing_failure(
                stage_trace,
                "CRITICAL PIPELINE HANDLER NOT AVAILABLE - Command will NOT execute!",
                "generate_critical_pipeline_stream",
                "Check if app/llm/critical_pipeline_stream.py exists and imports correctly"
            )
            _log_handler_availability()
    
    # Overwatcher
    if intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES:
        if _OVERWATCHER_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("overwatcher", provider=routing_info["provider"], model=routing_info["model"])
            return StreamingResponse(
                generate_overwatcher_stream(
                    project_id=req.project_id, message=req.message, db=db, trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        else:
            _log_routing_failure(stage_trace, "Overwatcher handler not available", "generate_overwatcher_stream")
    
    return None


def _handle_legacy_triggers(req, db, trace):
    """Handle legacy triggers when translation layer unavailable."""
    
    if _SANDBOX_AVAILABLE and is_sandbox_trigger(req.message):
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    if is_update_arch_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    if is_archmap_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    if is_zobie_map_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    if _INTROSPECTION_AVAILABLE and is_introspection_trigger(req.message):
        return StreamingResponse(
            generate_introspection_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    return None


def _handle_normal_routing(req, project, db, trace):
    """Handle normal job-type routing."""
    
    context_block = build_context_block(db, req.project_id)
    semantic_context = get_semantic_context(db, req.project_id, req.message) if req.use_semantic_search else ""
    doc_context = build_document_context(db, req.project_id)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===" + doc_context
    
    # Job continuation
    if req.continue_job_id and req.job_state == "needs_spec_clarification":
        provider = "anthropic"
        model = DEFAULT_MODELS["anthropic_opus"]
        messages = _build_messages(req, db)
        system_prompt = _build_system_prompt(project, full_context)
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id, message=req.message, provider=provider, model=model,
                system_prompt=system_prompt, messages=messages, full_context=full_context,
                job_type_str="architecture_design", db=db, trace=trace,
                enable_reasoning=req.enable_reasoning, continue_job_id=req.continue_job_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Normal routing
    job_type = classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value
    
    if req.provider and req.model:
        provider, model = req.provider, req.model
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    else:
        provider, model = select_provider_for_job_type(job_type)
    
    # Provider availability check
    available = get_available_streaming_provider()
    if not available:
        raise HTTPException(status_code=503, detail="No LLM provider available")
    
    from .streaming import get_available_streaming_providers
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    
    messages = _build_messages(req, db)
    system_prompt = _build_system_prompt(project, full_context)
    
    # High-stakes routing
    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id, message=req.message, provider=provider, model=model,
                system_prompt=system_prompt, messages=messages, full_context=full_context,
                job_type_str=job_type_value, db=db, trace=trace, enable_reasoning=req.enable_reasoning,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Normal stream
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id, message=req.message, provider=provider, model=model,
            system_prompt=system_prompt, messages=messages, db=db, trace=trace,
            enable_reasoning=req.enable_reasoning,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _build_messages(req, db) -> List[dict]:
    """Build message list from history + current message."""
    messages = []
    if req.include_history:
        history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        messages = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append({"role": "user", "content": req.message})
    return messages


def _build_system_prompt(project, full_context: str) -> str:
    """
    Build system prompt with project context and ASTRA capability layer.
    
    v4.9: Injects capability layer at the top of every system prompt.
    """
    # v4.9: Start with ASTRA capability layer
    capability_layer = ""
    if _CAPABILITIES_AVAILABLE and get_capability_context:
        try:
            capability_layer = get_capability_context()
        except Exception as e:
            print(f"[CAPABILITY_INJECTION] Error getting capability context: {e}")
    
    # Build project context
    system_prompt = f"Project: {project.name}."
    if project.description:
        system_prompt += f" {project.description}"
    if full_context:
        system_prompt += f"\n\nYou have access to the following context:\n\n{full_context}"
    
    # Combine: capabilities first, then project context
    if capability_layer:
        return f"{capability_layer}\n\n{system_prompt}"
    return system_prompt