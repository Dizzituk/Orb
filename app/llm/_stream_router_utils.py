# Purpose: stream router utils
# Called-by: app.llm.routing.weaver_reweave_handler, app.llm.stream_router
# Depends-on: app.llm.project_scoping_stream, app.llm.routing.chat_routing, app.llm.routing.command_dispatch, app.llm.routing.handler_registry (+3 more)
# Last-renovated: 2026-06-11
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
    file_upload_uri: Optional[str] = None             # v2.3: Debug file upload — Gemini Files API URI
    file_upload_mime: Optional[str] = None             # v2.3: Debug file upload — MIME type
    file_upload_name: Optional[str] = None             # v2.3: Debug file upload — original filename
    file_upload_local_path: Optional[str] = None       # v2.3: Debug file upload — local file for cleanup
    file_upload_gemini_name: Optional[str] = None      # v2.3: Debug file upload — Gemini file name for cleanup
    documents: Optional[list] = None                    # v3.0: Structured document blocks from large text pastes

def _is_scoping_cancel(msg_lower: str) -> bool:
    """Check if a message is an explicit cancel/exit request.

    Uses word-boundary matching to prevent false positives like
    'existing' matching 'exit' or 'nonstop' matching 'stop'.
    Only triggers on short messages (≤10 words) that are clearly
    cancel intents, not long descriptions that happen to contain
    these words.
    """
    import re
    # Long messages are descriptions, not cancel requests
    if len(msg_lower.split()) > 10:
        return False
    _cancel_pattern = re.compile(
        r'\b(?:cancel|stop|exit|nevermind|never\s*mind|quit|abort)\b'
    )
    return bool(_cancel_pattern.search(msg_lower))


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
    CanonicalIntent.PROJECT_SCOPE_START,
}

def _handle_flow_state_routing(req, db, trace, conversation_id, stage_trace, translation_result=None):
    """Handle flow state routing (Spec Gate clarifications + project scoping).
    
    v2.2: Added project scoping session interception. If the user is in an
    active scoping conversation, route messages to the scoping handler.
    
    v5.1: Now checks for explicit command intents and skips flow state interception
    for commands like RUN_CRITICAL_PIPELINE_FOR_JOB, OVERWATCHER_EXECUTE_CHANGES, etc.
    This prevents explicit commands from being incorrectly routed to spec_gate_clarification.
    """
    # v2.2: Check for active project scoping session
    # IMPORTANT: Always key by project_id (not conversation_id) because the
    # scoping handler stores it that way, and conversation_id includes a
    # session hash that changes between requests.
    try:
        from app.shared_context.project_session import get_project_session
        _scope_key = str(req.project_id)
        _scope_session = get_project_session(_scope_key)
        if _scope_session.scoping_active:
            # Check if user is confirming the project
            msg_lower = req.message.strip().lower()
            if msg_lower in ("yes", "y", "confirm", "confirmed", "go", "do it", "create it"):
                from app.llm.project_scoping_stream import handle_scoping_confirmation
                logger.info("[flow_state] Scoping CONFIRMED — creating project")
                return StreamingResponse(
                    handle_scoping_confirmation(
                        project_id=req.project_id,
                        message=req.message,
                        db=db,
                        conversation_id=_scope_key,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            # Check if user wants to exit scoping
            # v2.3: Use word-boundary check to prevent false positives
            # (e.g. "existing" matching "exit", "nonstop" matching "stop")
            elif _is_scoping_cancel(msg_lower):
                _scope_session.clear()
                logger.info("[flow_state] Scoping CANCELLED")
                # Fall through to normal chat
            else:
                # Continue scoping conversation
                from app.llm.project_scoping_stream import generate_project_scoping_stream
                logger.info("[flow_state] Scoping CONTINUE — routing to scoping handler")
                # v2.2: Set sticky model to GPT-5.4 for the duration of scoping
                try:
                    from app.llm.routing.chat_routing import _set_sticky_model
                    import os as _os
                    _set_sticky_model(
                        req.project_id,
                        _os.getenv('CHAT_DEEP_PROVIDER', 'openai'),
                        _os.getenv('CHAT_DEEP_MODEL', 'gpt-5.4'),
                    )
                except Exception:
                    pass
                return StreamingResponse(
                    generate_project_scoping_stream(
                        project_id=req.project_id,
                        message=req.message,
                        db=db,
                        trace=trace,
                        conversation_id=_scope_key,
                    ),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
    except Exception as e:
        logger.debug("[flow_state] Scoping check failed: %s", e)

    # v2.4: "Work on existing project" shortcut.
    # If the user mentions a known project by name (e.g. "work on Driver CoPilot",
    # "let's work on the bridge app"), set the ProjectSession directly.
    # This skips the full scoping flow for projects already in the registry.
    try:
        import re as _re
        # Strip Astra address prefix before checking
        _raw_msg = req.message.strip()
        _stripped_msg = _re.sub(
            r'^(?:[Hh]ey[,.]?\s+)?[Aa][Ss][Tt][Rr][AaOoEe][,.]?\s+', '', _raw_msg
        ).strip()
        _msg_lower = _stripped_msg.lower()
        _work_on_patterns = (
            "work on ", "switch to ", "let's work on ", "lets work on ",
            "open ", "load ", "pick up ", "continue with ",
        )
        _wants_project_switch = any(_msg_lower.startswith(p) for p in _work_on_patterns)

        if _wants_project_switch:
            from app.pipeline_v2.target_registry import resolve_project_from_message
            from app.shared_context.project_session import get_project_session

            _profile = resolve_project_from_message(_stripped_msg)
            _scope_key = str(req.project_id)
            _session = get_project_session(_scope_key)

            # Only switch if we found a specific project (not the default fallback)
            if _profile and _profile.project_id != "astra-backend":
                _session.project_id = _profile.project_id
                _session.project_name = _profile.project_name
                _session.project_type = _profile.language
                _session.project_root = _profile.project_root
                _session.scoping_active = False
                logger.info(
                    "[flow_state] Project switched to: %s (%s) at %s",
                    _profile.project_id, _profile.project_name, _profile.project_root,
                )
                # Return a simple confirmation as SSE stream
                async def _confirm_switch():
                    yield f'data: {{"type": "metadata", "provider": "system", "model": "system"}}\n\n'.encode()
                    yield f'data: {{"type": "token", "content": "\U0001f3af Switched to **{_profile.project_name}** (`{_profile.project_id}`)\\n- Root: `{_profile.project_root}`\\n- Language: {_profile.language}\\n- Framework: {_profile.framework}\\n\\nPipeline will now target this project. What would you like to do?\\n"}}\n\n'.encode()
                    yield f'data: {{"type": "done", "provider": "system", "model": "system"}}\n\n'.encode()
                return StreamingResponse(
                    _confirm_switch(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
    except Exception as e:
        logger.debug("[flow_state] Project switch check failed: %s", e)

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
