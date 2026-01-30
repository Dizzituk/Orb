# FILE: app/llm/routing/command_dispatch.py
"""
Command dispatch for stream routing - routes intents to stream handlers.

v1.1 (2026-01-28): Added MULTI_FILE_SEARCH and MULTI_FILE_REFACTOR handlers (Level 3)
v1.0 (2026-01-20): Extracted from stream_router.py for modularity.

This module provides:
- `handle_command_execution()` - Main command dispatcher
- `create_error_stream()` - Generates error SSE streams
- Stage trace helpers

The dispatch table maps CanonicalIntent → stream handler.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Any, AsyncIterator

from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.llm.translation_routing import (
    CanonicalIntent,
    intent_to_routing_info,
    _get_spec_gate_config,
    _get_critical_pipeline_config,
)

from app.llm.legacy_triggers import ARCHMAP_PROVIDER, ARCHMAP_MODEL

from .handler_registry import (
    # Availability flags
    _WEAVER_AVAILABLE,
    _SPEC_GATE_STREAM_AVAILABLE,
    _CRITICAL_PIPELINE_AVAILABLE,
    _OVERWATCHER_AVAILABLE,
    _LOCAL_TOOLS_AVAILABLE,
    _SANDBOX_AVAILABLE,
    _RAG_STREAM_AVAILABLE,
    _EMBEDDING_STREAM_AVAILABLE,
    _STAGE_TRACE_AVAILABLE,
    # Handlers
    generate_weaver_stream,
    generate_spec_gate_stream,
    generate_critical_pipeline_stream,
    generate_overwatcher_stream,
    generate_sandbox_stream,
    generate_local_architecture_map_stream,
    generate_full_architecture_map_stream,
    generate_update_architecture_stream,
    generate_sandbox_structure_scan_stream,
    generate_latest_architecture_map_stream,
    generate_latest_codebase_report_full_stream,
    generate_filesystem_query_stream,
    generate_codebase_report_stream,
    generate_rag_query_stream,
    generate_embedding_status_stream,
    generate_embeddings_stream,
    StageTrace,
    get_env_model_audit,
    log_handler_availability,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE TRACE HELPERS
# =============================================================================

def create_stage_trace(
    command_type: str,
    project_id: int,
    job_id: Optional[str] = None,
) -> Optional[StageTrace]:
    """Create a stage trace if available."""
    if _STAGE_TRACE_AVAILABLE and StageTrace:
        return StageTrace.start(command_type, project_id=project_id, job_id=job_id)
    return None


def log_routing_failure(
    stage_trace: Optional[StageTrace],
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
    
    logger.warning(f"[command_dispatch] ROUTING_FAILURE: {reason} (handler={expected_handler})")
    
    if stage_trace:
        stage_trace.record_routing_failure(reason, expected_handler, fallback_action)


# =============================================================================
# ERROR STREAM GENERATORS
# =============================================================================

async def create_error_stream(
    error_type: str,
    error_message: str,
    intent: Optional[str] = None,
    reason: Optional[str] = None,
) -> AsyncIterator[str]:
    """Generate an error SSE stream."""
    yield "data: " + json.dumps({'type': 'error', 'error': error_type}) + "\n\n"
    yield "data: " + json.dumps({'type': 'token', 'content': error_message}) + "\n\n"
    
    if intent and reason:
        yield "data: " + json.dumps({
            'type': 'command_blocked',
            'intent': intent,
            'reason': reason,
        }) + "\n\n"
    
    yield "data: " + json.dumps({'type': 'done', 'provider': 'system', 'model': 'command_router'}) + "\n\n"


def _make_unavailable_error(handler_name: str, module_path: str) -> str:
    """Generate standard handler unavailable error message."""
    return (
        f"⚠️ **{handler_name} Not Available**\n\n"
        f"The {handler_name.lower()} module failed to import.\n"
        f"Check server logs for details.\n\n"
        f"**Possible solutions:**\n"
        f"1. Ensure `{module_path}` exists\n"
        f"2. Check for import errors in the module\n"
        f"3. Restart the backend server"
    )


# =============================================================================
# MAIN COMMAND DISPATCHER
# =============================================================================

def handle_command_execution(
    req: Any,  # StreamRequest
    translation_result: Any,  # TranslationResult
    db: Session,
    trace: Any,
    conversation_id: str,
    stage_trace: Optional[StageTrace] = None,
) -> Optional[StreamingResponse]:
    """
    Handle approved command execution by dispatching to the correct stream handler.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        translation_result: Translation layer result with resolved_intent
        db: Database session
        trace: Audit trace
        conversation_id: Conversation ID string
        stage_trace: Optional stage trace for debugging
    
    Returns:
        StreamingResponse if command handled, None if should fall through
    """
    intent = translation_result.resolved_intent
    routing_info = intent_to_routing_info(intent)
    
    if not routing_info:
        log_routing_failure(
            stage_trace,
            f"No routing info for intent: {intent.value if intent else 'None'}",
            "intent_to_routing_info",
            "falling through to normal routing"
        )
        return None
    
    # Log the resolved routing
    print(f"[COMMAND_ROUTE] Intent: {intent.value if intent else 'None'}")
    print(f"[COMMAND_ROUTE] Routing: type={routing_info['type']}, provider={routing_info['provider']}, model={routing_info['model']}")
    print(f"[COMMAND_ROUTE] Reason: {routing_info['reason']}")
    
    if trace:
        trace.log_request_start(
            job_type=req.job_type or "",
            resolved_job_type=routing_info["type"],
            provider=routing_info["provider"],
            model=routing_info["model"],
            reason=routing_info["reason"],
            frontier_override=False,
            file_map_injected=False,
            attachments=None,
        )
    
    # Standard SSE headers
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    
    # =========================================================================
    # DISPATCH TABLE
    # =========================================================================
    
    # --- Sandbox ---
    if intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF and _SANDBOX_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("sandbox", provider="local", model="sandbox_manager")
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    # --- Update Architecture ---
    if intent == CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY and _LOCAL_TOOLS_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("update_architecture", provider="local", model="architecture_scanner")
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    # --- Sandbox Structure Scan ---
    if intent == CanonicalIntent.SCAN_SANDBOX_STRUCTURE and _LOCAL_TOOLS_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("sandbox_structure_scan", provider="local", model="sandbox_structure_scanner")
        return StreamingResponse(
            generate_sandbox_structure_scan_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    # --- Architecture Map (ALL CAPS) - Full scan ---
    if intent == CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES and _LOCAL_TOOLS_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("full_architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL)
        return StreamingResponse(
            generate_full_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    # --- Architecture Map (lowercase) - DB only ---
    if intent == CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY and _LOCAL_TOOLS_AVAILABLE:
        if stage_trace:
            stage_trace.enter_stage("architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL)
        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    # --- Weaver ---
    if intent == CanonicalIntent.WEAVER_BUILD_SPEC:
        if _WEAVER_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("weaver", provider=routing_info["provider"], model=routing_info["model"])
            return StreamingResponse(
                generate_weaver_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Weaver handler not available", "generate_weaver_stream")
    
    # --- Spec Gate ---
    if intent == CanonicalIntent.SEND_TO_SPEC_GATE:
        if _SPEC_GATE_STREAM_AVAILABLE:
            spec_provider, spec_model = _get_spec_gate_config()
            print(f"[SPEC_GATE_ROUTE] Using provider={spec_provider}, model={spec_model}")
            if stage_trace:
                stage_trace.enter_stage("spec_gate", provider=spec_provider, model=spec_model)
            return StreamingResponse(
                generate_spec_gate_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Spec Gate stream handler not available", "generate_spec_gate_stream")
    
    # --- Critical Pipeline ---
    if intent == CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB:
        if _CRITICAL_PIPELINE_AVAILABLE:
            crit_provider, crit_model = _get_critical_pipeline_config()
            print(f"[CRITICAL_PIPELINE_ROUTE] Using provider={crit_provider}, model={crit_model}")
            if stage_trace:
                stage_trace.enter_stage("critical_pipeline", provider=crit_provider, model=crit_model)
            return StreamingResponse(
                generate_critical_pipeline_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(
                stage_trace,
                "CRITICAL PIPELINE HANDLER NOT AVAILABLE - Command will NOT execute!",
                "generate_critical_pipeline_stream",
                "Check if app/llm/critical_pipeline_stream.py exists and imports correctly"
            )
            log_handler_availability()
    
    # --- Overwatcher ---
    if intent == CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES:
        if _OVERWATCHER_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("overwatcher", provider=routing_info["provider"], model=routing_info["model"])
            return StreamingResponse(
                generate_overwatcher_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Overwatcher handler not available", "generate_overwatcher_stream")
    
    # --- RAG Codebase Query ---
    if intent == CanonicalIntent.RAG_CODEBASE_QUERY:
        if _RAG_STREAM_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("rag_query", provider="local", model="rag_answerer")
            return StreamingResponse(
                generate_rag_query_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "RAG stream handler not available", "generate_rag_query_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("RAG Query Handler", "app/llm/rag_stream.py")
            return StreamingResponse(
                create_error_stream("RAG handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Embedding Status ---
    if intent == CanonicalIntent.EMBEDDING_STATUS:
        if _EMBEDDING_STREAM_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("embedding_status", provider="local", model="embedding_manager")
            return StreamingResponse(
                generate_embedding_status_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Embedding stream handler not available", "generate_embedding_status_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Embedding Status Handler", "app/llm/embedding_stream.py")
            return StreamingResponse(
                create_error_stream("Embedding status handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Generate Embeddings ---
    if intent == CanonicalIntent.GENERATE_EMBEDDINGS:
        if _EMBEDDING_STREAM_AVAILABLE:
            if stage_trace:
                stage_trace.enter_stage("generate_embeddings", provider="local", model="embedding_manager")
            return StreamingResponse(
                generate_embeddings_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Embedding stream handler not available", "generate_embeddings_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Generate Embeddings Handler", "app/llm/embedding_stream.py")
            return StreamingResponse(
                create_error_stream("Generate embeddings handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Filesystem Query ---
    if intent == CanonicalIntent.FILESYSTEM_QUERY:
        if _LOCAL_TOOLS_AVAILABLE and generate_filesystem_query_stream:
            if stage_trace:
                stage_trace.enter_stage("filesystem_query", provider="local", model="filesystem_scanner")
            return StreamingResponse(
                generate_filesystem_query_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Local tools not available for filesystem query", "generate_filesystem_query_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Filesystem Query Handler", "app/llm/local_tools/zobie_tools.py")
            return StreamingResponse(
                create_error_stream("Filesystem query handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Codebase Report ---
    if intent == CanonicalIntent.CODEBASE_REPORT:
        if _LOCAL_TOOLS_AVAILABLE and generate_codebase_report_stream:
            if stage_trace:
                stage_trace.enter_stage("codebase_report", provider="local", model="codebase_report")
            return StreamingResponse(
                generate_codebase_report_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Local tools not available for codebase report", "generate_codebase_report_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Codebase Report Handler", "app/llm/local_tools/zobie_tools.py")
            return StreamingResponse(
                create_error_stream("Codebase report handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Latest Architecture Map ---
    if intent == CanonicalIntent.LATEST_ARCHITECTURE_MAP:
        if _LOCAL_TOOLS_AVAILABLE and generate_latest_architecture_map_stream:
            if stage_trace:
                stage_trace.enter_stage("latest_architecture_map", provider="local", model="latest_report_resolver")
            return StreamingResponse(
                generate_latest_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Local tools not available for latest architecture map", "generate_latest_architecture_map_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Latest Architecture Map Handler", "app/llm/local_tools/zobie_tools.py")
            return StreamingResponse(
                create_error_stream("Latest architecture map handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Latest Codebase Report Full ---
    if intent == CanonicalIntent.LATEST_CODEBASE_REPORT_FULL:
        if _LOCAL_TOOLS_AVAILABLE and generate_latest_codebase_report_full_stream:
            if stage_trace:
                stage_trace.enter_stage("latest_codebase_report_full", provider="local", model="latest_report_resolver")
            return StreamingResponse(
                generate_latest_codebase_report_full_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Local tools not available for latest codebase report full", "generate_latest_codebase_report_full_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Latest Codebase Report Handler", "app/llm/local_tools/zobie_tools.py")
            return StreamingResponse(
                create_error_stream("Latest codebase report handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Multi-File Search (v1.1 - Level 3) ---
    if intent == CanonicalIntent.MULTI_FILE_SEARCH:
        # Multi-file search goes through Weaver → SpecGate flow
        # SpecGate will detect multi-file intent and run file discovery
        if _WEAVER_AVAILABLE:
            print(f"[MULTI_FILE_ROUTE] MULTI_FILE_SEARCH -> Weaver/SpecGate flow")
            if stage_trace:
                stage_trace.enter_stage("multi_file_search", provider=routing_info["provider"], model=routing_info["model"])
            return StreamingResponse(
                generate_weaver_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Weaver handler not available for multi-file search", "generate_weaver_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Multi-File Search Handler", "app/llm/weaver_stream.py")
            return StreamingResponse(
                create_error_stream("Multi-file search handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # --- Multi-File Refactor (v1.1 - Level 3) ---
    if intent == CanonicalIntent.MULTI_FILE_REFACTOR:
        # Multi-file refactor goes through Weaver → SpecGate flow
        # SpecGate will detect multi-file intent, run discovery, and request confirmation
        if _WEAVER_AVAILABLE:
            print(f"[MULTI_FILE_ROUTE] MULTI_FILE_REFACTOR -> Weaver/SpecGate flow")
            if stage_trace:
                stage_trace.enter_stage("multi_file_refactor", provider=routing_info["provider"], model=routing_info["model"])
            return StreamingResponse(
                generate_weaver_stream(
                    project_id=req.project_id,
                    message=req.message,
                    db=db,
                    trace=trace,
                    conversation_id=str(req.project_id),
                ),
                media_type="text/event-stream",
                headers=sse_headers,
            )
        else:
            log_routing_failure(stage_trace, "Weaver handler not available for multi-file refactor", "generate_weaver_stream")
            log_handler_availability()
            
            error_msg = _make_unavailable_error("Multi-File Refactor Handler", "app/llm/weaver_stream.py")
            return StreamingResponse(
                create_error_stream("Multi-file refactor handler not available", error_msg),
                media_type="text/event-stream",
                headers=sse_headers,
            )
    
    # No handler matched - return None to fall through
    return None


__all__ = [
    "handle_command_execution",
    "create_stage_trace",
    "log_routing_failure",
    "create_error_stream",
]
