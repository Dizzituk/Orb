# FILE: app/llm/routing/chat_routing.py
"""
Chat and normal routing handlers for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.

This module provides:
- `handle_chat_mode()` - Lightweight chat routing
- `handle_normal_routing()` - Standard job-type routing
- `handle_legacy_triggers()` - Fallback for when translation layer unavailable
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.memory import service as memory_service
# Import from pipeline.high_stakes to avoid circular import with router.py
from app.llm.pipeline.high_stakes import is_high_stakes_job, is_opus_model
from app.llm.streaming import get_available_streaming_provider, get_available_streaming_providers

from app.llm.stream_utils import (
    DEFAULT_MODELS,
    classify_job_type,
    select_provider_for_job_type,
)

from app.llm.legacy_triggers import (
    is_zobie_map_trigger,
    is_archmap_trigger,
    is_update_arch_trigger,
    is_introspection_trigger,
    is_sandbox_trigger,
)

from .handler_registry import (
    # Availability flags
    _LOCAL_TOOLS_AVAILABLE,
    _SANDBOX_AVAILABLE,
    _INTROSPECTION_AVAILABLE,
    _RAG_STREAM_AVAILABLE,
    # Handlers
    generate_sse_stream,
    generate_sandbox_stream,
    generate_introspection_stream,
    generate_local_architecture_map_stream,
    generate_local_zobie_map_stream,
    generate_update_architecture_stream,
    generate_rag_query_stream,
    generate_high_stakes_critique_stream,
)

from .prompt_builders import (
    build_system_prompt,
    build_messages,
    build_full_context,
)

from .rag_fallback import is_architecture_query

logger = logging.getLogger(__name__)


# =============================================================================
# CHAT MODE HANDLER
# =============================================================================

def handle_chat_mode(
    req: Any,  # StreamRequest
    project: Any,  # Project model
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """
    Handle CHAT mode - lightweight model, no commands.
    
    v4.8: Uses stage_models for provider/model selection with debug logging.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        project: Project ORM object
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse for chat
    """
    print(f"[CHAT_MODE] Handling chat for project={req.project_id}, message={req.message[:50]}...")
    
    # Build context
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    
    # Get provider/model from stage_models
    try:
        from app.llm.stage_models import get_chat_config
        chat_config = get_chat_config()
        provider = chat_config.provider
        model = chat_config.model
        print(f"[CHAT_MODE] Using stage_models: provider={provider}, model={model}")
    except ImportError:
        provider = "openai"
        model = DEFAULT_MODELS.get("openai", "gpt-4.1-mini")
        print(f"[CHAT_MODE] stage_models unavailable, using fallback: provider={provider}, model={model}")
    
    # Check provider availability
    available = get_available_streaming_provider()
    print(f"[CHAT_MODE] Available streaming provider: {available}")
    
    if provider not in ("openai", "anthropic", "google", "gemini"):
        print(f"[CHAT_MODE] WARNING: Unknown provider '{provider}', defaulting to available: {available}")
        provider = available or "openai"
    
    providers_available = get_available_streaming_providers()
    print(f"[CHAT_MODE] Provider availability: {providers_available}")
    
    if not providers_available.get(provider, False) and not providers_available.get("gemini" if provider == "google" else provider, False):
        print(f"[CHAT_MODE] Provider {provider} not available, falling back to {available}")
        if available:
            provider = available
    
    # Build messages
    messages = build_messages(
        message=req.message,
        project_id=req.project_id,
        db=db,
        include_history=req.include_history,
        history_limit=req.history_limit,
    )
    
    # Build system prompt (includes capability layer)
    system_prompt = build_system_prompt(project, full_context)
    
    print(f"[CHAT_MODE] Calling generate_sse_stream: provider={provider}, model={model}, messages={len(messages)}")
    
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
            trace=trace,
            enable_reasoning=req.enable_reasoning,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# NORMAL ROUTING HANDLER
# =============================================================================

def handle_normal_routing(
    req: Any,  # StreamRequest
    project: Any,  # Project model
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """
    Handle normal job-type routing with RAG fallback.
    
    v4.12: Includes RAG fallback for architecture queries.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        project: Project ORM object
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse for the routed job
    """
    
    # =========================================================================
    # RAG FALLBACK: Detect architecture questions when translation layer fails
    # =========================================================================
    if _RAG_STREAM_AVAILABLE and is_architecture_query(req.message):
        print(f"[NORMAL_ROUTING] RAG fallback: detected architecture query")
        print(f"[NORMAL_ROUTING]   message={req.message[:80]}...")
        return StreamingResponse(
            generate_rag_query_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
                trace=trace,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Build context
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    
    # Job continuation
    if req.continue_job_id and req.job_state == "needs_spec_clarification":
        provider = "anthropic"
        model = DEFAULT_MODELS["anthropic_opus"]
        messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
        system_prompt = build_system_prompt(project, full_context)
        
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str="architecture_design",
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
                continue_job_id=req.continue_job_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Normal job classification
    job_type = classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value
    
    # Determine provider/model
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
    
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    
    # Build messages and system prompt
    messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
    system_prompt = build_system_prompt(project, full_context)
    
    # High-stakes routing
    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str=job_type_value,
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Normal stream
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
            trace=trace,
            enable_reasoning=req.enable_reasoning,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# LEGACY TRIGGERS HANDLER
# =============================================================================

def handle_legacy_triggers(
    req: Any,  # StreamRequest
    db: Session,
    trace: Any,
) -> Optional[StreamingResponse]:
    """
    Handle legacy triggers when translation layer unavailable.
    
    Args:
        req: StreamRequest with project_id and message
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse if trigger matched, None otherwise
    """
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    
    if _SANDBOX_AVAILABLE and is_sandbox_trigger(req.message):
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_update_arch_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_archmap_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_zobie_map_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if _INTROSPECTION_AVAILABLE and is_introspection_trigger(req.message):
        return StreamingResponse(
            generate_introspection_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    return None


__all__ = [
    "handle_chat_mode",
    "handle_normal_routing",
    "handle_legacy_triggers",
]
