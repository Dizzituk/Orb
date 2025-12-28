# FILE: app/llm/stream_router.py
r"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

v3.2 (2025-12): Split into multiple files for maintainability
- stream_utils.py: Helper functions
- high_stakes_stream.py: High-stakes critique generator
- stream_router.py: Main router (this file)
"""

import os
import json
import uuid
import logging
import asyncio
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas

from .streaming import stream_llm, get_available_streaming_provider
from app.llm.audit_logger import get_audit_logger, RoutingTrace
from app.llm.router import is_high_stakes_job, is_opus_model

# Import from split modules
from .stream_utils import (
    DEFAULT_MODELS,
    parse_reasoning_tags,
    make_session_id,
    extract_usage_tokens,
    build_context_block,
    build_document_context,
    get_semantic_context,
    classify_job_type,
    select_provider_for_job_type,
)
from .high_stakes_stream import generate_high_stakes_critique_stream

# v0.16.0: Log introspection integration
try:
    from app.introspection.chat_integration import (
        detect_log_intent,
        handle_log_request,
        format_log_response_for_chat,
    )
    _INTROSPECTION_AVAILABLE = True
except ImportError:
    _INTROSPECTION_AVAILABLE = False

router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)


# =============================================================================
# LOCAL TOOL TRIGGERS
# =============================================================================

_ZOBIE_TRIGGER_SET = {"zobie map", "zombie map", "zobie_map", "/zobie_map", "/zombie_map"}

ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\tools\zobie_mapper\zobie_map.py")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", r"D:\tools\zobie_mapper\out")
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "300"))
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "").strip()
ZOBIE_MAPPER_ARGS = ZOBIE_MAPPER_ARGS_RAW.split() if ZOBIE_MAPPER_ARGS_RAW else []

from app.llm.local_tools.archmap_helpers import (
    _ARCHMAP_TRIGGER_SET,
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
)
from app.llm.local_tools.zobie_tools import (
    generate_local_architecture_map_stream,
    generate_local_zobie_map_stream,
)


def _is_zobie_map_trigger(msg: str) -> bool:
    return (msg or "").strip().lower() in _ZOBIE_TRIGGER_SET


def _is_archmap_trigger(msg: str) -> bool:
    return (msg or "").strip().lower() in _ARCHMAP_TRIGGER_SET


def _is_introspection_trigger(msg: str) -> bool:
    if not _INTROSPECTION_AVAILABLE:
        return False
    intent = detect_log_intent(msg)
    return intent.is_log_request


# =============================================================================
# INTROSPECTION STREAM
# =============================================================================

async def generate_introspection_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    """Generate SSE stream for log introspection results."""
    try:
        intent = detect_log_intent(message)
        
        yield "data: " + json.dumps({
            'type': 'token',
            'content': '📋 Fetching logs...\n\n'
        }) + "\n\n"
        
        summary, structured = await handle_log_request(db, intent)
        response = format_log_response_for_chat(summary, structured)
        
        chunk_size = 80
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield "data: " + json.dumps({'type': 'token', 'content': chunk}) + "\n\n"
            await asyncio.sleep(0.01)

        from app.memory import service as mem_svc, schemas as mem_schemas
        mem_svc.create_message(db, mem_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        mem_svc.create_message(db, mem_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=response, provider="introspection", model="log_query"
        ))

        if trace:
            trace.finalize(success=True)

        yield "data: " + json.dumps({
            'type': 'done',
            'provider': 'introspection',
            'model': 'log_query',
            'total_length': len(response)
        }) + "\n\n"

    except Exception as e:
        logger.exception("[introspection] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({'type': 'error', 'error': str(e)}) + "\n\n"


# =============================================================================
# NORMAL SSE STREAM
# =============================================================================

async def generate_sse_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    db: Session,
    trace: Optional[RoutingTrace] = None,
    enable_reasoning: bool = False,
):
    """Generate SSE stream for normal (non-high-stakes) LLM responses."""
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    current_provider = provider
    current_model = model
    final_usage = None
    accumulated = ""
    reasoning_content = ""

    try:
        async for event in stream_llm(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            enable_reasoning=enable_reasoning,
        ):
            event_type = event.get("type")
            if event_type == "token":
                chunk = event.get("content", "")
                accumulated += chunk
                yield "data: " + json.dumps(event) + "\n\n"
            elif event_type == "reasoning":
                reasoning_content += event.get("content", "")
                yield "data: " + json.dumps(event) + "\n\n"
            elif event_type == "usage":
                final_usage = event.get("usage")
            elif event_type == "error":
                yield "data: " + json.dumps(event) + "\n\n"
                if trace and not trace_finished:
                    trace.log_model_call("primary", current_provider, current_model, "primary", 0, 0, 0, success=False, error=event.get("error"))
                    trace.finalize(success=False, error_message=event.get("error"))
                    trace_finished = True
                return
            elif event_type == "done":
                current_provider = event.get("provider", current_provider)
                current_model = event.get("model", current_model)

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        logger.exception("[stream] Stream failed: %s", e)
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        if trace and not trace_finished:
            trace.log_model_call("primary", current_provider, current_model, "primary", 0, 0, duration_ms, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        yield "data: " + json.dumps({'type': 'error', 'error': str(e)}) + "\n\n"
        return

    duration_ms = max(0, int(loop.time() * 1000) - started_ms)

    answer_content, extracted_reasoning = parse_reasoning_tags(accumulated)
    if extracted_reasoning:
        reasoning_content = extracted_reasoning

    memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
    memory_service.create_message(
        db,
        memory_schemas.MessageCreate(
            project_id=project_id,
            role="assistant",
            content=answer_content,
            provider=current_provider,
            model=current_model,
            reasoning=reasoning_content or None,
        ),
    )

    if trace and not trace_finished:
        prompt_tokens, completion_tokens = extract_usage_tokens(final_usage)
        trace.log_model_call("primary", current_provider, current_model, "primary", prompt_tokens, completion_tokens, duration_ms, success=True)
        trace.finalize(success=True)
        trace_finished = True

    yield "data: " + json.dumps({'type': 'done', 'provider': current_provider, 'model': current_model, 'total_length': len(answer_content)}) + "\n\n"


# =============================================================================
# REQUEST MODEL
# =============================================================================

class StreamChatRequest(BaseModel):
    project_id: int
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    job_type: Optional[str] = None
    requested_type: Optional[str] = None
    include_history: bool = True
    history_limit: int = 20
    use_semantic_search: bool = False
    enable_reasoning: bool = False
    continue_job_id: Optional[str] = None
    job_state: Optional[str] = None


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@router.post("/chat")
async def stream_chat(req: StreamChatRequest, db: Session = Depends(get_db), auth: AuthResult = Depends(require_auth)):
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")

    audit = get_audit_logger()
    trace: Optional[RoutingTrace] = None
    request_id = str(uuid.uuid4())
    if audit:
        trace = audit.start_trace(session_id=make_session_id(auth), project_id=req.project_id, user_text=req.message, request_id=request_id)

    # Local tool triggers
    if _is_archmap_trigger(req.message):
        routing_reason = "Local tool trigger: CREATE ARCHITECTURE MAP"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if _is_zobie_map_trigger(req.message):
        routing_reason = "Local tool trigger: ZOBIE MAP"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.zobie_map", provider="local", model="zobie_mapper", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.zobie_map", provider="local", model="zobie_mapper", reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if _INTROSPECTION_AVAILABLE and _is_introspection_trigger(req.message):
        routing_reason = "Local tool trigger: LOG INTROSPECTION"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.introspection", provider="introspection", model="log_query", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.introspection", provider="introspection", model="log_query", reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_introspection_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Build context
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
        print(f"[stream_router] Continuing job {req.continue_job_id} (state: {req.job_state})")
        
        provider = "anthropic"
        model = DEFAULT_MODELS["anthropic_opus"]
        job_type_value = "architecture_design"
        routing_reason = f"Job continuation: {req.continue_job_id} (spec clarification)"
        
        if trace:
            trace.log_request_start(job_type="architecture_design", resolved_job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False)
        
        messages: List[dict] = []
        if req.include_history:
            history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
            messages = [{"role": msg.role, "content": msg.content} for msg in history]
        messages.append({"role": "user", "content": req.message})
        
        system_prompt = f"Project: {project.name}."
        if project.description:
            system_prompt += f" {project.description}"
        if full_context:
            system_prompt += f"\n\nYou have access to the following context from this project:\n\n{full_context}"
        
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
                continue_job_id=req.continue_job_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Normal routing
    job_type = classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value

    if req.provider and req.model:
        provider = req.provider
        model = req.model
        routing_reason = "Explicit provider+model from request"
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = "Explicit provider from request (default model)"
    else:
        provider, model = select_provider_for_job_type(job_type)
        routing_reason = f"Job-type routing: {job_type_value} -> {provider}/{model}"

    available = get_available_streaming_provider()
    if not available:
        if trace:
            trace.log_error("STREAM", "no_provider_available")
            trace.finalize(success=False, error_message="No LLM provider available")
        raise HTTPException(status_code=503, detail="No LLM provider available")

    from .streaming import get_available_streaming_providers
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = f"{routing_reason} | fallback_to={provider}/{model}"

    if trace:
        trace.log_request_start(job_type=req.job_type or "", resolved_job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
        trace.log_routing_decision(job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False)

    messages: List[dict] = []
    if req.include_history:
        history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        messages = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append({"role": "user", "content": req.message})

    system_prompt = f"Project: {project.name}."
    if project.description:
        system_prompt += f" {project.description}"

    if full_context:
        system_prompt += f"""

You have access to the following context from this project:

{full_context}

Use this context to answer the user's questions. If asked about people, documents,
or information that appears in the context above, use that information to respond.
Do NOT claim you don't have information if it's present in the context."""

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