# FILE: app/llm/stream_router.py
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

v4.1 (2026-01): CRITICAL FIX - CHAT mode returns early, bypasses job classification
v4.0 (2026-01): ASTRA Translation Layer integration - prevents misfires
v3.4 (2025-12): Added UPDATE ARCHITECTURE command
v3.3 (2025-12): Added sandbox/zombie control integration
v3.2 (2025-12): Split into multiple files for maintainability
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
from .stream_memory import inject_memory_for_stream

# =============================================================================
# TRANSLATION LAYER IMPORT
# =============================================================================
try:
    from app.translation import (
        translate_message_sync,
        TranslationMode,
        CanonicalIntent,
        LatencyTier,
        TranslationResult,
    )
    _TRANSLATION_LAYER_AVAILABLE = True
except ImportError:
    _TRANSLATION_LAYER_AVAILABLE = False
    logging.warning("[stream_router] Translation layer not available - using legacy trigger detection")

# Log introspection integration
try:
    from app.introspection.chat_integration import (
        detect_log_intent,
        handle_log_request,
        format_log_response_for_chat,
    )
    _INTROSPECTION_AVAILABLE = True
except ImportError:
    _INTROSPECTION_AVAILABLE = False

# Sandbox/zombie control integration
try:
    from app.sandbox import handle_sandbox_prompt, detect_sandbox_intent
    _SANDBOX_AVAILABLE = True
except ImportError:
    _SANDBOX_AVAILABLE = False

router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)


# =============================================================================
# LOCAL TOOL TRIGGERS (Legacy - kept for fallback if translation layer unavailable)
# =============================================================================

_ZOBIE_TRIGGER_SET = {"zobie map", "zombie map", "zobie_map", "/zobie_map", "/zombie_map"}

ZOBIE_CONTROLLER_URL = (os.getenv("ORB_ZOBIE_CONTROLLER_URL") or "").rstrip("/")
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\tools\zobie_mapper\zobie_map.py")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", "").strip()
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "300"))
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "").strip()
ZOBIE_MAPPER_ARGS = ZOBIE_MAPPER_ARGS_RAW.split() if ZOBIE_MAPPER_ARGS_RAW else []

from app.llm.local_tools.archmap_helpers import (
    _ARCHMAP_TRIGGER_SET,
    _UPDATE_ARCH_TRIGGER_SET,
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    default_controller_base_url,
    default_zobie_mapper_out_dir,
)

# Resolve sandbox defaults (drive-agnostic) if env vars not provided.
if not ZOBIE_CONTROLLER_URL:
    ZOBIE_CONTROLLER_URL = default_controller_base_url(__file__)
if not ZOBIE_MAPPER_OUT_DIR:
    ZOBIE_MAPPER_OUT_DIR = default_zobie_mapper_out_dir(__file__)

from app.llm.local_tools.zobie_tools import (
    generate_local_architecture_map_stream,
    generate_local_zobie_map_stream,
    generate_update_architecture_stream,
)

# Architecture query (file-based, no service)
from app.llm.local_tools import arch_query


# =============================================================================
# LEGACY TRIGGER DETECTION (used if translation layer not available)
# =============================================================================

def _is_zobie_map_trigger(msg: str) -> bool:
    return (msg or "").strip().lower() in _ZOBIE_TRIGGER_SET


def _is_archmap_trigger(msg: str) -> bool:
    return (msg or "").strip().lower() in _ARCHMAP_TRIGGER_SET


def _is_update_arch_trigger(msg: str) -> bool:
    return (msg or "").strip().lower() in _UPDATE_ARCH_TRIGGER_SET


def _is_introspection_trigger(msg: str) -> bool:
    if not _INTROSPECTION_AVAILABLE:
        return False
    intent = detect_log_intent(msg)
    return intent.is_log_request


def _is_sandbox_trigger(msg: str) -> bool:
    if not _SANDBOX_AVAILABLE:
        return False
    tool, _ = detect_sandbox_intent(msg)
    return tool is not None


def _is_arch_query_trigger(msg: str) -> bool:
    """Detect architecture/signature queries."""
    if not arch_query.is_service_available():
        return False
    msg_lower = (msg or "").lower()
    triggers = ["structure of", "signatures of", "signatures in", "find function", 
                "find class", "find method", "what's in", "whats in", "search for"]
    has_py = ".py" in msg_lower
    has_struct = any(w in msg_lower for w in ["structure", "signature", "function", "class", "method"])
    return any(t in msg_lower for t in triggers) or (has_py and has_struct)


# =============================================================================
# TRANSLATION LAYER ROUTING
# =============================================================================

def _route_via_translation_layer(
    message: str,
    user_id: str = "default",
    conversation_id: Optional[str] = None,
) -> Optional["TranslationResult"]:
    """
    Route message through translation layer.
    Returns TranslationResult or None if layer unavailable.
    """
    if not _TRANSLATION_LAYER_AVAILABLE:
        return None
    
    try:
        result = translate_message_sync(
            text=message,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        logger.debug(
            f"[translation] Mode={result.mode.value}, Intent={result.resolved_intent.value}, "
            f"Execute={result.should_execute}, Tier={result.latency_tier.value}"
        )
        return result
    except Exception as e:
        logger.warning(f"[translation] Layer failed, falling back to legacy: {e}")
        return None


def _intent_to_routing_info(intent: "CanonicalIntent") -> Optional[dict]:
    """Map canonical intent to routing information."""
    mapping = {
        CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES: {
            "type": "local.architecture_map",
            "provider": ARCHMAP_PROVIDER,
            "model": ARCHMAP_MODEL,
            "reason": "Translation layer: CREATE ARCHITECTURE MAP (full)",
        },
        CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY: {
            "type": "local.architecture_map_structure",
            "provider": ARCHMAP_PROVIDER,
            "model": ARCHMAP_MODEL,
            "reason": "Translation layer: Create architecture map (structure only)",
        },
        CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY: {
            "type": "local.update_architecture",
            "provider": "local",
            "model": "architecture_scanner",
            "reason": "Translation layer: UPDATE ARCHITECTURE",
        },
        CanonicalIntent.START_SANDBOX_ZOMBIE_SELF: {
            "type": "local.sandbox",
            "provider": "local",
            "model": "sandbox_manager",
            "reason": "Translation layer: START SANDBOX ZOMBIE",
        },
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB: {
            "type": "high_stakes.critical_pipeline",
            "provider": "anthropic",
            "model": DEFAULT_MODELS.get("anthropic_opus", "claude-opus-4-5-20251101"),
            "reason": "Translation layer: RUN CRITICAL PIPELINE",
        },
    }
    return mapping.get(intent, None)


# =============================================================================
# SANDBOX STREAM
# =============================================================================

async def generate_sandbox_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    """Generate SSE stream for sandbox control commands."""
    try:
        yield "data: " + json.dumps({"type": "token", "content": "🧟 "}) + "\n\n"
        response_text = handle_sandbox_prompt(message)
        chunk_size = 50
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            await asyncio.sleep(0.01)
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=response_text,
            provider="local", model="sandbox_manager"
        ))
        if trace:
            trace.finalize(success=True)
        yield "data: " + json.dumps({
            "type": "done", "provider": "local", "model": "sandbox_manager",
            "total_length": len(response_text)
        }) + "\n\n"
    except Exception as e:
        logger.exception("[sandbox] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


async def generate_arch_query_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    """Generate SSE stream for architecture queries."""
    try:
        yield "data: " + json.dumps({"type": "token", "content": "📐 "}) + "\n\n"
        response_text = arch_query.query_architecture(message)
        chunk_size = 100
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            await asyncio.sleep(0.005)
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=response_text,
            provider="local", model="arch_query"
        ))
        if trace:
            trace.finalize(success=True)
        yield "data: " + json.dumps({
            "type": "done", "provider": "local", "model": "arch_query",
            "total_length": len(response_text)
        }) + "\n\n"
    except Exception as e:
        logger.error(f"[arch_query] Error: {e}")
        if trace:
            trace.log_error("ARCH_QUERY", str(e))
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "message": str(e)}) + "\n\n"


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
        yield "data: " + json.dumps({"type": "token", "content": "📋 Fetching logs...\n\n"}) + "\n\n"
        summary, structured = await handle_log_request(db, intent)
        response = format_log_response_for_chat(summary, structured)
        chunk_size = 80
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            yield "data: " + json.dumps({"type": "token", "content": chunk}) + "\n\n"
            await asyncio.sleep(0.01)
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=response,
            provider="local", model="log_query"
        ))
        if trace:
            trace.finalize(success=True)
        yield "data: " + json.dumps({
            "type": "done", "provider": "introspection", "model": "log_query",
            "total_length": len(response)
        }) + "\n\n"
    except Exception as e:
        logger.exception("[introspection] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


# =============================================================================
# FEEDBACK STREAM
# =============================================================================

async def generate_feedback_stream(
    project_id: int,
    message: str,
    translation_result: "TranslationResult",
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    """Generate SSE stream acknowledging feedback."""
    try:
        response = (
            "✅ Feedback received. This will be used to improve command detection.\n\n"
            f"Original message: \"{message[:100]}{'...' if len(message) > 100 else ''}\"\n"
        )
        
        yield "data: " + json.dumps({"type": "token", "content": response}) + "\n\n"
        
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=response,
            provider="local", model="feedback_handler"
        ))
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done", "provider": "local", "model": "feedback_handler",
            "total_length": len(response)
        }) + "\n\n"
    except Exception as e:
        logger.exception("[feedback] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


# =============================================================================
# CONFIRMATION STREAM
# =============================================================================

async def generate_confirmation_stream(
    project_id: int,
    message: str,
    translation_result: "TranslationResult",
    db: Session,
    trace: Optional[RoutingTrace] = None,
):
    """Generate SSE stream requesting confirmation for high-stakes operation."""
    try:
        prompt = translation_result.confirmation_gate.confirmation_prompt if translation_result.confirmation_gate else None
        if not prompt:
            prompt = f"⚠️ HIGH-STAKES OPERATION\nYou are about to execute: {translation_result.resolved_intent.value}\nType 'Yes' to confirm."
        
        yield "data: " + json.dumps({"type": "token", "content": prompt}) + "\n\n"
        yield "data: " + json.dumps({
            "type": "confirmation_required",
            "intent": translation_result.resolved_intent.value,
            "context": translation_result.extracted_context,
        }) + "\n\n"
        
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=prompt,
            provider="local", model="confirmation_gate"
        ))
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done", "provider": "local", "model": "confirmation_gate",
            "total_length": len(prompt)
        }) + "\n\n"
    except Exception as e:
        logger.exception("[confirmation] Stream failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


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
    """Generate SSE stream for normal LLM chat."""
    full_response = ""
    reasoning_text = ""
    
    try:
        async for chunk in stream_llm(
            provider=provider,
            model=model,
            messages=messages,
            system_prompt=system_prompt,
        ):
            if isinstance(chunk, dict):
                content = chunk.get("text", "") or chunk.get("content", "")
                if chunk.get("type") == "reasoning":
                    reasoning_text += content
                    if enable_reasoning:
                        yield "data: " + json.dumps({"type": "reasoning", "content": content}) + "\n\n"
                    continue
            else:
                content = str(chunk)
            
            if content:
                full_response += content
                yield "data: " + json.dumps({"type": "token", "content": content}) + "\n\n"
        
        # Save to memory
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider=provider
        ))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=full_response,
            provider=provider, model=model
        ))
        
        if trace:
            trace.finalize(success=True)
        
        yield "data: " + json.dumps({
            "type": "done",
            "provider": provider,
            "model": model,
            "total_length": len(full_response),
        }) + "\n\n"
        
    except Exception as e:
        logger.exception("[stream] Failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


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
    
    # =========================================================================
    # TRANSLATION LAYER ROUTING (v4.0 - prevents misfires)
    # =========================================================================
    
    user_id = str(auth.user_id) if hasattr(auth, 'user_id') else "default"
    conversation_id = f"{req.project_id}-{make_session_id(auth)}"
    
    translation_result = _route_via_translation_layer(
        message=req.message,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if translation_result is not None:
        # Translation layer is available - use it for routing decisions
        
        # =====================================================================
        # ARCH QUERY: Check BEFORE chat mode - structure/signature queries
        # =====================================================================
        if _is_arch_query_trigger(req.message):
            routing_reason = "Architecture query (pre-translation check)"
            if trace:
                trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.arch_query", provider="local", model="arch_query", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
                trace.log_routing_decision(job_type="local.arch_query", provider="local", model="arch_query", reason=routing_reason, frontier_override=False, file_map_injected=False)
            return StreamingResponse(
                generate_arch_query_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # =====================================================================
        # CHAT MODE: RETURN EARLY - bypass job classification entirely (v4.1)
        # This prevents "Tell me about Overwatcher" from triggering SpecGate
        # =====================================================================
        if translation_result.mode == TranslationMode.CHAT:
            logger.info(f"[translation] CHAT MODE - returning early, bypassing job classification")
            
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
            
            # Force lightweight provider - NEVER Opus, NEVER high-stakes
            provider = "openai"
            model = DEFAULT_MODELS.get("openai", "gpt-4.1-mini")
            routing_reason = "Translation layer: CHAT mode - forced lightweight (bypassed classification)"
            
            # Check if provider is available, fallback if not
            available = get_available_streaming_provider()
            if available and available != provider:
                from .streaming import get_available_streaming_providers
                providers_available = get_available_streaming_providers()
                if not providers_available.get(provider, False):
                    provider = available
                    model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS.get("openai", "gpt-4.1-mini"))
                    routing_reason += f" | fallback_to={provider}/{model}"
            
            if trace:
                trace.log_request_start(
                    job_type="chat_light", resolved_job_type="chat_light",
                    provider=provider, model=model, reason=routing_reason,
                    frontier_override=False, file_map_injected=False, attachments=None
                )
                trace.log_routing_decision(
                    job_type="chat_light", provider=provider, model=model,
                    reason=routing_reason, frontier_override=False, file_map_injected=False
                )
            
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
            
            # ASTRA Memory injection
            system_prompt = inject_memory_for_stream(db, messages, system_prompt, 'chat_light')

            # RETURN EARLY - never reaches job classification
            return StreamingResponse(
                generate_sse_stream(
                    project_id=req.project_id, message=req.message, provider=provider, model=model,
                    system_prompt=system_prompt, messages=messages, db=db, trace=trace,
                    enable_reasoning=req.enable_reasoning,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # FEEDBACK MODE: Log feedback and acknowledge
        elif translation_result.mode == TranslationMode.FEEDBACK:
            routing_reason = "Translation layer: USER FEEDBACK"
            if trace:
                trace.log_request_start(
                    job_type=req.job_type or "", resolved_job_type="local.feedback",
                    provider="local", model="feedback_handler", reason=routing_reason,
                    frontier_override=False, file_map_injected=False, attachments=None
                )
                trace.log_routing_decision(
                    job_type="local.feedback", provider="local", model="feedback_handler",
                    reason=routing_reason, frontier_override=False, file_map_injected=False
                )
            return StreamingResponse(
                generate_feedback_stream(
                    project_id=req.project_id, message=req.message,
                    translation_result=translation_result, db=db, trace=trace
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # COMMAND MODE: Check if we should execute
        elif translation_result.mode == TranslationMode.COMMAND_CAPABLE:
            
            # Check if awaiting confirmation (high-stakes)
            if (translation_result.confirmation_gate and 
                translation_result.confirmation_gate.awaiting_confirmation):
                routing_reason = f"Translation layer: Awaiting confirmation for {translation_result.resolved_intent.value}"
                if trace:
                    trace.log_request_start(
                        job_type=req.job_type or "", resolved_job_type="local.confirmation",
                        provider="local", model="confirmation_gate", reason=routing_reason,
                        frontier_override=False, file_map_injected=False, attachments=None
                    )
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
                intent = translation_result.resolved_intent
                routing_info = _intent_to_routing_info(intent)
                
                if routing_info:
                    if trace:
                        trace.log_request_start(
                            job_type=req.job_type or "", resolved_job_type=routing_info["type"],
                            provider=routing_info["provider"], model=routing_info["model"],
                            reason=routing_info["reason"], frontier_override=False,
                            file_map_injected=False, attachments=None
                        )
                        trace.log_routing_decision(
                            job_type=routing_info["type"], provider=routing_info["provider"],
                            model=routing_info["model"], reason=routing_info["reason"],
                            frontier_override=False, file_map_injected=False
                        )
                    
                    # Route to appropriate handler
                    if intent == CanonicalIntent.START_SANDBOX_ZOMBIE_SELF:
                        if _SANDBOX_AVAILABLE:
                            return StreamingResponse(
                                generate_sandbox_stream(
                                    project_id=req.project_id, message=req.message, db=db, trace=trace
                                ),
                                media_type="text/event-stream",
                                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                            )
                    
                    elif intent == CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY:
                        return StreamingResponse(
                            generate_update_architecture_stream(
                                project_id=req.project_id, message=req.message, db=db, trace=trace
                            ),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
                    
                    elif intent in (CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES, 
                                    CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY):
                        return StreamingResponse(
                            generate_local_architecture_map_stream(
                                project_id=req.project_id, message=req.message, db=db, trace=trace
                            ),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
            
            # If command mode but not executing (blocked by gates), fall through to lightweight chat
            if not translation_result.should_execute:
                logger.debug(
                    f"[translation] Command blocked: {translation_result.execution_blocked_reason}"
                )
                # Fall through to legacy/normal routing, but this should be rare
    
    # =========================================================================
    # LEGACY FALLBACK ROUTING (if translation layer unavailable)
    # Only reaches here if translation layer is None
    # =========================================================================
    
    # Only use legacy triggers if translation layer is not available
    if translation_result is None:
        # Sandbox control (legacy)
        if _SANDBOX_AVAILABLE and _is_sandbox_trigger(req.message):
            routing_reason = "Local tool trigger: SANDBOX CONTROL"
            if trace:
                trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.sandbox", provider="local", model="sandbox_manager", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
                trace.log_routing_decision(job_type="local.sandbox", provider="local", model="sandbox_manager", reason=routing_reason, frontier_override=False, file_map_injected=False)
            return StreamingResponse(
                generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # UPDATE ARCHITECTURE (legacy)
        if _is_update_arch_trigger(req.message):
            routing_reason = "Local tool trigger: UPDATE ARCHITECTURE"
            if trace:
                trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.update_architecture", provider="local", model="architecture_scanner", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
                trace.log_routing_decision(job_type="local.update_architecture", provider="local", model="architecture_scanner", reason=routing_reason, frontier_override=False, file_map_injected=False)
            return StreamingResponse(
                generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # ARCHITECTURE QUERY (structure/signature queries)
        if _is_arch_query_trigger(req.message):
            routing_reason = "Local tool trigger: ARCHITECTURE QUERY"
            if trace:
                trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.arch_query", provider="local", model="arch_query", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
                trace.log_routing_decision(job_type="local.arch_query", provider="local", model="arch_query", reason=routing_reason, frontier_override=False, file_map_injected=False)
            return StreamingResponse(
                generate_arch_query_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        
        # CREATE ARCHITECTURE MAP (legacy)
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
        
        # Zobie map (legacy)
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
        
        # Log introspection (legacy)
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
    
    # =========================================================================
    # BUILD CONTEXT (only reached if translation layer blocked a command
    # or translation layer unavailable)
    # =========================================================================
    
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
    
    # =========================================================================
    # JOB CONTINUATION
    # =========================================================================
    
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
                project_id=req.project_id, message=req.message, provider=provider, model=model,
                system_prompt=system_prompt, messages=messages, full_context=full_context,
                job_type_str=job_type_value, db=db, trace=trace,
                enable_reasoning=req.enable_reasoning, continue_job_id=req.continue_job_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # =========================================================================
    # NORMAL ROUTING (only if translation layer unavailable or blocked command)
    # =========================================================================
    
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
    
    # ASTRA Memory injection
    system_prompt = inject_memory_for_stream(db, messages, system_prompt, job_type_value)

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