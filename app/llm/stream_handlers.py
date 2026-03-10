# FILE: app/llm/stream_handlers.py
"""
SSE stream generator functions for stream router.

Contains all async generator functions that produce SSE events:
- generate_sse_stream: Normal LLM chat
- generate_sandbox_stream: Sandbox control
- generate_introspection_stream: Log queries
- generate_feedback_stream: Feedback acknowledgment
- generate_confirmation_stream: High-stakes confirmation

v1.1 (2026-01): Added debug logging, proper error event handling from stream_llm
v1.0 (2026-01): Extracted from stream_router.py
"""

import json
import asyncio
import logging
from typing import Optional, List, TYPE_CHECKING

from sqlalchemy.orm import Session

from app.memory import service as memory_service, schemas as memory_schemas
from .streaming import stream_llm

if TYPE_CHECKING:
    from app.llm.audit_logger import RoutingTrace
    from app.translation import TranslationResult

logger = logging.getLogger(__name__)


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
    trace: Optional["RoutingTrace"] = None,
    enable_reasoning: bool = False,
    tools: Optional[List[dict]] = None,
):
    """Generate SSE stream for normal LLM chat."""
    # v1.1: Debug logging
    print(f"[SSE_STREAM] Starting: provider={provider}, model={model}, messages={len(messages)}")
    
    full_response = ""
    reasoning_text = ""
    chunk_count = 0
    
    try:
        # v2.0: Use tool loop when tools are provided, plain stream otherwise
        if tools:
            from app.llm.chat_tool_loop import stream_with_tools
            _stream = stream_with_tools(
                messages=messages,
                system_prompt=system_prompt,
                provider=provider,
                model=model,
                tools=tools,
                enable_reasoning=enable_reasoning,
                max_tokens=16384,  # v2.0: Tool sessions need room for exploration + response
            )
        else:
            _stream = stream_llm(
                provider=provider,
                model=model,
                messages=messages,
                system_prompt=system_prompt,
            )

        async for chunk in _stream:
            # v1.1: Log chunk types for debugging
            if chunk_count == 0:
                print(f"[SSE_STREAM] First chunk received: {type(chunk)}")
            chunk_count += 1
            
            if isinstance(chunk, dict):
                # v1.1: Handle error events from stream_llm
                if chunk.get("type") == "error":
                    error_msg = chunk.get("message", "Unknown error")
                    print(f"[SSE_STREAM] ERROR from stream_llm: {error_msg}")
                    yield "data: " + json.dumps({"type": "error", "error": error_msg}) + "\n\n"
                    if trace:
                        trace.finalize(success=False, error_message=error_msg)
                    return
                
                # v1.1: Handle metadata events
                if chunk.get("type") == "metadata":
                    actual_provider = chunk.get("provider", provider)
                    actual_model = chunk.get("model", model)
                    print(f"[SSE_STREAM] Metadata: provider={actual_provider}, model={actual_model}")
                    yield "data: " + json.dumps({"type": "metadata", "provider": actual_provider, "model": actual_model}) + "\n\n"
                    continue
                
                content = chunk.get("text", "") or chunk.get("content", "")
                # v2.0: Handle tool events from chat_tool_loop
                if chunk.get("type") == "tool_call":
                    yield "data: " + json.dumps({
                        "type": "tool_call",
                        "name": chunk.get("name", ""),
                        "input": chunk.get("input", {}),
                    }) + "\n\n"
                    continue

                if chunk.get("type") == "tool_result":
                    yield "data: " + json.dumps({
                        "type": "tool_result",
                        "name": chunk.get("name", ""),
                        "result_preview": chunk.get("result_preview", ""),
                    }) + "\n\n"
                    continue

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
        
        print(f"[SSE_STREAM] Completed: chunks={chunk_count}, response_len={len(full_response)}")
        
        # v2.5: Extract and save any HTML files from the response
        try:
            from app.llm.html_extractor import extract_and_save_html
            saved_files = extract_and_save_html(full_response)
            if saved_files:
                print(f"[SSE_STREAM] Extracted {len(saved_files)} file(s): {[f['filename'] for f in saved_files]}")
                from app.llm.file_output import sse_file_outputs
                yield sse_file_outputs(saved_files)
        except Exception as extract_err:
            print(f"[SSE_STREAM] File extraction failed (non-fatal): {extract_err}")
            logger.warning("[stream] File extraction failed: %s", extract_err)

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
        print(f"[SSE_STREAM] EXCEPTION: {e}")
        logger.exception("[stream] Failed: %s", e)
        if trace:
            trace.finalize(success=False, error_message=str(e))
        yield "data: " + json.dumps({"type": "error", "error": str(e)}) + "\n\n"


# =============================================================================
# SANDBOX STREAM
# =============================================================================

async def generate_sandbox_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional["RoutingTrace"] = None,
):
    """Generate SSE stream for sandbox control commands."""
    try:
        from app.sandbox import handle_sandbox_prompt
        
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


# =============================================================================
# INTROSPECTION STREAM
# =============================================================================

async def generate_introspection_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional["RoutingTrace"] = None,
):
    """Generate SSE stream for log introspection results."""
    try:
        from app.introspection.chat_integration import (
            detect_log_intent,
            handle_log_request,
            format_log_response_for_chat,
        )
        
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
    trace: Optional["RoutingTrace"] = None,
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
    trace: Optional["RoutingTrace"] = None,
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


__all__ = [
    "generate_sse_stream",
    "generate_sandbox_stream",
    "generate_introspection_stream",
    "generate_feedback_stream",
    "generate_confirmation_stream",
]
