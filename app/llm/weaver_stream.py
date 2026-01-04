# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA

Provides streaming spec generation with incremental updates.
Matches stream_router.py calling convention.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, Dict, List, Optional, Any

from sqlalchemy.orm import Session

from app.llm.audit_logger import RoutingTrace
from app.specs import service as specs_service

from app.llm.weaver_stream_core import (
    WeaverContext,
    gather_weaver_context,
    gather_weaver_delta_context,
    build_weaver_prompt,
    build_weaver_update_prompt,
    parse_weaver_response,
    build_spec_from_dict,
    _get_last_consumed_message_id_from_spec,
    _to_jsonable,
)

# Import streaming functions for all providers
try:
    from app.llm.streaming import stream_openai, stream_anthropic, stream_gemini
    _STREAMING_AVAILABLE = True
except ImportError:
    try:
        from .streaming import stream_openai, stream_anthropic, stream_gemini  # type: ignore
        _STREAMING_AVAILABLE = True
    except ImportError:
        stream_openai = None  # type: ignore
        stream_anthropic = None  # type: ignore
        stream_gemini = None  # type: ignore
        _STREAMING_AVAILABLE = False

logger = logging.getLogger(__name__)


def _serialize_sse(data: Dict[str, Any]) -> bytes:
    """Serialize dict to SSE format."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _get_weaver_config() -> tuple[str, str]:
    """
    Get provider and model for weaver from environment.
    
    Returns (provider, model)
    """
    provider = os.getenv("WEAVER_PROVIDER", "openai")
    model = os.getenv("WEAVER_MODEL", "gpt-4.1-mini")
    
    return provider, model


def _get_streaming_function(provider: str):
    """Get the appropriate streaming function for the provider."""
    provider_lower = provider.lower()
    
    if provider_lower in ("openai", "openai-compatible"):
        return stream_openai
    elif provider_lower in ("anthropic", "claude"):
        return stream_anthropic
    elif provider_lower in ("google", "gemini"):
        return stream_gemini
    else:
        logger.warning("[WEAVER] Unknown provider '%s', defaulting to OpenAI", provider)
        return stream_openai


async def generate_weaver_stream(
    *,
    project_id: int,
    message: str,
    db: Session,
    trace: RoutingTrace,
    conversation_id: str,
) -> AsyncIterator[bytes]:
    """
    Weaver handler for incremental spec building.
    
    Called by stream_router.py for WEAVER_BUILD_SPEC commands.
    
    Incremental behavior:
    - First run: builds spec from recent context
    - Subsequent runs with no new messages: reuses previous spec
    - Subsequent runs with new messages: updates spec incrementally
    
    Checkpoint persists across restarts in Spec database records.
    """
    logger.info("[WEAVER] Starting for project_id=%s", project_id)
    
    # Get provider and model from environment
    provider, model = _get_weaver_config()
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        logger.error("[WEAVER] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    try:
        logger.info("[WEAVER] Using provider=%s, model=%s (from env)", provider, model)
        
        # Get the streaming function for this provider
        stream_fn = _get_streaming_function(provider)
        if stream_fn is None:
            error_msg = f"Streaming function not available for provider: {provider}"
            logger.error("[WEAVER] %s", error_msg)
            yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # Load previous spec (if any)
        last_spec = specs_service.get_latest_spec(db, project_id=project_id)
        
        # Determine mode and gather context
        mode = "create"
        last_consumed_message_id: Optional[int] = None
        previous_spec_core: Optional[Dict[str, Any]] = None
        previous_weak_spots: List[str] = []
        
        if last_spec:
            # Extract checkpoint from previous spec
            last_consumed_message_id = _get_last_consumed_message_id_from_spec(last_spec)
            logger.info("[WEAVER] Found previous spec_id=%s, last_consumed_message_id=%s",
                       last_spec.spec_id, last_consumed_message_id)
            
            # Extract spec core for update prompt
            try:
                spec_schema = specs_service.get_spec_schema(last_spec)
                previous_spec_core = _to_jsonable(spec_schema)
                
                # Extract weak spots from content_json metadata
                if hasattr(last_spec, "content_json") and last_spec.content_json:
                    if isinstance(last_spec.content_json, dict):
                        metadata = last_spec.content_json.get("metadata", {})
                        if isinstance(metadata, dict):
                            previous_weak_spots = metadata.get("weak_spots", [])
                
                mode = "update"
            except Exception as e:
                logger.warning("[WEAVER] Failed to load previous spec core: %s", e)
                mode = "create"
        
        # Gather context based on mode
        if mode == "update" and last_consumed_message_id is not None:
            context = gather_weaver_delta_context(
                db=db,
                project_id=project_id,
                since_message_id=last_consumed_message_id,
            )
            logger.info("[WEAVER] Delta mode: %d messages since message_id=%d",
                       len(context.messages), last_consumed_message_id)
        else:
            context = gather_weaver_context(db=db, project_id=project_id)
            logger.info("[WEAVER] Fresh mode: %d messages", len(context.messages))
        
        # If no new messages in update mode → reuse previous spec
        if mode == "update" and len(context.messages) == 0:
            logger.info("[WEAVER] No new messages; reusing spec_id=%s", last_spec.spec_id)
            
            reuse_message = f"""📋 Spec already up to date

Using existing spec `{last_spec.spec_id[:12]}...` (no new messages since last weave).

Ready for Spec Gate review."""
            
            yield _serialize_sse({"type": "token", "content": reuse_message})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # Build prompt
        if mode == "update" and previous_spec_core:
            prompt_text = build_weaver_update_prompt(
                previous_spec_core=previous_spec_core,
                previous_weak_spots=previous_weak_spots,
                delta_context=context,
            )
        else:
            prompt_text = build_weaver_prompt(context)
        
        logger.info("[WEAVER] Built %s prompt (%d tokens estimated)",
                   mode, len(prompt_text) // 4)
        
        # Show user-facing start message
        start_message = f"🧵 Weaving spec from conversation\n\nAnalyzing {len(context.messages)} messages..."
        yield _serialize_sse({"type": "token", "content": start_message})
        
        # Build messages for LLM
        messages = [{"role": "user", "content": prompt_text}]
        
        # Stream from provider (silently - don't show raw JSON to user)
        response_chunks: List[str] = []
        
        logger.info("[WEAVER] Calling stream function for provider=%s, model=%s", provider, model)
        
        # Call provider stream function (collect silently, don't stream to user)
        async for chunk in stream_fn(messages=messages, model=model):
            # Extract content from chunk
            content = None
            
            if isinstance(chunk, dict):
                # Check for 'text' key first (what OpenAI provider returns)
                content = chunk.get("text") or chunk.get("content")
                
                # Skip metadata chunks
                if chunk.get("type") == "metadata":
                    continue
                    
            elif hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
            
            if content:
                response_chunks.append(content)
                # Don't stream raw JSON to user
        
        # Parse response
        full_response = "".join(response_chunks).strip()
        logger.info("[WEAVER] Generated %d chars", len(full_response))
        
        spec_dict, summary_text = parse_weaver_response(full_response)
        
        if not spec_dict:
            error_msg = summary_text or "Failed to parse LLM response as JSON"
            logger.error("[WEAVER] Parse error: %s", error_msg)
            logger.error("[WEAVER] Full response (first 500 chars): %s", full_response[:500])
            
            error_message = f"❌ Weaver failed to generate valid spec\n\n{error_msg}"
            yield _serialize_sse({"type": "token", "content": error_message})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # Extract weak spots for checkpoint
        weak_spots = spec_dict.get("weak_spots", []) or []
        
        # Build SpecSchema
        spec_schema = build_spec_from_dict(
            spec_dict=spec_dict,
            context=context,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=model,
        )
        
        # Store checkpoint metadata in spec
        # Calculate last_consumed_message_id as max of message_ids we just used
        if context.message_ids:
            new_last_consumed = max(context.message_ids)
        elif last_consumed_message_id is not None:
            new_last_consumed = last_consumed_message_id
        else:
            new_last_consumed = None
        
        # Store checkpoint in content_json metadata
        spec_dict_full = _to_jsonable(spec_schema)
        
        if "metadata" not in spec_dict_full:
            spec_dict_full["metadata"] = {}
        
        spec_dict_full["metadata"]["weak_spots"] = weak_spots
        spec_dict_full["metadata"]["weaver_last_consumed_message_id"] = new_last_consumed
        
        # Rebuild SpecSchema with checkpoint
        spec_schema_with_checkpoint = build_spec_from_dict(
            spec_dict=spec_dict_full,
            context=context,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=model,
        )
        
        # Persist spec
        db_spec = specs_service.create_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema_with_checkpoint,
            generator_model=model,
        )
        
        logger.info("[WEAVER] Persisted spec_id=%s (db_id=%d), checkpoint_message_id=%s",
                   db_spec.spec_id, db_spec.id, new_last_consumed)
        
        # Format weak spots section if any
        weak_spots_section = ""
        if weak_spots:
            weak_spots_section = "\n\n⚠️ Weak spots to address:\n" + "\n".join([f"  • {ws}" for ws in weak_spots[:5]])
            if len(weak_spots) > 5:
                weak_spots_section += f"\n  ... and {len(weak_spots) - 5} more"
        
        # Format nice summary message for user
        action_verb = "created" if mode == "create" else "updated"
        summary_message = f"""📋 Spec {action_verb}: `{db_spec.spec_id[:12]}...`

**Title:** {spec_dict.get('title', 'Untitled')}
**Summary:** {spec_dict.get('summary', 'No summary')[:150]}{'...' if len(spec_dict.get('summary', '')) > 150 else ''}

📊 Based on {len(context.messages)} messages{weak_spots_section}

Ready for Spec Gate review. Say **"yes"** to proceed or ask me to refine the spec."""

        # Send the formatted message to user
        yield _serialize_sse({"type": "token", "content": summary_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER] Error during streaming")
        error_message = f"❌ Weaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})