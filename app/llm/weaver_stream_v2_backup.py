# FILE: app/llm/weaver_stream_v2_backup.py
"""
BACKUP: Weaver Stream Handler for ASTRA - v2.1 ORIGINAL

This is a backup of the v2.x Weaver implementation BEFORE the v3.0 simplification.
The v2.x version does full spec building + DB persistence.

Kept for reference in case the complex behaviour is needed.

v2.1 (2026-01-04): Content Preservation Fix
- Preserves content_verbatim, location, scope_constraints through to DB
- Fixed create_spec call signature (removed invalid content_json param)
- Enriches metadata BEFORE rebuild to ensure all fields survive
- Shows content preservation status in UI output

v2.0: Incremental weaving with checkpointing.
"""
from __future__ import annotations

import json
import logging
import os
from typing import AsyncIterator, Dict, List, Optional, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports with graceful fallbacks
# ---------------------------------------------------------------------------

try:
    from app.llm.audit_logger import RoutingTrace
except ImportError:
    RoutingTrace = None

try:
    from app.specs import service as specs_service
    _SPECS_SERVICE_AVAILABLE = True
except ImportError:
    specs_service = None
    _SPECS_SERVICE_AVAILABLE = False

try:
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
    _CORE_AVAILABLE = True
except ImportError as e:
    _CORE_AVAILABLE = False
    logger.warning(f"[weaver_stream_v2] Core module not available: {e}")

# Import streaming functions for all providers
try:
    from app.llm.streaming import stream_openai, stream_anthropic, stream_gemini
    _STREAMING_AVAILABLE = True
except ImportError:
    try:
        from .streaming import stream_openai, stream_anthropic, stream_gemini
        _STREAMING_AVAILABLE = True
    except ImportError:
        stream_openai = None
        stream_anthropic = None
        stream_gemini = None
        _STREAMING_AVAILABLE = False


def _serialize_sse(data: Dict[str, Any]) -> bytes:
    """Serialize dict to SSE format."""
    return f"data: {json.dumps(data)}\n\n".encode("utf-8")


def _get_weaver_config() -> tuple[str, str]:
    """Get provider and model for weaver from environment."""
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
        logger.warning("[WEAVER_V2] Unknown provider '%s', defaulting to OpenAI", provider)
        return stream_openai


async def generate_weaver_stream_v2(
    *,
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: str,
) -> AsyncIterator[bytes]:
    """
    V2 Weaver handler for incremental spec building (BACKUP VERSION).
    
    This is the ORIGINAL v2.1 implementation that does full spec building.
    Kept for reference - use generate_weaver_stream from weaver_stream.py
    for the new simplified behaviour.
    """
    logger.info("[WEAVER_V2] Starting for project_id=%s", project_id)
    
    provider, model = _get_weaver_config()
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        logger.error("[WEAVER_V2] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _CORE_AVAILABLE:
        error_msg = "Weaver core module not available"
        logger.error("[WEAVER_V2] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _SPECS_SERVICE_AVAILABLE or not specs_service:
        error_msg = "Specs service not available - cannot persist to DB"
        logger.error("[WEAVER_V2] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    try:
        logger.info("[WEAVER_V2] Using provider=%s, model=%s (from env)", provider, model)
        
        stream_fn = _get_streaming_function(provider)
        if stream_fn is None:
            error_msg = f"Streaming function not available for provider: {provider}"
            logger.error("[WEAVER_V2] %s", error_msg)
            yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # Load previous spec (for incremental mode)
        last_spec = specs_service.get_latest_spec(db, project_id=project_id)
        
        mode = "create"
        last_consumed_message_id: Optional[int] = None
        previous_spec_core: Optional[Dict[str, Any]] = None
        previous_weak_spots: List[str] = []
        
        if last_spec:
            last_consumed_message_id = _get_last_consumed_message_id_from_spec(last_spec)
            logger.info("[WEAVER_V2] Found previous spec_id=%s, last_consumed_message_id=%s",
                       last_spec.spec_id, last_consumed_message_id)
            
            try:
                spec_schema = specs_service.get_spec_schema(last_spec)
                previous_spec_core = _to_jsonable(spec_schema)
                
                if hasattr(last_spec, "content_json") and last_spec.content_json:
                    if isinstance(last_spec.content_json, dict):
                        metadata = last_spec.content_json.get("metadata", {})
                        if isinstance(metadata, dict):
                            previous_weak_spots = metadata.get("weak_spots", [])
                
                mode = "update"
            except Exception as e:
                logger.warning("[WEAVER_V2] Failed to load previous spec core: %s", e)
                mode = "create"
        
        # Gather context
        if mode == "update" and last_consumed_message_id is not None:
            context = gather_weaver_delta_context(
                db=db,
                project_id=project_id,
                since_message_id=last_consumed_message_id,
            )
            logger.info("[WEAVER_V2] Delta mode: %d messages since message_id=%d",
                       len(context.messages), last_consumed_message_id)
        else:
            context = gather_weaver_context(db=db, project_id=project_id)
            logger.info("[WEAVER_V2] Fresh mode: %d messages", len(context.messages))
        
        if mode == "update" and len(context.messages) == 0:
            logger.info("[WEAVER_V2] No new messages; reusing spec_id=%s", last_spec.spec_id)
            
            reuse_message = f"""📋 Spec already up to date

Using existing spec `{last_spec.spec_id[:12]}...` (no new messages since last weave).

**Spec ID:** `{last_spec.spec_id}`
**Hash:** `{last_spec.spec_hash[:16] if last_spec.spec_hash else 'N/A'}...`

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
        
        logger.info("[WEAVER_V2] Built %s prompt (%d tokens estimated)", mode, len(prompt_text) // 4)
        
        start_message = f"🧵 Weaving spec from conversation\n\nAnalyzing {len(context.messages)} messages..."
        yield _serialize_sse({"type": "token", "content": start_message})
        
        # Stream from LLM
        messages = [{"role": "user", "content": prompt_text}]
        response_chunks: List[str] = []
        
        async for chunk in stream_fn(messages=messages, model=model):
            content = None
            if isinstance(chunk, dict):
                content = chunk.get("text") or chunk.get("content")
                if chunk.get("type") == "metadata":
                    continue
            elif hasattr(chunk, "choices") and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    content = delta.content
            if content:
                response_chunks.append(content)
        
        full_response = "".join(response_chunks).strip()
        logger.info("[WEAVER_V2] Generated %d chars", len(full_response))
        
        spec_dict, summary_text = parse_weaver_response(full_response)
        
        if not spec_dict:
            error_msg = summary_text or "Failed to parse LLM response as JSON"
            logger.error("[WEAVER_V2] Parse error: %s", error_msg)
            error_message = f"\n\n❌ Weaver failed to generate valid spec\n\n{error_msg}"
            yield _serialize_sse({"type": "token", "content": error_message})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # Build and persist spec (the v2.x behaviour)
        spec_schema = build_spec_from_dict(
            spec_dict=spec_dict,
            context=context,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=model,
        )
        
        # Enrich metadata
        spec_dict_full = _to_jsonable(spec_schema)
        if "metadata" not in spec_dict_full:
            spec_dict_full["metadata"] = {}
        
        weak_spots = spec_dict.get("weak_spots", []) or []
        if context.message_ids:
            new_last_consumed = max(context.message_ids)
        elif last_consumed_message_id is not None:
            new_last_consumed = last_consumed_message_id
        else:
            new_last_consumed = None
        
        spec_dict_full["metadata"]["weak_spots"] = weak_spots
        spec_dict_full["metadata"]["weaver_last_consumed_message_id"] = new_last_consumed
        
        # Rebuild and persist
        spec_schema_with_checkpoint = build_spec_from_dict(
            spec_dict=spec_dict_full,
            context=context,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=model,
        )
        
        db_spec = specs_service.create_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema_with_checkpoint,
            generator_model=model,
        )
        
        logger.info("[WEAVER_V2] Persisted spec_id=%s (db_id=%d)", db_spec.spec_id, db_spec.id)
        
        action_verb = "created" if mode == "create" else "updated"
        summary_message = f"""

📋 **Spec {action_verb}:** `{db_spec.spec_id[:12]}...`

**Spec ID:** `{db_spec.spec_id}`
**Hash:** `{db_spec.spec_hash[:16] if db_spec.spec_hash else 'N/A'}...`

**Title:** {spec_dict.get('title', 'Untitled')}
**Summary:** {spec_dict.get('summary', 'No summary')[:150]}

📊 Based on {len(context.messages)} messages

---
💾 **Spec saved to database** - Ready for Spec Gate."""

        yield _serialize_sse({"type": "token", "content": summary_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER_V2] Error during streaming")
        error_message = f"\n\n❌ Weaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})


# Export for reference - use this if you need the old behaviour
__all__ = ["generate_weaver_stream_v2"]
