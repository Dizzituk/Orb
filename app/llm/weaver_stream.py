# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA

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
    logger.warning(f"[weaver_stream] Core module not available: {e}")

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

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
        logger.warning("[WEAVER] Unknown provider '%s', defaulting to OpenAI", provider)
        return stream_openai


# ---------------------------------------------------------------------------
# Main Stream Generator
# ---------------------------------------------------------------------------

async def generate_weaver_stream(
    *,
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[Any] = None,
    conversation_id: str,
) -> AsyncIterator[bytes]:
    """
    Weaver handler for incremental spec building.
    
    Called by stream_router.py for WEAVER_BUILD_SPEC commands.
    
    Incremental behavior:
    - First run: builds spec from recent context
    - Subsequent runs with no new messages: reuses previous spec
    - Subsequent runs with new messages: updates spec incrementally
    
    CRITICAL: This function must successfully persist to DB so Spec Gate
    can load the spec. The PoT (Point of Truth) chain depends on this.
    """
    logger.info("[WEAVER] Starting for project_id=%s", project_id)
    
    provider, model = _get_weaver_config()
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        logger.error("[WEAVER] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _CORE_AVAILABLE:
        error_msg = "Weaver core module not available"
        logger.error("[WEAVER] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _SPECS_SERVICE_AVAILABLE or not specs_service:
        error_msg = "Specs service not available - cannot persist to DB"
        logger.error("[WEAVER] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    try:
        logger.info("[WEAVER] Using provider=%s, model=%s (from env)", provider, model)
        
        stream_fn = _get_streaming_function(provider)
        if stream_fn is None:
            error_msg = f"Streaming function not available for provider: {provider}"
            logger.error("[WEAVER] %s", error_msg)
            yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # =====================================================================
        # Step 1: Load previous spec (for incremental mode)
        # =====================================================================
        
        last_spec = specs_service.get_latest_spec(db, project_id=project_id)
        
        mode = "create"
        last_consumed_message_id: Optional[int] = None
        previous_spec_core: Optional[Dict[str, Any]] = None
        previous_weak_spots: List[str] = []
        
        if last_spec:
            last_consumed_message_id = _get_last_consumed_message_id_from_spec(last_spec)
            logger.info("[WEAVER] Found previous spec_id=%s, last_consumed_message_id=%s",
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
                logger.warning("[WEAVER] Failed to load previous spec core: %s", e)
                mode = "create"
        
        # =====================================================================
        # Step 2: Gather context
        # =====================================================================
        
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

**Spec ID:** `{last_spec.spec_id}`
**Hash:** `{last_spec.spec_hash[:16] if last_spec.spec_hash else 'N/A'}...`

Ready for Spec Gate review."""
            
            yield _serialize_sse({"type": "token", "content": reuse_message})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # =====================================================================
        # Step 3: Build prompt
        # =====================================================================
        
        if mode == "update" and previous_spec_core:
            prompt_text = build_weaver_update_prompt(
                previous_spec_core=previous_spec_core,
                previous_weak_spots=previous_weak_spots,
                delta_context=context,
            )
        else:
            prompt_text = build_weaver_prompt(context)
        
        logger.info("[WEAVER] Built %s prompt (%d tokens estimated)", mode, len(prompt_text) // 4)
        
        # Show user-facing start message
        start_message = f"🧵 Weaving spec from conversation\n\nAnalyzing {len(context.messages)} messages..."
        yield _serialize_sse({"type": "token", "content": start_message})
        
        # =====================================================================
        # Step 4: Stream from LLM
        # =====================================================================
        
        messages = [{"role": "user", "content": prompt_text}]
        response_chunks: List[str] = []
        
        logger.info("[WEAVER] Calling stream function for provider=%s, model=%s", provider, model)
        
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
        
        # =====================================================================
        # Step 5: Parse response
        # =====================================================================
        
        full_response = "".join(response_chunks).strip()
        logger.info("[WEAVER] Generated %d chars", len(full_response))
        
        spec_dict, summary_text = parse_weaver_response(full_response)
        
        if not spec_dict:
            error_msg = summary_text or "Failed to parse LLM response as JSON"
            logger.error("[WEAVER] Parse error: %s", error_msg)
            error_message = f"\n\n❌ Weaver failed to generate valid spec\n\n{error_msg}"
            yield _serialize_sse({"type": "token", "content": error_message})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        # =====================================================================
        # Step 6: Build SpecSchema (first pass)
        # =====================================================================
        
        weak_spots = spec_dict.get("weak_spots", []) or []
        
        # v2.1: Extract content preservation fields from parsed spec
        content_verbatim = spec_dict.get("content_verbatim")
        location = spec_dict.get("location")
        scope_constraints = spec_dict.get("scope_constraints", [])
        outputs = spec_dict.get("outputs", [])
        steps = spec_dict.get("steps", [])
        
        # v2.2: CRITICAL - If in UPDATE mode and LLM dropped content fields, restore from previous spec
        # This prevents the "Chinese Whispers" problem where clarifications wipe key data
        if mode == "update" and previous_spec_core:
            prev_meta = previous_spec_core.get("metadata", {}) or {}
            
            # Preserve content_verbatim if LLM dropped it (unless user explicitly cleared it)
            if not content_verbatim:
                prev_cv = prev_meta.get("content_verbatim") or previous_spec_core.get("content_verbatim")
                if prev_cv:
                    content_verbatim = prev_cv
                    spec_dict["content_verbatim"] = prev_cv
                    logger.info("[WEAVER] Restored content_verbatim from previous spec: '%s'", prev_cv[:50] if prev_cv else "")
            
            # Preserve location if LLM dropped it
            if not location:
                prev_loc = prev_meta.get("location") or previous_spec_core.get("location")
                if prev_loc:
                    location = prev_loc
                    spec_dict["location"] = prev_loc
                    logger.info("[WEAVER] Restored location from previous spec: '%s'", prev_loc[:50] if prev_loc else "")
            
            # Preserve outputs if LLM dropped them
            if not outputs:
                prev_out = prev_meta.get("outputs") or previous_spec_core.get("outputs")
                if prev_out:
                    outputs = prev_out
                    spec_dict["outputs"] = prev_out
                    logger.info("[WEAVER] Restored outputs from previous spec: %d items", len(prev_out) if prev_out else 0)
            
            # Merge scope_constraints (add new ones, keep old)
            prev_constraints = prev_meta.get("scope_constraints") or previous_spec_core.get("scope_constraints", [])
            if prev_constraints:
                merged_constraints = list(prev_constraints)
                for c in scope_constraints:
                    if c not in merged_constraints:
                        merged_constraints.append(c)
                scope_constraints = merged_constraints
                spec_dict["scope_constraints"] = merged_constraints
        
        spec_schema = build_spec_from_dict(
            spec_dict=spec_dict,
            context=context,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=model,
        )
        
        # =====================================================================
        # Step 7: Enrich with checkpoint + content preservation metadata
        # =====================================================================
        
        if context.message_ids:
            new_last_consumed = max(context.message_ids)
        elif last_consumed_message_id is not None:
            new_last_consumed = last_consumed_message_id
        else:
            new_last_consumed = None
        
        # Convert to dict for enrichment
        spec_dict_full = _to_jsonable(spec_schema)
        
        if "metadata" not in spec_dict_full:
            spec_dict_full["metadata"] = {}
        
        # CRITICAL: Store ALL fields that downstream stages need
        spec_dict_full["metadata"]["weak_spots"] = weak_spots
        spec_dict_full["metadata"]["weaver_last_consumed_message_id"] = new_last_consumed
        
        # v2.1: Content preservation fields - these flow through to Spec Gate
        spec_dict_full["metadata"]["content_verbatim"] = content_verbatim
        spec_dict_full["metadata"]["location"] = location
        spec_dict_full["metadata"]["scope_constraints"] = scope_constraints
        spec_dict_full["metadata"]["outputs"] = outputs
        spec_dict_full["metadata"]["steps"] = steps
        
        # Also store at top level for direct access
        spec_dict_full["content_verbatim"] = content_verbatim
        spec_dict_full["location"] = location
        spec_dict_full["scope_constraints"] = scope_constraints
        spec_dict_full["outputs"] = outputs
        spec_dict_full["steps"] = steps
        
        # =====================================================================
        # Step 8: Rebuild SpecSchema with enriched data
        # =====================================================================
        
        spec_schema_with_checkpoint = build_spec_from_dict(
            spec_dict=spec_dict_full,
            context=context,
            project_id=project_id,
            conversation_id=conversation_id,
            generator_model=model,
        )
        
        # =====================================================================
        # Step 9: Persist to DB (CRITICAL for PoT chain)
        # =====================================================================
        
        # IMPORTANT: create_spec signature is (db, project_id, spec_schema, generator_model)
        # Do NOT pass content_json - it doesn't exist as a parameter
        db_spec = specs_service.create_spec(
            db=db,
            project_id=project_id,
            spec_schema=spec_schema_with_checkpoint,
            generator_model=model,
        )
        
        logger.info("[WEAVER] Persisted spec_id=%s (db_id=%d), checkpoint_message_id=%s",
                   db_spec.spec_id, db_spec.id, new_last_consumed)
        
        # =====================================================================
        # Step 10: Format user-facing output
        # =====================================================================
        
        # Content preservation summary (v2.1)
        preservation_status = "\n\n### Content Preservation\n"
        if content_verbatim:
            preview = content_verbatim[:50] + ("..." if len(content_verbatim) > 50 else "")
            preservation_status += f"✅ **Content Verbatim:** `\"{preview}\"`\n"
        else:
            preservation_status += "⚠️ **Content Verbatim:** (not specified)\n"
        
        if location:
            preservation_status += f"✅ **Location:** `{location}`\n"
        else:
            preservation_status += "⚠️ **Location:** (not specified)\n"
        
        if scope_constraints:
            preservation_status += f"✅ **Scope Constraints:** {len(scope_constraints)} defined\n"
        
        if outputs:
            preservation_status += f"✅ **Outputs:** {len(outputs)} artifact(s)\n"
        else:
            preservation_status += "⚠️ **Outputs:** (none - Spec Gate may block)\n"
        
        if steps:
            preservation_status += f"✅ **Steps:** {len(steps)} step(s)\n"
        else:
            preservation_status += "⚠️ **Steps:** (none - Spec Gate may block)\n"
        
        # Weak spots section
        weak_spots_section = ""
        if weak_spots:
            weak_spots_section = "\n\n⚠️ **Weak spots to address:**\n" + "\n".join([f"  • {ws}" for ws in weak_spots[:5]])
            if len(weak_spots) > 5:
                weak_spots_section += f"\n  ... and {len(weak_spots) - 5} more"
        
        # Final summary
        action_verb = "created" if mode == "create" else "updated"
        summary_message = f"""

📋 **Spec {action_verb}:** `{db_spec.spec_id[:12]}...`

**Spec ID:** `{db_spec.spec_id}`
**Hash:** `{db_spec.spec_hash[:16] if db_spec.spec_hash else 'N/A'}...`

**Title:** {spec_dict.get('title', 'Untitled')}
**Summary:** {spec_dict.get('summary', 'No summary')[:150]}{'...' if len(spec_dict.get('summary', '')) > 150 else ''}

📊 Based on {len(context.messages)} messages{preservation_status}{weak_spots_section}

---
💾 **Spec saved to database** - Ready for Spec Gate.
Say **'Astra, command: critical architecture'** to validate and create PoT spec."""

        yield _serialize_sse({"type": "token", "content": summary_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER] Error during streaming")
        error_message = f"\n\n❌ Weaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})