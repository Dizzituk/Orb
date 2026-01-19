# FILE: app/llm/weaver_stream.py
"""
Weaver Stream Handler for ASTRA - SIMPLIFIED VERSION

v3.0 (2026-01-19): LOCKED BEHAVIOUR SPEC IMPLEMENTATION
- Weaver is now a SIMPLE TEXT ORGANIZER
- No spec building, no JSON parsing, no DB persistence
- Reads conversation (ramble) and organizes it into a job description
- Stores output in flow state for Spec Gate to build spec from

LOCKED WEAVER BEHAVIOUR:
- Purpose: Convert human rambling into a structured job outline
- NOT a full spec builder - just a text organizer
- Reads messages to get input (the ramble)
- Does NOT persist to specs table
- Does NOT build JSON specs
- Does NOT resolve ambiguities or contradictions
- May ask ONE question ONLY if core goal is completely missing

Previous versions (v2.x) did full spec building - that logic is now
in weaver_stream_v2_backup.py if needed for reference.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
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

# Memory service for reading conversation
try:
    from app.memory import service as memory_service
    _MEMORY_AVAILABLE = True
except ImportError:
    memory_service = None
    _MEMORY_AVAILABLE = False

# Flow state for storing output (NOT specs table)
try:
    from app.llm.spec_flow_state import start_weaver_flow, SpecFlowStage
    _FLOW_STATE_AVAILABLE = True
except ImportError:
    start_weaver_flow = None
    SpecFlowStage = None
    _FLOW_STATE_AVAILABLE = False

# Simple weaver function
try:
    from app.llm.weaver_simple import weave, WEAVER_SYSTEM_PROMPT, _format_messages_as_ramble
    _SIMPLE_WEAVER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"[weaver_stream] weaver_simple not available: {e}")
    _SIMPLE_WEAVER_AVAILABLE = False
    weave = None

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


def _is_control_message(role: str, content: str) -> bool:
    """Check if message is a control/system message to skip."""
    c = (content or "").strip()
    rl = (role or "").strip().lower()
    
    if not c:
        return True
    
    if rl == "system":
        return True
    
    # Skip command triggers
    if rl == "user":
        lc = c.lower()
        if any(lc.startswith(prefix) for prefix in [
            "astra, command:", "astra command:", "astra, cmd:", "orb, command:",
            "how does that look all together",
        ]):
            return True
    
    # Skip Weaver/Orb output messages
    if rl in ("assistant", "orb"):
        markers = (
            "🧵 weaving", "📋 spec", "📋 job description",
            "shall i send", "say yes to proceed", "⚠️ weak spots",
            "ready for spec gate", "provenance",
        )
        lc = c.lower()
        if any(m in lc for m in markers):
            return True
    
    return False


def _gather_ramble_messages(db: Session, project_id: int, max_messages: int = 50) -> List[Dict[str, Any]]:
    """
    Gather recent conversation messages as the ramble input.
    
    This is the ONLY DB access Weaver does - reading its input.
    """
    if not _MEMORY_AVAILABLE or not memory_service:
        return []
    
    try:
        messages_raw = memory_service.list_messages(db, project_id, limit=max_messages)
        messages_raw = list(reversed(messages_raw))  # Chronological order
        
        messages: List[Dict[str, Any]] = []
        for msg in messages_raw:
            role = getattr(msg, "role", "user")
            content = getattr(msg, "content", "") or ""
            
            if _is_control_message(role, content):
                continue
            
            messages.append({
                "role": role,
                "content": content,
            })
        
        return messages
    except Exception as e:
        logger.error("[WEAVER] Failed to gather messages: %s", e)
        return []


def _format_ramble(messages: List[Dict[str, Any]]) -> str:
    """Format messages into a ramble text block."""
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue
        speaker = "Human" if role == "user" else "Assistant"
        lines.append(f"[{speaker}]: {content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Main Stream Generator - SIMPLIFIED VERSION
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
    Weaver handler - SIMPLIFIED TEXT ORGANIZER.
    
    This is the v3.0 implementation per the LOCKED BEHAVIOUR SPEC:
    1. Reads conversation from DB (the ramble input)
    2. Uses simple LLM call to organize text into job description
    3. Displays result to user
    4. Stores output in flow state (NOT specs table)
    
    DOES NOT:
    - Build JSON specs
    - Persist to specs table
    - Resolve ambiguities
    - Infer missing details
    - Do feasibility checking
    """
    logger.info("[WEAVER] Starting SIMPLE weaver for project_id=%s", project_id)
    
    provider, model = _get_weaver_config()
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    if not _STREAMING_AVAILABLE:
        error_msg = "Streaming providers not available - check imports"
        logger.error("[WEAVER] %s", error_msg)
        yield _serialize_sse({"type": "token", "content": f"❌ {error_msg}"})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        return
    
    if not _MEMORY_AVAILABLE:
        error_msg = "Memory service not available - cannot read conversation"
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
        # Step 1: Gather conversation (the ramble input)
        # =====================================================================
        
        messages = _gather_ramble_messages(db, project_id)
        
        if not messages:
            no_messages_msg = (
                "🧵 **No conversation to weave**\n\n"
                "I don't see any recent messages to organize into a job description.\n\n"
                "**What to do:**\n"
                "Share what you want to build or change, then say "
                "`how does that look all together` again."
            )
            yield _serialize_sse({"type": "token", "content": no_messages_msg})
            yield _serialize_sse({"type": "done", "provider": provider, "model": model})
            return
        
        ramble_text = _format_ramble(messages)
        logger.info("[WEAVER] Gathered %d messages, %d chars of ramble", len(messages), len(ramble_text))
        
        # =====================================================================
        # Step 2: Show starting message
        # =====================================================================
        
        start_message = f"🧵 **Organizing your thoughts...**\n\nAnalyzing {len(messages)} messages to create a job description.\n\n"
        yield _serialize_sse({"type": "token", "content": start_message})
        
        # =====================================================================
        # Step 3: Build SIMPLE organizer prompt (per locked spec)
        # =====================================================================
        
        system_prompt = """You are Weaver, a text organizer.

Your ONLY job: Take the human's rambling and restructure it into a clear, readable document.

## What You DO:
- Group related ideas together
- Rephrase for clarity and structure ONLY (meaning stays exactly the same)
- Preserve ambiguities and contradictions (do NOT resolve them)
- Write down explicit implications if clearly stated (e.g., "this must be secure")

## What You DO NOT DO:
- No adding detail
- No removing ambiguity  
- No resolving contradictions
- No inferring intent
- No inventing implications
- No technical feasibility checking
- No system/file/architecture awareness
- No suggesting improvements
- No optimisation

## Output Format:
Produce a structured job description. Use ONLY sections that are relevant:

## What is being built or changed
(Core subject)

## Intended outcome
(What success looks like)

## Constraints
(Only if explicitly stated by the human)

## Platform/Environment
(Only if mentioned)

## Priority notes
(If the human mentioned priorities)

## Unresolved ambiguities
(List any contradictions or gaps - DO NOT RESOLVE THEM)

## Critical Rule:
If the human didn't say it, it doesn't appear in your output.

## Exception - Single Question:
If the core goal is COMPLETELY missing and impossible to infer, ask ONE clarifying question.
Otherwise, produce the job description even if vague."""

        user_prompt = f"""Organize this conversation into a job description:

{ramble_text}

Remember: Only include what was actually said. Preserve any ambiguities or contradictions."""
        
        # =====================================================================
        # Step 4: Stream from LLM
        # =====================================================================
        
        llm_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        response_chunks: List[str] = []
        
        logger.info("[WEAVER] Calling stream function for provider=%s, model=%s", provider, model)
        
        async for chunk in stream_fn(messages=llm_messages, model=model):
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
                # Stream tokens to user
                yield _serialize_sse({"type": "token", "content": content})
        
        # =====================================================================
        # Step 5: Store output in flow state (NOT specs table)
        # =====================================================================
        
        job_description = "".join(response_chunks).strip()
        logger.info("[WEAVER] Generated job description: %d chars", len(job_description))
        
        # Generate a simple ID for tracking
        weaver_output_id = f"weaver-{uuid.uuid4().hex[:12]}"
        
        if _FLOW_STATE_AVAILABLE and start_weaver_flow:
            try:
                flow_state = start_weaver_flow(
                    project_id=project_id,
                    weaver_spec_id=weaver_output_id,
                    weaver_job_description=job_description,
                )
                logger.info("[WEAVER] Stored in flow state: %s", weaver_output_id)
            except Exception as e:
                logger.warning("[WEAVER] Failed to store in flow state: %s", e)
        
        # =====================================================================
        # Step 6: Show completion message
        # =====================================================================
        
        completion_message = f"""

---

📋 **Job description ready** (`{weaver_output_id}`)

This is a structured outline of what you described. Review it above.

**Next step:** Say **'Send to Spec Gate'** to validate and build a full specification."""

        yield _serialize_sse({"type": "token", "content": completion_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})
        
    except Exception as e:
        logger.exception("[WEAVER] Error during streaming")
        error_message = f"\n\n❌ Weaver error: {str(e)}"
        yield _serialize_sse({"type": "token", "content": error_message})
        yield _serialize_sse({"type": "done", "provider": provider, "model": model})


# ---------------------------------------------------------------------------
# LEGACY COMPATIBILITY - Export old function name
# ---------------------------------------------------------------------------

# If other modules import the old complex version, they'll get this instead
__all__ = ["generate_weaver_stream"]
