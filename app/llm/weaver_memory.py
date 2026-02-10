# FILE: app/llm/weaver_memory.py
"""
Progressive Memory for Weaver — Context Compaction for Long Conversations.

Phase 4C of Pipeline Evolution.

In long conversations (20+ messages), the full ramble exceeds practical
context limits and drowns the LLM in repetition. Progressive memory
solves this by compacting older messages into key points while keeping
recent messages verbatim.

The compaction strategy:
    - Recent messages (last N) are kept VERBATIM — the user's freshest
      thinking, exact wording, nuance all preserved.
    - Older messages are DISTILLED into a structured summary: key
      decisions made, requirements stated, constraints identified,
      questions asked. This is a lossy compression that preserves
      WHAT was decided but not HOW it was discussed.

The threshold and window sizes are configurable. Compaction is an LLM
call (lightweight — it's summarising, not reasoning).

Integration:
    - Called from weaver_stream.py before the Weaver LLM call
    - Replaces the raw ramble_text with compacted_context + recent_ramble
    - Transparent to the Weaver: it just sees a shorter, denser input

v1.0 (2026-02-10): Initial implementation — Phase 4C.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WEAVER_MEMORY_BUILD_ID = "2026-02-10-v1.0-progressive-memory"
print(f"[WEAVER_MEMORY_LOADED] BUILD_ID={WEAVER_MEMORY_BUILD_ID}")

# =============================================================================
# CONFIGURATION — all tuneable via env vars
# =============================================================================

# Minimum messages before compaction kicks in
COMPACTION_THRESHOLD = int(os.getenv("WEAVER_COMPACTION_THRESHOLD", "15"))

# How many recent messages to keep verbatim
RECENT_WINDOW = int(os.getenv("WEAVER_RECENT_WINDOW", "8"))

# Maximum chars for the distilled summary
MAX_SUMMARY_CHARS = int(os.getenv("WEAVER_MAX_SUMMARY_CHARS", "3000"))


# =============================================================================
# RESULT SCHEMA
# =============================================================================

@dataclass
class CompactedContext:
    """Result of progressive memory compaction."""
    distilled_summary: str = ""       # Structured summary of older messages
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    total_messages: int = 0
    compacted_count: int = 0          # How many messages were distilled
    preserved_count: int = 0          # How many kept verbatim
    was_compacted: bool = False       # Whether compaction actually ran
    skip_reason: str = ""             # Why compaction was skipped (if not run)
    model_used: str = ""

    def format_for_weaver(self) -> str:
        """
        Format as a single text block that replaces the raw ramble.

        Structure:
            [CONTEXT FROM EARLIER DISCUSSION]
            ...distilled summary...

            [RECENT CONVERSATION — VERBATIM]
            Human: ...
            Human: ...
        """
        parts = []

        if self.distilled_summary:
            parts.append("=" * 50)
            parts.append("CONTEXT FROM EARLIER DISCUSSION (distilled)")
            parts.append("=" * 50)
            parts.append(self.distilled_summary)
            parts.append("")

        if self.recent_messages:
            parts.append("=" * 50)
            parts.append("RECENT CONVERSATION — VERBATIM (preserve exact wording)")
            parts.append("=" * 50)
            for msg in self.recent_messages:
                role = msg.get("role", "user")
                content = msg.get("content", "").strip()
                if content:
                    speaker = "Human" if role == "user" else "Assistant"
                    parts.append(f"{speaker}: {content}")
                    parts.append("")

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_messages": self.total_messages,
            "compacted_count": self.compacted_count,
            "preserved_count": self.preserved_count,
            "was_compacted": self.was_compacted,
            "skip_reason": self.skip_reason,
            "summary_chars": len(self.distilled_summary),
            "model_used": self.model_used,
        }


# =============================================================================
# DISTILLATION PROMPT
# =============================================================================

DISTILL_SYSTEM_PROMPT = """\
You are a conversation summariser. You receive older messages from a \
conversation between a user and an AI assistant. Your job is to distill \
these into a structured summary of KEY POINTS ONLY.

Extract and organise:
1. **Decisions made** — what the user has decided (tech stack, features, approach)
2. **Requirements stated** — specific features, behaviours, or constraints
3. **Constraints identified** — limitations, must-haves, must-not-haves
4. **Open questions** — things still undecided or ambiguous
5. **Context** — project name, domain, target platform, user preferences

RULES:
- Be CONCISE. Each point = 1 line.
- Preserve the user's terminology exactly (don't paraphrase domain terms).
- Only include things the user actually said — don't infer or add.
- Drop small talk, greetings, pleasantries, meta-discussion about the AI.
- Drop repetition — if the user said the same thing 3 times, list it once.
- If the user changed their mind, only include the LATEST decision.

OUTPUT FORMAT:
Use the exact section headers below. Omit empty sections.

## Decisions
- <decision 1>
- <decision 2>

## Requirements
- <requirement 1>
- <requirement 2>

## Constraints
- <constraint 1>

## Open Questions
- <question 1>

## Context
- <context point 1>
"""


# =============================================================================
# COMPACTION LOGIC
# =============================================================================

def _should_compact(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Check if compaction should run. Returns skip reason if not."""
    if len(messages) < COMPACTION_THRESHOLD:
        return f"Only {len(messages)} messages (threshold: {COMPACTION_THRESHOLD})"

    # Count user messages specifically
    user_count = sum(1 for m in messages if m.get("role") == "user")
    if user_count < COMPACTION_THRESHOLD // 2:
        return f"Only {user_count} user messages"

    return None


async def compact_conversation(
    messages: List[Dict[str, Any]],
    provider_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> CompactedContext:
    """
    Main entry point: compact a long conversation for Weaver consumption.

    If the conversation is short enough, returns it unchanged. If long,
    distills older messages and keeps recent ones verbatim.

    Args:
        messages: All conversation messages (chronological order)
        provider_id/model_id: Override model for distillation

    Returns:
        CompactedContext with distilled summary + recent messages
    """
    result = CompactedContext(
        total_messages=len(messages),
    )

    # Check threshold
    skip_reason = _should_compact(messages)
    if skip_reason:
        result.skip_reason = skip_reason
        result.recent_messages = messages
        result.preserved_count = len(messages)
        return result

    # Split into old (to distill) and recent (to keep)
    split_point = max(0, len(messages) - RECENT_WINDOW)
    old_messages = messages[:split_point]
    recent_messages = messages[split_point:]

    result.recent_messages = recent_messages
    result.preserved_count = len(recent_messages)
    result.compacted_count = len(old_messages)

    if not old_messages:
        result.skip_reason = "No old messages to distill"
        return result

    # Format old messages for distillation
    old_text_parts = []
    for msg in old_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if content:
            speaker = "Human" if role == "user" else "Assistant"
            old_text_parts.append(f"{speaker}: {content}")

    old_text = "\n\n".join(old_text_parts)

    # Trim if extremely long
    if len(old_text) > 20000:
        old_text = old_text[:20000] + "\n\n... (truncated) ..."

    # Resolve model
    _provider = provider_id
    _model = model_id

    if not _provider or not _model:
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("WEAVER_COMPACTION")
            _provider = _provider or config.provider
            _model = _model or config.model
        except (ImportError, Exception) as _cfg_err:
            logger.warning("[weaver_memory] stage_models unavailable: %s", _cfg_err)

    if not _provider or not _model:
        # Can't distill without a model — return all messages uncompacted
        logger.warning("[weaver_memory] Model not configured — skipping compaction")
        result.recent_messages = messages
        result.preserved_count = len(messages)
        result.compacted_count = 0
        result.skip_reason = "WEAVER_COMPACTION model not configured"
        return result

    logger.info(
        "[weaver_memory] Compacting: %d old messages → distill, %d recent → keep verbatim",
        len(old_messages), len(recent_messages),
    )

    user_prompt = f"""\
Distill these {len(old_messages)} older conversation messages into key points:

{old_text}

Output the structured summary (Decisions, Requirements, Constraints, Open Questions, Context). Be concise.
"""

    try:
        from app.providers.registry import llm_call

        llm_result = await llm_call(
            provider_id=_provider,
            model_id=_model,
            messages=[{"role": "user", "content": user_prompt}],
            system_prompt=DISTILL_SYSTEM_PROMPT,
            max_tokens=1500,
            timeout_seconds=45,
        )

        if not llm_result.is_success():
            logger.warning("[weaver_memory] Distillation LLM call failed: %s", llm_result.error_message)
            result.recent_messages = messages
            result.preserved_count = len(messages)
            result.compacted_count = 0
            result.skip_reason = f"LLM call failed: {llm_result.error_message}"
            return result

        summary = (llm_result.content or "").strip()

        # Cap the summary length
        if len(summary) > MAX_SUMMARY_CHARS:
            summary = summary[:MAX_SUMMARY_CHARS] + "\n\n... (summary truncated) ..."

        result.distilled_summary = summary
        result.was_compacted = True
        result.model_used = f"{_provider}/{_model}"

        logger.info(
            "[weaver_memory] Compaction complete: %d messages → %d chars summary + %d verbatim",
            len(old_messages), len(summary), len(recent_messages),
        )

        return result

    except ImportError:
        logger.warning("[weaver_memory] Provider registry unavailable")
        result.recent_messages = messages
        result.preserved_count = len(messages)
        result.compacted_count = 0
        result.skip_reason = "Provider registry unavailable"
        return result
    except Exception as e:
        logger.exception("[weaver_memory] Distillation error: %s", e)
        result.recent_messages = messages
        result.preserved_count = len(messages)
        result.compacted_count = 0
        result.skip_reason = f"Exception: {e}"
        return result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CompactedContext",
    "compact_conversation",
    "WEAVER_MEMORY_BUILD_ID",
]
