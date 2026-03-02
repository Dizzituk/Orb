# FILE: app/memory/knowledge_extractor.py
"""
Layer 3: Extract durable knowledge from conversation summaries
and promote it to ASTRA's permanent memory system.

Called when a session transitions to 'archived' (Phase 5 lifecycle).
Uses a cheap LLM to classify summary content into extraction categories:
  - biographical:  permanent (user facts, background)
  - preference:    evolving  (style, tool, workflow preferences)
  - philosophy:    permanent (development principles, design rules)
  - project:       ephemeral (current task context, WIP decisions)
  - learning:      evolving  (knowledge gaps, skill building)

Each extraction is written to ASTRA memory via the preference service
with appropriate confidence tiers. New extractions start at low confidence
and get promoted after 2+ reinforcements (existing ASTRA behaviour).

Part of CONV-MEMORY-001: Phase 6 (RAG Memory Extraction).
"""

import json
import logging
import os
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session as DbSession

from app.memory.conversation_models import ConversationSession
from app.memory.conversation_service import get_latest_summary

logger = logging.getLogger(__name__)


# =========================================================================
# Configuration
# =========================================================================

def _get_extraction_model() -> tuple[str, str]:
    """
    Return (provider, model) for knowledge extraction.
    Reuses the summary model config — same cost tier.
    """
    provider = os.getenv("SUMMARY_PROVIDER", "google")
    model = os.getenv("SUMMARY_MODEL", "gemini-2.0-flash-lite")
    return provider, model


# =========================================================================
# Extraction prompt
# =========================================================================

_EXTRACTION_SYSTEM_PROMPT = """\
You are a knowledge extractor. Given a conversation summary, identify \
facts worth storing in long-term memory.

RULES:
- Output ONLY valid JSON array — no markdown fences, no preamble.
- Each item must have: "category", "key", "value".
- Category must be one of: biographical, preference, philosophy, project, learning.
- Key should be a short identifier (e.g. "preferred_language", "current_project").
- Value should be a concise statement of the fact.
- Only extract CLEAR, DEFINITE facts — not speculation or possibilities.
- Skip anything already obvious (e.g. "user is chatting with AI").
- Maximum 10 extractions per summary.
- Deduplicate — don't extract the same fact twice with different wording.

CATEGORIES:
- biographical:  Personal facts (name, job, location, experience)
- preference:    Style/tool/workflow preferences expressed by the user
- philosophy:    Development principles, design rules, hard constraints
- project:       Current project context, active decisions, WIP items
- learning:      Knowledge gaps identified, skills being built

OUTPUT FORMAT:
[
  {"category": "preference", "key": "code_style", "value": "Prefers small files under 20KB"},
  {"category": "biographical", "key": "occupation", "value": "Delivery driver building AI platform"}
]
"""


# =========================================================================
# Response parsing
# =========================================================================

def _parse_extractions(raw: str) -> List[Dict[str, str]]:
    """Parse LLM extraction response into a list of fact dicts."""
    cleaned = raw.strip()

    # Strip markdown fences
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(
            "[knowledge_ext] Failed to parse extraction JSON: %s", e,
        )
        return []

    if not isinstance(data, list):
        logger.warning("[knowledge_ext] Extraction response is not a list")
        return []

    valid = []
    for item in data:
        if (
            isinstance(item, dict)
            and "category" in item
            and "key" in item
            and "value" in item
            and item["category"] in (
                "biographical", "preference", "philosophy",
                "project", "learning",
            )
        ):
            valid.append(item)

    return valid[:10]  # Hard cap


# =========================================================================
# ASTRA memory writing
# =========================================================================

# Map extraction categories to ASTRA preference namespaces and strengths
_CATEGORY_CONFIG = {
    "biographical": {
        "namespace": "user_personal",
        "source": "conversation_extraction",
        "is_permanent": True,
    },
    "preference": {
        "namespace": "user_personal",
        "source": "conversation_extraction",
        "is_permanent": False,
    },
    "philosophy": {
        "namespace": "dev_principles",
        "source": "conversation_extraction",
        "is_permanent": True,
    },
    "project": {
        "namespace": "project_context",
        "source": "conversation_extraction",
        "is_permanent": False,
    },
    "learning": {
        "namespace": "user_personal",
        "source": "conversation_extraction",
        "is_permanent": False,
    },
}


def _write_to_astra_memory(
    db: DbSession,
    extractions: List[Dict[str, str]],
    session_id: int,
) -> int:
    """
    Write extracted facts to ASTRA memory via preference service.

    Uses create_preference for new facts and append_preference_evidence
    for existing ones (reinforcement). Returns count of items written.
    """
    written = 0

    try:
        from app.astra_memory.preference_service import create_preference
        from app.astra_memory.preference_models import (
            PreferenceRecord,
            PreferenceStrength,
            SignalType,
        )
        from app.astra_memory.confidence_scoring import (
            append_preference_evidence,
        )
    except ImportError:
        logger.debug("[knowledge_ext] ASTRA preference service not available")
        return 0

    for item in extractions:
        category = item["category"]
        key = item["key"]
        value = item["value"]
        config = _CATEGORY_CONFIG.get(category, _CATEGORY_CONFIG["project"])

        pref_key = f"conv_extract:{category}:{key}"
        context_ptr = f"session:{session_id}"

        try:
            # Check if this fact already exists
            existing = (
                db.query(PreferenceRecord)
                .filter(PreferenceRecord.preference_key == pref_key)
                .first()
            )

            if existing:
                # Reinforce existing fact with new evidence
                append_preference_evidence(
                    db=db,
                    preference_key=pref_key,
                    signal_type=SignalType.IMPLICIT,
                    context_pointer=context_ptr,
                    details={
                        "action": "reinforce_from_conversation",
                        "new_value": value,
                    },
                )
                logger.debug(
                    "[knowledge_ext] Reinforced: %s", pref_key,
                )
            else:
                # Create new preference
                strength = (
                    PreferenceStrength.HARD_RULE
                    if config["is_permanent"]
                    else PreferenceStrength.SOFT
                )
                create_preference(
                    db=db,
                    preference_key=pref_key,
                    preference_value=value,
                    strength=strength,
                    source=config["source"],
                    namespace=config["namespace"],
                    context_pointer=context_ptr,
                )
                logger.debug(
                    "[knowledge_ext] Created: %s = %s", pref_key, value[:60],
                )

            written += 1

        except Exception as e:
            logger.warning(
                "[knowledge_ext] Failed to write %s: %s", pref_key, e,
            )

    return written


# =========================================================================
# Main entry point
# =========================================================================

async def extract_and_promote_async(
    db: DbSession,
    session: ConversationSession,
) -> bool:
    """
    Async version of extract_and_promote.
    Called from the async lifecycle scheduler.
    """
    latest = get_latest_summary(db, session.id)
    if not latest or not latest.summary_json:
        logger.debug(
            "[knowledge_ext] No summary for session %d — skipping",
            session.id,
        )
        return False

    summary_json = latest.summary_json
    summary_text = json.dumps(summary_json, indent=2)

    provider, model = _get_extraction_model()

    try:
        from app.llm._streaming_utils_3 import call_llm_text
        raw = await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=(
                f"Extract durable knowledge from this conversation summary:\n\n"
                f"{summary_text}"
            ),
            max_tokens=1500,
            timeout_seconds=30,
        )
    except Exception as e:
        logger.error(
            "[knowledge_ext] LLM extraction failed for session %d: %s",
            session.id, e,
        )
        return False

    extractions = _parse_extractions(raw)
    if not extractions:
        logger.info(
            "[knowledge_ext] No extractable facts from session %d",
            session.id,
        )
        return False

    written = _write_to_astra_memory(db, extractions, session.id)
    logger.info(
        "[knowledge_ext] Session %d: extracted %d facts, wrote %d to ASTRA",
        session.id, len(extractions), written,
    )
    return written > 0


def extract_and_promote(
    db: DbSession,
    session: ConversationSession,
) -> bool:
    """
    Extract knowledge from a session's summary and write to ASTRA memory.

    Called by session_lifecycle.py when a session is archived.
    Returns True if any extractions were written.
    """
    latest = get_latest_summary(db, session.id)
    if not latest or not latest.summary_json:
        logger.debug(
            "[knowledge_ext] No summary for session %d — skipping",
            session.id,
        )
        return False

    summary_json = latest.summary_json
    summary_text = json.dumps(summary_json, indent=2)

    # Call LLM for extraction
    provider, model = _get_extraction_model()

    import asyncio

    async def _extract():
        from app.llm._streaming_utils_3 import call_llm_text
        return await call_llm_text(
            provider=provider,
            model=model,
            system_prompt=_EXTRACTION_SYSTEM_PROMPT,
            user_prompt=(
                f"Extract durable knowledge from this conversation summary:\n\n"
                f"{summary_text}"
            ),
            max_tokens=1500,
            timeout_seconds=30,
        )

    try:
        # Run the async LLM call.
        # We may be called from either a sync or async context:
        #   - From lifecycle scheduler (async loop running but called from sync func)
        #   - From a test or manual invocation (no loop)
        try:
            loop = asyncio.get_running_loop()
            # We're inside a running loop — can't use run_until_complete.
            # Create a task and use a thread to wait for it.
            import concurrent.futures
            future = asyncio.run_coroutine_threadsafe(_extract(), loop)
            raw = future.result(timeout=45)
        except RuntimeError:
            # No running loop — create one
            raw = asyncio.run(_extract())
    except Exception as e:
        logger.error(
            "[knowledge_ext] LLM extraction failed for session %d: %s",
            session.id, e,
        )
        return False

    # Parse and write
    extractions = _parse_extractions(raw)
    if not extractions:
        logger.info(
            "[knowledge_ext] No extractable facts from session %d",
            session.id,
        )
        return False

    written = _write_to_astra_memory(db, extractions, session.id)
    logger.info(
        "[knowledge_ext] Session %d: extracted %d facts, wrote %d to ASTRA",
        session.id, len(extractions), written,
    )
    return written > 0
