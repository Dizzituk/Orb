# FILE: app/memory/domains/confidence_learning.py
# Purpose: Confidence learning loop (Spec Section 5.5).
# Called-by: app.memory.domains, app.memory.integration
# Depends-on: app.memory.domains.confidence
# Last-renovated: 2026-06-11
"""
Confidence learning loop (Spec Section 5.5).

Bridges the translation layer's confirmation/correction flow
with the ConfidenceStore. When the user confirms or corrects
an intent, this module processes the signal and updates the
confidence scores.

Three signal types:
    confirmation — User said "yes" to a proposed intent
    correction   — User said "no, I meant X"
    timeout      — User didn't respond (weak negative signal)

The learning loop also handles:
    - Creating new mappings from corrections
    - Parameterised phrase patterns (e.g. "chuck that over to {person}")
    - Logging learning events for audit

Usage:
    from app.memory.domains.confidence_learning import (
        on_confirmation,
        on_correction,
        on_timeout,
    )

    # User confirmed "run the pipeline"
    on_confirmation("send it to the pipeline", "RUN_PIPELINE")

    # User corrected: "no, I meant send to Dave"
    on_correction(
        phrase="chuck that over to dave",
        wrong_intent="RUN_PIPELINE",
        right_intent="SEND_TO_CONTACT",
    )
"""

import logging
import re
from datetime import datetime
from typing import Optional

from app.memory.domains.confidence import ConfidenceStore

logger = logging.getLogger(__name__)

# Module-level store instance (lazy init)
_store: Optional[ConfidenceStore] = None


def _get_store() -> ConfidenceStore:
    """Get or create the ConfidenceStore singleton."""
    global _store
    if _store is None:
        _store = ConfidenceStore()
    return _store


# =========================================================================
# Learning signals
# =========================================================================

def on_confirmation(
    phrase: str,
    intent: str,
    context_tags: Optional[list[str]] = None,
) -> dict:
    """
    Process a user confirmation.

    Called when the user confirms a proposed intent:
      ASTRA: "Did you mean: run the pipeline?"
      User: "yes"

    This is the primary positive learning signal.

    Args:
        phrase: The original user phrase.
        intent: The intent that was confirmed.
        context_tags: Optional context qualifiers.

    Returns:
        Dict with new_confidence and mapping info.
    """
    store = _get_store()
    new_conf = store.reinforce(phrase, intent, context_tags)

    # Also reinforce any generalised pattern
    pattern = _extract_pattern(phrase)
    if pattern and pattern != phrase:
        store.reinforce(pattern, intent, context_tags)

    logger.info(
        "[learning] CONFIRMED: '%s' → %s (conf=%.3f)",
        phrase[:50], intent, new_conf,
    )

    return {
        "signal": "confirmation",
        "phrase": phrase,
        "intent": intent,
        "new_confidence": round(new_conf, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }


def on_correction(
    phrase: str,
    wrong_intent: str,
    right_intent: str,
    context_tags: Optional[list[str]] = None,
) -> dict:
    """
    Process a user correction.

    Called when the user corrects ASTRA's interpretation:
      ASTRA: "Did you mean: run the pipeline?"
      User: "no, I meant send the document to Dave"

    This penalises the wrong mapping AND creates/reinforces the
    correct one. Per spec section 5.5:
      - Wrong mapping gets +1 correction, confidence decreases
      - New mapping is created with initial low confidence

    Args:
        phrase: The original user phrase.
        wrong_intent: What ASTRA incorrectly resolved to.
        right_intent: What the user actually meant.
        context_tags: Optional context qualifiers.

    Returns:
        Dict with both confidence scores.
    """
    store = _get_store()
    wrong_conf, right_conf = store.correct(
        phrase, wrong_intent, right_intent, context_tags
    )

    logger.info(
        "[learning] CORRECTED: '%s' %s(%.3f) → %s(%.3f)",
        phrase[:50], wrong_intent, wrong_conf,
        right_intent, right_conf,
    )

    return {
        "signal": "correction",
        "phrase": phrase,
        "wrong_intent": wrong_intent,
        "right_intent": right_intent,
        "wrong_confidence": round(wrong_conf, 4),
        "right_confidence": round(right_conf, 4),
        "timestamp": datetime.utcnow().isoformat(),
    }


def on_timeout(
    phrase: str,
    proposed_intent: str,
    timeout_seconds: float = 30.0,
) -> dict:
    """
    Process a confirmation timeout.

    If the user doesn't respond to a confirmation request within
    the timeout window, treat it as a weak negative signal. The
    mapping isn't penalised as hard as a correction, but we note
    it wasn't confirmed.

    This does NOT add a full correction — it's a soft signal
    that something might be off.

    Args:
        phrase: The original user phrase.
        proposed_intent: The intent that was proposed.
        timeout_seconds: How long we waited.

    Returns:
        Dict with timeout info.
    """
    # Timeouts are logged but don't modify scores directly.
    # Repeated timeouts for the same pattern could be promoted
    # to corrections in a future enhancement.
    logger.info(
        "[learning] TIMEOUT: '%s' → %s (waited %.0fs)",
        phrase[:50], proposed_intent, timeout_seconds,
    )

    return {
        "signal": "timeout",
        "phrase": phrase,
        "proposed_intent": proposed_intent,
        "timeout_seconds": timeout_seconds,
        "timestamp": datetime.utcnow().isoformat(),
    }


# =========================================================================
# Batch operations
# =========================================================================

def run_decay() -> int:
    """
    Run confidence decay on all stale mappings.

    Should be called periodically (e.g. daily cron or startup).
    Returns count of entries affected.
    """
    store = _get_store()
    count = store.decay_stale(days_threshold=30)
    logger.info("[learning] Decay run: %d entries affected", count)
    return count


def get_learning_stats() -> dict:
    """
    Get summary statistics for the confidence learning system.

    Returns counts and confidence distribution.
    """
    store = _get_store()
    stats = store.get_stats()
    return {
        "total_mappings": stats.total_entries,
        "high_confidence": stats.embedded_entries,  # Reused field
        "domain": stats.domain,
    }


# =========================================================================
# Pattern extraction
# =========================================================================

def _extract_pattern(phrase: str) -> Optional[str]:
    """
    Extract a generalised pattern from a specific phrase.

    Replaces specific values with placeholders so the learning
    can generalise across similar phrases:

      "chuck that over to dave" → "chuck that over to {person}"
      "send file report.pdf"   → "send file {filename}"
      "run job 42"             → "run job {number}"

    Returns None if no generalisation is possible.
    """
    result = phrase.lower().strip()
    changed = False

    # Replace names after "to" (likely person references)
    new = re.sub(
        r'\bto\s+([a-z]+)\s*$',
        r'to {person}',
        result,
    )
    if new != result:
        result = new
        changed = True

    # Replace filenames with extensions
    new = re.sub(
        r'\b[\w\-]+\.\w{1,5}\b',
        '{filename}',
        result,
    )
    if new != result:
        result = new
        changed = True

    # Replace numbers
    new = re.sub(r'\b\d+\b', '{number}', result)
    if new != result:
        result = new
        changed = True

    # Replace UUIDs
    new = re.sub(
        r'\b[a-f0-9\-]{36}\b',
        '{uuid}',
        result,
    )
    if new != result:
        result = new
        changed = True

    return result if changed else None
