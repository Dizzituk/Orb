# FILE: app/self_model/injection.py
"""
Self-Model → Prompt injection filter.

Single responsibility: decide which facts from the Self-Model store
deserve a slot in the LLM system context, and format them.

Rule set (Phase 1):
  - biographical facts: ALWAYS injected (any confidence)
  - philosophy facts:   ALWAYS injected (any confidence)
  - preference facts:   injected if confidence >= MEDIUM
  - project facts:      injected if confidence == HIGH
  - learning facts:     injected if confidence == HIGH
  - pattern facts:      injected if confidence >= MEDIUM

Override: set env SELF_MODEL_INJECT_ALL=1 to inject every fact
(useful for debugging what's actually in the store).

Later phases will plug the Tier 1 identity store in above this.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable, List, Optional

logger = logging.getLogger(__name__)

# Confidence rank for comparisons (higher = more confident)
_CONF_RANK = {"low": 0, "medium": 1, "high": 2}

# Per-category minimum confidence required for injection.
# "any" means inject regardless of confidence.
_CATEGORY_RULES = {
    "biographical": "any",
    "philosophy":   "any",
    "preference":   "medium",
    "pattern":      "medium",
    "project":      "high",
    "learning":     "high",
}

_DEFAULT_RULE = "high"  # unknown category → require HIGH


def _conf_str(fact) -> str:
    """Extract confidence as a lowercase string from a UserFact."""
    c = getattr(fact, "confidence", "low")
    return c.value.lower() if hasattr(c, "value") else str(c).lower()


def _should_inject(fact) -> bool:
    """Apply per-category confidence gating."""
    if os.getenv("SELF_MODEL_INJECT_ALL") == "1":
        return True

    rule = _CATEGORY_RULES.get(fact.category, _DEFAULT_RULE)
    if rule == "any":
        return True

    fact_rank = _CONF_RANK.get(_conf_str(fact), 0)
    required_rank = _CONF_RANK.get(rule, 2)
    return fact_rank >= required_rank


def filter_facts(facts: Iterable) -> List:
    """Return the subset of facts that should be injected."""
    return [f for f in facts if _should_inject(f)]


def format_injection_block(facts: Iterable) -> Optional[str]:
    """
    Build the [SELF-MODEL: USER FACTS] block from already-filtered facts.
    Returns None if no facts qualify.
    """
    facts = list(facts)
    if not facts:
        return None

    lines = ["[SELF-MODEL: USER FACTS]"]
    for f in facts:
        conf = _conf_str(f)
        lines.append(f"  {f.category}/{f.key}: {f.value} (confidence: {conf})")
    lines.append("[/SELF-MODEL: USER FACTS]")
    return "\n".join(lines)


def build_self_model_block() -> Optional[str]:
    """
    One-call convenience: load the user model, filter, format.
    Returns a ready-to-append string or None.

    Safe to call from anywhere — all failures logged and swallowed.
    """
    try:
        from app.self_model.user_model import get_user_model
        all_facts = get_user_model().get_all()
    except Exception as exc:
        logger.debug("[self_model.injection] load failed: %s", exc)
        return None

    filtered = filter_facts(all_facts)
    total = len(all_facts)
    kept = len(filtered)
    logger.info(
        "[self_model.injection] %d/%d facts passed filter (dropped %d as low-signal)",
        kept, total, total - kept,
    )

    return format_injection_block(filtered)