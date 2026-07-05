# FILE: app/memory/escalation.py
# Purpose: Model escalation pathway (Spec Section 10, Job 6C).
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Model escalation pathway (Spec Section 10, Job 6C).

When a lower-tier model determines the query exceeds its
capability, it returns an escalation signal. This module
processes that signal and re-routes to the next tier up.

Escalation chain (automatic):
    local → local_rag → sonnet → gemini_deep
    opus = manual only (via UI model switcher)
    specialist stays at specialist (no escalation)

The escalation is always preferred over a bad answer.
Extra latency from re-routing is acceptable.

Escalation triggers:
    - Model returns ESCALATE token in response
    - Response confidence below threshold
    - Response contains "I don't have enough context"
    - Token limit hit before completion

Usage:
    from app.memory.escalation import (
        check_escalation,
        escalate,
        EscalationResult,
    )

    # After getting a response from a model
    esc = check_escalation(response_text, current_tier="local")
    if esc.should_escalate:
        new_tier = escalate(current_tier="local")
        # Re-route query to new_tier
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Escalation chain
# =========================================================================

# v2.0: Escalation chain updated.
# "gemini_deep" is now the automatic ceiling — Gemini 3.1 Pro customtools
# handles codebase scanning, structural queries, and tool-driven exploration
# at ~50% the cost of Opus.
# "opus" is still in the map but NOT in the chain — only reachable via
# manual UI model switcher when the user decides Gemini isn't cutting it.
ESCALATION_CHAIN = {
    "local": "local_rag",
    "local_rag": "sonnet",
    "sonnet": "gemini_deep",
    "gemini_deep": None,    # Top of automatic chain
    "opus": None,           # Manual escalation only (via UI model switcher)
    "specialist": None,     # Specialist stays specialist
}

# Map model targets to provider/model combos
# v2.0: Added gemini_deep tier, reads from env for easy model swaps
# v2.1 (2026-07-02): de-hardcoded — gemini_deep falls back to the MULTIMODAL
# role model and "specialist" resolves the IMAGE_VISION role (the old
# hardcoded "gemini-pro" id no longer exists upstream). Env reads happen at
# import time (restart-gated); try/except keeps import safe if env is bare.
import os as _os
try:
    from app.llm.frontier_models import get_role_model as _get_role_model
    _deep_model = _os.getenv("CHAT_DEEP_MODEL") or _get_role_model("MULTIMODAL")[1]
    _specialist_provider, _specialist_model = _get_role_model("IMAGE_VISION")
except Exception:  # env-only resolver unavailable/unconfigured — fail soft
    logger.error("[escalation] role-model resolution failed at import", exc_info=True)
    _deep_model = _os.getenv("CHAT_DEEP_MODEL") or _os.getenv("MULTIMODAL_MODEL", "")
    _specialist_provider = _os.getenv("IMAGE_VISION_PROVIDER", "google")
    _specialist_model = _os.getenv("IMAGE_VISION_MODEL", "")
MODEL_TARGET_MAP = {
    "local": {"provider": "local", "model": "ollama"},
    "local_rag": {"provider": "local", "model": "ollama", "rag": True},
    "sonnet": {"provider": "anthropic", "model": "sonnet"},
    "gemini_deep": {
        "provider": _os.getenv("CHAT_DEEP_PROVIDER", "google"),
        "model": _deep_model,
    },
    "opus": {"provider": "anthropic", "model": "opus"},
    "specialist": {"provider": _specialist_provider, "model": _specialist_model},
}


# =========================================================================
# Escalation result
# =========================================================================

@dataclass
class EscalationResult:
    """Result of an escalation check."""
    should_escalate: bool
    reason: Optional[str]
    current_tier: str
    next_tier: Optional[str]
    signals: dict


# =========================================================================
# Detection patterns
# =========================================================================

# Patterns in model output that signal it can't handle the query
ESCALATION_PATTERNS = [
    r"\[ESCALATE\]",
    r"\[NEEDS_ESCALATION\]",
    r"I don'?t have enough context",
    r"this (?:is |seems? )?(?:too complex|beyond my)",
    r"I'?m not (?:able|equipped) to",
    r"you (?:should|might want to) (?:ask|use) a (?:more|larger)",
    r"insufficient (?:context|information) to (?:provide|answer)",
]

# Confidence patterns — model expressing uncertainty
LOW_CONFIDENCE_PATTERNS = [
    r"I'?m not (?:entirely? |completely? )?(?:sure|certain)",
    r"I (?:think|believe|guess) (?:but|however)",
    r"take this with a grain of salt",
    r"I might be wrong",
    r"I'?m guessing",
]


# =========================================================================
# Public API
# =========================================================================

def check_escalation(
    response_text: str,
    current_tier: str,
    was_truncated: bool = False,
    response_confidence: Optional[float] = None,
) -> EscalationResult:
    """
    Check if a model response signals escalation is needed.

    Args:
        response_text: The model's response text.
        current_tier: Current model tier (local, sonnet, etc.)
        was_truncated: True if response hit token limit.
        response_confidence: Model's self-reported confidence (if available).

    Returns:
        EscalationResult with should_escalate flag and reason.
    """
    signals = {}
    next_tier = ESCALATION_CHAIN.get(current_tier)

    # No escalation possible from top of chain
    if next_tier is None:
        return EscalationResult(
            should_escalate=False,
            reason=None,
            current_tier=current_tier,
            next_tier=None,
            signals={"at_top": True},
        )

    # Check 1: Explicit escalation token
    for pattern in ESCALATION_PATTERNS:
        if re.search(pattern, response_text, re.IGNORECASE):
            signals["escalation_pattern"] = pattern
            return EscalationResult(
                should_escalate=True,
                reason=f"Model signalled escalation: matched '{pattern}'",
                current_tier=current_tier,
                next_tier=next_tier,
                signals=signals,
            )

    # Check 2: Token limit truncation
    if was_truncated:
        signals["truncated"] = True
        return EscalationResult(
            should_escalate=True,
            reason="Response truncated by token limit",
            current_tier=current_tier,
            next_tier=next_tier,
            signals=signals,
        )

    # Check 3: Low self-reported confidence
    if response_confidence is not None and response_confidence < 0.4:
        signals["low_confidence"] = response_confidence
        return EscalationResult(
            should_escalate=True,
            reason=f"Model confidence too low: {response_confidence:.2f}",
            current_tier=current_tier,
            next_tier=next_tier,
            signals=signals,
        )

    # Check 4: Uncertainty language (soft signal — only for local tier)
    if current_tier in ("local", "local_rag"):
        uncertainty_count = sum(
            1 for p in LOW_CONFIDENCE_PATTERNS
            if re.search(p, response_text, re.IGNORECASE)
        )
        if uncertainty_count >= 2:
            signals["uncertainty_patterns"] = uncertainty_count
            return EscalationResult(
                should_escalate=True,
                reason=f"Multiple uncertainty signals ({uncertainty_count})",
                current_tier=current_tier,
                next_tier=next_tier,
                signals=signals,
            )

    # No escalation needed
    return EscalationResult(
        should_escalate=False,
        reason=None,
        current_tier=current_tier,
        next_tier=next_tier,
        signals=signals,
    )


def escalate(current_tier: str) -> Optional[str]:
    """
    Get the next tier in the escalation chain.

    Returns None if already at top or tier is unknown.
    """
    return ESCALATION_CHAIN.get(current_tier)


def get_model_target(tier: str) -> dict:
    """
    Get provider/model info for a tier.

    Returns dict with provider, model, and optional rag flag.
    """
    return MODEL_TARGET_MAP.get(tier, MODEL_TARGET_MAP["sonnet"])


def get_escalation_chain(start_tier: str) -> list[str]:
    """
    Get the full escalation chain from a starting tier.

    Returns list of tiers from start to top.
    """
    chain = [start_tier]
    current = start_tier
    while True:
        next_tier = ESCALATION_CHAIN.get(current)
        if next_tier is None:
            break
        chain.append(next_tier)
        current = next_tier
    return chain
