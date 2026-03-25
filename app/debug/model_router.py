# FILE: app/debug/model_router.py
"""
Model Router: Routes debug queries to the appropriate LLM based on complexity.

Tier 1 (Triage):  OPENAI_DEFAULT_MODEL (env) — log reading, status, simple diagnostics.
Tier 2 (Analysis): Claude Sonnet — root cause analysis, multi-file reasoning.
Tier 3 (Agentic):  Claude Sonnet — implementing fixes, running commands, iterating.

Escalation is automatic based on query classification and conversation context.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class DebugTier(str, Enum):
    """Routing tier for debug queries."""
    TRIAGE = "triage"
    ANALYSIS = "analysis"
    AGENTIC = "agentic"


@dataclass
class RoutingDecision:
    """Result of query classification."""
    tier: DebugTier
    provider: str
    model: str
    reason: str
    enable_tools: bool = False


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

TIER_MODELS = {
    DebugTier.TRIAGE: {
        "provider": "openai",
        "model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.4-mini"),
    },
    DebugTier.ANALYSIS: {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
    },
    DebugTier.AGENTIC: {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
    },
}


# =============================================================================
# CLASSIFICATION PATTERNS
# =============================================================================

# Keywords that signal the user wants an action, not just information
AGENTIC_PATTERNS = [
    r"\bfix\s+(it|this|that)\b",
    r"\bimplement\b",
    r"\bmake\s+(that|the)\s+change\b",
    r"\bapply\s+(the\s+)?fix\b",
    r"\bwrite\s+(the|a)\s+fix\b",
    r"\bedit\s+(the\s+)?file\b",
    r"\bmodify\b",
    r"\bpatch\b",
    r"\brefactor\b",
    r"\bupdate\s+(the|this)\s+(file|code|module)\b",
    r"\bcreate\s+(a|the)\s+file\b",
    r"\brun\s+(the\s+)?(command|script|test)\b",
    r"\bexecute\b",
    r"\binstall\b",
    r"\bdelete\s+(the|this)\b",
    r"\bremove\s+(the|this)\b",
]

# Keywords that suggest deeper reasoning is needed
ANALYSIS_PATTERNS = [
    r"\bwhy\s+(did|does|is|was|would|doesn.t)\b",
    r"\broot\s+cause\b",
    r"\bexplain\s+(the|this|why)\b",
    r"\bdiagnose\b",
    r"\banalyse\b",
    r"\banalyze\b",
    r"\binvestigate\b",
    r"\btrace\s+(the|this)\b",
    r"\bdebug\s+(the|this)\b",
    r"\bcompare\s+(the|these)\b",
    r"\brelationship\s+between\b",
    r"\bhow\s+(does|do|did|would|should)\s+.+\s+(work|interact|connect|relate)\b",
    r"\bwhat.s\s+(causing|wrong\s+with)\b",
    r"\bsugg(est|estion)\b",
    r"\brecommend\b",
]

_AGENTIC_RX = [re.compile(p, re.IGNORECASE) for p in AGENTIC_PATTERNS]
_ANALYSIS_RX = [re.compile(p, re.IGNORECASE) for p in ANALYSIS_PATTERNS]


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def classify_query(
    message: str,
    conversation_history: Optional[List[dict]] = None,
) -> RoutingDecision:
    """
    Classify a debug query and determine the appropriate model tier.

    Args:
        message: The user's current message.
        conversation_history: Previous messages in the conversation (for context).

    Returns:
        RoutingDecision with tier, provider, model, and reasoning.
    """
    msg_lower = message.lower().strip()

    # Check for agentic intent first (highest tier)
    for rx in _AGENTIC_RX:
        if rx.search(msg_lower):
            cfg = TIER_MODELS[DebugTier.AGENTIC]
            return RoutingDecision(
                tier=DebugTier.AGENTIC,
                provider=cfg["provider"],
                model=cfg["model"],
                reason=f"Agentic pattern matched: {rx.pattern}",
                enable_tools=True,
            )

    # Check for analysis intent
    for rx in _ANALYSIS_RX:
        if rx.search(msg_lower):
            cfg = TIER_MODELS[DebugTier.ANALYSIS]
            return RoutingDecision(
                tier=DebugTier.ANALYSIS,
                provider=cfg["provider"],
                model=cfg["model"],
                reason=f"Analysis pattern matched: {rx.pattern}",
                enable_tools=False,
            )

    # Check conversation context for escalation signals
    if conversation_history and len(conversation_history) > 2:
        # If the conversation has been going back and forth, the simple model
        # may not be cutting it — escalate to analysis
        recent_assistant = [
            m for m in conversation_history[-4:]
            if m.get("role") == "assistant"
        ]
        for msg in recent_assistant:
            content = msg.get("content", "").lower()
            if any(phrase in content for phrase in [
                "i'm not sure", "i can't determine", "unclear",
                "need more context", "difficult to say",
            ]):
                cfg = TIER_MODELS[DebugTier.ANALYSIS]
                return RoutingDecision(
                    tier=DebugTier.ANALYSIS,
                    provider=cfg["provider"],
                    model=cfg["model"],
                    reason="Escalated: previous response showed uncertainty",
                    enable_tools=False,
                )

    # Default: triage tier
    cfg = TIER_MODELS[DebugTier.TRIAGE]
    return RoutingDecision(
        tier=DebugTier.TRIAGE,
        provider=cfg["provider"],
        model=cfg["model"],
        reason="Default triage routing",
        enable_tools=False,
    )


def get_tier_cost_estimate(tier: DebugTier) -> dict:
    """Get estimated cost per query for a given tier."""
    estimates = {
        DebugTier.TRIAGE:  {"min_gbp": 0.001, "max_gbp": 0.005, "avg_gbp": 0.003},
        DebugTier.ANALYSIS: {"min_gbp": 0.01,  "max_gbp": 0.05,  "avg_gbp": 0.03},
        DebugTier.AGENTIC:  {"min_gbp": 0.05,  "max_gbp": 0.20,  "avg_gbp": 0.12},
    }
    return estimates.get(tier, {"min_gbp": 0, "max_gbp": 0, "avg_gbp": 0})
