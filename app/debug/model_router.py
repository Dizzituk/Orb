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
    # Reasoning effort for GPT-5.x / extended-thinking-capable models.
    # Format: {"effort": "low"|"medium"|"high"} or None for no reasoning.
    # Bound to tier: Triage=None, Analysis=high, Agentic=high.
    reasoning: Optional[dict] = None


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

TIER_MODELS = {
    DebugTier.TRIAGE: {
        "provider": "openai",
        "model": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.4-mini"),
    },
    # v2 (2026-04-15): Analysis + Agentic moved from Claude Sonnet to GPT-5.4
    # per user preference: GPT-5.4 with reasoning_effort=high is better AND
    # cheaper than Sonnet extended thinking for debug workloads.
    DebugTier.ANALYSIS: {
        "provider": "openai",
        "model": "gpt-5.4",
    },
    DebugTier.AGENTIC: {
        "provider": "openai",
        "model": "gpt-5.4",
    },
}

# Reasoning effort bound to tier. Passed through to providers via the
# registry's `reasoning` parameter. Triage stays cheap; Analysis + Agentic
# get high-effort reasoning because by the time the classifier has decided
# a query is "think about this", it always deserves the thinking budget.
TIER_REASONING = {
    DebugTier.TRIAGE:   None,
    DebugTier.ANALYSIS: {"effort": "high"},
    DebugTier.AGENTIC:  {"effort": "high"},
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

# Keywords that suggest deeper reasoning is needed.
# v2 (2026-04-15): expanded for natural / voice-dictated phrasing.
# The formal words (analyse, diagnose, root cause) rarely appear in
# real speech — people say "have a look", "go through", "dig into".
ANALYSIS_PATTERNS = [
    # Original formal patterns
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
    # v2: natural / voice-dictated phrasing
    r"\bhave\s+a\s+(look|think|gander)\b",
    r"\btake\s+a\s+(look|think)\b",
    r"\bgo\s+(and\s+)?(have\s+a\s+look|look|through|over|into)\b",
    r"\blook\s+(at|through|over|into)\b",
    r"\bgo\s+through\s+(the|this|that|my|your)\b",
    r"\bdig\s+(in|into|through)\b",
    r"\bwork\s+out\s+(what|why|how|where)\b",
    r"\bfigure\s+out\s+(what|why|how|where)\b",
    r"\bunderstand(ing)?\b",
    r"\bdeep\s+dive\b",
    r"\bthink\s+(about|through)\b",
    r"\breason\s+(about|through)\b",
    r"\bgive\s+(me\s+)?a\s+(good|deep|full|proper|thorough)\b",
    r"\b(check|review)\s+(the|this|that|my)\s+(code|codebase|base|project|logic|flow)\b",
    r"\bwhat.s\s+(going\s+on|happening|up)\b",
    r"\bwhat\s+could\s+be\b",
    r"\b(audit|inspect)\s+(the|this|my)\b",
]

_AGENTIC_RX = [re.compile(p, re.IGNORECASE) for p in AGENTIC_PATTERNS]
_ANALYSIS_RX = [re.compile(p, re.IGNORECASE) for p in ANALYSIS_PATTERNS]


# =============================================================================
# ROUTING LOGIC
# =============================================================================

def _make_decision(tier: DebugTier, reason: str, enable_tools: bool = False) -> RoutingDecision:
    """Build a RoutingDecision from a tier — single source of truth for
    model + reasoning binding. Adding a new tier means one place to update."""
    cfg = TIER_MODELS[tier]
    return RoutingDecision(
        tier=tier,
        provider=cfg["provider"],
        model=cfg["model"],
        reason=reason,
        enable_tools=enable_tools,
        reasoning=TIER_REASONING.get(tier),
    )


# Long messages (typically voice-dictated deep-dive requests) almost never
# belong in triage, even if none of the regex patterns happen to match.
# This is a safety net for natural phrasing the patterns miss.
_LONG_MESSAGE_CHAR_THRESHOLD = 400


def classify_query(
    message: str,
    conversation_history: Optional[List[dict]] = None,
) -> RoutingDecision:
    """
    Classify a debug query and determine the appropriate model tier.

    Decision order:
      1. Agentic patterns (fix/implement/edit) → AGENTIC
      2. Analysis patterns (why/look/dig/understand) → ANALYSIS
      3. Long message escalation (>400 chars) → ANALYSIS
      4. Prior-turn uncertainty from assistant → ANALYSIS
      5. Default → TRIAGE

    Reasoning effort is bound to tier via TIER_REASONING, so any match
    that escalates past triage automatically gets reasoning enabled.

    Args:
        message: The user's current message.
        conversation_history: Previous messages in the conversation.

    Returns:
        RoutingDecision with tier, provider, model, reasoning, tools.
    """
    msg_lower = message.lower().strip()

    # 1. Agentic intent (highest tier — user wants an action)
    for rx in _AGENTIC_RX:
        if rx.search(msg_lower):
            return _make_decision(
                DebugTier.AGENTIC,
                reason=f"Agentic pattern matched: {rx.pattern}",
                enable_tools=True,
            )

    # 2. Analysis intent (reasoning required)
    for rx in _ANALYSIS_RX:
        if rx.search(msg_lower):
            return _make_decision(
                DebugTier.ANALYSIS,
                reason=f"Analysis pattern matched: {rx.pattern}",
            )

    # 3. Length-based escalation. Voice-dictated deep-dive requests often
    # wander past any single keyword but are clearly not triage material.
    if len(message) >= _LONG_MESSAGE_CHAR_THRESHOLD:
        return _make_decision(
            DebugTier.ANALYSIS,
            reason=f"Long message escalation ({len(message)} chars >= {_LONG_MESSAGE_CHAR_THRESHOLD})",
        )

    # 4. Prior-turn uncertainty — if the last assistant reply hedged,
    # escalate so the next turn has reasoning to work with.
    if conversation_history and len(conversation_history) > 2:
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
                return _make_decision(
                    DebugTier.ANALYSIS,
                    reason="Escalated: previous response showed uncertainty",
                )

    # 5. Default: triage
    return _make_decision(DebugTier.TRIAGE, reason="Default triage routing")


def get_tier_cost_estimate(tier: DebugTier) -> dict:
    """Get estimated cost per query for a given tier."""
    estimates = {
        DebugTier.TRIAGE:  {"min_gbp": 0.001, "max_gbp": 0.005, "avg_gbp": 0.003},
        DebugTier.ANALYSIS: {"min_gbp": 0.01,  "max_gbp": 0.05,  "avg_gbp": 0.03},
        DebugTier.AGENTIC:  {"min_gbp": 0.05,  "max_gbp": 0.20,  "avg_gbp": 0.12},
    }
    return estimates.get(tier, {"min_gbp": 0, "max_gbp": 0, "avg_gbp": 0})
