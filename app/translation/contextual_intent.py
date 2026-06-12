# FILE: app/translation/contextual_intent.py
# Purpose: Contextual Intent Router.
# Called-by: app.translation.translator
# Depends-on: app.translation.schemas
# Last-renovated: 2026-06-11
"""
Contextual Intent Router.

Handles messages that address ASTRA directly but don't match hard commands.
When the directive gate blocks a message (past tense, question, future planning),
this module runs a lightweight classification to detect *implied* actions.

Intent Tiers:
    Tier 1 — Direct Command: "Astra, run the pipeline" → hard route
    Tier 2 — Conversational Intent: "Astra, I want to build a bridge app" → scoping
    Tier 3 — Contextual Intelligence: "Astra, the surf was light today" → web search (surf report)
    Tier 4 — Future Intent / Parking: "Astra, next week I might..." → memory capture
    Tier 5 — Pure Chat: "Astra, what do you think about AI?" → conversation

Tiers 1-2 are handled by existing tier0/tier1 rules + scoping.
Tier 5 is the default fallback.
This module handles Tiers 3 and 4 — the gap between hard commands and pure chat.

v1.0 (2026-03-11): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.translation.schemas import CanonicalIntent

logger = logging.getLogger(__name__)


class ImpliedIntentType(str, Enum):
    """What the user implicitly wants from a non-command statement."""
    WEB_SEARCH = "web_search"           # "surf was light" → get surf report
    WEATHER = "weather"                 # "it's raining" → weather forecast
    MEMORY_STORE = "memory_store"       # "I've been thinking about X" → park for later
    FUTURE_PARKING = "future_parking"   # "next week I might" → park in memory
    NEWS_CHECK = "news_check"           # "what's happening with AI" → web search news
    NONE = "none"                       # Pure chat, no implied action


@dataclass
class ContextualIntentResult:
    """Result of contextual intent analysis."""
    implied_type: ImpliedIntentType
    canonical_intent: Optional[CanonicalIntent]
    confidence: float
    reasoning: str
    extracted_query: Optional[str] = None  # e.g. "surf report Cornwall"


# ═══════════════════════════════════════════════════════════════════
# Keyword-based fast detection (no LLM)
# ═══════════════════════════════════════════════════════════════════

# Topic → implied action mapping
# When "Astra, <statement about topic>", what does the user likely want?
_CONTEXTUAL_TOPIC_MAP = {
    # Surf / weather / outdoor activity → get current conditions
    "surf": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "surf report"},
    "waves": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "surf forecast"},
    "swell": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "swell forecast"},
    "tide": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "tide times"},
    "rain": {"type": ImpliedIntentType.WEATHER, "query_suffix": "weather forecast"},
    "raining": {"type": ImpliedIntentType.WEATHER, "query_suffix": "weather forecast"},
    "weather": {"type": ImpliedIntentType.WEATHER, "query_suffix": "weather forecast"},
    "sunny": {"type": ImpliedIntentType.WEATHER, "query_suffix": "weather forecast"},
    "wind": {"type": ImpliedIntentType.WEATHER, "query_suffix": "wind forecast"},
    "storm": {"type": ImpliedIntentType.WEATHER, "query_suffix": "storm forecast"},
    "snow": {"type": ImpliedIntentType.WEATHER, "query_suffix": "weather forecast"},
    "cold": {"type": ImpliedIntentType.WEATHER, "query_suffix": "weather forecast"},

    # Traffic / delivery context → route conditions
    "traffic": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "traffic report"},
    "roadworks": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "roadworks"},
    "roads": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "road conditions"},

    # Finance / markets
    "market": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "stock market today"},
    "bitcoin": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "bitcoin price today"},
    "crypto": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "crypto market today"},
    "stocks": {"type": ImpliedIntentType.WEB_SEARCH, "query_suffix": "stock market today"},

    # News / current events
    "news": {"type": ImpliedIntentType.NEWS_CHECK, "query_suffix": "news today"},
    "election": {"type": ImpliedIntentType.NEWS_CHECK, "query_suffix": "election news"},
}

# Future-intent markers — these indicate something to park, not act on
_FUTURE_MARKERS = [
    "next week", "next month", "one day", "eventually",
    "someday", "in the future", "later", "down the line",
    "at some point", "when I get round to", "I might",
    "I should", "we should", "I could", "might want to",
    "been thinking about", "thinking about",
    "been considering", "considering",
]

# Thinking/reflection markers — capture in memory but don't act
_THINKING_MARKERS = [
    "I've been thinking", "I was thinking", "I had a thought",
    "what if we", "what about", "wouldn't it be cool if",
    "imagine if", "you know what would be good",
]


def detect_contextual_intent(
    message: str,
    was_astra_addressed: bool = False,
) -> ContextualIntentResult:
    """Detect implied intent from a non-command message.

    This runs AFTER the directive gate has blocked the message from being
    a hard command. It looks for contextual clues about what the user
    actually wants.

    Args:
        message: The user's message (stripped of Astra prefix).
        was_astra_addressed: Whether the user addressed Astra directly.

    Returns:
        ContextualIntentResult with the implied intent and suggested action.
    """
    text_lower = message.lower().strip()

    # Only do contextual inference if the user addressed Astra
    if not was_astra_addressed:
        return ContextualIntentResult(
            implied_type=ImpliedIntentType.NONE,
            canonical_intent=None,
            confidence=0.0,
            reasoning="Not addressed to Astra — no contextual inference",
        )

    # ── Check thinking/reflection markers first (more specific) ──
    for marker in _THINKING_MARKERS:
        if marker.lower() in text_lower:
            return ContextualIntentResult(
                implied_type=ImpliedIntentType.MEMORY_STORE,
                canonical_intent=CanonicalIntent.MEMORY_STORE,
                confidence=0.75,
                reasoning=f"Thinking/reflection detected: '{marker}'",
                extracted_query=message.strip(),
            )

    # ── Check future parking markers ──
    for marker in _FUTURE_MARKERS:
        if marker in text_lower:
            return ContextualIntentResult(
                implied_type=ImpliedIntentType.FUTURE_PARKING,
                canonical_intent=CanonicalIntent.MEMORY_STORE,
                confidence=0.80,
                reasoning=f"Future intent detected: '{marker}'",
                extracted_query=message.strip(),
            )

    # ── Check topic-based contextual actions ──
    # Look for topic keywords that imply an action
    for keyword, action in _CONTEXTUAL_TOPIC_MAP.items():
        if keyword in text_lower:
            # Build a search query from context
            location = os.getenv("ASTRA_USER_LOCATION", "Cornwall")
            query = f"{action['query_suffix']} {location}"

            intent_type = action["type"]
            if intent_type == ImpliedIntentType.WEATHER:
                canonical = CanonicalIntent.WEB_SEARCH
            elif intent_type == ImpliedIntentType.NEWS_CHECK:
                canonical = CanonicalIntent.WEB_SEARCH
            else:
                canonical = CanonicalIntent.WEB_SEARCH

            return ContextualIntentResult(
                implied_type=intent_type,
                canonical_intent=canonical,
                confidence=0.85,
                reasoning=f"Contextual topic '{keyword}' → {intent_type.value}",
                extracted_query=query,
            )

    # ── No contextual intent detected ──
    return ContextualIntentResult(
        implied_type=ImpliedIntentType.NONE,
        canonical_intent=None,
        confidence=0.0,
        reasoning="No contextual intent detected — treating as pure chat",
    )
