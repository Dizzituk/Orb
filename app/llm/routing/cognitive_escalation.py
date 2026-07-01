# FILE: app/llm/routing/cognitive_escalation.py
# Purpose: Cognitive escalation cue detection.
# Called-by: app.llm.routing.chat_model_selection
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-07-01
"""
Cognitive escalation cue detection.

Detects provider-agnostic phrases that signal the user wants a higher-
capability model — e.g. "think about this", "take your time",
"use a bigger model" — without naming a specific provider.

This is separate from EXPLICIT_MODEL_PATTERNS in app/memory/complexity.py.
Those patterns require a named provider ("use Claude", "switch to GPT").
This module handles the provider-agnostic case that would otherwise fall
through to the cheap chat tier by mistake.

Routing rule:
    - Cognitive cue + no named provider + sticky is small → escalate to the
      frontier tier (COGNITIVE_ESCALATION_* / FRONTIER_* env vars)
    - Cognitive cue + no named provider + no sticky → escalate likewise
    - Cognitive cue + sticky already capable → keep sticky
    - Cognitive cue + named provider → defer to explicit pattern handling
      (do not short-circuit here)

The caller (chat_model_selection.select_chat_model) checks
`_detect_explicit_model_request` BEFORE acting on a cognitive cue,
so "use Claude to think about this" still routes correctly to Opus.
"""

from __future__ import annotations

import os
import re
from typing import Optional

# =============================================================================
# Escalation cue patterns
# =============================================================================
#
# Grouped by theme. Add new phrases to the relevant group. Word-boundary
# anchors (\b) prevent false matches inside larger words.

COGNITIVE_ESCALATION_PATTERNS: list[str] = [
    # --- "think" variants ---
    r"\bthink (?:about|through|on|over) (?:this|it|that)\b",
    r"\bthink (?:hard|harder|carefully|deeply|properly|longer|more)\b",
    r"\b(?:really|actually|carefully|properly) think\b",
    r"\bgive (?:this|that|it) (?:some )?(?:real |proper |serious |more )?thought\b",
    r"\bthink (?:this|that|it) through\b",
    r"\bneed to think\b",

    # --- Time / effort cues ---
    r"\btake your time\b",
    r"\btake a (?:minute|moment|second|sec)\b",
    r"\bno rush\b",
    r"\btake (?:a bit )?longer\b",
    r"\bspend (?:some |a bit of )?time\b",

    # --- Care / quality cues ---
    r"\bbe careful (?:with|about|on|here)\b",
    r"\bdo (?:this|that|it) (?:properly|carefully|right|well|thoroughly)\b",
    r"\b(?:really|actually) work on (?:this|it|that)\b",
    r"\bfocus (?:hard|carefully|properly) on\b",
    r"\bget this right\b",

    # --- Importance / difficulty cues ---
    r"\b(?:this is|it'?s|that'?s) (?:important|complex|tricky|critical|hard|difficult)\b",
    r"\bthis needs (?:real |proper )?(?:care|thought|focus|attention)\b",
    r"\bhigh[\s-]stakes\b",

    # --- Explicit size / capability cues ---
    r"\b(?:bigger|smarter|better|stronger|more powerful|flagship|top)\s+model\b",
    r"\buse (?:a|the|some|something) (?:bigger|smarter|better|stronger|more powerful)\b",
    r"\buse the best (?:one|model|you have)\b",
    r"\bswitch to (?:a |the )?(?:bigger|better|smarter) (?:one|model)\b",
    r"\bupgrade (?:the )?model\b",
    r"\bneed a bigger\b",
]

# Pre-compile once at import time for speed.
_COMPILED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in COGNITIVE_ESCALATION_PATTERNS
]


# =============================================================================
# Small-model classification (env-driven — Lane A de-hardcode, 2026-07-01)
# =============================================================================
#
# Models considered "small" — a cognitive cue should escalate off these.
# Anything not classified small is treated as capable enough to satisfy a
# cognitive cue (so we don't downgrade Opus sessions on a "think about
# this" cue).
#
# Two signals, no hardcoded model IDs:
#   1. ASTRA_SMALL_MODELS — comma-separated exact IDs from .env.
#   2. Tier-marker heuristic — small tiers are consistently NAMED small
#      (mini/nano/flash/lite/haiku). Markers are segment-anchored so
#      "gemini-3.1-pro" does NOT match on the "mini" inside "gemini".

_SMALL_TIER_MARKER = re.compile(
    r"(?:^|[-_.])(?:mini|nano|flash|lite|haiku)(?:$|[-_.\d])", re.IGNORECASE
)


def get_small_models() -> set[str]:
    """Exact small-tier model IDs from ASTRA_SMALL_MODELS in .env."""
    raw = os.getenv("ASTRA_SMALL_MODELS", "")
    return {m.strip() for m in raw.split(",") if m.strip()}


# =============================================================================
# Public API
# =============================================================================

def detect_cognitive_escalation(text: str) -> Optional[str]:
    """Return the first matching pattern string, or None.

    Returning the pattern (rather than a bool) lets the caller log which
    cue fired — useful when tuning the table.
    """
    if not text:
        return None
    for pattern, compiled in zip(COGNITIVE_ESCALATION_PATTERNS, _COMPILED_PATTERNS):
        if compiled.search(text):
            return pattern
    return None


def is_small_model(model_id: str) -> bool:
    """Return True if model_id is small-tier (env list or tier-marker).

    Unknown model IDs without a small tier-marker are treated as NOT
    small — we'd rather keep a current session on an unfamiliar model
    than escalate needlessly.
    """
    if not model_id:
        return False
    if model_id in get_small_models():
        return True
    return bool(_SMALL_TIER_MARKER.search(model_id))
