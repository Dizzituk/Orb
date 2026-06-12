# FILE: app/invocation/classifier.py
# Purpose: Intent Classifier - deterministic pattern matching for ambient invocations.
# Called-by: app.invocation, app.invocation.router, app.invocation.tier_gate
# Depends-on: app.invocation.models
# Last-renovated: 2026-06-11
"""
Intent Classifier - deterministic pattern matching for ambient invocations.

Runs in microseconds before any LLM call. Matches the user's transcript
against a rule table and returns the most specific matching Intent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from app.invocation.models import Intent, ClassifiedIntent


@dataclass
class _Rule:
    name: str
    patterns: List[re.Pattern]
    intent: Intent
    confidence: float
    needs_screenshot: bool


def _compile(*patterns: str) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_RULES: List[_Rule] = [
    _Rule(
        name="takeover",
        patterns=_compile(r"\btake(\s+it)?\s+over\b|\bdo\s+(this|it)\s+for\s+me\b|\bhandle\s+this\s+for\s+me\b"),
        intent=Intent.TAKEOVER,
        confidence=0.95,
        needs_screenshot=True,
    ),
    _Rule(
        name="diagnose_why",
        patterns=_compile(r"\bwhy\s+(is|isn't|is\s+not|won't)\b|\bwhat's\s+wrong\b|\bnot\s+working\b"),
        intent=Intent.LOOK_DIAGNOSE,
        confidence=0.9,
        needs_screenshot=True,
    ),
    _Rule(
        name="explain",
        patterns=_compile(r"\bexplain\s+this\b|\bexplain\s+(to\s+)?me\b|\bhelp\s+me\s+understand\b"),
        intent=Intent.LOOK_EXPLAIN,
        confidence=0.9,
        needs_screenshot=True,
    ),
    _Rule(
        name="whats_going_on",
        patterns=_compile(r"\bwhat's\s+going\s+on\b|\bwhat\s+is\s+going\s+on\b|\bwhat\s+am\s+i\s+(doing|looking\s+at)\b"),
        intent=Intent.LOOK_EXPLAIN,
        confidence=0.85,
        needs_screenshot=True,
    ),
    _Rule(
        name="read_aloud",
        patterns=_compile(r"\bread\s+(this|it|that)\b|\bread\s+(to|for)\s+me\b"),
        intent=Intent.LOOK_READ,
        confidence=0.95,
        needs_screenshot=True,
    ),
    _Rule(
        name="what_does_this_say",
        patterns=_compile(r"\bwhat\s+does\s+(this|it|that)\s+say\b|\bwhat's\s+(this|that)\b|\bwhat\s+is\s+this\b"),
        intent=Intent.LOOK_DESCRIBE,
        confidence=0.85,
        needs_screenshot=True,
    ),
    _Rule(
        name="weather",
        patterns=_compile(r"\bweather\b|\btemperature\b|\bforecast\b"),
        intent=Intent.CONVERSE,
        confidence=0.9,
        needs_screenshot=False,
    ),
    _Rule(
        name="news",
        patterns=_compile(r"\b(the\s+)?news\b|\bheadlines\b"),
        intent=Intent.CONVERSE,
        confidence=0.9,
        needs_screenshot=False,
    ),
    _Rule(
        name="remind_or_schedule",
        patterns=_compile(r"\bremind\s+me\b|\bset\s+(a\s+)?reminder\b|\bschedule\b"),
        intent=Intent.CONVERSE,
        confidence=0.9,
        needs_screenshot=False,
    ),
]


def classify(transcript: str) -> ClassifiedIntent:
    """Classify a transcript against the rule table."""
    if not transcript or not transcript.strip():
        return ClassifiedIntent(intent=Intent.UNKNOWN, confidence=0.0, needs_screenshot=False)

    text = transcript.strip()
    for rule in _RULES:
        if all(p.search(text) for p in rule.patterns):
            return ClassifiedIntent(
                intent=rule.intent,
                confidence=rule.confidence,
                matched_rule=rule.name,
                needs_screenshot=rule.needs_screenshot,
            )

    return ClassifiedIntent(
        intent=Intent.UNKNOWN,
        confidence=0.0,
        matched_rule=None,
        needs_screenshot=True,
    )


def intent_to_default_tier(intent: Intent):
    """Map an intent to its default (un-promoted) tier."""
    from app.invocation.models import Tier
    return {
        Intent.CONVERSE:      Tier.T1_MINI,
        Intent.LOOK_READ:     Tier.T1_MINI,
        Intent.LOOK_DESCRIBE: Tier.T1_MINI,
        Intent.LOOK_EXPLAIN:  Tier.T2_STD,
        Intent.LOOK_DIAGNOSE: Tier.T2_STD,
        Intent.TAKEOVER:      Tier.T2_STD,
        Intent.NOTIFY:        Tier.T1_MINI,
        Intent.UNKNOWN:       Tier.T1_MINI,
    }[intent]