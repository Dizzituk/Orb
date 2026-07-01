# FILE: app/llm/routing/conversational_gate.py
# Purpose: Conversational-vs-actionable gate — talking ABOUT a topic must not
#          escalate the model tier or divert the message out of chat.
# Called-by: app.llm.routing.chat_model_selection, app.llm.routing.rag_fallback
# Depends-on: stdlib only
# Last-renovated: 2026-07-01
"""
Conversational-vs-actionable gate (Lane A, 2026-07-01).

Evidence from the 2026-07-01 chat log: "can we talk about mapping out the
repo" (04:35) escalated to the architect tier because TIER 4 fires on
keyword counts, and a musing about "tidying the memory... mapping the
repos" (05:08) was diverted out of chat into the RAG answerer by guard
keywords. Both were pure conversation. Talking ABOUT a topic (repo,
memory, architecture) is not a request to DO work on it.

One shared gate, consulted in two places:
  (a) chat_model_selection._select_by_complexity — BEFORE the TIER 4
      architect branch (and the TIER 2 reasoning branches), so a
      conversational turn stays on the normal chat model no matter how
      many architecture keywords it contains.
  (b) rag_fallback.is_architecture_query / needs_llm_codebase_check —
      so a conversational turn is never diverted into the RAG answerer
      (nor pays for the LLM safety-net round-trip).

Decision rule (deterministic, no model calls):
  1. A command-shaped sentence anywhere in the message wins: the turn is
     ACTIONABLE and the gate stands aside ("map the repo", "refactor X",
     "where is Y implemented", "wondering if you could add Z").
  2. Otherwise, a conversational lead-in makes the turn CONVERSATIONAL
     ("can we talk about", "I'm thinking about", "what do you think",
     "wondering if", "thinking out loud", ...).
  3. Otherwise the gate is neutral (returns False) and the existing
     keyword tiers apply unchanged.

Keyword presence alone is never sufficient to defeat the gate: once a
conversational lead-in is present, only a genuine imperative re-enables
escalation/diversion.

Scope notes (deliberate, 2026-07-01):
  - Cognitive-escalation cues ("think hard", "take your time", "no rush")
    run BEFORE this gate in select_chat_model and are NOT suppressed —
    they are an explicit user request for more compute, even mid-musing.
  - rag_fallback consults the gate only for the broad NATURAL patterns;
    the high-precision locate shapes ("where is X implemented", "who
    calls Y") are already command-like and bypass it.

Kill-switch: CONVERSATIONAL_GATE_ENABLED=0 disables the gate entirely
(gate reports every turn as not-conversational, restoring old routing).
"""

from __future__ import annotations

import os
import re
from typing import List

# =============================================================================
# Conversational lead-ins — talk ABOUT a topic
# =============================================================================
# Matched anywhere in the message (word-boundary anchored). These phrases
# frame the turn as discussion, opinion-seeking, or musing.

CONVERSATIONAL_LEADIN_PATTERNS: List[str] = [
    # --- "let's discuss" forms ---
    r"\b(?:can|could|shall|should) we (?:have a )?(?:talk|chat|natter) about\b",
    r"\blet'?s (?:talk|chat|have a chat|have a natter) about\b",
    r"\b(?:we|i) should (?:talk|chat) about\b",
    r"\bi(?:'d| would) (?:love|like) to (?:talk|chat) about\b",

    # --- musing / thinking-out-loud forms ---
    r"\bi'?m thinking (?:about|of|on|that)\b",
    r"\bi(?:'ve| have) been thinking\b",
    r"\bi was thinking\b",
    r"\bbeen thinking about\b",
    r"\bthinking out loud\b",
    r"\bjust (?:thinking|musing|mulling|pondering)\b",
    r"\bmusing (?:about|on|over)\b",
    r"\btoying with the idea\b",
    r"\bplaying with the idea\b",

    # --- wondering / curiosity forms ---
    r"\b(?:just )?wondering (?:if|whether|about|how|what)\b",
    r"\bi wonder (?:if|whether|how|what)\b",
    r"\bi'?m curious (?:about|if|whether|how)\b",
    r"\bcurious about\b",

    # --- opinion-seeking forms ---
    r"\bwhat do you think\b",
    r"\bwhat are your thoughts\b",
    r"\byour thoughts on\b",
    r"\bwhat'?s your (?:take|view|opinion|read) on\b",
    r"\bhow do you feel about\b",
    r"\bdo you reckon\b",
    r"\bwould you say\b",

    # --- concept-level explanation ---
    r"\bexplain the (?:idea|concept|philosophy|thinking) (?:of|behind)\b",
]

_LEADIN_COMPILED: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in CONVERSATIONAL_LEADIN_PATTERNS
]


# =============================================================================
# Command shapes — a genuine request to DO work
# =============================================================================
# Verbs that, in command position, mean "do the work". Word-boundary
# anchored so gerunds don't fire ("tidying" does not match \btidy\b).

_ACTION_VERBS = (
    r"(?:map|refactor|build|implement|fix|write|create|generate|rename|"
    r"migrate|split|extract|redesign|restructure|(?:re)?organi[sz]e|optimi[sz]e|"
    r"audit|analy[sz]e|design|plan|draft|add|remove|delete|update|convert|"
    r"port|deploy|debug|diagnose|investigate|trace|profile|benchmark|"
    r"document|test|run|scaffold|wire|integrate|tidy|clean|"
    r"overhaul|rework|rewrite|replace|merge|consolidate|swap|upgrade|"
    r"automate|install|configure|connect|sort\s+out|set\s+up|hook\s+up)"
)

# Politeness/delegation prefixes that still make it a command. A leading
# interjection ("actually, redesign X now") keeps command position too.
_COMMAND_PREFIX = (
    r"(?:(?:actually|ok|okay|right|now|then|also|and|but|so)[,\s]+)?"
    r"(?:please\s+|go ahead and\s+|can you\s+|could you\s+|would you\s+|"
    r"i need you to\s+|i want you to\s+|let'?s\s+|astra[,:]?\s+)?"
)

ACTIONABLE_COMMAND_PATTERNS: List[str] = [
    # Action verb in command position at the start of a sentence, optionally
    # behind a bullet marker and/or a politeness prefix.
    r"^\s*(?:[-*•]\s+|\d+[.)]\s+)?" + _COMMAND_PREFIX + _ACTION_VERBS + r"\b",
    # Polite-request phrasing anywhere — declarative ("if you could add X",
    # "you can refactor Y") AND interrogative ("could you map the repo",
    # "can you please migrate Z").
    r"\b(?:if\s+)?(?:you\s+(?:could|can|would|will)|(?:could|can|would|will)\s+you)\s+"
    r"(?:please\s+)?(?:go ahead and\s+)?" + _ACTION_VERBS + r"\b",
    # Locate-in-code questions are actionable lookups, not musings.
    r"^\s*where (?:is|are|does|do)\b.+\b(?:implemented|defined|located|handled|triggered|called|routed|live|lives|stored|kept)\b",
    r"^\s*who calls\b",
    r"^\s*show me where\b",
    r"^\s*find (?:where|the file|all|every)\b",
    r"^\s*walk me through\b",
]

_COMMAND_COMPILED: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in ACTIONABLE_COMMAND_PATTERNS
]

# Em/en dashes split too — "can we talk about X — actually, refactor it now"
# must expose the trailing imperative to the ^-anchored command patterns.
# Plain hyphen is NOT a splitter (compound words). Known trade-off: '?' splits
# mean an elliptical "What do you think? Refactor A or rebuild B?" counts as
# actionable — that re-enables the ordinary keyword tiers, which is the safe
# direction (asking to choose between two pieces of work).
_SENTENCE_SPLIT = re.compile(r"[.!?;\n—–]+")


def _enabled() -> bool:
    return os.getenv("CONVERSATIONAL_GATE_ENABLED", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


# =============================================================================
# Public API
# =============================================================================

def has_actionable_command(message: str) -> bool:
    """True when any sentence in the message is command-shaped."""
    if not message:
        return False
    for sentence in _SENTENCE_SPLIT.split(message):
        sentence = sentence.strip()
        if not sentence:
            continue
        for pattern in _COMMAND_COMPILED:
            if pattern.search(sentence):
                return True
    return False


def has_conversational_leadin(message: str) -> bool:
    """True when the message contains a conversational lead-in phrase."""
    if not message:
        return False
    return any(p.search(message) for p in _LEADIN_COMPILED)


def is_conversational_turn(message: str) -> bool:
    """True when this turn is talk ABOUT a topic, not a request to DO work.

    Callers use this to suppress tier escalation and RAG diversion. A
    command-shaped sentence always wins over a lead-in, so "can we talk
    about X — actually, refactor Y now" still routes up.
    """
    if not _enabled():
        return False
    if not message or not message.strip():
        return False
    if has_actionable_command(message):
        return False
    return has_conversational_leadin(message)


__all__ = [
    "is_conversational_turn",
    "has_actionable_command",
    "has_conversational_leadin",
    "CONVERSATIONAL_LEADIN_PATTERNS",
    "ACTIONABLE_COMMAND_PATTERNS",
]
