# FILE: tests/test_codebase_routing_negatives.py
# Purpose: Regression: personal/life-planning messages must NOT route to codebase RAG.
# Called-by: pytest (tests/)
# Depends-on: app.llm.routing.rag_fallback, app.astra_memory.topic_tagger,
#             app.translation._tier0_codebase_questions
# Last-renovated: 2026-06-16
"""
Regression test for Job J8 (P2).

A personal/life-planning message ("stack money -> caravan -> land -> solar ->
borehole -> retire; motivation at 45") was once reported as routing to the
codebase-intelligence responder. A 2026-06-14 RAG-tightening already narrowed
the patterns so the message now routes to general chat. This test LOCKS that
fix in.

It exercises ONLY the deterministic codebase-detection gates that sit in front
of the responder:

  * is_architecture_query()        — app.llm.routing.rag_fallback
  * needs_llm_codebase_check()     — app.llm.routing.rag_fallback
  * check_rag_codebase_query()     — app.translation._tier0_codebase_questions (Tier-0)
  * check_natural_codebase_question() — app.translation._tier0_codebase_questions (Tier-0)
  * extract_tags()                 — app.astra_memory.topic_tagger

All five are pure functions of a single string. NO DB, NO live LLM call. The
LLM safety-net (codebase_intent_llm) is NOT exercised here on purpose — these
tests assert exactly the deterministic gate that decides whether the model is
ever consulted.
"""
from __future__ import annotations

from app.llm.routing.rag_fallback import (
    is_architecture_query,
    needs_llm_codebase_check,
)
from app.translation._tier0_codebase_questions import (
    check_natural_codebase_question,
    check_rag_codebase_query,
)
from app.astra_memory.topic_tagger import extract_tags

# The exact J8 message that was misrouted.
J8_MESSAGE = (
    "I want to stack money, buy a caravan, then land, put in solar and a "
    "borehole, then retire. How do I stay motivated at 45?"
)

# Reflective / personal variants: life planning, motivation, finance goals,
# philosophy. None of these are codebase questions.
PERSONAL_VARIANTS = [
    "I'm 45 and need to stack money for a caravan then land then retire",
    "how do I stay motivated at this age",
    "should I buy land or a caravan first",
    "what's the best way to save up for early retirement",
    "I want to live off-grid with solar and a borehole one day",
    "how do I find motivation to keep grinding at work",
    "is it smarter to buy a plot of land or rent forever",
    "I keep losing motivation, how do other people stay driven",
    "planning my future: caravan, then land, then a house",
    "what gives life meaning once you hit your mid forties",
    "should I focus on stacking cash or enjoying life now",
]

ALL_PERSONAL = [J8_MESSAGE] + PERSONAL_VARIANTS

# Topic tags that would (incorrectly) signal a code/dev message.
CODE_DEV_TAGS = {
    "code",
    "python",
    "testing",
    "architecture",
    "documentation",
    "debugging",
    "astra_dev",
}


# ---------------------------------------------------------------------------
# NEGATIVES — personal messages must NOT trip any codebase gate
# ---------------------------------------------------------------------------

def test_is_architecture_query_false_for_personal_messages():
    for msg in ALL_PERSONAL:
        assert is_architecture_query(msg) is False, (
            f"is_architecture_query falsely fired on personal message: {msg!r}"
        )


def test_needs_llm_codebase_check_false_for_personal_messages():
    for msg in ALL_PERSONAL:
        assert needs_llm_codebase_check(msg) is False, (
            f"needs_llm_codebase_check falsely fired (would pay for an LLM "
            f"round-trip) on personal message: {msg!r}"
        )


def test_tier0_rigid_no_match_for_personal_messages():
    for msg in ALL_PERSONAL:
        assert check_rag_codebase_query(msg).matched is False, (
            f"Tier-0 rigid RAG pattern falsely matched personal message: {msg!r}"
        )


def test_tier0_natural_no_match_for_personal_messages():
    for msg in ALL_PERSONAL:
        assert check_natural_codebase_question(msg).matched is False, (
            f"Tier-0 natural codebase pattern falsely matched personal "
            f"message: {msg!r}"
        )


def test_j8_message_has_no_code_tag():
    tags = extract_tags(J8_MESSAGE)
    assert not (set(tags) & CODE_DEV_TAGS), (
        f"extract_tags tagged the J8 personal message as code/dev: {tags!r}"
    )


# ---------------------------------------------------------------------------
# POSITIVE CONTROLS — the gate still catches real codebase questions
# ---------------------------------------------------------------------------

def test_control_architecture_question_still_detected():
    # High-precision "where is ... router" shape -> deterministic arch route.
    assert is_architecture_query("where is the stream router defined") is True


def test_control_feature_question_routes_to_llm_net():
    # "how does THE finance panel work" lacks a guard keyword, so the
    # deterministic gate declines it (is_architecture_query False) but flags it
    # for the LLM safety-net (needs_llm_codebase_check True). This is the exact
    # live behaviour — the controls must match reality, not an assumption.
    feature_q = "how does the finance panel work"
    assert is_architecture_query(feature_q) is False
    assert needs_llm_codebase_check(feature_q) is True
