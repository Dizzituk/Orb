# FILE: tests/test_recall_guard.py
# Purpose: Guard the 2026-06-24 fix — recall questions about our own conversation
#          ("what was I mentioning to you about anthropic", "what topics have we
#          spoken about today") must NOT trigger a web search, while genuine
#          "what's the latest on X" searches still fire.
# Last-renovated: 2026-06-24
import asyncio

from app.memory.recall_intent import is_recall_question
from app.translation._tier0_web_search import (
    check_web_search_trigger,
    check_deep_research_trigger,
)
from app.translation.schemas import CanonicalIntent


# Real phrasings from the 06-24 conversation + close variants — all RECALL.
_RECALL = [
    "give me a solid read back of all the topics and everything that we spoken about in this chat thread",
    "What was I mentioning to you about anthropic",
    "what topics have we spoken about today",
    "what's my very first message about",
    "what did we talk about today",
    "did we talk about anthropic",
    "remind me what we discussed",
    "do you remember what we said earlier",
    "catch me up",
    "what have we been talking about",
]

# Genuine lookups / non-recall — must STILL be searchable (or simply not recall).
_NOT_RECALL = [
    "what's the latest on anthropic",
    "search the web for anthropic news",
    "what is the current bitcoin price",
    "how do I make a techno bassline",
    "research the fable model thoroughly",
    "log my lunch",
    # mixed-intent / coincidental phrasing that must NOT suppress a search
    # (false positives caught + fixed 2026-06-24):
    "we were talking about going to the shops, find me one nearby",
    "keep going on the highway then turn left",
    "do you remember to buy milk",
    # un-anchored interrogatives / generic-noun trivia must still reach the web
    # (review fixes #19 / #17, 2026-06-24):
    "do you remember how photosynthesis works",
    "do you remember when the eiffel tower was built",
    "what was the first message ever sent on the internet",
]


def test_predicate_matches_recall():
    for t in _RECALL:
        assert is_recall_question(t), f"should be recall: {t!r}"


def test_predicate_rejects_non_recall():
    for t in _NOT_RECALL:
        assert not is_recall_question(t), f"should NOT be recall: {t!r}"


def test_tier0_suppresses_web_search_on_recall():
    # The exact message that hit Brave twice in the real log.
    r = check_web_search_trigger("what topics have we spoken about today")
    assert r.matched is False, r.rule_name
    r2 = check_web_search_trigger("What was I mentioning to you about anthropic")
    assert r2.matched is False, r2.rule_name


def test_tier0_still_searches_genuine_queries():
    # Staleness/explicit paths must survive the new guard.
    latest = check_web_search_trigger("what's the latest on anthropic")
    assert latest.matched is True and latest.intent == CanonicalIntent.WEB_SEARCH
    explicit = check_web_search_trigger("search the web for anthropic news")
    assert explicit.matched is True and explicit.intent == CanonicalIntent.WEB_SEARCH


def test_deep_research_skips_recall():
    # A recall question must not launch an expensive deep-research run...
    r = check_deep_research_trigger("investigate what was I mentioning about anthropic")
    assert r.matched is False, r.rule_name
    # ...but a genuine deep-research request still fires.
    g = check_deep_research_trigger("investigate the fable model thoroughly")
    assert g.matched is True and g.intent == CanonicalIntent.DEEP_RESEARCH


def test_grounding_gate_skips_recall(monkeypatch):
    import app.grounding.grounding_gate as gg
    monkeypatch.setenv("GROUNDING_GATE_ENABLED", "true")
    res = asyncio.run(gg.evaluate_grounding("What was I mentioning to you about anthropic"))
    assert res.grounding_applied is False
    assert "recall_question" in res.classification_signals


def test_bridge_records_classifier_decision(monkeypatch):
    # The phone/bridge path now writes a classifier-decision trail so the chat
    # LLM can explain "why did you search?" (read side was already wired). Mock
    # the translation layer and assert _run_translation records under str(pid).
    import types
    import app.translation as translation_pkg
    import app.bridge.chat_helpers as ch
    from app.translation import recent_decisions as rd

    pid = 778899
    rd.clear_conversation(str(pid))
    fake = types.SimpleNamespace(
        resolved_intent=types.SimpleNamespace(value="WEB_SEARCH"),
        extracted_context={"_classifier_rule": "web_search_staleness_auto",
                           "_classifier_reason": "stale topic"},
        confirmation_gate=types.SimpleNamespace(requires_confirmation=False, passed=True),
        intent_confidence=0.75,
    )
    monkeypatch.setattr(translation_pkg, "translate_message_sync",
                        lambda *a, **k: fake, raising=False)
    ch._run_translation("whats the latest on anthropic", object(), project_id=pid)
    block = rd.build_decisions_block(str(pid))
    assert block and "WEB_SEARCH" in block, block
    # legacy callers (no project_id) must still work and record nothing extra.
    ch._run_translation("hello", object())
    rd.clear_conversation(str(pid))


def test_coverage_still_exports_predicate():
    # coverage.is_recall_question is re-exported from the shared module.
    from app.memory.nat_jobs import coverage
    assert coverage.is_recall_question("what did we talk about today") is True
    assert coverage.is_recall_question("what's the latest on bitcoin") is False
