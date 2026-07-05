# FILE: tests/test_conversational_gate.py
# Purpose: Lane A (2026-07-01) — conversational turns never escalate the tier
#          nor divert to RAG; genuine imperatives still route up; RAG-as-context
#          keeps codebase questions in chat with a grounded block injected.
# Called-by: pytest
# Depends-on: app.llm.routing.conversational_gate, app.llm.routing.rag_fallback,
#             app.llm.routing.chat_model_selection, app.llm.routing.codebase_context_bridge
# Last-renovated: 2026-07-01
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm.routing.conversational_gate import (
    has_actionable_command,
    has_conversational_leadin,
    is_conversational_turn,
)
from app.llm.routing.rag_fallback import (
    is_architecture_query,
    needs_llm_codebase_check,
)
from app.llm.routing import tier_momentum
from app.llm.routing import chat_model_selection as cms

PID = 4242

CHEAP = "gpt-5.4-mini"
ARCHITECT = "claude-opus-4-8"
MID = "gpt-5.4"


@pytest.fixture(autouse=True)
def _env_and_state(monkeypatch):
    """Deterministic env (models are env-only now) + clean routing state."""
    monkeypatch.setenv("CHAT_PROVIDER", "openai")
    monkeypatch.setenv("CHAT_MODEL", CHEAP)
    monkeypatch.setenv("ARCHITECT_PROVIDER", "anthropic")
    monkeypatch.setenv("ARCHITECT_MODEL", ARCHITECT)
    monkeypatch.setenv("REASONING_PROVIDER", "openai")
    monkeypatch.setenv("REASONING_MODEL", MID)
    monkeypatch.setenv("FRONTIER_PROVIDER", "openai")
    monkeypatch.setenv("FRONTIER_MODEL", MID)
    monkeypatch.setenv("DEFAULT_PROVIDER", "openai")
    monkeypatch.setenv("DEFAULT_MODEL", CHEAP)
    monkeypatch.delenv("CONVERSATIONAL_GATE_ENABLED", raising=False)
    # No history / no sticky between tests.
    monkeypatch.setattr(cms.memory_service, "list_messages", lambda *a, **k: [])
    tier_momentum.reset()
    cms._session_model_cache.clear()
    cms._sticky_source.clear()
    yield
    tier_momentum.reset()
    cms._session_model_cache.clear()
    cms._sticky_source.clear()


def _req(message: str):
    return SimpleNamespace(
        message=message, project_id=PID,
        provider=None, model=None,
        ui_context={"view_type": "chat"},
    )


# =========================================================================
# Gate unit behaviour
# =========================================================================

CONVERSATIONAL = [
    "can we talk about mapping out the repo",                      # 04:35 replay
    "Can we have a chat about the memory architecture?",
    "let's chat about the pipeline redesign",
    "I'm thinking about tidying the memory and mapping out the repos",  # 05:08 replay
    "been thinking about the architecture a lot lately",
    "thinking out loud here — the repo structure feels off",
    "what do you think about the migration plan?",
    "wondering if the pipeline integration is worth redesigning",
    "explain the idea of a dependency graph to me",
    "just musing on the blast radius of that refactor",
    "what's your take on the memory system architecture?",
]

ACTIONABLE = [
    "map the repo",
    "map out the repo architecture and redesign the pipeline integration now",
    "refactor the recall module",
    "please refactor the memory pipeline",
    "can you migrate the routing tables",
    "where is the stream router implemented",
    "who calls build_full_context",
    "build me a dashboard for the metrics",
    "let's refactor the router",
    "I was wondering if you could add a retry to the fetcher",   # polite request
    "can we talk about the roadmap. actually, redesign the pipeline now",  # command wins
    # 2026-07-01 review fixes — grammar holes locked in:
    "could you map the repo when you get a chance?",             # interrogative polite
    "can you please migrate the routing tables",                 # 'please' inside request
    "analyze the memory pipeline and optimize the recall module", # US spellings
    "overhaul the memory system",                                # vocabulary gap
    "sort out the recall module please",                         # phrasal verb
    "- refactor the tier code\n- test the recall module",        # bulleted work spec
    "can we talk about the router — actually, refactor it now",  # em-dash command wins
]

NEUTRAL = [
    "alright mate, how's it going?",
    "I want to redesign the memory architecture — walk me through the trade-off analysis.",
    "how would the migration plan handle the pipeline integration?",
    "should I buy land or a caravan first",
]


@pytest.mark.parametrize("msg", CONVERSATIONAL)
def test_conversational_leadins_detected(msg):
    assert is_conversational_turn(msg) is True, msg


@pytest.mark.parametrize("msg", ACTIONABLE)
def test_actionable_commands_defeat_the_gate(msg):
    assert is_conversational_turn(msg) is False, msg


@pytest.mark.parametrize("msg", NEUTRAL)
def test_neutral_messages_leave_existing_routing_alone(msg):
    assert is_conversational_turn(msg) is False, msg


def test_kill_switch(monkeypatch):
    monkeypatch.setenv("CONVERSATIONAL_GATE_ENABLED", "0")
    assert is_conversational_turn("can we talk about the repo") is False


def test_gerunds_do_not_count_as_commands():
    # "tidying"/"mapping" inside a musing are not command-position verbs.
    msg = "I'm thinking about tidying the memory and mapping the repos"
    assert has_actionable_command(msg) is False
    assert has_conversational_leadin(msg) is True


# =========================================================================
# (a) Escalation gate — chat_model_selection replay (04:35 log evidence)
# =========================================================================

def test_0435_replay_conversational_arch_talk_stays_on_chat_model():
    """'can we talk about mapping out the repo …' — heavy arch keywords, but
    pure conversation. Must stay on the chat model: no Opus, no reasoning tier."""
    msg = ("can we talk about mapping out the repo — the architecture, "
           "the pipeline integration and a possible redesign?")
    provider, model, extras = cms.select_chat_model(_req(msg), None)
    assert model == CHEAP, f"conversational turn escalated to {model}"
    # And it must not have primed the held-model slot with an architect model.
    decisions = tier_momentum.get_recent_decisions()
    assert decisions and decisions[-1]["tier"] in ("casual", "explore")


def test_same_keywords_with_imperative_still_escalate():
    msg = "map out the repo architecture and redesign the pipeline integration now"
    provider, model, extras = cms.select_chat_model(_req(msg), None)
    assert model == ARCHITECT
    assert provider == "anthropic"


def test_architect_model_comes_from_env_not_a_literal(monkeypatch):
    """THE 04:35 bug: ARCHITECT_MODEL missing from .env ran a stale hardcoded
    claude-opus-4-6. Now the env value always wins."""
    monkeypatch.setenv("ARCHITECT_MODEL", "claude-test-arch")
    _, model, _ = cms.select_chat_model(
        _req("redesign the architecture: trade-off analysis of the migration "
             "pipeline please"), None,
    )
    assert model == "claude-test-arch"


def test_missing_role_and_default_fails_loud(monkeypatch):
    from app.llm.frontier_models import get_role_model
    for var in ("ARCHITECT_PROVIDER", "ARCHITECT_MODEL",
                "DEFAULT_PROVIDER", "DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError) as exc:
        get_role_model("ARCHITECT")
    assert "ARCHITECT" in str(exc.value)
    assert ".env" in str(exc.value)


# =========================================================================
# (b) RAG divert gate — rag_fallback replay (05:08 log evidence)
# =========================================================================

def test_0508_replay_musing_not_diverted_to_rag():
    # Without the gate this matches natural pattern "tell me about" + guard
    # keywords (memory/pipeline) and leaves chat for the RAG answerer.
    msg = "Tell me about the memory pipeline — I'm just thinking out loud here"
    assert is_architecture_query(msg) is False
    assert needs_llm_codebase_check(msg) is False


def test_conversational_arch_musing_not_diverted():
    assert is_architecture_query(
        "I'm thinking about tidying the memory and mapping out the repos"
    ) is False


def test_genuine_codebase_questions_still_divert():
    # High-precision control (same as test_codebase_routing_negatives).
    assert is_architecture_query("where is the stream router defined") is True
    # Natural + guard keyword control, no conversational lead-in.
    assert is_architecture_query("Tell me about the memory pipeline") is True


def test_locate_questions_keep_grounding_despite_trailing_musing():
    """2026-07-01 review fix: high-precision locate shapes bypass the gate —
    a trailing musing clause must not strip a locate question of grounding."""
    assert is_architecture_query(
        "where is the stream router defined? I'm just curious about it"
    ) is True


# =========================================================================
# RAG-as-context bridge (Lane A<->C contract) + divert wiring
# =========================================================================

def test_bridge_returns_block_from_contract_fn(monkeypatch):
    import app.rag.answerer as answerer
    from app.llm.routing.codebase_context_bridge import try_build_codebase_context

    async def fake_build(question, db):
        return "[CODEBASE_CONTEXT]\nfiles: app/llm/routing/chat_routing.py"

    monkeypatch.setattr(answerer, "build_codebase_context", fake_build)
    block = try_build_codebase_context("where is the stream router defined", None)
    assert block.startswith("[CODEBASE_CONTEXT]")


def test_bridge_never_raises(monkeypatch):
    import app.rag.answerer as answerer
    from app.llm.routing.codebase_context_bridge import try_build_codebase_context

    async def boom(question, db):
        raise RuntimeError("lane C exploded")

    monkeypatch.setattr(answerer, "build_codebase_context", boom)
    assert try_build_codebase_context("q", None) == ""


def test_bridge_kill_switch(monkeypatch):
    from app.llm.routing.codebase_context_bridge import try_build_codebase_context
    monkeypatch.setenv("RAG_AS_CONTEXT_ENABLED", "0")
    assert try_build_codebase_context("q", None) == ""


def test_bridge_clamps_oversized_blocks(monkeypatch):
    import app.rag.answerer as answerer
    from app.llm.routing.codebase_context_bridge import try_build_codebase_context

    async def huge(question, db):
        return "x" * 10_000

    monkeypatch.setattr(answerer, "build_codebase_context", huge)
    assert len(try_build_codebase_context("q", None)) == 3000


def test_bridge_timeout_is_effective_inside_event_loop(monkeypatch):
    """2026-07-01 review fix: on timeout the daemon worker is ABANDONED — the
    caller gets \"\" within ~the cap instead of blocking until the build ends."""
    import asyncio as _asyncio
    import time
    import app.rag.answerer as answerer
    from app.llm.routing.codebase_context_bridge import try_build_codebase_context

    async def slow(question, db):
        await _asyncio.sleep(3)
        return "too late"

    monkeypatch.setattr(answerer, "build_codebase_context", slow)
    monkeypatch.setenv("RAG_CONTEXT_TIMEOUT_S", "0.3")

    async def _call_on_loop_thread():
        # Called on the event-loop thread (exactly like the sync chat
        # handlers) -> get_running_loop() succeeds -> daemon-worker path.
        return try_build_codebase_context("q", None)

    start = time.monotonic()
    block = _asyncio.run(_call_on_loop_thread())
    elapsed = time.monotonic() - start
    assert block == ""
    assert elapsed < 2.0, f"timeout did not cut the wait (took {elapsed:.1f}s)"


def _normal_routing_harness(monkeypatch, context_block: str):
    """Run handle_normal_routing with everything stubbed; returns
    (captured_full_context, rag_divert_called)."""
    import app.llm.routing.chat_routing as cr
    import app.llm.routing.codebase_context_bridge as ccb

    captured = {}
    called = {"divert": False}

    monkeypatch.setattr(cr, "_RAG_STREAM_AVAILABLE", True)
    monkeypatch.setattr(cr, "is_architecture_query", lambda m: True)
    monkeypatch.setattr(ccb, "try_build_codebase_context", lambda q, db: context_block)

    def fake_rag_stream(**kwargs):
        # Record at CALL time — async generator bodies only run when iterated.
        called["divert"] = True

        async def _gen():
            yield "data: rag\n\n"

        return _gen()

    monkeypatch.setattr(cr, "generate_rag_query_stream", fake_rag_stream)
    monkeypatch.setattr(cr, "build_full_context", lambda *a, **k: "BASE_CONTEXT")
    monkeypatch.setattr(cr, "classify_job_type", lambda *a: SimpleNamespace(value="chat_light"))
    monkeypatch.setattr(cr, "select_provider_for_job_type", lambda jt: ("openai", "stub-model"))
    monkeypatch.setattr(cr, "ensure_provider_available", lambda p, m: (p, m))
    monkeypatch.setattr(cr, "build_messages", lambda *a, **k: [])
    monkeypatch.setattr(cr, "_run_grounding_gate", lambda req, sp, label: sp)
    monkeypatch.setattr(cr, "_inject_tools", lambda p, m, r, sp, **k: (None, sp))
    monkeypatch.setattr(cr, "is_high_stakes_job", lambda jt: False)

    def fake_system_prompt(project, full_context, **kwargs):
        captured["full_context"] = full_context
        return "SYS"

    monkeypatch.setattr(cr, "build_system_prompt", fake_system_prompt)

    async def fake_sse(**kwargs):
        yield "data: chat\n\n"

    monkeypatch.setattr(cr, "generate_sse_stream", fake_sse)

    req = SimpleNamespace(
        message="where is the stream router defined",
        project_id=PID, provider=None, model=None,
        continue_job_id=None, job_state=None, job_type="",
        use_semantic_search=False, include_history=True, history_limit=10,
        enable_reasoning=False,
    )
    resp = cr.handle_normal_routing(req, project=None, db=None, trace=None)
    return captured, called, resp


def test_divert_point_injects_context_and_stays_in_chat(monkeypatch):
    """With a grounded block available, the architecture question is answered
    by the NORMAL chat path (block injected into the prompt context) and the
    raw RAG stream is never used as the reply."""
    captured, called, resp = _normal_routing_harness(
        monkeypatch, "[CODEBASE_CONTEXT] grounded stuff",
    )
    assert resp is not None
    assert called["divert"] is False, "raw RAG stream must not be the reply"
    assert "BASE_CONTEXT" in captured["full_context"]
    assert "[CODEBASE_CONTEXT] grounded stuff" in captured["full_context"]


def test_divert_point_falls_back_to_legacy_rag_without_block(monkeypatch):
    """Block unavailable (Lane C missing / kill-switched / errored) → the
    legacy RAG divert still answers, exactly as before."""
    captured, called, resp = _normal_routing_harness(monkeypatch, "")
    assert resp is not None
    assert called["divert"] is True, "legacy divert must still work as fallback"
    assert "full_context" not in captured, "chat path must not have been built"
