# FILE: tests/test_tier_momentum.py
# Purpose: Job 6 (2026-06-12) — tier router moves BOTH directions: scripted
#          casual→arch→arch→casual→casual conversation rises to the architect
#          tier immediately and settles back to cheap within a bounded lag;
#          every decision logged with its reason.
# Called-by: pytest
# Depends-on: app.llm.routing.tier_momentum, app.llm.routing.chat_model_selection
# Last-renovated: 2026-06-12
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.memory.models import Project, Message
from app.memory.conversation_models import ConversationSession
from app.llm.routing import tier_momentum
from app.llm.routing import chat_model_selection as cms

_TABLES = [Project.__table__, Message.__table__, ConversationSession.__table__]

PID = 777


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # Deterministic defaults — strip any machine-level overrides
    for var in ("CHAT_MODEL", "CHAT_PROVIDER", "ARCHITECT_MODEL",
                "ARCHITECT_PROVIDER", "REASONING_MODEL", "REASONING_PROVIDER",
                "FRONTIER_MODEL", "FRONTIER_PROVIDER",
                "TIER_MOMENTUM_DECAY", "TIER_MOMENTUM_WINDOW",
                "TIER_MOMENTUM_HOLD"):
        monkeypatch.delenv(var, raising=False)
    tier_momentum.reset()
    cms._session_model_cache.clear()
    yield
    tier_momentum.reset()
    cms._session_model_cache.clear()


def _req(message: str):
    # ui_context present (plain chat panel) → confirmation gate skipped,
    # and it is not an agentic tab (no job/job_type fields)
    return SimpleNamespace(
        message=message, project_id=PID,
        provider=None, model=None,
        ui_context={"view_type": "chat"},
    )


CHEAP = "gpt-5.4-mini"
ARCHITECT = "claude-opus-4-6"


# =========================================================================
# Momentum maths
# =========================================================================

def test_momentum_decays_per_turn():
    tier_momentum.record_turn(PID, "architect", "anthropic", ARCHITECT, "t")
    first = tier_momentum.current_momentum(PID)
    tier_momentum.record_turn(PID, "casual", "openai", CHEAP, "t")
    second = tier_momentum.current_momentum(PID)
    tier_momentum.record_turn(PID, "casual", "openai", CHEAP, "t")
    third = tier_momentum.current_momentum(PID)
    assert first > second > third
    assert first == pytest.approx(2.0)
    assert second == pytest.approx(2.0 * tier_momentum.DECAY)


def test_hold_releases_after_decay():
    tier_momentum.record_turn(PID, "architect", "anthropic", ARCHITECT, "t")
    assert tier_momentum.momentum_hold(PID) is not None
    # Two casual turns drain a single architect spike below the threshold
    tier_momentum.record_turn(PID, "casual", "openai", CHEAP, "t")
    tier_momentum.record_turn(PID, "casual", "openai", CHEAP, "t")
    assert tier_momentum.momentum_hold(PID) is None


def test_hold_never_invents_escalation():
    # Casual-only history → no hold, regardless of threshold
    tier_momentum.record_turn(PID, "casual", "openai", CHEAP, "t")
    assert tier_momentum.momentum_hold(PID) is None


# =========================================================================
# Scripted conversation through the real selector
# =========================================================================

def test_scripted_conversation_rises_and_falls(db):
    script = [
        # (message, expected_model, phase)
        ("alright mate, how's it going?", CHEAP, "casual open"),
        ("I want to redesign the memory architecture — walk me through the "
         "trade-off analysis and the key design decision options.",
         ARCHITECT, "architectural turn 1"),
        ("And how would the migration plan handle the pipeline integration "
         "and the blast radius across the codebase architecture?",
         ARCHITECT, "architectural turn 2"),
        ("ha fair enough mate", ARCHITECT, "casual aside — held by momentum"),
        ("cheers. pizza tonight then?", ARCHITECT, "second casual — still held"),
        ("nice one", CHEAP, "third casual — momentum drained, de-escalated"),
    ]

    routed = []
    for message, expected_model, phase in script:
        provider, model, extras = cms.select_chat_model(_req(message), db)
        routed.append((phase, model))
        assert model == expected_model, (
            f"{phase}: expected {expected_model}, got {model} "
            f"(momentum={tier_momentum.current_momentum(PID)})"
        )

    # The architectural escalation must NOT have latched a permanent sticky —
    # that was the stagnation bug (one arch question pinned the session)
    assert cms.get_sticky_model(PID) is None

    # Momentum strictly decays across the casual tail
    decisions = tier_momentum.get_recent_decisions()
    tail = [d for d in decisions if d["tier"] == "casual"][-3:]
    momenta = [d["momentum"] for d in tail]
    assert momenta == sorted(momenta, reverse=True)


def test_every_decision_logged_with_reason(db):
    cms.select_chat_model(_req("quick one — what's the time?"), db)
    cms.select_chat_model(
        _req("redesign the architecture: trade-off analysis of the migration "
             "pipeline please"), db,
    )
    decisions = tier_momentum.get_recent_decisions()
    assert len(decisions) >= 2
    for d in decisions:
        assert d["reason"], "every tier decision must carry a reason"
        assert d["tier"] in ("architect", "reasoning", "explore", "casual")
        assert "momentum" in d and "model" in d


def test_explicit_user_choice_still_latches_sticky(db):
    """Momentum replaces ONLY complexity-driven latching — an explicit
    frontend model choice must remain permanently sticky."""
    req = _req("whatever you say")
    req.provider, req.model = "anthropic", "claude-opus-4-6"
    cms.select_chat_model(req, db)
    assert cms.get_sticky_model(PID) == ("anthropic", "claude-opus-4-6")
