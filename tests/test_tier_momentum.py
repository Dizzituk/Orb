# FILE: tests/test_tier_momentum.py
# Purpose: Job 6 (2026-06-12) + Lane A retune (2026-07-01) — tier router moves
#          BOTH directions: rises to architect immediately, and after the spike
#          a held casual turn steps DOWN to the reasoning tier (never Opus),
#          settling back to cheap within ~1 turn per spike. Models are env-only.
# Called-by: pytest
# Depends-on: app.llm.routing.tier_momentum, app.llm.routing.chat_model_selection
# Last-renovated: 2026-07-01
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

# Env-pinned models for deterministic assertions (Lane A 2026-07-01: routing
# is env-only — tests must SET the env, not rely on hardcoded defaults).
CHEAP = "gpt-5.4-mini"
MID = "gpt-5.4"
ARCHITECT = "claude-opus-4-8"


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
    # Deterministic env — routing models resolve ONLY from env now.
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
    for var in ("COGNITIVE_ESCALATION_MODEL", "COGNITIVE_ESCALATION_PROVIDER",
                "FILE_CREATION_MODEL", "FILE_CREATION_PROVIDER",
                "IMAGE_CHAT_MODEL", "IMAGE_CHAT_PROVIDER",
                "TIER_MOMENTUM_DECAY", "TIER_MOMENTUM_WINDOW",
                "TIER_MOMENTUM_HOLD", "TIER_MOMENTUM_STEPDOWN"):
        monkeypatch.delenv(var, raising=False)
    tier_momentum.reset()
    cms._session_model_cache.clear()
    cms._sticky_source.clear()
    yield
    tier_momentum.reset()
    cms._session_model_cache.clear()
    cms._sticky_source.clear()


def _req(message: str):
    # ui_context present (plain chat panel) → confirmation gate skipped,
    # and it is not an agentic tab (no job/job_type fields)
    return SimpleNamespace(
        message=message, project_id=PID,
        provider=None, model=None,
        ui_context={"view_type": "chat"},
    )


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
    assert first == pytest.approx(tier_momentum.SCORE_ARCHITECT)
    assert second == pytest.approx(tier_momentum.SCORE_ARCHITECT * tier_momentum.DECAY)


def test_single_spike_hold_lasts_at_most_one_casual_turn():
    """Lane A retune: DECAY=0.45 → a single architect spike (2.0) is below
    HOLD_THRESHOLD=1.0 after ONE casual turn."""
    tier_momentum.record_turn(PID, "architect", "anthropic", ARCHITECT, "t")
    assert tier_momentum.momentum_hold(PID) is not None      # turn 1 held
    tier_momentum.record_turn(PID, "casual", "openai", CHEAP, "t")
    assert tier_momentum.momentum_hold(PID) is None          # turn 2 released


def test_architect_hold_steps_down_to_reasoning_tier():
    """A held casual turn must NOT ride the top model — it steps down to the
    REASONING/FRONTIER tier (env-resolved)."""
    tier_momentum.record_turn(PID, "architect", "anthropic", ARCHITECT, "t")
    held = tier_momentum.momentum_hold(PID)
    assert held is not None
    provider, model, momentum = held
    assert model == MID, f"architect hold must step down to the mid tier, got {model}"
    assert provider == "openai"


def test_reasoning_hold_keeps_reasoning_model():
    tier_momentum.record_turn(PID, "reasoning", "openai", MID, "t")
    held = tier_momentum.momentum_hold(PID)
    assert held is not None
    assert held[1] == MID


def test_stepdown_kill_switch(monkeypatch):
    monkeypatch.setenv("TIER_MOMENTUM_STEPDOWN", "0")
    import importlib
    importlib.reload(tier_momentum)
    try:
        tier_momentum.record_turn(PID, "architect", "anthropic", ARCHITECT, "t")
        held = tier_momentum.momentum_hold(PID)
        assert held is not None
        assert held[1] == ARCHITECT, "kill-switch must restore hold-what-was-used"
    finally:
        monkeypatch.delenv("TIER_MOMENTUM_STEPDOWN", raising=False)
        importlib.reload(tier_momentum)
        tier_momentum.reset()


def test_hold_releases_after_decay():
    tier_momentum.record_turn(PID, "architect", "anthropic", ARCHITECT, "t")
    assert tier_momentum.momentum_hold(PID) is not None
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
    """The 04:35→05:00 replay shape: architect turns run Opus; the casual tail
    drops to the MID tier immediately (step-down) and to cheap right after —
    the top model never outlives the architect turns themselves."""
    script = [
        # (message, expected_model, phase)
        ("alright mate, how's it going?", CHEAP, "casual open"),
        ("I want to redesign the memory architecture — walk me through the "
         "trade-off analysis and the key design decision options.",
         ARCHITECT, "architectural turn 1"),
        ("And how would the migration plan handle the pipeline integration "
         "and the blast radius across the codebase architecture?",
         ARCHITECT, "architectural turn 2"),
        ("ha fair enough mate", MID, "casual aside — held at MID tier, not Opus"),
        ("cheers. pizza tonight then?", MID, "second casual — still MID"),
        ("nice one", CHEAP, "third casual — momentum drained, de-escalated"),
    ]

    for message, expected_model, phase in script:
        provider, model, extras = cms.select_chat_model(_req(message), db)
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


def test_single_arch_spike_drops_off_top_model_immediately(db):
    """One architect turn → the very next turn is already OFF the top model
    (mid tier at most), and the one after is back on cheap."""
    _, model1, _ = cms.select_chat_model(
        _req("redesign the architecture: trade-off analysis of the migration "
             "pipeline please"), db)
    assert model1 == ARCHITECT
    _, model2, _ = cms.select_chat_model(_req("ha nice one"), db)
    assert model2 == MID, f"first casual turn must step down off Opus, got {model2}"
    _, model3, _ = cms.select_chat_model(_req("cheers mate"), db)
    assert model3 == CHEAP, f"second casual turn must be back on cheap, got {model3}"


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
    req.provider, req.model = "anthropic", ARCHITECT
    cms.select_chat_model(req, db)
    assert cms.get_sticky_model(PID) == ("anthropic", ARCHITECT)


# =========================================================================
# De-escalation fix (2026-06-24): automatic escalations must NOT latch
# =========================================================================

def test_image_gen_does_not_latch_and_drops_back(db):
    """The logged 6h-hold incident: an image-gen turn escalates to write the
    prompt but must not pin — the very next casual turn is back on cheap."""
    _, model1, extras1 = cms.select_chat_model(_req("make me an image of a dragon"), db)
    assert extras1.get("image_route") is True
    assert model1 != CHEAP                       # escalated to write the prompt
    assert cms.get_sticky_model(PID) is None      # but did NOT permanently pin
    # image is recorded as 'explore' -> drops back on the very next casual turn
    _, model2, _ = cms.select_chat_model(_req("nice one cheers"), db)
    assert model2 == CHEAP                        # dropped back to mini immediately


def test_cognitive_cue_escalates_then_decays(db):
    """'think hard' escalates this turn but does not latch; it decays."""
    _, model1, extras1 = cms.select_chat_model(
        _req("think hard about this one for me please"), db)
    assert model1 != CHEAP
    assert extras1.get("model_source") == "cognitive_auto"
    assert cms.get_sticky_model(PID) is None
    cms.select_chat_model(_req("ok"), db)
    _, model3, _ = cms.select_chat_model(_req("ta"), db)
    assert model3 == CHEAP


def test_history_only_restores_explicit_pin(monkeypatch):
    """infer_sticky_from_history (the post-restart path) must restore ONLY an
    explicit pin — never an auto-escalation or a legacy NULL row (LATCH B)."""
    auto = [SimpleNamespace(role="assistant", model=MID,
                            provider="openai", model_source="image_auto")]
    monkeypatch.setattr(cms.memory_service, "list_messages", lambda *a, **k: auto)
    assert cms.infer_sticky_from_history(PID, None) is None

    legacy_null = [SimpleNamespace(role="assistant", model=MID,
                                   provider="openai", model_source=None)]
    monkeypatch.setattr(cms.memory_service, "list_messages", lambda *a, **k: legacy_null)
    assert cms.infer_sticky_from_history(PID, None) is None

    pinned = [SimpleNamespace(role="assistant", model=ARCHITECT,
                              provider="anthropic", model_source="explicit_user")]
    monkeypatch.setattr(cms.memory_service, "list_messages", lambda *a, **k: pinned)
    assert cms.infer_sticky_from_history(PID, None) == ("anthropic", ARCHITECT)

    # v2026-07-01: a pinned SMALL model is not "elevated" — nothing to restore
    # (falling through to normal tiering picks the cheap model anyway).
    pinned_small = [SimpleNamespace(role="assistant", model=CHEAP,
                                    provider="openai", model_source="explicit_user")]
    monkeypatch.setattr(cms.memory_service, "list_messages", lambda *a, **k: pinned_small)
    assert cms.infer_sticky_from_history(PID, None) is None


def test_history_newest_pin_decides_no_resurrection(monkeypatch):
    """2026-07-01 review fix: the NEWEST pinned row decides. A user who pinned
    Opus, then explicitly downgraded to mini, must NOT get Opus resurrected
    after a restart — the scan stops at the newer small pin and restores
    nothing."""
    history_chronological = [
        SimpleNamespace(role="assistant", model=ARCHITECT,
                        provider="anthropic", model_source="explicit_user"),
        SimpleNamespace(role="assistant", model=CHEAP,
                        provider="openai", model_source="frontend_override"),
        SimpleNamespace(role="assistant", model=CHEAP,
                        provider="openai", model_source="frontend_override"),
    ]
    monkeypatch.setattr(cms.memory_service, "list_messages",
                        lambda *a, **k: history_chronological)
    assert cms.infer_sticky_from_history(PID, None) is None

    # And unpinned auto rows BETWEEN pins are still skipped — an elevated
    # explicit pin newer than any small pin IS restored through auto noise.
    history2 = [
        SimpleNamespace(role="assistant", model=CHEAP,
                        provider="openai", model_source="frontend_override"),
        SimpleNamespace(role="assistant", model=ARCHITECT,
                        provider="anthropic", model_source="explicit_user"),
        SimpleNamespace(role="assistant", model=MID,
                        provider="openai", model_source="image_auto"),
    ]
    monkeypatch.setattr(cms.memory_service, "list_messages",
                        lambda *a, **k: history2)
    assert cms.infer_sticky_from_history(PID, None) == ("anthropic", ARCHITECT)


def test_explicit_pin_survives_casual_turns(db):
    """An explicit dropdown choice holds across casual turns (is_explicit_sticky)."""
    req = _req("whatever you say")
    req.provider, req.model = "anthropic", ARCHITECT
    cms.select_chat_model(req, db)
    assert cms.is_explicit_sticky(PID) is True
    _, model2, _ = cms.select_chat_model(_req("ok cheers"), db)
    assert model2 == ARCHITECT
