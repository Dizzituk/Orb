# FILE: tests/test_scene_research.py
# Purpose: Units for the v3 agentic research rail — gate decisions, fact-pack shape from the
#          reused web-search plumbing (mocked), maybe_research wiring + graceful degradation.
# Called-by: pytest
# Depends-on: app.scene_director.research.*
# Last-renovated: 2026-06-13
from __future__ import annotations

import types

import pytest

from app.scene_director.research import gate, researcher
from app.scene_director.research import maybe_research


# ── gate ─────────────────────────────────────────────────────────────────────

def test_gate_skips_imaginative():
    needs, _ = gate.should_research("a friendly dragon flying over a candy mountain")
    assert needs is False


def test_gate_triggers_on_year():
    assert gate.should_research("the streets of London in 1666")[0] is True


def test_gate_triggers_on_history_cue():
    assert gate.should_research("a medieval village market")[0] is True


def test_gate_triggers_on_landmark():
    assert gate.should_research("a walk across Tower Bridge")[0] is True


def test_gate_triggers_on_explicit_era():
    assert gate.should_research("a quiet street", era="tudor")[0] is True


def test_gate_skips_generic_street():
    needs, _ = gate.should_research("a man walks down a city street")
    assert needs is False


# ── researcher (search mocked) ───────────────────────────────────────────────

def _fake_resp(ok=True, answer="The Thames is tidal in London [1]. Timber houses were common [2].",
               sources=None):
    src = sources if sources is not None else [
        types.SimpleNamespace(title="Britannica", url="https://britannica.com/x"),
        types.SimpleNamespace(title="Museum of London", url="https://museumoflondon.org/y"),
    ]
    return types.SimpleNamespace(ok=ok, answer=answer, sources=src)


@pytest.mark.asyncio
async def test_researcher_builds_fact_pack(monkeypatch):
    import app.llm.web_search as ws

    async def fake_search(req, context=None):
        assert req.query  # query built
        return _fake_resp()
    monkeypatch.setattr(ws, "search_and_answer", fake_search, raising=False)

    pack = await researcher.research("the Great Fire of London", era="tudor")
    assert pack is not None
    assert pack["summary"] and isinstance(pack["facts"], list) and pack["facts"]
    assert pack["sources"][0]["url"].startswith("https://")
    # citation markers stripped from facts
    assert all("[1]" not in f and "[2]" not in f for f in pack["facts"])


@pytest.mark.asyncio
async def test_researcher_none_on_empty_answer(monkeypatch):
    import app.llm.web_search as ws

    async def fake_search(req, context=None):
        return _fake_resp(ok=True, answer="")
    monkeypatch.setattr(ws, "search_and_answer", fake_search, raising=False)
    assert await researcher.research("Tower Bridge") is None


@pytest.mark.asyncio
async def test_researcher_none_on_exception(monkeypatch):
    import app.llm.web_search as ws

    async def boom(req, context=None):
        raise RuntimeError("network down")
    monkeypatch.setattr(ws, "search_and_answer", boom, raising=False)
    assert await researcher.research("Tower Bridge") is None


# ── maybe_research wiring ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_maybe_research_runs_when_gated_in(monkeypatch):
    monkeypatch.setenv("SCENE_RESEARCH_ENABLED", "true")
    import app.llm.web_search as ws

    async def fake_search(req, context=None):
        return _fake_resp()
    monkeypatch.setattr(ws, "search_and_answer", fake_search, raising=False)

    pack = await maybe_research("the streets of London in 1666")
    assert pack is not None and pack["sources"]


@pytest.mark.asyncio
async def test_maybe_research_skips_imaginative(monkeypatch):
    monkeypatch.setenv("SCENE_RESEARCH_ENABLED", "true")
    assert await maybe_research("a friendly dragon over candy mountain") is None


@pytest.mark.asyncio
async def test_maybe_research_disabled(monkeypatch):
    monkeypatch.setenv("SCENE_RESEARCH_ENABLED", "false")
    assert await maybe_research("the Great Fire of London 1666") is None
