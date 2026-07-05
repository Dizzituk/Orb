# FILE: tests/test_study_state_and_map_convergence.py
# Purpose: 2026-07-03 improvements — self-confirming selector map (promote +
#          stamp on success) and the mid-study "continue means the course" flag.
# Called-by: pytest
# Depends-on: app.content.distribution.posting_drivers.driver_runner, app.web_automation.study_state, app.web_automation.coursera_study_tools
# Last-renovated: 2026-07-03
"""
Two follow-ups to the Amendment A study loop:

  * StepRunner._record_success — the winning candidate moves to the front
    of its step and gets verified/observed stamps, so the map converges to
    live-proven selectors through ordinary use (and the 10s dead-candidate
    wait from the first live resume becomes a one-time cost).
  * study_state — while a Coursera study tool ran recently, the chat
    context carries a [STUDY SESSION] block so bare "continue / what's
    next" deterministically means the course.
"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import pytest

from app.content.distribution.posting_drivers import driver_runner
from app.content.distribution.posting_drivers.driver_runner import StepRunner
from app.web_automation import study_state
from app.web_automation.study_state import (
    STUDY_ACTIVE_WINDOW_S,
    active_study_session,
    build_study_session_block,
    clear_study_activity,
    mark_study_activity,
)


class FakeBridge:
    """Scripts responses by action_type. `dead` substrings make wait_for miss."""

    def __init__(self, dead=None):
        self.dead = dead or []
        self.calls = []

    async def execute_action(self, sid, action_type, payload=None, timeout_seconds=None):
        payload = payload or {}
        self.calls.append((action_type, payload))
        if action_type == "wait_for":
            key = payload.get("selector") or payload.get("text") or ""
            matched = not any(d in key for d in self.dead)
            return {"ok": True, "result": {"matched": matched, "timeout": not matched, "waited_ms": 1}}
        if action_type == "click":
            return {"ok": True, "result": {"changed": True}}
        if action_type == "dom_snapshot":
            return {"ok": True, "result": {"elements": []}}
        return {"ok": True, "result": {}}


def _runner(smap, *, dead=None, persist=False):
    return StepRunner(
        "s", "coursera", "coursera",
        bridge=FakeBridge(dead=dead), pace_range=(0, 0), settle_s=0,
        persist_heal=persist, selector_map=smap,
    )


# ── self-confirming map ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_success_promotes_winner_and_stamps_it():
    dead = {"text": "Go To Course", "role": "button", "cite": "draft", "verified": False}
    winner = {"css": "div.resume", "cite": "draft", "verified": False}
    smap = {"steps": {"go": [dead, winner]}}

    res = await _runner(smap, dead=["Go To Course"]).act("go", "click")
    assert res["ok"] is True

    cands = smap["steps"]["go"]
    assert cands[0] is winner                       # promoted to front
    assert cands[0]["verified"] is True
    assert cands[0]["observed"] == date.today().isoformat()
    assert cands[1] is dead and cands[1]["verified"] is False  # draft untouched


@pytest.mark.asyncio
async def test_converged_candidate_causes_no_disk_write(monkeypatch):
    writes = []
    monkeypatch.setattr(driver_runner, "save_selector_map", lambda n, d: writes.append(n))

    winner = {"css": "div.resume", "cite": "live", "verified": True, "observed": "2026-07-03"}
    smap = {"steps": {"go": [winner]}}
    res = await _runner(smap, persist=True).act("go", "click")
    assert res["ok"] is True
    assert writes == []                              # already converged → no write

    fresh = {"css": "div.resume", "cite": "draft", "verified": False}
    smap2 = {"steps": {"go": [{"css": "div.dead"}, fresh]}}
    res2 = await _runner(smap2, dead=["div.dead"], persist=True).act("go", "click")
    assert res2["ok"] is True
    assert writes == ["coursera"]                    # convergence → exactly one write


@pytest.mark.asyncio
async def test_healed_candidate_gets_stamped_on_success():
    smap = {"steps": {"go": [{"css": "div.dead", "verified": False, "cite": "draft"}]}}

    async def fake_heal(sid, step, goal, els):
        return "div.healed"

    runner = StepRunner(
        "s", "coursera", "coursera",
        bridge=FakeBridge(dead=["div.dead"]), heal=fake_heal,
        pace_range=(0, 0), settle_s=0, persist_heal=False, selector_map=smap,
    )
    res = await runner.act("go", "click", goal="the button")
    assert res["ok"] is True and res.get("healed") is True
    front = runner.map["steps"]["go"][0]
    assert front["css"] == "div.healed"
    assert front["verified"] is True                 # proven the moment it worked
    assert front["observed"] == date.today().isoformat()


def test_live_coursera_map_untouched_by_tests():
    """Guard: the shipped map on disk still parses and none of these tests
    wrote to it (persist_heal=False everywhere above except the monkeypatched
    write counter)."""
    p = Path("app/content/distribution/posting_drivers/selector_maps/coursera.json")
    assert p.exists()


# ── study-session flag ───────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_study_state():
    clear_study_activity()
    yield
    clear_study_activity()


def test_mark_and_active_window():
    assert active_study_session() is None
    mark_study_activity(course="Economics Of Ai", item_title="Intro Video", item_type="video")
    s = active_study_session()
    assert s is not None
    assert s["item_title"] == "Intro Video" and s["course"] == "Economics Of Ai"
    assert 0 <= s["age_s"] <= 5


def test_marker_expires_after_window():
    mark_study_activity(item_title="Intro Video")
    study_state._state["ts"] = time.time() - STUDY_ACTIVE_WINDOW_S - 5
    assert active_study_session() is None
    assert build_study_session_block() is None


def test_block_names_item_and_routes_to_study_tools():
    mark_study_activity(course="Economics Of Ai", item_title="Intro Video", item_type="video")
    block = build_study_session_block()
    assert block is not None
    assert block.startswith("[STUDY SESSION]")
    assert "Intro Video" in block
    assert "coursera_next_item" in block
    assert "coursera_read_lesson" in block
    assert "web_click" in block  # explicit don't-improvise instruction


def test_block_absent_when_not_studying():
    assert build_study_session_block() is None


@pytest.mark.asyncio
async def test_handlers_mark_activity_and_enrich_on_success(monkeypatch):
    from app.web_automation import coursera_study_tools as study

    async def fake_resume(course=None, **kw):
        return {"ok": True, "state": "ok", "course": "Economics Of Ai",
                "item_title": "Intro Video", "item_type": "video"}

    monkeypatch.setattr(study.coursera_driver, "resume_course", fake_resume)
    out = await study.coursera_resume_handler({}, None)
    assert out["ok"] is True
    s = active_study_session()
    assert s and s["item_title"] == "Intro Video"


@pytest.mark.asyncio
async def test_failed_tool_still_marks_but_does_not_enrich(monkeypatch):
    from app.web_automation import coursera_study_tools as study

    async def fake_next(**kw):
        return {"ok": False, "state": "logged_out", "message": "log in first"}

    monkeypatch.setattr(study.coursera_driver, "next_item", fake_next)
    out = await study.coursera_next_item_handler({}, None)
    assert out["ok"] is False
    s = active_study_session()
    assert s is not None                # user is still trying to study
    assert s["item_title"] == ""        # but nothing false was recorded


def test_prompt_builders_wires_the_block():
    src = Path("app/llm/routing/prompt_builders.py").read_text(encoding="utf-8")
    assert "build_study_session_block" in src
