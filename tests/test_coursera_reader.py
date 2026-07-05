# FILE: tests/test_coursera_reader.py
# Purpose: Unit tests for the Coursera login health check + vision progress
#          composite, and the desktop-offline timeout mapping (bridge mocked).
# Called-by: pytest
# Depends-on: app.web_automation.coursera_reader, app.web_automation.action_queue
# Last-renovated: 2026-07-01
"""
Coursera reader fix (2026-07-01) — the three failure modes that used to
produce wrong answers must now produce the right message:

  * logged out       -> "needs a fresh login", never the public page as truth
  * desktop offline  -> "desktop browser is offline", never a vague timeout
  * progress reads   -> vision composite, short-circuiting before vision
                        when the session is logged out
"""
from __future__ import annotations

import asyncio

import pytest

from app.web_automation import action_queue, coursera_reader
from app.web_automation.action_queue import DESKTOP_OFFLINE_PREFIX
from app.web_automation.models import ActionStatus


# ── Fakes ────────────────────────────────────────────────────────────

class _FakeSession:
    id = "fake-session-id"
    platform = "coursera"


def _fake_dispatch(script: dict):
    """Return an async _dispatch stand-in serving canned per-action replies."""
    calls = []

    async def dispatch(ref, action_type, payload, timeout_seconds=30.0):
        calls.append(action_type)
        return script.get(action_type, {"ok": True, "result": {}})

    dispatch.calls = calls
    return dispatch


def _wire(monkeypatch, *, opened=None, script=None):
    monkeypatch.setattr(coursera_reader, "_resolve_session", lambda ref: _FakeSession())

    async def ensure_open(session_id, timeout_seconds=10.0):
        return opened if opened is not None else {"ok": True, "result": {}}

    monkeypatch.setattr(coursera_reader.bridge, "ensure_session_open", ensure_open)
    dispatch = _fake_dispatch(script or {})
    monkeypatch.setattr(coursera_reader, "_dispatch", dispatch)
    return dispatch


def _dom(*texts: str) -> dict:
    return {"ok": True, "result": {"elements": [{"text": t} for t in texts]}}


_MY_LEARNING_NAV = {
    "ok": True,
    "result": {"current_url": "https://www.coursera.org/my-learning"},
}
_BOUNCED_NAV = {
    "ok": True,
    "result": {"current_url": "https://www.coursera.org/"},
}


# ── Login state classification ───────────────────────────────────────

@pytest.mark.asyncio
async def test_logged_out_by_bounce_and_markers(monkeypatch):
    _wire(monkeypatch, script={
        "navigate": _BOUNCED_NAV,
        "dom_snapshot": _dom("Join for Free", "Log In", "What do you want to learn?"),
    })
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "logged_out"
    assert result["ok"] is False
    assert result["logged_in"] is False
    assert "fresh login" in result["message"]


@pytest.mark.asyncio
async def test_logged_out_by_bounce_alone(monkeypatch):
    # Redirected off my-learning and no learner nav — logged out even if
    # the marketing page's button text changes wording some day.
    _wire(monkeypatch, script={
        "navigate": _BOUNCED_NAV,
        "dom_snapshot": _dom("Explore careers", "For universities"),
    })
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "logged_out"


@pytest.mark.asyncio
async def test_logged_in_on_my_learning(monkeypatch):
    _wire(monkeypatch, script={
        "navigate": _MY_LEARNING_NAV,
        "dom_snapshot": _dom("My Learning", "The Economics of AI", "Continue Learning"),
    })
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "ok"
    assert result["ok"] is True
    assert result["logged_in"] is True


@pytest.mark.asyncio
async def test_unreadable_dom_falls_back_to_vision(monkeypatch):
    # DOM snapshot failed outright. The old behaviour returned "unknown"
    # and handed verification back to the caller, causing stop/loop cycles
    # (2026-07-03 fix). An unreadable DOM on My Learning must now fall to
    # the vision check and PROCEED on its verdict rather than stalling.
    _wire(monkeypatch, script={
        "navigate": _MY_LEARNING_NAV,
        "dom_snapshot": {"ok": False, "error": "snapshot failed"},
    })
    vision_called = []

    async def fake_vision(input_data, context):
        vision_called.append(True)
        return {"ok": True, "answer": "LOGGED_IN - personal course cards visible"}

    monkeypatch.setattr(coursera_reader, "vision_check_handler", fake_vision)
    result = await coursera_reader.coursera_health_handler({}, None)
    assert vision_called == [True]
    assert result["state"] == "ok"
    assert result["logged_in"] is True


@pytest.mark.asyncio
async def test_unreadable_dom_still_catches_logout_via_vision(monkeypatch):
    # The flip side: a dead DOM must not become a free pass. If the vision
    # model sees the public/marketing page, the check still halts.
    _wire(monkeypatch, script={
        "navigate": _MY_LEARNING_NAV,
        "dom_snapshot": {"ok": False, "error": "snapshot failed"},
    })

    async def fake_vision(input_data, context):
        return {"ok": True, "answer": "LOGGED_OUT"}

    monkeypatch.setattr(coursera_reader, "vision_check_handler", fake_vision)
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "logged_out"
    assert "fresh login" in result["message"]


@pytest.mark.asyncio
async def test_desktop_offline_from_open(monkeypatch):
    _wire(monkeypatch, opened={
        "ok": False,
        "error": f"{DESKTOP_OFFLINE_PREFIX} — the ASTRA desktop app never picked this action up",
    })
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "desktop_offline"
    assert "desktop" in result["message"].lower()


# ── No-double-load behaviour (live incident 2026-07-01: redundant ───
#    navigate raced Coursera's client redirect → ERR_ABORTED wedge)

@pytest.mark.asyncio
async def test_fresh_view_on_my_learning_skips_navigate_and_reload(monkeypatch):
    dispatch = _wire(
        monkeypatch,
        opened={"ok": True, "result": {
            "current_url": "https://www.coursera.org/my-learning",
            "view_was_fresh": True,
        }},
        script={"dom_snapshot": _dom("My Learning", "Continue Learning")},
    )
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "ok"
    assert "navigate" not in dispatch.calls
    assert "reload" not in dispatch.calls


@pytest.mark.asyncio
async def test_old_view_on_my_learning_reloads_instead_of_navigating(monkeypatch):
    dispatch = _wire(
        monkeypatch,
        opened={"ok": True, "result": {
            "current_url": "https://www.coursera.org/my-learning?myLearningTab=IN_PROGRESS",
            "view_was_fresh": False,
        }},
        script={
            "reload": {"ok": True, "result": {}},
            "dom_snapshot": _dom("My Learning", "Continue Learning"),
        },
    )
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "ok"
    assert "reload" in dispatch.calls
    assert "navigate" not in dispatch.calls


@pytest.mark.asyncio
async def test_failed_navigate_is_benign_when_page_landed_anyway(monkeypatch):
    # ERR_ABORTED / slow-fail race: navigate errors but current_state
    # shows we're sitting on My Learning — must classify, not error out.
    dispatch = _wire(
        monkeypatch,
        opened={"ok": True, "result": {
            "current_url": "https://www.coursera.org/learn/economics-of-ai/lecture/x",
            "view_was_fresh": False,
        }},
        script={
            "navigate": {"ok": False, "error": "page did not respond — the desktop browser accepted the action but returned no result within 25s."},
            "current_state": {"ok": True, "result": {
                "url": "https://www.coursera.org/my-learning?myLearningTab=IN_PROGRESS",
            }},
            "dom_snapshot": _dom("My Learning", "Continue Learning"),
        },
    )
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "ok"
    assert "current_state" in dispatch.calls


@pytest.mark.asyncio
async def test_failed_navigate_and_unreadable_state_is_error(monkeypatch):
    _wire(
        monkeypatch,
        opened={"ok": True, "result": {
            "current_url": "https://www.coursera.org/learn/economics-of-ai/lecture/x",
            "view_was_fresh": False,
        }},
        script={
            "navigate": {"ok": False, "error": "executor threw: boom"},
            "current_state": {"ok": False, "error": "executor threw: boom"},
        },
    )
    result = await coursera_reader.coursera_health_handler({}, None)
    assert result["state"] == "error"
    assert "could not reach My Learning" in result["error"]


def test_on_my_learning_is_path_based():
    assert coursera_reader._on_my_learning("https://www.coursera.org/my-learning")
    assert coursera_reader._on_my_learning(
        "https://www.coursera.org/my-learning?myLearningTab=IN_PROGRESS")
    # Login page carrying my-learning only in the redirect param must NOT count.
    assert not coursera_reader._on_my_learning(
        "https://www.coursera.org/login?redirectTo=%2Fmy-learning")
    assert not coursera_reader._on_my_learning("https://www.coursera.org/")
    assert not coursera_reader._on_my_learning("")


# ── Progress composite ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_progress_short_circuits_before_vision_when_logged_out(monkeypatch):
    _wire(monkeypatch, script={
        "navigate": _BOUNCED_NAV,
        "dom_snapshot": _dom("Join for Free"),
    })
    vision_called = []

    async def fake_vision(input_data, context):
        vision_called.append(True)
        return {"ok": True, "answer": "should never run"}

    monkeypatch.setattr(coursera_reader, "vision_check_handler", fake_vision)
    result = await coursera_reader.coursera_progress_handler({}, None)
    assert result["state"] == "logged_out"
    assert vision_called == []          # vision must NOT read a public page
    assert "fresh login" in result["message"]


@pytest.mark.asyncio
async def test_progress_returns_vision_answer_when_logged_in(monkeypatch):
    _wire(monkeypatch, script={
        "navigate": _MY_LEARNING_NAV,
        "dom_snapshot": _dom("My Learning", "Continue Learning"),
    })

    async def fake_vision(input_data, context):
        return {
            "ok": True,
            "answer": "The Economics of AI — module 3 of 6, next: The Teleological Stance",
            "screenshot_path": "C:/shots/coursera.png",
        }

    monkeypatch.setattr(coursera_reader, "vision_check_handler", fake_vision)
    result = await coursera_reader.coursera_progress_handler({}, None)
    assert result["state"] == "ok"
    assert "module 3 of 6" in result["answer"]
    assert result["screenshot_path"] == "C:/shots/coursera.png"


@pytest.mark.asyncio
async def test_progress_trusts_vision_logged_out_verdict(monkeypatch):
    # DOM said 'unknown' (empty snapshot), vision sees the public page.
    _wire(monkeypatch, script={
        "navigate": _MY_LEARNING_NAV,
        "dom_snapshot": _dom(),
    })

    async def fake_vision(input_data, context):
        return {"ok": True, "answer": "LOGGED_OUT", "screenshot_path": ""}

    monkeypatch.setattr(coursera_reader, "vision_check_handler", fake_vision)
    result = await coursera_reader.coursera_progress_handler({}, None)
    assert result["state"] == "logged_out"
    assert "fresh login" in result["message"]


# ── Desktop-offline timeout mapping (action_queue) ───────────────────

@pytest.mark.asyncio
async def test_await_result_pending_dead_maps_to_desktop_offline(monkeypatch):
    loop = asyncio.get_event_loop()
    action_queue._result_futures["act-pending"] = loop.create_future()
    monkeypatch.setattr(
        action_queue, "_mark_timed_out",
        lambda action_id: (ActionStatus.pending, False),
    )
    result = await action_queue.await_result("act-pending", timeout_seconds=0.05)
    assert result["ok"] is False
    assert result["error"].startswith(DESKTOP_OFFLINE_PREFIX)
    assert "desktop app" in result["error"]


@pytest.mark.asyncio
async def test_await_result_pending_alive_maps_to_busy_not_offline(monkeypatch):
    # Live incident 2026-07-01 23:52: a wedged navigate held the serial
    # queue, current_state (5s) expired undelivered, and the user was
    # told the desktop was offline while it was demonstrably running.
    loop = asyncio.get_event_loop()
    action_queue._result_futures["act-busy"] = loop.create_future()
    monkeypatch.setattr(
        action_queue, "_mark_timed_out",
        lambda action_id: (ActionStatus.pending, True),
    )
    result = await action_queue.await_result("act-busy", timeout_seconds=0.05)
    assert result["ok"] is False
    assert result["error"].startswith("browser is busy on a previous action")
    assert DESKTOP_OFFLINE_PREFIX not in result["error"]


@pytest.mark.asyncio
async def test_await_result_in_flight_maps_to_page_stall(monkeypatch):
    loop = asyncio.get_event_loop()
    action_queue._result_futures["act-inflight"] = loop.create_future()
    monkeypatch.setattr(
        action_queue, "_mark_timed_out",
        lambda action_id: (ActionStatus.in_flight, False),
    )
    result = await action_queue.await_result("act-inflight", timeout_seconds=0.05)
    assert result["ok"] is False
    assert result["error"].startswith("page did not respond")
    assert DESKTOP_OFFLINE_PREFIX not in result["error"]


# ── Wiring parity (would have caught the web_vision_check KeyError) ──

def test_registry_parity_handlers_schemas_descriptions():
    from app.web_automation.coursera_reader import (
        COURSERA_DESCRIPTIONS, COURSERA_HANDLERS, COURSERA_SCHEMAS,
    )
    from app.web_automation.tool_handlers import HANDLERS, TOOL_DESCRIPTIONS
    from app.web_automation.tool_schemas import TOOL_SCHEMAS

    all_handlers = {**HANDLERS, **COURSERA_HANDLERS}
    all_schemas = {**TOOL_SCHEMAS, **COURSERA_SCHEMAS}
    all_descriptions = {**TOOL_DESCRIPTIONS, **COURSERA_DESCRIPTIONS}
    assert set(all_handlers) - set(all_schemas) == set()
    assert set(all_handlers) - set(all_descriptions) == set()


def test_chat_layer_exposes_coursera_tools():
    from app.debug.web_tool_definitions import get_web_tools

    names = {t["name"] for t in get_web_tools()}
    assert {"web_coursera_health", "web_coursera_progress"} <= names
