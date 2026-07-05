# FILE: tests/test_agentic_routing_tooltrust.py
# Purpose: 2026-07-03 incident fixes — agentic tab no longer frontier/pinned,
#          claude-* tool-trust prefix guard, Coursera progress DOM fallback.
# Called-by: pytest
# Depends-on: app.llm.routing.chat_model_selection, app.llm.chat_tool_registry, app.web_automation.coursera_reader
# Last-renovated: 2026-07-03
"""
Live incident 2026-07-03 00:17 ("resume course" answered by a toolless
claude-fable-5): a browser-tab turn routed to FRONTIER via the agentic-tab
branch, PINNED it onto the project, and the frontier model wasn't in the
Anthropic tool-trust list so every tool was silently stripped. These tests
pin the fixed behaviour:

  1. Agentic tab -> AGENTIC_CONTEXT/REASONING tier, momentum only, no pin.
  2. 'agentic_tab' rows never latch (in-memory or via history restore).
  3. is_tool_eligible trusts any claude-* (mirrors the gpt-5 prefix rule).
  4. web_coursera_progress degrades to a DOM course list when the visual
     read is unavailable, instead of dead-ending.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.llm.chat_tool_registry import is_tool_eligible
from app.llm.routing import chat_model_selection as cms


class _Req:
    def __init__(self, message, project_id, ui_context=None):
        self.message = message
        self.project_id = project_id
        self.provider = None
        self.model = None
        self.ui_context = ui_context


@pytest.fixture(autouse=True)
def _clean_sticky_state():
    cms._session_model_cache.clear()
    cms._sticky_source.clear()
    yield
    cms._session_model_cache.clear()
    cms._sticky_source.clear()


def _wire(monkeypatch, role_log):
    def fake_get_role_model(role, *fallbacks):
        role_log.append((role, fallbacks))
        return ("openai", "gpt-5.4")

    monkeypatch.setattr(cms, "get_role_model", fake_get_role_model)
    monkeypatch.setattr(cms, "ensure_provider_available", lambda p, m: (p, m))


# ── 1+2 · agentic tab routing ────────────────────────────────────────

def test_agentic_tab_resolves_reasoning_fallback_not_frontier(monkeypatch):
    roles = []
    _wire(monkeypatch, roles)
    req = _Req("carry on with the course please", 990001, ui_context={"job": "coursera"})
    provider, model, extras = cms.select_chat_model(req, db=None)

    assert roles == [("AGENTIC_CONTEXT", ("REASONING",))], roles  # never FRONTIER
    assert (provider, model) == ("openai", "gpt-5.4")
    assert extras["model_source"] == "agentic_tab"


def test_agentic_tab_does_not_pin_the_session(monkeypatch):
    _wire(monkeypatch, [])
    req = _Req("carry on with the course please", 990002, ui_context={"job_type": "website"})
    cms.select_chat_model(req, db=None)

    # No sticky latch: the next turn re-routes normally.
    assert cms.get_sticky_model(990002) is None
    assert cms.is_explicit_sticky(990002) is False


def test_agentic_tab_source_is_not_pinned():
    assert "agentic_tab" not in cms._PINNED_SOURCES
    cms.set_sticky_model(990003, "anthropic", "claude-fable-5", source="agentic_tab")
    assert cms.is_explicit_sticky(990003) is False  # would decay, never short-circuit


def test_history_restore_skips_agentic_tab_rows(monkeypatch):
    """The poisoned row class (model_source='agentic_tab', e.g. live row 5203)
    must NOT re-latch the frontier model after a restart."""
    monkeypatch.setattr(cms, "is_small_model", lambda m: False)

    def fake_list(db, project_id, limit=10):
        return [SimpleNamespace(role="assistant", model="claude-fable-5",
                                provider="anthropic", model_source="agentic_tab")]

    monkeypatch.setattr(cms.memory_service, "list_messages", fake_list)
    assert cms.infer_sticky_from_history(990004, db=None) is None


def test_history_restore_still_honours_real_pins(monkeypatch):
    monkeypatch.setattr(cms, "is_small_model", lambda m: False)

    def fake_list(db, project_id, limit=10):
        return [SimpleNamespace(role="assistant", model="claude-fable-5",
                                provider="anthropic", model_source="frontend_override")]

    monkeypatch.setattr(cms.memory_service, "list_messages", fake_list)
    assert cms.infer_sticky_from_history(990005, db=None) == ("anthropic", "claude-fable-5")


def test_non_agentic_ui_context_untouched(monkeypatch):
    roles = []
    _wire(monkeypatch, roles)
    req = _Req("carry on with the course please", 990006, ui_context={"job": "health"})
    cms.select_chat_model(req, db=None)
    # Fell through the agentic branch entirely (roles logged only by later
    # tiers, none of which is AGENTIC_CONTEXT-first).
    assert all(r[0] != "AGENTIC_CONTEXT" for r in roles)


# ── 3 · tool-trust prefix guard ──────────────────────────────────────

def test_claude_prefix_is_tool_eligible_even_off_list(monkeypatch):
    monkeypatch.setenv("ASTRA_TOOL_TRUSTED_MODELS_ANTHROPIC", "claude-opus-4-8,claude-sonnet-5")
    assert is_tool_eligible("anthropic", "claude-fable-5") is True   # the live gap
    assert is_tool_eligible("anthropic", "claude-opus-4-8") is True  # list still works
    assert is_tool_eligible("anthropic", "grok-9") is False          # unknown stays out


def test_claude_prefix_survives_empty_env(monkeypatch):
    monkeypatch.setenv("ASTRA_TOOL_TRUSTED_MODELS_ANTHROPIC", "")
    assert is_tool_eligible("anthropic", "claude-fable-5") is True
    assert is_tool_eligible("anthropic", "CLAUDE-FABLE-5") is True  # case-insensitive
    assert is_tool_eligible("anthropic", "fable-5") is False


# ── 4 · Coursera progress DOM fallback ───────────────────────────────

class _FakeSession:
    id = "fake-session-id"
    platform = "coursera"


def _wire_reader(monkeypatch, elements, vision_error):
    from app.web_automation import coursera_reader as cr

    monkeypatch.setattr(cr, "_resolve_session", lambda ref: _FakeSession())

    async def ensure_open(session_id, timeout_seconds=10.0):
        return {"ok": True, "result": {"current_url": cr.MY_LEARNING_URL,
                                       "view_was_fresh": True}}

    monkeypatch.setattr(cr.bridge, "ensure_session_open", ensure_open)

    async def dispatch(ref, action_type, payload, timeout_seconds=30.0):
        if action_type == "dom_snapshot":
            return {"ok": True, "result": {"elements": elements}}
        return {"ok": True, "result": {}}

    monkeypatch.setattr(cr, "_dispatch", dispatch)

    async def dead_vision(input_data, context):
        return {"ok": False, "error": vision_error}

    monkeypatch.setattr(cr, "vision_check_handler", dead_vision)
    return cr


@pytest.mark.asyncio
async def test_progress_falls_back_to_dom_titles_when_screenshot_dead(monkeypatch):
    elements = [
        {"text": "My Learning"},
        {"text": "Continue Learning"},
        {"text": "The Economics of AI", "href": "/learn/economics-of-ai/home/module/2"},
        {"text": "AI For Everyone", "href": "/learn/ai-for-everyone"},
    ]
    cr = _wire_reader(monkeypatch, elements,
                      "screenshot failed: screenshot came back empty — view not visible")
    out = await cr.coursera_progress_handler({}, None)
    assert out["state"] == "ok"
    assert "The Economics of AI" in out["answer"]
    assert "AI For Everyone" in out["answer"]
    assert "Browser tab" in out["message"]  # honest about the missing ticks


@pytest.mark.asyncio
async def test_progress_fallback_stays_error_when_dom_has_no_courses(monkeypatch):
    elements = [{"text": "My Learning"}, {"text": "Continue Learning"}]
    cr = _wire_reader(monkeypatch, elements, "screenshot returned no image data")
    out = await cr.coursera_progress_handler({}, None)
    assert out["state"] == "error"
    assert "no image data" in out["error"]


@pytest.mark.asyncio
async def test_progress_offline_vision_still_maps_to_desktop_offline(monkeypatch):
    from app.web_automation.action_queue import DESKTOP_OFFLINE_PREFIX

    elements = [
        {"text": "My Learning"},
        {"text": "The Economics of AI", "href": "/learn/economics-of-ai"},
    ]
    cr = _wire_reader(monkeypatch, elements,
                      f"{DESKTOP_OFFLINE_PREFIX} — never picked up")
    out = await cr.coursera_progress_handler({}, None)
    assert out["state"] == "desktop_offline"
    assert out["message"] == cr.DESKTOP_OFFLINE_MESSAGE
