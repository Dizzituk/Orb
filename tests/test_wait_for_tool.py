# FILE: tests/test_wait_for_tool.py
# Purpose: wait_for action constructor + backend tool handler (Job 1).
# Called-by: pytest
# Depends-on: app.web_automation.action_types, app.web_automation.tool_handlers
# Last-renovated: 2026-07-02
import pytest

from app.web_automation import action_types, tool_handlers


def test_wait_for_constructor_shapes_payload():
    t, p = action_types.wait_for("div.x", state="gone", timeout_ms=5000, poll_ms=300)
    assert t == "wait_for"
    assert p["selector"] == "div.x" and p["state"] == "gone"
    assert p["timeout_ms"] == 5000 and p["poll_ms"] == 300
    assert "wait_for" in action_types.ALL_ACTIONS


def test_wait_for_constructor_text_and_url():
    t, p = action_types.wait_for(text="Published", url_pattern="reel")
    assert p["text"] == "Published" and p["url_pattern"] == "reel"
    assert "selector" not in p


def test_wait_for_requires_a_condition():
    with pytest.raises(ValueError):
        action_types.wait_for()


@pytest.mark.asyncio
async def test_handler_matched(monkeypatch):
    captured = {}

    async def fake_dispatch(ref, at, payload, timeout_seconds=30.0):
        captured["at"] = at
        captured["payload"] = payload
        return {"ok": True, "result": {"matched": True, "timeout": False, "waited_ms": 120}}

    monkeypatch.setattr(tool_handlers, "_dispatch", fake_dispatch)
    out = await tool_handlers.wait_for_handler(
        {"session": "meta_business", "selector": "div.x", "state": "visible", "timeout_ms": 8000}, None
    )
    assert out["ok"] and out["matched"] and not out["timeout"] and out["waited_ms"] == 120
    assert captured["at"] == "wait_for"
    assert captured["payload"]["selector"] == "div.x" and captured["payload"]["timeout_ms"] == 8000


@pytest.mark.asyncio
async def test_handler_timeout_is_not_an_error(monkeypatch):
    async def fake_dispatch(ref, at, payload, timeout_seconds=30.0):
        return {"ok": True, "result": {"matched": False, "timeout": True, "waited_ms": 15000}}

    monkeypatch.setattr(tool_handlers, "_dispatch", fake_dispatch)
    out = await tool_handlers.wait_for_handler({"session": "x", "text": "hello"}, None)
    assert out["ok"] is True and out["matched"] is False and out["timeout"] is True


@pytest.mark.asyncio
async def test_handler_requires_condition():
    out = await tool_handlers.wait_for_handler({"session": "x"}, None)
    assert out["ok"] is False and "at least one" in out["error"]
