# FILE: tests/test_display_client.py
# Purpose: Unit tests for app/media/display_client.py + app/tools/display_tools.py
#          (bridge mocked — no desktop needed).
# Called-by: pytest
# Depends-on: app.media.display_client, app.tools.display_tools
# Last-renovated: 2026-06-14
"""
Display manager backend-half tests.

The desktop executes displays_* actions; here we assert the Python side
builds the right payloads, targets the "displays" control session, and the
chat tools validate input + degrade to {ok: False, error} instead of raising.
"""
from __future__ import annotations

import pytest

from app.media import display_client
from app.tools import display_tools


class _BridgeRecorder:
    """Stands in for web_automation.bridge.execute_by_platform."""

    def __init__(self, reply=None):
        self.calls = []
        self.reply = reply or {"ok": True, "result": {}}

    async def __call__(self, platform, action_type, payload, *, timeout_seconds=30.0):
        self.calls.append(
            {"platform": platform, "action": action_type,
             "payload": payload, "timeout": timeout_seconds}
        )
        return self.reply


@pytest.fixture()
def bridge_recorder(monkeypatch):
    rec = _BridgeRecorder()
    monkeypatch.setattr(display_client.bridge, "execute_by_platform", rec)
    return rec


@pytest.mark.asyncio
async def test_list_displays_targets_control_session(bridge_recorder):
    out = await display_client.list_displays()
    assert out["ok"] is True
    call = bridge_recorder.calls[0]
    assert call["platform"] == "displays"
    assert call["action"] == "displays_list"


@pytest.mark.asyncio
async def test_open_on_display_builds_payload(bridge_recorder):
    await display_client.open_on_display(
        "https://example.com", "bed screen", fullscreen=False
    )
    call = bridge_recorder.calls[0]
    assert call["action"] == "displays_open"
    assert call["payload"] == {
        "url": "https://example.com",
        "display": "bed screen",
        "fullscreen": False,
        "cdp": False,
    }


@pytest.mark.asyncio
async def test_open_on_display_resolves_session_for_cdp(bridge_recorder, monkeypatch):
    monkeypatch.setattr(
        display_client, "_resolve_session_id", lambda session: "sess-1234"
    )
    await display_client.open_on_display(
        "https://soundcloud.com/feed", "screen 2", cdp=True, session="soundcloud"
    )
    payload = bridge_recorder.calls[0]["payload"]
    assert payload["cdp"] is True
    assert payload["session_id"] == "sess-1234"


@pytest.mark.asyncio
async def test_open_on_display_cdp_without_known_session_still_opens(bridge_recorder, monkeypatch):
    monkeypatch.setattr(display_client, "_resolve_session_id", lambda session: None)
    await display_client.open_on_display(
        "https://example.com", cdp=True, session="nonexistent"
    )
    payload = bridge_recorder.calls[0]["payload"]
    assert "session_id" not in payload  # opens uncontrolled rather than failing


@pytest.mark.asyncio
async def test_close_window_variants(bridge_recorder):
    await display_client.close_window(window_id=7)
    await display_client.close_window(display="bed screen")
    await display_client.close_window()
    payloads = [c["payload"] for c in bridge_recorder.calls]
    assert payloads == [{"window_id": 7}, {"display": "bed screen"}, {}]


@pytest.mark.asyncio
async def test_move_and_save_alias_payloads(bridge_recorder):
    await display_client.move_window(3, "main")
    await display_client.save_alias("Bed Screen", 2)
    move, save = bridge_recorder.calls
    assert move["action"] == "displays_move"
    assert move["payload"] == {"window_id": 3, "display": "main"}
    assert save["action"] == "displays_save_alias"
    assert save["payload"] == {"alias": "Bed Screen", "display_index": 2}


@pytest.mark.asyncio
async def test_save_alias_by_display_id(bridge_recorder):
    await display_client.save_alias("watch screen", display_id=42)
    call = bridge_recorder.calls[0]
    assert call["action"] == "displays_save_alias"
    assert call["payload"] == {"alias": "watch screen", "display_id": 42}


@pytest.mark.asyncio
async def test_calibrate_and_close_payloads(bridge_recorder):
    await display_client.calibrate()
    await display_client.calibrate(duration_ms=30000)
    await display_client.close_calibration()
    actions = [c["action"] for c in bridge_recorder.calls]
    assert actions == [
        "displays_calibrate", "displays_calibrate", "displays_close_calibration",
    ]
    assert bridge_recorder.calls[0]["payload"] == {}
    assert bridge_recorder.calls[1]["payload"] == {"duration_ms": 30000}


# ── Tool-handler validation (no bridge round-trip on bad input) ────────────

@pytest.mark.asyncio
async def test_open_tool_requires_url():
    out = await display_tools.open_on_display_handler({}, None)
    assert out["ok"] is False and "url" in out["error"]


@pytest.mark.asyncio
async def test_move_tool_requires_both_args():
    out = await display_tools.move_display_window_handler({"window_id": 1}, None)
    assert out["ok"] is False


@pytest.mark.asyncio
async def test_save_alias_tool_requires_both_args():
    out = await display_tools.save_display_alias_handler({"alias": "x"}, None)
    assert out["ok"] is False


@pytest.mark.asyncio
async def test_save_alias_tool_accepts_display_id(monkeypatch):
    captured = {}

    async def fake_save(alias, display_index=None, *, display_id=None):
        captured.update(alias=alias, display_index=display_index, display_id=display_id)
        return {"ok": True, "result": {"saved": True}}

    monkeypatch.setattr(display_client, "save_alias", fake_save)
    out = await display_tools.save_display_alias_handler(
        {"alias": "watch screen", "display_id": 7}, None
    )
    assert out["ok"] is True
    assert captured == {"alias": "watch screen", "display_index": None, "display_id": 7}


@pytest.mark.asyncio
async def test_tool_handler_passes_through_unresolved_alias(monkeypatch):
    async def fake_open(url, display="main", **kwargs):
        return {
            "ok": False,
            "error": 'I don\'t know a screen called "garage".',
            "error_code": "UNRESOLVED_ALIAS",
            "known_aliases": ["bed screen", "screen two"],
        }

    monkeypatch.setattr(display_client, "open_on_display", fake_open)
    out = await display_tools.open_on_display_handler(
        {"url": "https://x.com", "display": "garage"}, None
    )
    assert out["error_code"] == "UNRESOLVED_ALIAS"
    assert "bed screen" in out["known_aliases"]


def test_display_tools_register_cleanly():
    """Registration is idempotent and names land in the global registry."""
    from app.tools.registry import list_tools
    display_tools.register_display_tools()
    names = {t["name"] for t in list_tools()}
    assert {"list_displays", "open_on_display", "close_display_window",
            "move_display_window", "list_display_windows",
            "save_display_alias", "calibrate_displays",
            "end_display_calibration"} <= names
