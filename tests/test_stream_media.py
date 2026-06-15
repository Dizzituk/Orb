# FILE: tests/test_stream_media.py
# Purpose: Unit tests for app/media/stream_sites.py (play ladder, toggle/skip
#          routing) + app/media/youtube_search.py (duration parse, resolution).
# Called-by: pytest
# Depends-on: app.media.stream_sites, app.media.youtube_search
# Last-renovated: 2026-06-12
from __future__ import annotations

import pytest

from app.media import stream_sites, youtube_search
from app.tools import stream_tools


class _ScriptedBridge:
    """Answers execute_by_platform per (action, selector) script."""

    def __init__(self, script):
        self.script = script          # list of (matcher, reply) in order OR dict fn
        self.calls = []

    async def __call__(self, platform, action_type, payload, *, timeout_seconds=30.0):
        self.calls.append({"platform": platform, "action": action_type,
                           "payload": payload})
        return self.script(platform, action_type, payload)


@pytest.fixture(autouse=True)
def reset_active():
    stream_sites._active.update(platform=None, window_id=None)
    yield


def _ok_open(monkeypatch):
    async def fake_open(url, display="main", **kwargs):
        return {"ok": True, "result": {"window_id": 9, "display_label": display,
                                       "note": None}}
    monkeypatch.setattr(stream_sites.display_client, "open_on_display", fake_open)


@pytest.mark.asyncio
async def test_play_soundcloud_happy_ladder(monkeypatch):
    _ok_open(monkeypatch)

    def script(platform, action, payload):
        if action == "click" and payload.get("selector") == ".playControl":
            return {"ok": True, "result": {"changed": True}}
        return {"ok": True, "result": {}}

    bridge = _ScriptedBridge(script)
    monkeypatch.setattr(stream_sites.bridge, "execute_by_platform", bridge)

    out = await stream_sites.play_soundcloud("bed screen")
    assert out["ok"] and out["platform"] == "soundcloud"
    assert out["ladder"] == ["play_toggle:ok"]
    actions = [c["action"] for c in bridge.calls]
    assert actions[0] == "wait"  # page settle before any clicking
    assert stream_sites.active_platform() == "soundcloud"


@pytest.mark.asyncio
async def test_play_ladder_walks_to_key_and_login_hint(monkeypatch):
    _ok_open(monkeypatch)

    def script(platform, action, payload):
        return {"ok": False, "error": "selector not found"}  # every step misses

    monkeypatch.setattr(stream_sites.bridge, "execute_by_platform",
                        _ScriptedBridge(script))
    out = await stream_sites.play_soundcloud()
    assert out["ok"] is False
    assert "log in once" in out["error"]
    assert out["ladder"] == ["play_toggle:miss", "first_item:miss", "key:miss"]


@pytest.mark.asyncio
async def test_toggle_routes_to_last_opened(monkeypatch):
    _ok_open(monkeypatch)

    def script(platform, action, payload):
        if action == "click":
            return {"ok": True, "result": {}}
        return {"ok": True, "result": {}}

    bridge = _ScriptedBridge(script)
    monkeypatch.setattr(stream_sites.bridge, "execute_by_platform", bridge)
    await stream_sites.play_mixcloud()
    out = await stream_sites.toggle_playback()
    assert out["ok"] and out["platform"] == "mixcloud"


@pytest.mark.asyncio
async def test_toggle_with_nothing_open_is_honest():
    out = await stream_sites.toggle_playback()
    assert out["ok"] is False and "nothing" in out["error"]


@pytest.mark.asyncio
async def test_skip_uses_centralised_selectors(monkeypatch):
    _ok_open(monkeypatch)
    used = []

    def script(platform, action, payload):
        if action == "click":
            used.append(payload["selector"])
            return {"ok": payload["selector"] == ".skipControl__next", "error": "x"}
        return {"ok": True}

    monkeypatch.setattr(stream_sites.bridge, "execute_by_platform",
                        _ScriptedBridge(script))
    await stream_sites.play_soundcloud()
    used.clear()
    out = await stream_sites.skip("next")
    assert out["ok"] and ".skipControl__next" in used
    assert used[0] in stream_sites.SELECTORS["soundcloud"]["next"]


# ── youtube ────────────────────────────────────────────────────────────────

def test_parse_iso8601_duration():
    assert youtube_search.parse_iso8601_duration("PT12M34S") == 754
    assert youtube_search.parse_iso8601_duration("PT1H2M3S") == 3723
    assert youtube_search.parse_iso8601_duration("PT45S") == 45
    assert youtube_search.parse_iso8601_duration("P1DT1S") == 86401
    assert youtube_search.parse_iso8601_duration("garbage") is None


@pytest.mark.asyncio
async def test_search_caches_results_for_resolution(monkeypatch):
    def fake_blocking(query, max_results):
        return [
            {"video_id": "abc123def45", "title": "Agrivoltaics explained",
             "channel": "C1", "published": "", "duration_seconds": 720},
            {"video_id": "xyz987uvw65", "title": "Solar grazing", "channel": "C2",
             "published": "", "duration_seconds": 300},
        ]

    monkeypatch.setattr(youtube_search, "_search_blocking", fake_blocking)
    rows = await youtube_search.search("agrivoltaics")
    assert rows[0]["index"] == 1
    assert youtube_search.resolve_video(2)["video_id"] == "xyz987uvw65"
    assert youtube_search.resolve_video("solar")["video_id"] == "xyz987uvw65"
    assert youtube_search.resolve_video(None)["video_id"] == "abc123def45"
    assert youtube_search.resolve_video(
        "https://www.youtube.com/watch?v=qqqqqqqqqqq")["video_id"] == "qqqqqqqqqqq"


@pytest.mark.asyncio
async def test_play_video_opens_watch_url_and_notes_active(monkeypatch):
    opened = {}

    async def fake_open(url, display="main", **kwargs):
        opened["url"] = url
        opened["session"] = kwargs.get("session")
        return {"ok": True, "result": {"window_id": 3}}

    async def quiet_cookies(platform):
        return None

    from app.media import display_client
    monkeypatch.setattr(display_client, "open_on_display", fake_open)
    monkeypatch.setattr(stream_sites, "_dismiss_cookies", quiet_cookies)
    youtube_search._recent_videos.clear()
    out = await youtube_search.play_video("https://youtu.be/abcdefghijk", "bed screen")
    assert out["ok"]
    assert opened["url"] == "https://www.youtube.com/watch?v=abcdefghijk"
    assert opened["session"] == "youtube_watch"
    assert stream_sites.active_platform() == "youtube_watch"


@pytest.mark.asyncio
async def test_find_videos_tool_requires_query():
    out = await stream_tools.find_videos_handler({}, None)
    assert out["ok"] is False


def test_stream_tools_register_cleanly():
    from app.tools.registry import list_tools
    stream_tools.register_stream_tools()
    names = {t["name"] for t in list_tools()}
    assert {"play_soundcloud_desktop", "play_mixcloud_desktop",
            "toggle_desktop_playback", "skip_desktop_media",
            "find_videos", "play_video"} <= names
