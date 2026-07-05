# FILE: tests/test_posting_drivers.py
# Purpose: posting driver_runner gating/retry/self-heal + meta_driver flow (Jobs 3, 5).
# Called-by: pytest
# Depends-on: app.content.distribution.posting_drivers.*
# Last-renovated: 2026-07-02
import pytest

from app.content.distribution.posting_drivers import self_heal
from app.content.distribution.posting_drivers.driver_runner import (
    StepRunner, find_in_snapshot,
)
from app.content.distribution.posting_drivers import meta_driver


class FakeBridge:
    """Scripts responses by action_type. `dead` substrings make wait_for miss."""

    def __init__(self, dead=None, changed=True, elements=None):
        self.dead = dead or []
        self.changed = changed
        self.elements = elements if elements is not None else []
        self.calls = []

    async def ensure_session_open(self, sid, timeout_seconds=20.0):
        return {"ok": True}

    async def execute_action(self, sid, action_type, payload=None, timeout_seconds=None):
        payload = payload or {}
        self.calls.append((action_type, payload))
        if action_type == "wait_for":
            key = payload.get("selector") or payload.get("text") or ""
            matched = not any(d in key for d in self.dead)
            return {"ok": True, "result": {"matched": matched, "timeout": not matched, "waited_ms": 5}}
        if action_type == "click":
            return {"ok": True, "result": {"changed": self.changed}}
        if action_type == "type":
            return {"ok": True, "result": {}}
        if action_type == "upload_file":
            return {"ok": True, "result": {"attached": payload.get("file_path")}}
        if action_type == "dom_snapshot":
            return {"ok": True, "result": {"elements": self.elements}}
        if action_type == "screenshot":
            return {"ok": True, "result": {"image_base64": ""}}
        if action_type == "current_state":
            return {"ok": True, "result": {"url": "https://business.facebook.com/x", "title": "t"}}
        return {"ok": True, "result": {}}


# ── pure helpers ─────────────────────────────────────────────────────

def test_find_in_snapshot_prefers_largest():
    els = [
        {"tag": "div", "role": "button", "text": "Create reel", "x": 10, "y": 10, "w": 20, "h": 10},
        {"tag": "div", "role": "button", "text": "Create reel now", "x": 300, "y": 300, "w": 184, "h": 36},
    ]
    c = find_in_snapshot(els, "create reel", "button")
    assert c == {"x": 392, "y": 318}
    assert find_in_snapshot(els, "nonexistent") is None


def test_parse_selector_variants():
    assert self_heal._parse_selector("`div[aria-label=\"Publish\"]`") == 'div[aria-label="Publish"]'
    assert self_heal._parse_selector("selector: input[type='file']") == "input[type='file']"
    assert self_heal._parse_selector("NONE") is None
    assert self_heal._parse_selector("this is clearly a prose sentence not a selector at all") is None


# ── driver_runner ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_act_happy_path_css_click():
    smap = {"steps": {"go": [{"css": "div.ok"}]}}
    runner = StepRunner("s", "meta_business", "meta_business",
                        bridge=FakeBridge(), pace_range=(0, 0),
                        persist_heal=False, selector_map=smap)
    res = await runner.act("go", "click")
    assert res["ok"] and res["candidate"] == "css:div.ok"


@pytest.mark.asyncio
async def test_act_click_no_change_fails():
    smap = {"steps": {"go": [{"css": "div.ok"}]}}
    runner = StepRunner("s", "meta_business", "meta_business",
                        bridge=FakeBridge(changed=False), pace_range=(0, 0),
                        settle_s=0, persist_heal=False, selector_map=smap)
    res = await runner.act("go", "click")
    assert res["ok"] is False


@pytest.mark.asyncio
async def test_self_heal_repairs_and_prepends():
    smap = {"steps": {"go": [{"css": "div.dead"}, {"text": "DeadText", "role": "button"}]}}
    bridge = FakeBridge(dead=["div.dead", "DeadText"])

    async def fake_heal(sid, step, goal, els):
        return "div.healed"

    runner = StepRunner("s", "meta_business", "meta_business",
                        bridge=bridge, heal=fake_heal, pace_range=(0, 0),
                        settle_s=0, persist_heal=False, selector_map=smap)
    res = await runner.act("go", "click", goal="the button")
    assert res["ok"] and res.get("healed") is True
    # map now has the healed css as the first candidate
    assert runner.map["steps"]["go"][0]["css"] == "div.healed"


@pytest.mark.asyncio
async def test_no_heal_returns_failure():
    smap = {"steps": {"go": [{"css": "div.dead"}]}}
    runner = StepRunner("s", "meta_business", "meta_business",
                        bridge=FakeBridge(dead=["div.dead"]), pace_range=(0, 0),
                        settle_s=0, persist_heal=False, selector_map=smap)
    res = await runner.act("go", "click")
    assert res["ok"] is False and res["step"] == "go"


@pytest.mark.asyncio
async def test_relocate_uses_injected_vision(monkeypatch):
    async def fake_vision(b64, goal, tree):
        return "div[aria-label='Publish']"

    css = await self_heal.relocate(
        "s", "publish_button", "the publish button", [{"tag": "div"}],
        bridge=FakeBridge(), vision_fn=fake_vision,
    )
    assert css == "div[aria-label='Publish']"


# ── meta_driver ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_post_reel_happy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_driver, "AUDIT_ROOT", tmp_path / "audit")
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00\x00")

    async def no_vision(*a, **k):
        return None

    res = await meta_driver.post_reel(
        str(mp4), "caption #astra",
        bridge=FakeBridge(), session_id="s1", vision_fn=no_vision,
        pace_range=(0, 0), persist_heal=False,
    )
    assert res.ok is True and res.platform == "meta_business"
    assert res.audit_dir and res.steps


@pytest.mark.asyncio
async def test_post_reel_fails_when_composer_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(meta_driver, "AUDIT_ROOT", tmp_path / "audit")
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00\x00")

    # Kill the reel composer-open candidates; home_ready still matches.
    bridge = FakeBridge(dead=["Create reel"])

    async def no_vision(*a, **k):
        return None

    res = await meta_driver.post_reel(
        str(mp4), "cap",
        bridge=bridge, session_id="s1", vision_fn=no_vision,
        pace_range=(0, 0), persist_heal=False,
    )
    assert res.ok is False
    assert res.failed_step == "reel_composer_open"
    assert res.audit_dir  # audit artifacts written on failure


@pytest.mark.asyncio
async def test_post_image_missing_file():
    res = await meta_driver.post_image("Z:/nope/missing.png", "cap", session_id="s1")
    assert res.ok is False and res.failed_step == "preflight"
