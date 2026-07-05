# FILE: tests/test_eyes_play.py
# Purpose: live11 — the eyes drive the built app (whitelisted keys, own PID only) and evidence records the session.
# Called-by: pytest
# Depends-on: app.pipeline_v2.verifier_agent.eyes_judge, app.pipeline_v2.verifier_agent.host_perception
# Last-renovated: 2026-07-05
"""Taz's trust call (2026-07-05): the final checkout must literally play the
game — keys in, screenshots between, judge correlates. These tests pin the
injection surface (whitelist + own-PID-only) and the play choreography."""

import asyncio

import pytest

from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.verifier_agent import eyes_judge
from app.pipeline_v2.verifier_agent import host_perception as hp
from app.pipeline_v2.verifier_agent.host_perception import build_sendkeys_script


# ---------------------------------------------------------------------------
# build_sendkeys_script — the entire injection surface
# ---------------------------------------------------------------------------

def test_whitelist_filters_dangerous_tokens():
    script = build_sendkeys_script(1234, ["{LEFT}", "%{F4}", "{DELETE}", "^c", "{UP}"])
    assert script is not None
    assert "'{LEFT}'" in script and "'{UP}'" in script
    assert "%{F4}" not in script and "{DELETE}" not in script and "^c" not in script


def test_all_bad_tokens_or_bad_pid_returns_none():
    assert build_sendkeys_script(1234, ["%{F4}", "^{ESC}"]) is None
    assert build_sendkeys_script(0, ["{LEFT}"]) is None
    assert build_sendkeys_script(-5, ["{LEFT}"]) is None


def test_pid_and_delay_floor_embedded():
    script = build_sendkeys_script(9876, ["{DOWN}"], delay_ms=5)
    assert "AppActivate(9876)" in script
    assert "Milliseconds 30" in script  # floor, not 5


# ---------------------------------------------------------------------------
# run_eyes play choreography
# ---------------------------------------------------------------------------

class _FakeProc:
    def __init__(self, alive=True):
        self._alive = alive

    def poll(self):
        return None if self._alive else 0


def _profile(tmp_path):
    return BuildTargetProfile(
        project_id="play-fixture", project_name="Play Fixture",
        project_root=str(tmp_path), language="python", build_system="pip",
        framework="generic", source_root="src", package_name="",
        architecture_pattern="flat",
    )


@pytest.fixture
def rig(monkeypatch, tmp_path):
    """Fake perception rig: records every injection, screenshots are stubs."""
    calls = {"send_keys": [], "stopped": []}

    async def fake_launch(root, cmd=None, settle_seconds=4.0):
        return {"launched": True, "running": True, "pid": 4242,
                "returncode": None, "stdout": "", "stderr": "",
                "_proc": _FakeProc(alive=True)}

    async def fake_send_keys(pid, keys, delay_ms=120):
        calls["send_keys"].append((pid, list(keys)))
        return True

    async def fake_capture(evidence_dir, label="shot", rect=None):
        n = len(calls.get("shots", []))
        calls.setdefault("shots", []).append(n)
        return f"{evidence_dir}\\fake_{n}.png"

    async def fake_titles():
        return ["Tazza's Tetris"]

    def fake_stop(info):
        calls["stopped"].append(info.get("pid"))

    async def fake_wait_ready(info, timeout_s=None, poll_s=1.0, probe=None):
        calls.setdefault("waited", []).append(info.get("pid"))
        return {"ready": True, "exited": False, "returncode": None,
                "waited_s": 1.2, "window_title": "Tazza's Tetris"}

    async def fake_find(pid, foreground=False):
        return {"found": True, "hwnd": 7, "rect": (0, 0, 640, 720), "title": "Tazza's Tetris"}

    monkeypatch.setattr(hp, "launch_app", fake_launch)
    monkeypatch.setattr(hp, "send_keys", fake_send_keys)
    monkeypatch.setattr(hp, "capture_screen", fake_capture)
    monkeypatch.setattr(hp, "window_titles", fake_titles)
    monkeypatch.setattr(hp, "stop_app", fake_stop)
    monkeypatch.setattr(hp, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(hp, "find_pid_window", fake_find)

    async def fake_describe(path, hint):
        return f"(vision stub) {path}"
    monkeypatch.setattr(eyes_judge, "_describe_screenshot", fake_describe)
    return calls


def test_eyes_play_the_app_when_enabled(monkeypatch, tmp_path, rig):
    monkeypatch.setenv("ASTRA_EYES_INPUT", "1")
    ev = asyncio.run(eyes_judge.run_eyes(_profile(tmp_path), "tetris", str(tmp_path)))

    # 4 bursts injected, all to the eyes' own PID, arrows only
    assert len(rig["send_keys"]) == 4
    assert all(pid == 4242 for pid, _ in rig["send_keys"])
    assert rig["send_keys"][0][1] == ["{LEFT}", "{LEFT}", "{LEFT}"]
    assert rig["send_keys"][-1][1] == ["{DOWN}"] * 6
    # baseline + one shot after each burst
    assert len(ev.screenshots) == 5
    assert ev.input_driven is True
    assert len(ev.input_log) == 4
    assert "INPUT DRIVEN BY EYES" in ev.render()
    # live13: boot signal awaited and recorded before play
    assert rig["waited"] == [4242]
    assert ev.boot_ready is True
    assert "BOOT SIGNAL: window 'Tazza's Tetris'" in ev.render()
    # app always stopped afterwards
    assert rig["stopped"] == [4242]


def test_kill_switch_disables_play(monkeypatch, tmp_path, rig):
    monkeypatch.setenv("ASTRA_EYES_INPUT", "0")
    monkeypatch.setenv("ASTRA_EYES_SHOTS", "2")
    ev = asyncio.run(eyes_judge.run_eyes(_profile(tmp_path), "tetris", str(tmp_path)))

    assert rig["send_keys"] == []
    assert len(ev.screenshots) == 2
    assert ev.input_driven is False
    assert ev.input_log == []


def test_dead_app_never_gets_keys(monkeypatch, tmp_path, rig):
    monkeypatch.setenv("ASTRA_EYES_INPUT", "1")
    monkeypatch.setenv("ASTRA_EYES_SHOTS", "1")

    async def dead_launch(root, cmd=None, settle_seconds=4.0):
        return {"launched": True, "running": False, "pid": 4242,
                "returncode": 1, "stdout": "", "stderr": "boom",
                "_proc": _FakeProc(alive=False)}
    monkeypatch.setattr(hp, "launch_app", dead_launch)

    ev = asyncio.run(eyes_judge.run_eyes(_profile(tmp_path), "tetris", str(tmp_path)))
    assert rig["send_keys"] == []
    assert ev.still_running is False
