# FILE: tests/test_window_capture.py
# Purpose: live21 — the eyes find + foreground + capture the GAME window rect, not the user's full screen.
# Called-by: pytest
# Depends-on: app.pipeline_v2.verifier_agent.host_perception, app.pipeline_v2.verifier_agent.eyes_judge
# Last-renovated: 2026-07-05
"""Run #10 built a clean game (boot probe GREEN, app alive) but FAILED because
the eyes' full-screen screenshot captured the ASTRA desktop app the user was
working in — not the Tetris window. The eyes now locate the launched PID's
window via Win32 EnumWindows, foreground it, and capture ITS rectangle."""

import asyncio

from app.pipeline_v2.verifier_agent import host_perception as hp
from app.pipeline_v2.verifier_agent import eyes_judge
from app.pipeline_v2.verifier_agent.host_perception import parse_window_line
from app.pipeline_v2.build_targets import BuildTargetProfile


# ---------------------------------------------------------------------------
# parse_window_line — the Win32 output seam
# ---------------------------------------------------------------------------

def test_parse_valid_window_line():
    got = parse_window_line("12345|100,80,740,800|Tazza's Tetris")
    assert got["found"] is True
    assert got["hwnd"] == 12345
    assert got["rect"] == (100, 80, 740, 800)
    assert got["title"] == "Tazza's Tetris"


def test_parse_borderless_empty_title():
    got = parse_window_line("999|0,0,640,720|")
    assert got["found"] is True
    assert got["rect"] == (0, 0, 640, 720)
    assert got["title"] == ""


def test_parse_no_window():
    for raw in ("", "   ", "0|1,2,3,4|x", "garbage", "12|1,2,3|x"):
        assert parse_window_line(raw)["found"] is False


# ---------------------------------------------------------------------------
# capture_screen — rect vs full-screen branch (script selection, no real grab)
# ---------------------------------------------------------------------------

def test_capture_uses_rect_script_when_rect_given(monkeypatch, tmp_path):
    seen = {}

    class _Proc:
        async def communicate(self):
            return (b"SCREENSHOT_OK 640x720", b"")

    async def fake_exec(*args, **kwargs):
        seen["script"] = args[-1]
        # emulate the file being written
        import re
        m = re.search(r"Save\('([^']+)'\)", args[-1])
        if m:
            open(m.group(1), "wb").write(b"\x89PNG")
        return _Proc()

    monkeypatch.setattr(hp.asyncio, "create_subprocess_exec", fake_exec)
    out = asyncio.run(hp.capture_screen(str(tmp_path), label="eyes", rect=(10, 20, 650, 740)))
    assert out is not None
    assert "$l=10" in seen["script"] and "$t=20" in seen["script"]  # rect capture
    assert "PrimaryScreen" not in seen["script"]


def test_capture_full_screen_when_no_rect(monkeypatch, tmp_path):
    seen = {}

    class _Proc:
        async def communicate(self):
            return (b"SCREENSHOT_OK 1920x1080", b"")

    async def fake_exec(*args, **kwargs):
        seen["script"] = args[-1]
        import re
        m = re.search(r"Save\('([^']+)'\)", args[-1])
        if m:
            open(m.group(1), "wb").write(b"\x89PNG")
        return _Proc()

    monkeypatch.setattr(hp.asyncio, "create_subprocess_exec", fake_exec)
    out = asyncio.run(hp.capture_screen(str(tmp_path), label="eyes"))
    assert out is not None
    assert "PrimaryScreen" in seen["script"]


# ---------------------------------------------------------------------------
# run_eyes — foregrounds the window and captures its rect
# ---------------------------------------------------------------------------

def _profile(tmp_path):
    return BuildTargetProfile(
        project_id="t", project_name="T", project_root=str(tmp_path),
        language="python", build_system="pip", framework="generic",
        source_root="src", package_name="", architecture_pattern="flat",
    )


class _AliveProc:
    def poll(self):
        return None


def test_eyes_capture_targets_the_game_rect(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_EYES_INPUT", "1")
    calls = {"rects": [], "foreground": []}

    async def fake_launch(root, cmd=None, settle_seconds=4.0):
        return {"launched": True, "running": True, "pid": 5150,
                "returncode": None, "stdout": "", "stderr": "", "_proc": _AliveProc()}

    async def fake_wait_ready(info, timeout_s=None, poll_s=1.0, probe=None):
        return {"ready": True, "exited": False, "returncode": None,
                "waited_s": 1.0, "window_title": ""}  # borderless: empty title

    async def fake_find(pid, foreground=False):
        calls["foreground"].append(foreground)
        return {"found": True, "hwnd": 42, "rect": (100, 80, 740, 800), "title": ""}

    async def fake_send_keys(pid, keys, delay_ms=120):
        return True

    async def fake_capture(evidence_dir, label="shot", rect=None):
        calls["rects"].append(rect)
        return f"{evidence_dir}/x_{len(calls['rects'])}.png"

    async def fake_titles():
        return []

    async def fake_describe(path, hint):
        return "(vision) game visible"

    monkeypatch.setattr(hp, "launch_app", fake_launch)
    monkeypatch.setattr(hp, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(hp, "find_pid_window", fake_find)
    monkeypatch.setattr(hp, "send_keys", fake_send_keys)
    monkeypatch.setattr(hp, "capture_screen", fake_capture)
    monkeypatch.setattr(hp, "window_titles", fake_titles)
    monkeypatch.setattr(hp, "stop_app", lambda info: None)
    monkeypatch.setattr(eyes_judge, "_describe_screenshot", fake_describe)

    ev = asyncio.run(eyes_judge.run_eyes(_profile(tmp_path), "tetris", str(tmp_path)))

    assert ev.boot_ready is True
    assert True in calls["foreground"], "the game window must be foregrounded"
    # every screenshot targeted the game's rect, never the full screen
    assert calls["rects"], "screenshots were taken"
    assert all(r == (100, 80, 740, 800) for r in calls["rects"]), calls["rects"]
