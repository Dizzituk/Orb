# FILE: tests/test_gui_boot_signal.py
# Purpose: live20 — GUI apps never construct at import; boot signal keys on window existence not title text.
# Called-by: pytest
# Depends-on: app.pipeline_v2.builder_prompts, app.pipeline_v2.verifier_agent.host_perception
# Last-renovated: 2026-07-05
"""Run #9 froze on a title-less window: the pipeline's `from main import app`
boot check (a uvicorn convention) made the GUI worker build the app at MODULE
IMPORT, so importing double-inited and blocked the window's message pump — no
title — and the user had ALSO asked for a borderless no-X-button window, which
has no title by design. Two fixes, both pinned here."""

import asyncio

from app.pipeline_v2.builder_prompts import build_system_prompt, _PYTHON_SECTION
from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.verifier_agent.host_perception import wait_ready


def _profile(root, framework):
    return BuildTargetProfile(
        project_id="t", project_name="T", project_root=root,
        language="python", build_system="pip", framework=framework,
        source_root="src", package_name="", architecture_pattern="flat",
    )


# ---------------------------------------------------------------------------
# builder prompt: greenfield GUI vs the ASTRA backend
# ---------------------------------------------------------------------------

def test_greenfield_python_app_gets_gui_safe_section():
    prompt = build_system_prompt(_profile("C:/Games/T", "generic"))
    assert "NEVER construct the app" in prompt
    assert "MODULE IMPORT TIME" in prompt
    assert "from main import app" not in prompt  # the poison instruction is gone
    assert "Open the window FIRST" in prompt


def test_astra_backend_keeps_uvicorn_boot_check():
    prompt = build_system_prompt(_profile("D:/Orb", "fastapi"))
    assert "from main import app" in prompt  # uvicorn convention preserved


def test_orb_root_treated_as_backend_even_generic_framework():
    prompt = build_system_prompt(_profile("D:/Orb", "generic"))
    assert "from main import app" in prompt


# ---------------------------------------------------------------------------
# wait_ready: window existence, not title text
# ---------------------------------------------------------------------------

class _AliveProc:
    def poll(self):
        return None


def _run_wait(probe):
    info = {"pid": 4321, "_proc": _AliveProc()}
    return asyncio.run(wait_ready(info, timeout_s=1, poll_s=0.01, probe=probe))


def test_borderless_window_no_title_is_ready():
    async def probe(pid):
        return {"has_window": True, "title": ""}  # borderless: handle set, title empty
    got = _run_wait(probe)
    assert got["ready"] is True
    assert got["window_title"] == ""


def test_titled_window_still_ready_and_keeps_title():
    async def probe(pid):
        return {"has_window": True, "title": "Tazza's Tetris"}
    got = _run_wait(probe)
    assert got["ready"] is True
    assert got["window_title"] == "Tazza's Tetris"


def test_no_window_times_out():
    async def probe(pid):
        return {"has_window": False, "title": ""}
    got = _run_wait(probe)
    assert got["ready"] is False and got["exited"] is False


def test_legacy_string_probe_still_supported():
    async def probe(pid):
        return "Tazza's Tetris"  # old-style probe returning a title
    got = _run_wait(probe)
    assert got["ready"] is True
    assert got["window_title"] == "Tazza's Tetris"
