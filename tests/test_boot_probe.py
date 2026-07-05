# FILE: tests/test_boot_probe.py
# Purpose: live17 — the integrator's deterministic boot probe: crashes are caught and attributed before any token is spent.
# Called-by: pytest
# Depends-on: app.pipeline_v2.segmented.integrator, app.pipeline_v2.verifier_agent.host_perception
# Last-renovated: 2026-07-05
"""Every checkout failure to date was visible in a subprocess traceback within
seconds. The builder now proves the app STARTS before handing over; the
checkout only ever judges behaviour."""

import asyncio

import pytest

from app.pipeline_v2.segmented import integrator
from app.pipeline_v2.segmented.integrator import boot_probe, _owning_segment_for_traceback
from app.pipeline_v2.verifier_agent import host_perception as hp


class _Seg:
    def __init__(self, segment_id, file_scope):
        self.segment_id = segment_id
        self.file_scope = file_scope


class _Profile:
    project_root = "C:/Games/T"
    language = "python"


SEGS = [
    _Seg("seg-01-core", ["src/config.py", "src/board.py"]),
    _Seg("seg-08-audio", ["src/audio.py"]),
    _Seg("seg-09-entry", ["src/main.py"]),
]

TRACEBACK = (
    'Traceback (most recent call last):\n'
    '  File "C:\\Games\\T\\src\\main.py", line 8, in <module>\n'
    '  File "C:\\Games\\T\\src\\audio.py", line 4, in <module>\n'
    "ModuleNotFoundError: No module named 'config'"
)


@pytest.fixture
def _host_lane(monkeypatch):
    monkeypatch.setattr("app.pipeline_v2.android_sandbox.is_host_build", lambda p: True)
    monkeypatch.setattr("app.pipeline_v2.clone_freshness.is_self_build", lambda p: False)
    monkeypatch.setattr(hp, "guess_entrypoint", lambda root: ["python", "src/main.py"])
    stopped = []
    monkeypatch.setattr(hp, "stop_app", lambda info: stopped.append(True))
    return stopped


def test_crash_yields_attributed_boot_failure(monkeypatch, _host_lane):
    async def fake_launch(root, cmd=None, settle_seconds=4.0):
        return {"launched": True, "running": False, "returncode": 1,
                "stderr": TRACEBACK, "stdout": "", "_proc": None}
    monkeypatch.setattr(hp, "launch_app", fake_launch)

    failures = asyncio.run(boot_probe(SEGS, _Profile()))
    assert len(failures) == 1
    f = failures[0]
    assert f.kind == "boot_crash"
    assert f.segment_id == "seg-08-audio"  # LAST file in the traceback owns it
    assert "ModuleNotFoundError" in f.detail
    assert _host_lane, "probe must stop the launched process"


def test_healthy_app_probe_green_and_killed(monkeypatch, _host_lane):
    async def fake_launch(root, cmd=None, settle_seconds=4.0):
        return {"launched": True, "running": True, "returncode": None,
                "stderr": "", "stdout": "", "_proc": None}
    monkeypatch.setattr(hp, "launch_app", fake_launch)

    assert asyncio.run(boot_probe(SEGS, _Profile())) == []
    assert _host_lane, "healthy probe target must still be stopped"


def test_cli_framework_skips_probe(monkeypatch):
    """Bare-launching an argv CLI exits nonzero legitimately — no probe."""
    monkeypatch.setattr("app.pipeline_v2.android_sandbox.is_host_build", lambda p: True)
    monkeypatch.setattr("app.pipeline_v2.clone_freshness.is_self_build", lambda p: False)

    class _Cli(_Profile):
        framework = "cli"
    assert asyncio.run(boot_probe(SEGS, _Cli())) == []


def test_kotlin_and_selfbuild_lanes_skip(monkeypatch):
    class _K:
        project_root = "x"; language = "kotlin"
    assert asyncio.run(boot_probe(SEGS, _K())) == []

    monkeypatch.setattr("app.pipeline_v2.android_sandbox.is_host_build", lambda p: True)
    monkeypatch.setattr("app.pipeline_v2.clone_freshness.is_self_build", lambda p: True)
    assert asyncio.run(boot_probe(SEGS, _Profile())) == []


def test_traceback_attribution_fallback_to_entry():
    # traceback naming no scoped file -> the entry-owning segment answers
    sid = _owning_segment_for_traceback('File "C:\\elsewhere\\thing.py"', SEGS)
    assert sid == "seg-09-entry"


def test_worker_prompt_pins_import_convention(tmp_path):
    from app.pipeline_v2.segmented.worker import build_worker_prompt
    from app.pot_spec.grounded.segment_schemas import SegmentSpec

    class _P:
        project_root = str(tmp_path); project_name = "T"; language = "python"

    seg = SegmentSpec(segment_id="seg-01", title="core", file_scope=["src/board.py"])
    prompt = build_worker_prompt(seg, [], "", profile=_P())
    assert "IMPORT CONVENTION" in prompt
    assert "NEVER `from src.board import" in prompt or "NEVER `from src." in prompt

    class _K(_P):
        language = "kotlin"
    prompt_k = build_worker_prompt(seg, [], "", profile=_K())
    assert "IMPORT CONVENTION" not in prompt_k
