# FILE: tests/test_checkout_repair.py
# Purpose: live13 — boot-signal wait + the verifier's surgical repair loop (FAIL -> fix -> re-test).
# Called-by: pytest
# Depends-on: app.pipeline_v2.verifier_agent.eyes_judge, app.pipeline_v2.verifier_agent.host_perception
# Last-renovated: 2026-07-05
"""Taz's verifier directive: wait for a real boot signal (not an assumed
instant boot), and a FAIL verdict must trigger diagnosis -> surgical fix ->
re-test instead of sitting in the ledger. First real target: the run where
the game died on ModuleNotFoundError at launch."""

import asyncio

import pytest

from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.verifier_agent import eyes_judge
from app.pipeline_v2.verifier_agent.eyes_judge import EyesEvidence, JudgeReport
from app.pipeline_v2.verifier_agent.host_perception import wait_ready


class _FakeProc:
    def __init__(self, alive=True, rc=None):
        self._alive = alive
        self._rc = rc

    def poll(self):
        return None if self._alive else self._rc

    def communicate(self, timeout=None):
        return b"", b"ModuleNotFoundError: No module named 'config'"


def _profile(tmp_path):
    return BuildTargetProfile(
        project_id="repair-fixture", project_name="Repair Fixture",
        project_root=str(tmp_path), language="python", build_system="pip",
        framework="generic", source_root="src", package_name="",
        architecture_pattern="flat",
    )


# ---------------------------------------------------------------------------
# wait_ready — the boot signal
# ---------------------------------------------------------------------------

def test_ready_when_window_appears():
    async def probe(pid):
        return "Tazza's Tetris"
    info = {"pid": 77, "_proc": _FakeProc(alive=True)}
    got = asyncio.run(wait_ready(info, timeout_s=5, poll_s=0.01, probe=probe))
    assert got["ready"] is True
    assert got["window_title"] == "Tazza's Tetris"


def test_exit_during_wait_collects_traceback():
    async def probe(pid):
        return ""
    info = {"pid": 77, "_proc": _FakeProc(alive=False, rc=1)}
    got = asyncio.run(wait_ready(info, timeout_s=5, poll_s=0.01, probe=probe))
    assert got["ready"] is False and got["exited"] is True
    assert got["returncode"] == 1
    assert "ModuleNotFoundError" in got["stderr"]


def test_timeout_when_windowless_but_alive():
    async def probe(pid):
        return ""
    info = {"pid": 77, "_proc": _FakeProc(alive=True)}
    got = asyncio.run(wait_ready(info, timeout_s=0.05, poll_s=0.01, probe=probe))
    assert got["ready"] is False and got["exited"] is False


# ---------------------------------------------------------------------------
# run_checkout_with_repair — FAIL -> fix -> re-test
# ---------------------------------------------------------------------------

def _wire(monkeypatch, verdict_sequence):
    """Fake eyes (running app — judge path) + judge yielding the verdicts.

    live17: crashed evidence now short-circuits to a deterministic FAIL and
    never reaches the judge, so judge-path fixtures present a healthy app."""
    evidence = EyesEvidence(launched=True, still_running=True, returncode=None,
                            boot_ready=True, boot_window="Tazza's Tetris")

    async def fake_eyes(profile, spec_hint, job_dir, emit=None):
        return evidence

    verdicts = list(verdict_sequence)

    async def fake_judge(ev, spec_text, job_id, job_dir, ledger=None, emit=None):
        v = verdicts.pop(0) if verdicts else verdict_sequence[-1]
        r = JudgeReport(verdict=v, reasoning=f"fixture verdict {v}")
        if v == "FAIL":
            r.failures = [{"description": "boot crash", "evidence": "stderr",
                           "suspected_files": ["src/main.py"]}]
            r.redispatch_signal = "- [checkout_failure] boot crash"
        return r

    monkeypatch.setattr(eyes_judge, "run_eyes", fake_eyes)
    monkeypatch.setattr(eyes_judge, "run_judge", fake_judge)


def _run(tmp_path, fix_calls, **env):
    async def fix_fn(handover):
        fix_calls.append(handover)
        class _R:
            all_files_written = ["src/main.py"]
        return _R()
    return asyncio.run(eyes_judge.run_checkout_with_repair(
        _profile(tmp_path), "spec", "job-x", str(tmp_path), fix_fn=fix_fn,
    ))


def test_fail_then_pass_after_one_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")
    _wire(monkeypatch, ["FAIL", "PASS"])
    fix_calls = []
    report = _run(tmp_path, fix_calls)
    assert report.verdict == "PASS"
    assert len(fix_calls) == 1
    # the fix worker got the judge's evidence, not a vague nudge
    # (traceback-in-handover is covered by the deterministic crash tests)
    assert "boot crash" in fix_calls[0]
    assert "surgical" in fix_calls[0].lower()


def test_rounds_are_capped(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")
    _wire(monkeypatch, ["FAIL", "FAIL", "FAIL", "FAIL"])
    fix_calls = []
    report = _run(tmp_path, fix_calls)
    assert report.verdict == "FAIL"
    assert len(fix_calls) == 2


def test_blocked_never_triggers_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")
    _wire(monkeypatch, ["BLOCKED"])
    fix_calls = []
    report = _run(tmp_path, fix_calls)
    assert report.verdict == "BLOCKED"
    assert fix_calls == []


def test_pass_first_time_no_repair(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")
    _wire(monkeypatch, ["PASS"])
    fix_calls = []
    report = _run(tmp_path, fix_calls)
    assert report.verdict == "PASS"
    assert fix_calls == []


# ---------------------------------------------------------------------------
# live17: deterministic crash path — zero tokens, judge never consulted
# ---------------------------------------------------------------------------

def test_launch_crash_skips_judge_entirely(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "1")
    crashed = EyesEvidence(launched=True, still_running=False, returncode=1,
                           stderr_tail='File "src/main.py"\nModuleNotFoundError: No module named \'config\'')

    async def fake_eyes(profile, spec_hint, job_dir, emit=None):
        return crashed

    async def judge_must_not_run(*a, **k):
        raise AssertionError("run_judge must never be called for a launch crash")

    monkeypatch.setattr(eyes_judge, "run_eyes", fake_eyes)
    monkeypatch.setattr(eyes_judge, "run_judge", judge_must_not_run)

    fix_calls = []
    report = _run(tmp_path, fix_calls)
    assert report.verdict == "FAIL"
    assert report.model_used == "deterministic/boot-probe"
    # repair still fires, and it gets the actual traceback
    assert len(fix_calls) == 1
    assert "ModuleNotFoundError" in fix_calls[0]


def test_crash_then_healthy_pass_after_repair(monkeypatch, tmp_path):
    """Round 1 crashes (deterministic FAIL, no judge); the repair 'fixes' it;
    round 2 boots and the judge passes it."""
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")
    crashed = EyesEvidence(launched=True, still_running=False, returncode=1,
                           stderr_tail="ModuleNotFoundError: No module named 'config'")
    healthy = EyesEvidence(launched=True, still_running=True, returncode=None,
                           boot_ready=True, boot_window="Tazza's Tetris")
    sequence = [crashed, healthy]

    async def fake_eyes(profile, spec_hint, job_dir, emit=None):
        return sequence.pop(0)

    judge_calls = []

    async def fake_judge(ev, spec_text, job_id, job_dir, ledger=None, emit=None):
        judge_calls.append(ev)
        return JudgeReport(verdict="PASS", reasoning="boots and plays")

    monkeypatch.setattr(eyes_judge, "run_eyes", fake_eyes)
    monkeypatch.setattr(eyes_judge, "run_judge", fake_judge)

    fix_calls = []
    report = _run(tmp_path, fix_calls)
    assert report.verdict == "PASS"
    assert len(fix_calls) == 1
    assert len(judge_calls) == 1  # judge only ever saw the HEALTHY round
