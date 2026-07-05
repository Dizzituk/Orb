# FILE: tests/test_eyes_judge.py
# Purpose: Derek phase 6 gate — broken fixture caught by eyes, diagnosed by judge, ledger verdicts, re-dispatch signal.
# Called-by: pytest
# Depends-on: app.pipeline_v2.verifier_agent.eyes_judge, app.pipeline_v2.verifier_agent.host_perception
# Last-renovated: 2026-07-04
"""Gate 6: the eyes launch a REAL deliberately-broken fixture app on the
host and capture the crash; the (mocked) judge diagnoses it, flags land in
a real Decision Ledger, and a Phase-5-shaped re-dispatch signal comes back."""

import asyncio
import json

import pytest

import app.providers.registry as registry
from app.pipeline_v2.build_targets import BuildTargetProfile
from app.pipeline_v2.verifier_agent import eyes_judge
from app.pipeline_v2.verifier_agent.eyes_judge import (
    EyesEvidence, run_eyes, run_judge, run_eyes_judge_checkout,
)

BROKEN_APP = (
    "print('booting toy app')\n"
    "raise RuntimeError('deliberately broken fixture: game board never initialised')\n"
)
WORKING_APP = "print('toy app ok')\n"

JUDGE_FAIL_RESPONSE = (
    "VERDICT: FAIL\n"
    "REASONING: The app crashes immediately on launch with a RuntimeError; "
    "no customer could use it.\n"
    'FAILURES_JSON: [{"description": "App crashes on launch (RuntimeError: '
    'game board never initialised)", "evidence": "stderr traceback, returncode 1", '
    '"suspected_files": ["main.py"]}]'
)
JUDGE_PASS_RESPONSE = "VERDICT: PASS\nREASONING: Launches and exits cleanly.\nFAILURES_JSON: []"


class _FakeResult:
    def __init__(self, content, ok=True):
        self.content = content
        self._ok = ok
        self.error_message = "" if ok else "boom"

    def is_success(self):
        return self._ok


def _profile(tmp_path):
    return BuildTargetProfile(
        project_id="derek-eyes-fixture", project_name="Eyes Fixture",
        project_root=str(tmp_path), language="python", build_system="pip",
        framework="cli", source_root="", package_name="",
        architecture_pattern="flat",
    )


@pytest.fixture(autouse=True)
def _fast_eyes(monkeypatch):
    monkeypatch.setenv("ASTRA_EYES_SHOTS", "1")
    # Deterministic vision so the test needs no API key
    async def fake_describe(path, hint):
        return f"(vision) a desktop screenshot saved at {path}"
    monkeypatch.setattr(eyes_judge, "_describe_screenshot", fake_describe)
    yield


def test_eyes_catch_broken_fixture(tmp_path):
    (tmp_path / "main.py").write_text(BROKEN_APP, encoding="utf-8")
    job_dir = tmp_path / "job"
    ev = asyncio.run(run_eyes(_profile(tmp_path), "a toy app", str(job_dir)))
    assert ev.launched
    assert not ev.still_running
    assert ev.returncode not in (None, 0)
    assert "deliberately broken fixture" in ev.stderr_tail
    # Screenshot evidence captured on the host (interactive session)
    assert ev.screenshots, "eyes must capture at least one screenshot"
    # live17: a crashed launch spends ZERO vision tokens — the traceback is
    # the evidence and the descriptions say so explicitly.
    assert ev.descriptions and "vision skipped" in ev.descriptions[0]
    rendered = ev.render()
    assert "RuntimeError" in rendered and "SCREENSHOT 1" in rendered


def test_judge_diagnoses_and_signals_redispatch(tmp_path, monkeypatch):
    async def fake_llm_call(**kw):
        assert kw["stage"] == "checkout_judge"
        return _FakeResult(JUDGE_FAIL_RESPONSE)

    monkeypatch.setattr(registry, "llm_call", fake_llm_call)
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_MODEL", "heavy-standin")

    from app.pipeline_v2.ledger import load_or_create_ledger
    job_dir = str(tmp_path / "job")
    import os
    os.makedirs(job_dir, exist_ok=True)
    ledger = load_or_create_ledger("eyes-job-1", job_dir)

    ev = EyesEvidence(
        launched=True, still_running=False, returncode=1,
        stderr_tail="Traceback...\nRuntimeError: game board never initialised",
    )
    report = asyncio.run(run_judge(ev, "spec: playable game", "eyes-job-1", job_dir, ledger))

    assert report.verdict == "FAIL"
    assert report.failures and "crashes" in report.failures[0]["description"].lower()
    assert not report.escalated
    # Valid re-dispatch signal in the Phase-5 evidence shape
    assert "[checkout_failure]" in report.redispatch_signal
    assert "suspected files" in report.redispatch_signal
    # Verdict + failure flags landed in the ledger
    data = json.loads((tmp_path / "job" / "decision_ledger.json").read_text(encoding="utf-8"))
    entries = data["entries"]
    assert any(e["type"] == "verification" and e["stage"] == "checkout_judge" for e in entries)
    assert any(e["type"] == "flag" and e.get("category") == "checkout_failure" for e in entries)


def test_judge_escalates_to_apex_after_failed_rounds(tmp_path, monkeypatch):
    calls = []

    async def flaky_llm_call(**kw):
        calls.append(kw["model_id"])
        if kw["model_id"] == "apex-model":
            return _FakeResult(JUDGE_FAIL_RESPONSE)
        return _FakeResult("no verdict here at all")

    monkeypatch.setattr(registry, "llm_call", flaky_llm_call)
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_MODEL", "heavy-standin")
    monkeypatch.setenv("FRONTIER_PROVIDER", "anthropic")
    monkeypatch.setenv("FRONTIER_MODEL", "apex-model")
    monkeypatch.setenv("ASTRA_JUDGE_ESCALATE_AFTER", "2")

    report = asyncio.run(run_judge(EyesEvidence(), "spec", "j", str(tmp_path)))
    assert calls == ["heavy-standin", "heavy-standin", "apex-model"]
    assert report.escalated
    assert report.model_used == "anthropic/apex-model"
    assert report.verdict == "FAIL"


def test_end_to_end_pass_on_working_fixture(tmp_path, monkeypatch):
    (tmp_path / "main.py").write_text(WORKING_APP, encoding="utf-8")

    async def fake_llm_call(**kw):
        return _FakeResult(JUDGE_PASS_RESPONSE)

    monkeypatch.setattr(registry, "llm_call", fake_llm_call)
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("ASTRA_STAGE_CHECKOUT_JUDGE_MODEL", "heavy-standin")

    report = asyncio.run(run_eyes_judge_checkout(
        _profile(tmp_path), "spec: prints ok", "eyes-job-2", str(tmp_path / "job"),
    ))
    assert report.verdict == "PASS"
    assert report.redispatch_signal == ""
