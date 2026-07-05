# FILE: tests/test_verdict_honesty.py
# Purpose: live18 — 0-of-0 contract checks render SKIP not PASS; 0-file repair rounds stop the loop.
# Called-by: pytest
# Depends-on: app.pipeline_v2.final_verdict, app.pipeline_v2.verifier_agent.eyes_judge
# Last-renovated: 2026-07-05
"""The Builds tab showed 'Verifier PASSED' (0/0 targets, vacuous) beside
'Agentic Builder FAILED' (a repair round that wrote nothing) while the real
judge failed the build. Both cards lied in opposite directions."""

import asyncio

from app.pipeline_v2.final_verdict import build_final_verdict
from app.pipeline_v2.verifier_agent import eyes_judge
from app.pipeline_v2.verifier_agent.eyes_judge import EyesEvidence, JudgeReport
from app.pipeline_v2.build_targets import BuildTargetProfile


class _EmptyContractReport:
    target_results = []
    def is_passing(self):
        return True  # vacuous all() over empty list — the lie live18 kills
    def summary(self):
        return "VerificationReport(targets=0/0 passed, overall=PASS)"


class _RealContractReport(_EmptyContractReport):
    class _T:
        def is_passing(self):
            return True
    target_results = [_T()]
    def summary(self):
        return "VerificationReport(targets=1/1 passed, overall=PASS)"


def test_zero_target_contract_renders_skipped():
    v = build_final_verdict(pipeline_result=None, contract_report=_EmptyContractReport())
    contract = next(s for s in v.sources if s.source == "contract")
    assert contract.status == "skipped"
    assert "0 targets" in contract.detail
    assert "vacuous" in contract.detail


def test_real_target_contract_still_passes():
    v = build_final_verdict(pipeline_result=None, contract_report=_RealContractReport())
    contract = next(s for s in v.sources if s.source == "contract")
    assert contract.status == "pass"


def test_zero_file_repair_round_stops_the_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("ASTRA_CHECKOUT_REPAIR_ROUNDS", "2")
    eyes_calls = []
    healthy_fail = EyesEvidence(launched=True, still_running=True, returncode=None,
                                boot_ready=True, boot_window="T")

    async def fake_eyes(profile, spec_hint, job_dir, emit=None):
        eyes_calls.append(1)
        return healthy_fail

    async def fake_judge(ev, spec_text, job_id, job_dir, ledger=None, emit=None):
        r = JudgeReport(verdict="FAIL", reasoning="looks wrong")
        r.failures = [{"description": "missing bezel", "evidence": "shot", "suspected_files": []}]
        r.redispatch_signal = "- [checkout_failure] missing bezel"
        return r

    monkeypatch.setattr(eyes_judge, "run_eyes", fake_eyes)
    monkeypatch.setattr(eyes_judge, "run_judge", fake_judge)

    async def zero_file_fix(handover):
        class _R:
            all_files_written = []
        return _R()

    profile = BuildTargetProfile(
        project_id="t", project_name="T", project_root=str(tmp_path),
        language="python", build_system="pip", framework="generic",
        source_root="src", package_name="", architecture_pattern="flat",
    )
    report = asyncio.run(eyes_judge.run_checkout_with_repair(
        profile, "spec", "job-z", str(tmp_path), fix_fn=zero_file_fix,
    ))
    assert report.verdict == "FAIL"
    assert len(eyes_calls) == 1, "no re-test after a 0-file repair round"
