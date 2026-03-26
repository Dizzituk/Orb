from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from app.optimize.evidence_evaluator import EvidenceScore, evaluate_evidence, is_at_optimum
from app.optimize.scope_manager import ScopeFlag, detect_scope_expansion, list_scope_flags
from app.optimize.self_test import run_self_test
from app.optimize.trust_gate import TrustLevel, can_auto_continue

logger = logging.getLogger(__name__)


@dataclass
class ContinueDecision:
    continue_loop: bool
    reason: str


@dataclass
class PassResult:
    pass_number: int
    target_id: str
    improvement_delta: float
    evidence_strength: float
    evidence: EvidenceScore
    summary: str
    target_chunks: List[str] = field(default_factory=list)
    touched_subsystems: List[str] = field(default_factory=list)
    scope_flags: List[ScopeFlag] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evidence"] = self.evidence.to_dict()
        data["scope_flags"] = [flag.to_dict() for flag in self.scope_flags]
        return data


_LOOP_STATE: Dict[str, Any] = {
    "running": False,
    "target_id": None,
    "current_pass": 0,
    "max_passes": 0,
    "trust_level": TrustLevel.SEMI_AUTO.value,
    "history": [],
    "status": "idle",
    "stop_requested": False,
}


def _should_force_pass(metrics: Dict[str, float]) -> bool:
    return float(metrics.get("latency_ms", 0.0)) > 120 or float(metrics.get("complexity", 0.0)) > 12


def _evaluate_pass_evidence(report: PassResult) -> ContinueDecision:
    if report.evidence.regression_count > 0:
        return ContinueDecision(False, "A regression was detected, so the loop should stop.")
    if is_at_optimum(report.evidence):
        return ContinueDecision(False, "The system appears to be at its current optimum.")
    if report.evidence_strength >= 0.65:
        return ContinueDecision(True, "Evidence is strong enough to justify another pass.")
    return ContinueDecision(False, "The latest pass did not produce enough evidence for another pass.")


def _simulate_metrics(pass_number: int) -> tuple[Dict[str, float], Dict[str, float]]:
    before = {
        "latency_ms": max(60.0, 120.0 - ((pass_number - 1) * 10.0)),
        "complexity": max(4.0, 12.0 - ((pass_number - 1) * 1.5)),
    }
    after = {
        "latency_ms": max(55.0, before["latency_ms"] - (12.0 if pass_number == 1 else 4.0)),
        "complexity": max(3.0, before["complexity"] - (2.0 if pass_number == 1 else 0.8)),
    }
    return before, after


def run_recursive_loop(target_id: str, max_passes: int, trust_level: TrustLevel | str) -> Dict[str, Any]:
    level = trust_level if isinstance(trust_level, TrustLevel) else TrustLevel(trust_level)
    _LOOP_STATE.update({
        "running": True,
        "target_id": target_id,
        "current_pass": 0,
        "max_passes": max_passes,
        "trust_level": level.value,
        "history": [],
        "status": "running",
        "stop_requested": False,
    })

    for pass_number in range(1, max_passes + 1):
        if _LOOP_STATE["stop_requested"]:
            _LOOP_STATE["status"] = "stopped"
            break

        before_metrics, after_metrics = _simulate_metrics(pass_number)
        test_report = run_self_test(target_id).to_dict()
        evidence = evaluate_evidence(before_metrics, after_metrics, test_report)
        improvement_delta = round(sum(before_metrics.values()) - sum(after_metrics.values()), 4)
        touched_subsystems = [target_id]
        if pass_number == 1 and target_id == "routing":
            touched_subsystems.append("bridge")

        pass_result = PassResult(
            pass_number=pass_number,
            target_id=target_id,
            improvement_delta=improvement_delta,
            evidence_strength=evidence.improvement_score,
            evidence=evidence,
            summary=evidence.summary,
            target_chunks=[f"{target_id}/core.py"],
            touched_subsystems=touched_subsystems,
        )
        pass_result.scope_flags = detect_scope_expansion(pass_result, target_id)
        _LOOP_STATE["history"].append(pass_result)
        _LOOP_STATE["current_pass"] = pass_number

        if pass_result.scope_flags:
            _LOOP_STATE["status"] = "awaiting_scope_approval"
            break

        decision = _evaluate_pass_evidence(pass_result)
        if pass_number >= max_passes:
            _LOOP_STATE["status"] = "complete"
            break
        if _should_force_pass(before_metrics):
            continue
        if can_auto_continue(level, pass_result.evidence_strength) and decision.continue_loop:
            continue
        _LOOP_STATE["status"] = "system at optimum" if is_at_optimum(pass_result.evidence) else "awaiting_manual_continue"
        break
    else:
        _LOOP_STATE["status"] = "complete"

    _LOOP_STATE["running"] = _LOOP_STATE["status"] in {"running", "awaiting_scope_approval", "awaiting_manual_continue"}
    return get_loop_status()


def stop_loop() -> Dict[str, Any]:
    _LOOP_STATE["stop_requested"] = True
    _LOOP_STATE["running"] = False
    _LOOP_STATE["status"] = "stopped"
    return get_loop_status()


def get_loop_status() -> Dict[str, Any]:
    return {
        "running": _LOOP_STATE["running"],
        "target_id": _LOOP_STATE["target_id"],
        "current_pass": _LOOP_STATE["current_pass"],
        "max_passes": _LOOP_STATE["max_passes"],
        "trust_level": _LOOP_STATE["trust_level"],
        "status": _LOOP_STATE["status"],
        "flags": [flag.to_dict() for flag in list_scope_flags() if flag.status == "pending"],
        "history": [item.to_dict() for item in _LOOP_STATE["history"]],
    }


def get_loop_history() -> List[Dict[str, Any]]:
    return [item.to_dict() for item in _LOOP_STATE["history"]]
