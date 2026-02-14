# FILE: app/orchestrator/final_checkout.py
"""
Final Project Checkout — Stage 10.

Project-level verification that runs after ALL phases complete.
Catches issues that per-phase checkout (Stage 9) cannot:

1. Cross-phase import integrity — do files from Phase 2 correctly
   import from Phase 1 files?
2. Full project boot test — does the entire assembled project start?
3. Spec coverage — are all files from the original spec accounted for?
4. Deliverable summary — final report of what was built.

For single-phase jobs, this is largely redundant with Stage 9 but
still runs the spec coverage check and produces the summary report.

v1.0 (2026-02-14): Initial implementation — Stage 10.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .construction_planner_models import ConstructionPlan
from .construction_skeleton import verify_phase_deliverables
from .phase_checkout_checks import run_boot_test
from app.pot_spec.grounded.size_models import MAX_FILE_LINES

logger = logging.getLogger(__name__)

FINAL_CHECKOUT_BUILD_ID = "2026-02-14-v1.0-initial"
print(f"[FINAL_CHECKOUT_LOADED] BUILD_ID={FINAL_CHECKOUT_BUILD_ID}")


# =============================================================================
# RESULT MODEL
# =============================================================================

@dataclass
class SpecCoverageResult:
    """Did the build produce all files from the original spec?"""
    status: str  # "pass", "fail"
    expected_files: int = 0
    found_files: int = 0
    missing_files: List[str] = field(default_factory=list)
    extra_files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "expected_files": self.expected_files,
            "found_files": self.found_files,
            "missing_files": self.missing_files,
            "extra_files": self.extra_files,
        }


@dataclass
class CrossPhaseResult:
    """Did all phase contracts get honoured?"""
    status: str  # "pass", "fail", "skipped"
    phases_checked: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "phases_checked": self.phases_checked,
            "violations": self.violations,
        }


@dataclass
class FinalCheckoutResult:
    """Complete Stage 10 verification result."""
    job_id: str
    status: str = "pending"  # "pass", "fail", "error"
    boot_test_status: str = ""
    spec_coverage: Optional[SpecCoverageResult] = None
    cross_phase: Optional[CrossPhaseResult] = None
    total_files_built: int = 0
    total_phases: int = 0
    duration_ms: int = 0
    timestamp: str = ""
    checks_run: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "boot_test_status": self.boot_test_status,
            "spec_coverage": self.spec_coverage.to_dict() if self.spec_coverage else None,
            "cross_phase": self.cross_phase.to_dict() if self.cross_phase else None,
            "total_files_built": self.total_files_built,
            "total_phases": self.total_phases,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "checks_run": self.checks_run,
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_final_checkout(
    job_id: str,
    plan: ConstructionPlan,
    original_file_scope: List[str],
    job_dir: str,
    sandbox_base: str = r"D:\Orb",
    emit: Optional[Callable] = None,
) -> FinalCheckoutResult:
    """
    Run final project-level verification after all phases complete.

    Args:
        job_id: Job identifier
        plan: The completed construction plan
        original_file_scope: Full file scope from the original spec
        job_dir: Job directory for saving result
        sandbox_base: Root path for file checks
        emit: Progress callback

    Returns:
        FinalCheckoutResult with aggregated pass/fail
    """
    start = time.time()
    _emit = emit or (lambda msg: None)
    result = FinalCheckoutResult(job_id=job_id, total_phases=plan.total_phases)

    _emit(f"\n{'='*60}")
    _emit(f"🏆 FINAL PROJECT CHECKOUT — Stage 10")
    _emit(f"   {plan.total_phases} phase(s), {len(original_file_scope)} expected files")

    # --- Check 1: Spec coverage ---
    _emit("\n📋 Check 1: Spec file coverage...")
    result.spec_coverage = _check_spec_coverage(original_file_scope, sandbox_base)
    result.checks_run.append("spec_coverage")
    result.total_files_built = result.spec_coverage.found_files

    if result.spec_coverage.status == "pass":
        _emit(f"  ✅ All {result.spec_coverage.expected_files} spec files found on disk")
    else:
        _emit(f"  ❌ {len(result.spec_coverage.missing_files)} file(s) missing:")
        for mf in result.spec_coverage.missing_files[:10]:
            _emit(f"    • {mf}")

    # --- Check 2: Cross-phase contract verification ---
    if plan.is_multi_phase:
        _emit("\n🔗 Check 2: Cross-phase contract verification...")
        result.cross_phase = _check_cross_phase_contracts(plan, sandbox_base)
        result.checks_run.append("cross_phase")

        if result.cross_phase.status == "pass":
            _emit(f"  ✅ All {result.cross_phase.phases_checked} phase contracts honoured")
        else:
            _emit(f"  ❌ {len(result.cross_phase.violations)} contract violation(s)")
            for v in result.cross_phase.violations[:5]:
                _emit(f"    • Phase {v.get('phase_id', '?')}: {v.get('detail', '?')}")
    else:
        _emit("\n🔗 Check 2: Cross-phase contracts — SKIPPED (single phase)")

    # --- Check 3: Full project boot test ---
    _emit("\n🔧 Check 3: Full project boot test...")
    boot = run_boot_test(sandbox_base)
    result.boot_test_status = boot.status
    result.checks_run.append("boot_test")

    if boot.status == "pass":
        _emit("  ✅ Project boots cleanly")
    elif boot.status == "fail":
        _emit(f"  ❌ Boot failed: {boot.error_summary}")
    else:
        _emit(f"  ⚠️ Boot error: {boot.error_summary}")

    # --- Aggregate ---
    all_ok = (
        result.spec_coverage.status == "pass"
        and (not result.cross_phase or result.cross_phase.status == "pass")
        and result.boot_test_status == "pass"
    )
    result.status = "pass" if all_ok else "fail"
    result.duration_ms = int((time.time() - start) * 1000)

    if all_ok:
        _emit(f"\n🎉 FINAL CHECKOUT PASSED — project fully verified")
    else:
        _emit(f"\n❌ FINAL CHECKOUT FAILED")

    _emit(f"   Files built: {result.total_files_built}/{len(original_file_scope)}")
    _emit(f"   Duration: {result.duration_ms}ms")

    _save_result(result, job_dir)
    return result


# =============================================================================
# CHECK 1: SPEC COVERAGE
# =============================================================================

def _check_spec_coverage(
    file_scope: List[str],
    sandbox_base: str,
) -> SpecCoverageResult:
    """Verify all files from the original spec exist on disk."""
    missing = []
    found = 0

    for rel_path in file_scope:
        normalised = rel_path.replace("/", os.sep).replace("\\", os.sep)
        abs_path = os.path.join(sandbox_base, normalised)
        if os.path.isfile(abs_path):
            found += 1
        else:
            missing.append(rel_path)

    return SpecCoverageResult(
        status="fail" if missing else "pass",
        expected_files=len(file_scope),
        found_files=found,
        missing_files=missing,
    )


# =============================================================================
# CHECK 2: CROSS-PHASE CONTRACTS
# =============================================================================

def _check_cross_phase_contracts(
    plan: ConstructionPlan,
    sandbox_base: str,
) -> CrossPhaseResult:
    """Verify all phase contracts were honoured."""
    violations = []

    for phase in plan.phases:
        result = verify_phase_deliverables(plan, phase, sandbox_base)
        if result["status"] == "fail":
            for mf in result.get("missing", []):
                violations.append({
                    "phase_id": phase.phase_id,
                    "violation_type": "missing_deliverable",
                    "detail": f"Phase {phase.phase_number} promised '{mf}' but it's missing",
                })

    return CrossPhaseResult(
        status="fail" if violations else "pass",
        phases_checked=len(plan.phases),
        violations=violations,
    )


# =============================================================================
# PERSISTENCE
# =============================================================================

def _save_result(result: FinalCheckoutResult, job_dir: str) -> None:
    """Save final checkout result to job directory."""
    path = os.path.join(job_dir, "final_checkout_result.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("[final_checkout] Result saved to %s", path)
    except Exception as exc:
        logger.warning("[final_checkout] Failed to save: %s", exc)
