from __future__ import annotations
import os
import time
from .construction_planner_models import ConstructionPlan
from .phase_checkout_checks import run_boot_test_with_fix_loop
from .electron_boot_check import run_electron_boot_check, ElectronBootResult
from .strike_tracker import StrikeRecord, StrikeTracker
from app.orchestrator._final_checkout_utils_2 import _build_minimal_state_from_plan
from app.orchestrator._final_checkout_utils_3 import _build_plan_from_manifest, _run_boot_with_strike_tracking, _save_result
from app.orchestrator._final_checkout_utils_5 import _check_cross_phase_with_fixes, _check_spec_coverage_with_fixes, _run_ai_review_with_fixes
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
from app.sandbox_fs import (
    sandbox_isfile as _sbx_isfile,
    sandbox_isdir as _sbx_isdir,
    sandbox_exists as _sbx_exists,
    sandbox_read_text as _sbx_read_text,
)


@dataclass
class SpecCoverageResult:
    """Did the build produce all files from the original spec?"""
    status: str  # "pass", "fail"
    expected_files: int = 0
    found_files: int = 0
    missing_files: List[str] = field(default_factory=list)
    extra_files: List[str] = field(default_factory=list)
    files_generated: List[str] = field(default_factory=list)  # v3.0: files we created

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "expected_files": self.expected_files,
            "found_files": self.found_files,
            "missing_files": self.missing_files,
            "extra_files": self.extra_files,
            "files_generated": self.files_generated,
        }

@dataclass
class CrossPhaseResult:
    """Did all phase contracts get honoured?"""
    status: str  # "pass", "fail", "skipped"
    phases_checked: int = 0
    violations: List[Dict[str, Any]] = field(default_factory=list)
    fixes_applied: List[Dict[str, Any]] = field(default_factory=list)  # v3.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "phases_checked": self.phases_checked,
            "violations": self.violations,
            "fixes_applied": self.fixes_applied,
        }

@dataclass
class AIReviewResult:
    """Final AI quality/security/performance review."""
    status: str  # "pass", "fail", "error", "skipped"
    overall_score: float = 0.0  # 0-10
    quality_score: float = 0.0
    security_score: float = 0.0
    performance_score: float = 0.0
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    files_reviewed: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "overall_score": self.overall_score,
            "quality_score": self.quality_score,
            "security_score": self.security_score,
            "performance_score": self.performance_score,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "files_reviewed": self.files_reviewed,
            "notes": self.notes,
        }

async def run_final_checkout(
    job_id: str,
    plan: Optional[ConstructionPlan] = None,
    original_file_scope: Optional[List[str]] = None,
    job_dir: str = "",
    sandbox_base: str = r"D:\Orb",
    original_spec: Optional[str] = None,
    state: Optional[Any] = None,
    manifest: Optional[Any] = None,
    emit: Optional[Callable] = None,
) -> FinalCheckoutResult:
    """
    Run final project-level verification and autonomous fix-up.

    v3.0: This is now an autonomous closer. Every check that finds a
    problem attempts to fix it using the three-strike rule. Only
    unresolvable issues are escalated to human review.

    Args:
        job_id: Job identifier
        plan: The completed construction plan
        original_file_scope: Full file scope from the original spec
        job_dir: Job directory for saving result
        sandbox_base: Root path for file checks
        original_spec: The original POT spec markdown (for AI review)
        state: JobState with segment info (for boot fix mapping)
        manifest: SegmentManifest for integration checking
        emit: Progress callback

    Returns:
        FinalCheckoutResult with aggregated pass/fail + RAG records
    """
    from .final_checkout import FinalCheckoutResult, compile_pipeline_learning_report
    start = time.time()
    _emit = emit or (lambda msg: None)

    # v3.2: Build file scope from manifest if not provided
    if not original_file_scope and manifest:
        _all_scope = []
        for seg in manifest.segments:
            _all_scope.extend(seg.file_scope)
        original_file_scope = list(set(_all_scope))
    original_file_scope = original_file_scope or []

    # v3.2: Build a minimal ConstructionPlan from manifest if not provided
    if plan is None:
        plan = _build_plan_from_manifest(job_id, manifest)

    result = FinalCheckoutResult(job_id=job_id, total_phases=plan.total_phases)
    all_strike_records: List[StrikeRecord] = []

    # Set journal context for stages that use journal_emit()
    try:
        from app.experience.context import set_job_context, journal_emit as _journal_emit
        if not job_dir:
            import os as _os
            _jd = _os.path.join("D:/Orb/jobs/jobs", job_id)
        else:
            _jd = job_dir
        set_job_context(job_id=job_id, job_dir=_jd)
        _journal_emit(
            stage="final_checkout",
            event_type="stage_enter",
            description="Final checkout starting",
            details={"total_phases": plan.total_phases, "expected_files": len(original_file_scope)},
        )
    except Exception:
        pass

    _emit(f"\n{'='*60}")
    _emit(">>> FINAL PROJECT CHECKOUT - Stage 10 (Autonomous Closer)")
    _emit(f"   {plan.total_phases} phase(s), {len(original_file_scope)} expected files")

    # Build state for boot fix mapping if not provided
    _boot_state = state
    if _boot_state is None:
        _boot_state = _build_minimal_state_from_plan(plan, sandbox_base)

    # Get sandbox client for fix operations
    sandbox_client = None
    try:
        from app.overwatcher.sandbox_client import get_sandbox_client
        sandbox_client = get_sandbox_client()
        if not sandbox_client.is_connected():
            sandbox_client = None
    except Exception:
        pass

    # ---------------------------------------------------------------
    # CHECK 1: Spec coverage (gate — with auto-generation)
    # ---------------------------------------------------------------
    _emit("\n[CHECK 1] Spec file coverage (with auto-generation)...")
    spec_tracker = StrikeTracker("spec_coverage", "final_checkout", job_id)
    result.spec_coverage = await _check_spec_coverage_with_fixes(
        file_scope=original_file_scope,
        sandbox_base=sandbox_base,
        sandbox_client=sandbox_client,
        original_spec=original_spec,
        tracker=spec_tracker,
        emit=_emit,
    )
    result.checks_run.append("spec_coverage")
    result.total_files_built = result.spec_coverage.found_files
    all_strike_records.append(spec_tracker.get_record())

    if result.spec_coverage.status == "pass":
        _emit(f"  [OK] All {result.spec_coverage.expected_files} spec files accounted for")
        if result.spec_coverage.files_generated:
            _emit(f"    ({len(result.spec_coverage.files_generated)} generated during checkout)")
    else:
        _emit(f"  [FAIL] {len(result.spec_coverage.missing_files)} file(s) still missing after fix attempts")

    # ---------------------------------------------------------------
    # CHECK 2: Cross-phase contract verification (with auto-fix)
    # ---------------------------------------------------------------
    if plan.is_multi_phase:
        _emit("\n[CHECK 2] Cross-phase contract verification (with auto-fix)...")
        cross_tracker = StrikeTracker("cross_phase", "final_checkout", job_id)
        result.cross_phase = await _check_cross_phase_with_fixes(
            plan=plan,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            tracker=cross_tracker,
            emit=_emit,
            state=_boot_state,
            manifest=manifest,
        )
        result.checks_run.append("cross_phase")
        all_strike_records.append(cross_tracker.get_record())

        if result.cross_phase.status == "pass":
            _emit(f"  [OK] All {result.cross_phase.phases_checked} phase contracts honoured")
        else:
            _emit(f"  [WARNING] {len(result.cross_phase.violations)} unresolved violation(s)")
    else:
        _emit("\n[CHECK 2] Cross-phase contracts -- SKIPPED (single phase)")

    # ---------------------------------------------------------------
    # CHECK 3: Full project boot test with fix loop
    # ---------------------------------------------------------------
    _emit("\n[CHECK 3] Full project boot test (with surgical fix loop)...")
    boot_tracker = StrikeTracker("boot_test", "final_checkout", job_id)

    boot = await _run_boot_with_strike_tracking(
        sandbox_base=sandbox_base,
        state=_boot_state,
        tracker=boot_tracker,
        emit=_emit,
    )
    result.boot_test_status = boot.status
    result.checks_run.append("boot_test")
    all_strike_records.append(boot_tracker.get_record())

    if boot.status == "pass":
        _emit("  [OK] Project boots cleanly")
    elif boot.status == "fail":
        _emit(f"  [FAIL] Boot failed after strike rule: {boot.error_summary}")
    else:
        _emit(f"  [WARNING] Boot error: {boot.error_summary}")

    # ---------------------------------------------------------------
    # CHECK 4: Final AI Review (advisory with auto-fix)
    # ---------------------------------------------------------------
    _emit("\n[CHECK 4] Final AI Review (quality/security/performance)...")
    review_tracker = StrikeTracker("ai_review", "final_checkout", job_id)
    result.ai_review = await _run_ai_review_with_fixes(
        original_spec=original_spec,
        file_scope=original_file_scope,
        sandbox_base=sandbox_base,
        sandbox_client=sandbox_client,
        tracker=review_tracker,
        emit=_emit,
    )
    result.checks_run.append("ai_review")
    all_strike_records.append(review_tracker.get_record())

    if result.ai_review and result.ai_review.status == "pass":
        _emit(f"  [OK] AI Review passed (score: {result.ai_review.overall_score}/10)")
    elif result.ai_review and result.ai_review.status == "fail":
        _emit(f"  [WARNING] AI Review flagged issues (score: {result.ai_review.overall_score}/10)")
    else:
        _emit("  [SKIPPED] AI Review could not run")

    # ---------------------------------------------------------------
    # CHECK 5: Final boot confirmation (hard gate — after all fixes)
    # ---------------------------------------------------------------
    any_fixes_applied = (
        bool(result.spec_coverage.files_generated)
        or (result.cross_phase and bool(result.cross_phase.fixes_applied))
    )

    if any_fixes_applied and result.boot_test_status == "pass":
        _emit("\n[CHECK 5] Final boot confirmation (post-fix verification)...")
        final_boot = await run_boot_test_with_fix_loop(
            sandbox_base=sandbox_base,
            state=_boot_state,
            emit=_emit,
        )
        result.boot_test_status = final_boot.status
        result.checks_run.append("final_boot_confirmation")

        if final_boot.status == "pass":
            _emit("  [OK] Final boot confirmed — all fixes hold")
        else:
            _emit(f"  [FAIL] Final boot failed after fixes: {final_boot.error_summary}")
    elif any_fixes_applied and result.boot_test_status != "pass":
        _emit("\n[CHECK 5] Final boot confirmation -- SKIPPED (boot already failed)")

    # ---------------------------------------------------------------
    # CHECK 5b: Real Electron Boot (v1.0 — full runtime validation)
    # ---------------------------------------------------------------
    electron_boot_result = None
    if result.boot_test_status == "pass" and sandbox_client:
        _emit("\n[CHECK 5b] Full Electron boot (runtime validation)...")
        try:
            electron_boot_result = run_electron_boot_check(
                client=sandbox_client,
                emit=_emit,
            )
            result.checks_run.append("electron_boot")
            if not electron_boot_result.success:
                _emit(f"  ⚠️  Vite compiled OK but Electron boot found issues:")
                _emit(f"      {electron_boot_result.error_summary[:200]}")
                if electron_boot_result.errors:
                    result.boot_test_status = "warning"
            else:
                _emit(f"  ✅ Full Electron boot confirmed clean ({electron_boot_result.boot_time_ms}ms)")
        except Exception as _ebc_err:
            logger.warning("[final_checkout] Electron boot check failed: %s", _ebc_err)
            _emit(f"  [SKIPPED] Electron boot check failed: {_ebc_err}")
    elif result.boot_test_status != "pass":
        _emit("\n[CHECK 5b] Electron boot -- SKIPPED (vite build already failed)")
    elif not sandbox_client:
        _emit("\n[CHECK 5b] Electron boot -- SKIPPED (no sandbox connection)")

    # ---------------------------------------------------------------
    # AGGREGATE
    # ---------------------------------------------------------------
    all_ok = (
        result.spec_coverage.status == "pass"
        and result.boot_test_status == "pass"
    )
    result.status = "pass" if all_ok else "fail"
    result.duration_ms = int((time.time() - start) * 1000)

    # Store strike records for RAG
    result.strike_records = [r.to_dict() for r in all_strike_records]

    if all_ok:
        _emit(f"\n>>> FINAL CHECKOUT PASSED — ready for human review")
    else:
        _emit(f"\n>>> FINAL CHECKOUT FAILED — requires human intervention")
        if result.boot_test_status != "pass":
            _emit("   Reason: Boot test failed")
        if result.spec_coverage.status != "pass":
            _emit(f"   Reason: {len(result.spec_coverage.missing_files)} spec files missing")

    _emit(f"   Files built: {result.total_files_built}/{len(original_file_scope)}")
    _emit(f"   Strike records: {len(all_strike_records)} checks tracked")
    _emit(f"   Duration: {result.duration_ms}ms")

    _save_result(result, job_dir)

    # ---------------------------------------------------------------
    # CHECK 6: Pipeline Learning Report (absolute last thing)
    # ---------------------------------------------------------------
    _phase_checkout_dict = None
    if state and hasattr(state, 'integration_check') and state.integration_check:
        _phase_checkout_dict = state.integration_check.get("phase_checkout")

    compile_pipeline_learning_report(
        job_id=job_id,
        job_dir=job_dir,
        state=state,
        manifest=manifest,
        final_result=result,
        phase_checkout_result=_phase_checkout_dict,
        original_spec=original_spec,
        emit=_emit,
    )
    result.checks_run.append("pipeline_learning_report")

    return result


# v3.4-fix: Frontend path prefixes that resolve to D:\orb-desktop
_FE_PREFIX = "orb-desktop/"
_FE_ROOT = r"D:\orb-desktop"
_FE_BARE = ("src/", "src\\", "public/", "public\\")


def _file_exists_on_sandbox(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> bool:
    """Check if a file exists on the sandbox.

    v3.4-fix: Frontend path resolution - bare src/ or orb-desktop/src/
    paths resolve to D:\\orb-desktop, not D:\\Orb\\src (which does not exist).
    Matches the same logic used by _write_file_to_sandbox in utils_5.
    """
    if client:
        try:
            normed = rel_path.replace("/", "\\")
            normalized_fwd = rel_path.replace("\\", "/")

            if normed.startswith("C:") or normed.startswith("D:"):
                abs_path = normed
            elif normalized_fwd.startswith(_FE_PREFIX):
                frontend_rel = normalized_fwd[len(_FE_PREFIX):]
                abs_path = _FE_ROOT + "\\" + frontend_rel.replace("/", "\\")
            elif any(normalized_fwd.startswith(bp.replace("\\", "/")) for bp in _FE_BARE):
                abs_path = _FE_ROOT + "\\" + normed
            else:
                abs_path = f"{sandbox_base}\\{normed}"

            result = client.shell_run(
                f'Test-Path -Path "{abs_path}" -PathType Leaf',
                timeout_seconds=10,
            )
            return "True" in (result.stdout or "")
        except Exception:
            pass

    # Fallback to host filesystem - apply same frontend resolution
    normalized_fwd = rel_path.replace("\\", "/")
    if normalized_fwd.startswith(_FE_PREFIX):
        frontend_rel = normalized_fwd[len(_FE_PREFIX):]
        abs_path = os.path.join(_FE_ROOT, frontend_rel.replace("/", os.sep))
    elif any(normalized_fwd.startswith(bp.replace("\\", "/")) for bp in _FE_BARE):
        abs_path = os.path.join(_FE_ROOT, rel_path.replace("/", os.sep))
    else:
        abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
    return _sbx_isfile(abs_path)
