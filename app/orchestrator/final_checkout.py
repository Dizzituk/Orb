# FILE: app/orchestrator/final_checkout.py
"""
Final Project Checkout — Stage 10 (Autonomous Closer).

Project-level verification that runs after ALL phases complete.
Unlike Phase Checkout (Stage 9) which is a gatekeeper, Final Checkout
is an autonomous closer — every check that finds a problem attempts to
FIX it before moving on, using the three-strike rule.

Checks (in order):
1. Spec Coverage (gate) — verify all spec files exist.
   If missing: generate the file from spec + context, write to sandbox.
2. Cross-Phase Consistency (fix-on-find) — find import/signature drift.
   If found: read both files, fix the consumer, write to sandbox.
3. Boot Test with Fix Loop (gate) — surgical import/syntax repair.
4. AI Review (advisory with auto-fix) — quality/security/performance.
   If critical issues found: generate targeted fixes, apply, re-score.
5. Final Boot Confirmation — one last clean boot after all fixes.

Three-Strike Rule (via StrikeTracker):
  Strike 1: Error occurs → attempt fix.
  Strike 2: Same error → MUST use different strategy.
  Strike 3: Same error → hard stop, write review for human.
  Different error → strikes reset to 1.

Every attempt is recorded for RAG memory ingestion.

Token cost management:
  - No full codebase scan. Targeted reads only.
  - Boot test reads only failing files (from traceback).
  - AI review samples max 12 high-risk files, 8K chars each.
  - Surgical fixes write only the specific lines that need changing.

v3.0 (2026-02-15): Autonomous closer with StrikeTracker integration.
v2.0 (2026-02-15): Fix-loop boot test, surgical import repair, AI review.
v1.0 (2026-02-14): Initial implementation — Stage 10.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .construction_planner_models import ConstructionPlan
from .construction_skeleton import verify_phase_deliverables
from .phase_checkout_checks import (
    run_boot_test_with_fix_loop,
    _read_file_via_sandbox,
)
from .strike_tracker import (
    StrikeTracker,
    StrikeVerdict,
    FixOutcome,
    StrikeRecord,
)
from app.pot_spec.grounded.size_models import MAX_FILE_LINES

logger = logging.getLogger(__name__)

FINAL_CHECKOUT_BUILD_ID = "2026-02-15-v3.0-autonomous-closer"
print(f"[FINAL_CHECKOUT_LOADED] BUILD_ID={FINAL_CHECKOUT_BUILD_ID}")


# =============================================================================
# RESULT MODELS
# =============================================================================

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


@dataclass
class FinalCheckoutResult:
    """Complete result of Final Checkout (Stage 10)."""
    job_id: str
    status: str = "pending"  # "pass", "fail", "error"
    boot_test_status: str = ""
    spec_coverage: Optional[SpecCoverageResult] = None
    cross_phase: Optional[CrossPhaseResult] = None
    ai_review: Optional[AIReviewResult] = None
    total_files_built: int = 0
    total_phases: int = 0
    duration_ms: int = 0
    timestamp: str = ""
    checks_run: List[str] = field(default_factory=list)
    strike_records: List[Dict[str, Any]] = field(default_factory=list)  # v3.0: RAG data

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
            "ai_review": self.ai_review.to_dict() if self.ai_review else None,
            "total_files_built": self.total_files_built,
            "total_phases": self.total_phases,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "checks_run": self.checks_run,
            "strike_records": self.strike_records,
        }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_final_checkout(
    job_id: str,
    plan: ConstructionPlan,
    original_file_scope: List[str],
    job_dir: str,
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
    start = time.time()
    _emit = emit or (lambda msg: None)
    result = FinalCheckoutResult(job_id=job_id, total_phases=plan.total_phases)
    all_strike_records: List[StrikeRecord] = []

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
    return result


# =============================================================================
# CHECK 1: SPEC COVERAGE WITH AUTO-GENERATION
# =============================================================================

async def _check_spec_coverage_with_fixes(
    file_scope: List[str],
    sandbox_base: str,
    sandbox_client: Any,
    original_spec: Optional[str],
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> SpecCoverageResult:
    """
    Verify all spec files exist. If missing, attempt to generate them.

    Uses StrikeTracker: if generation fails for the same file repeatedly,
    escalate after 3 strikes.
    """
    _emit = emit or (lambda msg: None)

    # First pass: check what exists
    missing = []
    found = 0
    for rel_path in file_scope:
        exists = _file_exists_on_sandbox(sandbox_client, rel_path, sandbox_base)
        if exists:
            found += 1
        else:
            missing.append(rel_path)

    if not missing:
        tracker.record_resolution()
        return SpecCoverageResult(
            status="pass",
            expected_files=len(file_scope),
            found_files=found,
        )

    _emit(f"  {len(missing)} file(s) missing from spec — attempting generation...")
    generated = []

    for rel_path in missing:
        fix_start = time.time()
        error_text = f"missing_file:{rel_path}"
        verdict = tracker.report_error(error_text)

        if verdict == StrikeVerdict.HARD_STOP:
            _emit(f"    [STRIKE 3] Hard stop on {rel_path} — cannot generate")
            tracker.record_hard_stop(error_text)
            break

        strategy = "generate_from_spec" if verdict == StrikeVerdict.PROCEED else "generate_stub"
        _emit(f"    [{verdict.value}] Generating {rel_path} (strategy: {strategy})")

        success = await _generate_missing_file(
            rel_path=rel_path,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            original_spec=original_spec,
            strategy=strategy,
            emit=_emit,
        )

        duration = int((time.time() - fix_start) * 1000)

        if success:
            generated.append(rel_path)
            found += 1
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Generated {rel_path}",
                outcome=FixOutcome.RESOLVED,
                duration_ms=duration,
            )
        else:
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Failed to generate {rel_path}",
                outcome=FixOutcome.SAME_ERROR,
                duration_ms=duration,
            )

    # Recheck
    still_missing = [f for f in missing if f not in generated]

    if not still_missing:
        tracker.record_resolution()

    return SpecCoverageResult(
        status="pass" if not still_missing else "fail",
        expected_files=len(file_scope),
        found_files=found,
        missing_files=still_missing,
        files_generated=generated,
    )


async def _generate_missing_file(
    rel_path: str,
    sandbox_base: str,
    sandbox_client: Any,
    original_spec: Optional[str],
    strategy: str,
    emit: Optional[Callable] = None,
) -> bool:
    """
    Generate a missing file on the sandbox.

    strategy='generate_from_spec': Use LLM with spec context.
    strategy='generate_stub': Create minimal stub (fallback).
    """
    _emit = emit or (lambda msg: None)

    if strategy == "generate_stub":
        # Minimal stub — just enough to not break imports
        if rel_path.endswith(".py"):
            stub = f'# STUB: Auto-generated by Final Checkout\n# TODO: Implement per spec\n"""Stub for {rel_path}"""\n'
        elif rel_path.endswith("__init__.py"):
            stub = f'# Auto-generated __init__.py\n'
        else:
            stub = f'// STUB: Auto-generated by Final Checkout\n'

        return _write_file_to_sandbox(sandbox_client, rel_path, stub, sandbox_base)

    # strategy == "generate_from_spec": Use LLM
    if not original_spec:
        _emit(f"      No spec available — falling back to stub")
        return _write_file_to_sandbox_stub(sandbox_client, rel_path, sandbox_base)

    try:
        from app.providers.registry import llm_call

        prompt = (
            f"Generate the complete file content for: {rel_path}\n\n"
            f"Based on this specification:\n{original_spec[:4000]}\n\n"
            f"Output ONLY the file content, no markdown fences."
        )

        result = await llm_call(
            provider_id=os.getenv("ASTRA_FIX_PROVIDER", "anthropic"),
            model_id=os.getenv("ASTRA_FIX_MODEL", "claude-sonnet-4-5-20250929"),
            messages=[{"role": "user", "content": prompt}],
            system_prompt="You are a code generation agent. Output only file content.",
            max_tokens=8192,
            timeout_seconds=60,
        )

        content = result.content if result else None
        if content:
            return _write_file_to_sandbox(sandbox_client, rel_path, content, sandbox_base)

        return False
    except Exception as exc:
        _emit(f"      LLM generation failed: {exc}")
        return False


# =============================================================================
# CHECK 2: CROSS-PHASE WITH AUTO-FIX
# =============================================================================

async def _check_cross_phase_with_fixes(
    plan: ConstructionPlan,
    sandbox_base: str,
    sandbox_client: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
    state: Any = None,
    manifest: Any = None,
) -> CrossPhaseResult:
    """
    Verify cross-phase contracts and fix violations.

    Three layers of checking:
    1. Deliverable existence: does each promised file exist?
    2. Import resolution: do cross-segment imports resolve correctly?
    3. Interface contracts: do exposes/consumes match actual definitions?

    Fix strategy:
    - missing_deliverable: generate the file from spec
    - import_mismatch: read the provider file to see what it exports,
      fix the consumer's import to use the correct name. Provider is
      ground truth (written first), consumer is what gets fixed.
    - missing_export: add the missing name to the provider's __all__ or
      re-export it from __init__.py
    """
    _emit = emit or (lambda msg: None)
    violations = []
    fixes_applied = []

    # --- Layer 1: Deliverable existence ---
    for phase in plan.phases:
        result = verify_phase_deliverables(plan, phase, sandbox_base)
        if result["status"] == "fail":
            for mf in result.get("missing", []):
                violation = {
                    "phase_id": phase.phase_id,
                    "violation_type": "missing_deliverable",
                    "detail": f"Phase {phase.phase_number} promised '{mf}' but it's missing",
                }

                error_text = f"cross_phase_missing:{mf}"
                verdict = tracker.report_error(error_text)

                if verdict == StrikeVerdict.HARD_STOP:
                    violations.append(violation)
                    tracker.record_hard_stop(error_text)
                    continue

                fix_start = time.time()
                exists = _file_exists_on_sandbox(sandbox_client, mf, sandbox_base)

                if exists:
                    tracker.record_attempt(
                        error_detail=error_text,
                        fix_strategy="path_verification",
                        fix_description=f"File exists on sandbox: {mf}",
                        outcome=FixOutcome.RESOLVED,
                        duration_ms=int((time.time() - fix_start) * 1000),
                    )
                    fixes_applied.append({
                        "violation": violation,
                        "fix": "File found on sandbox (host path mismatch)",
                    })
                else:
                    tracker.record_attempt(
                        error_detail=error_text,
                        fix_strategy="verify_existence",
                        fix_description=f"File not found on sandbox either: {mf}",
                        outcome=FixOutcome.SAME_ERROR,
                        duration_ms=int((time.time() - fix_start) * 1000),
                    )
                    violations.append(violation)

    # --- Layer 2: Import resolution + interface contracts ---
    # Use the integration check's detection, then fix what it finds
    if state and manifest:
        import_issues = await _detect_and_fix_integration_issues(
            state=state,
            manifest=manifest,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            tracker=tracker,
            emit=_emit,
        )
        for issue_dict in import_issues.get("unresolved", []):
            violations.append(issue_dict)
        for fix_dict in import_issues.get("fixed", []):
            fixes_applied.append(fix_dict)

    if not violations:
        tracker.record_resolution()

    return CrossPhaseResult(
        status="pass" if not violations else "fail",
        phases_checked=len(plan.phases),
        violations=violations,
        fixes_applied=fixes_applied,
    )


async def _detect_and_fix_integration_issues(
    state: Any,
    manifest: Any,
    sandbox_base: str,
    sandbox_client: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run the integration check's import resolution and interface contract
    checks, then fix each issue found.

    Fix logic (provider is ground truth, consumer gets fixed):
    - import_resolution error (name not found in target):
      1. Read the provider file, get its actual exports
      2. Find the closest match to the missing name
      3. Rewrite the consumer's import line with the correct name
    - interface_contract error (expose promise broken):
      1. Check if the name exists under a different spelling
      2. Add re-export to __init__.py if needed
    """
    _emit = emit or (lambda msg: None)
    result = {"fixed": [], "unresolved": []}

    try:
        from .integration_check import (
            run_integration_check,
            IntegrationIssue,
        )
        from .ast_helpers import get_all_defined_names
    except ImportError as exc:
        _emit(f"  [SKIP] Cannot import integration_check: {exc}")
        return result

    # Run the read-only integration check to detect issues
    try:
        job_dir = os.path.join(sandbox_base, "jobs", "jobs")
        check_result = run_integration_check(
            manifest=manifest,
            state=state,
            job_dir=job_dir,
            on_progress=lambda msg: _emit(f"    {msg}"),
        )
    except Exception as exc:
        _emit(f"  [SKIP] Integration check failed: {exc}")
        return result

    errors = [i for i in check_result.tier1_issues if i.severity == "error"]
    if not errors:
        _emit("  [OK] No cross-segment integration errors")
        return result

    _emit(f"  {len(errors)} integration error(s) found — attempting fixes...")

    for issue in errors:
        error_text = f"{issue.check_type}:{issue.expected[:80]}"
        verdict = tracker.report_error(error_text)

        if verdict == StrikeVerdict.HARD_STOP:
            result["unresolved"].append({
                "check_type": issue.check_type,
                "detail": issue.message,
                "file_a": issue.file_a,
                "file_b": issue.file_b,
            })
            tracker.record_hard_stop(error_text)
            continue

        fix_start = time.time()
        strategies_tried = tracker.get_strategies_tried()

        if issue.check_type == "import_resolution":
            fixed = await _fix_import_resolution_issue(
                issue=issue,
                sandbox_base=sandbox_base,
                sandbox_client=sandbox_client,
                verdict=verdict,
                strategies_tried=strategies_tried,
                emit=_emit,
            )
        elif issue.check_type == "interface_contract":
            fixed = await _fix_interface_contract_issue(
                issue=issue,
                sandbox_base=sandbox_base,
                sandbox_client=sandbox_client,
                verdict=verdict,
                emit=_emit,
            )
        else:
            fixed = False
            _emit(f"    [SKIP] No auto-fix for {issue.check_type}")

        duration = int((time.time() - fix_start) * 1000)
        strategy = "import_rewrite" if issue.check_type == "import_resolution" else "contract_fix"

        if fixed:
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Fixed {issue.check_type} in {issue.file_b}",
                outcome=FixOutcome.RESOLVED,
                duration_ms=duration,
            )
            result["fixed"].append({
                "check_type": issue.check_type,
                "detail": issue.message,
                "fix": f"Rewrote import in {issue.file_b}",
            })
        else:
            tracker.record_attempt(
                error_detail=error_text,
                fix_strategy=strategy,
                fix_description=f"Could not fix {issue.check_type}",
                outcome=FixOutcome.SAME_ERROR,
                duration_ms=duration,
            )
            result["unresolved"].append({
                "check_type": issue.check_type,
                "detail": issue.message,
                "file_a": issue.file_a,
                "file_b": issue.file_b,
            })

    return result


async def _fix_import_resolution_issue(
    issue: Any,
    sandbox_base: str,
    sandbox_client: Any,
    verdict: StrikeVerdict,
    strategies_tried: List[str],
    emit: Optional[Callable] = None,
) -> bool:
    """
    Fix an import resolution issue.

    The provider file (file_a) is ground truth. The consumer file (file_b)
    imports a name that doesn't exist in the provider. We need to find
    the correct name and rewrite the consumer's import.

    Strike 1: Find closest match in provider exports, rewrite import.
    Strike 2: Ask LLM to fix the import with full context of both files.
    """
    _emit = emit or (lambda msg: None)

    consumer_file = issue.file_b
    provider_file = issue.file_a

    if not consumer_file or not os.path.isfile(consumer_file):
        # Try reading from sandbox
        if sandbox_client:
            consumer_content = _read_file_via_sandbox(
                sandbox_client, consumer_file, sandbox_base
            )
        else:
            return False
    else:
        try:
            with open(consumer_file, "r", encoding="utf-8", errors="replace") as f:
                consumer_content = f.read()
        except Exception:
            return False

    if not consumer_content:
        return False

    # Parse what name is missing from the issue
    expected = issue.expected  # e.g. "Name 'foo' should be defined in 'bar.py'"
    import re as _re
    name_match = _re.search(r"Name '([^']+)'", expected)
    if not name_match:
        return False
    missing_name = name_match.group(1)

    # Strategy 1: Find closest match in provider
    if verdict == StrikeVerdict.PROCEED or "import_rewrite_fuzzy" not in strategies_tried:
        try:
            from .ast_helpers import get_all_defined_names
        except ImportError:
            return False

        if provider_file and os.path.isfile(provider_file):
            provider_names = get_all_defined_names(provider_file)
        elif sandbox_client and provider_file:
            provider_content = _read_file_via_sandbox(
                sandbox_client, provider_file, sandbox_base
            )
            if provider_content:
                # Extract names from content using AST
                import ast
                try:
                    tree = ast.parse(provider_content)
                    provider_names = set()
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            provider_names.add(node.name)
                        elif isinstance(node, ast.ClassDef):
                            provider_names.add(node.name)
                        elif isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name):
                                    provider_names.add(target.id)
                except SyntaxError:
                    provider_names = set()
            else:
                return False
        else:
            return False

        # Find closest match
        best_match = _find_closest_name(missing_name, provider_names)
        if best_match:
            _emit(f"    Rewriting import: '{missing_name}' -> '{best_match}' in {os.path.basename(consumer_file)}")
            new_content = consumer_content.replace(
                f"import {missing_name}", f"import {best_match}"
            ).replace(
                f"{missing_name},", f"{best_match},"
            ).replace(
                f", {missing_name}", f", {best_match}"
            )

            if new_content != consumer_content:
                return _write_file_to_sandbox(
                    sandbox_client, consumer_file, new_content, sandbox_base
                )

    # Strategy 2 (strike 2): Ask LLM
    if verdict == StrikeVerdict.CHANGE_APPROACH:
        _emit(f"    [STRIKE 2] Asking LLM to fix import in {os.path.basename(consumer_file)}")
        return await _llm_fix_import(
            consumer_file=consumer_file,
            consumer_content=consumer_content,
            provider_file=provider_file,
            missing_name=missing_name,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
        )

    return False


async def _fix_interface_contract_issue(
    issue: Any,
    sandbox_base: str,
    sandbox_client: Any,
    verdict: StrikeVerdict,
    emit: Optional[Callable] = None,
) -> bool:
    """
    Fix an interface contract violation.

    If a segment promised to expose a name but doesn't, check if it
    exists under a slightly different name and add a re-export.
    """
    _emit = emit or (lambda msg: None)
    # Interface contract fixes are less common and more complex.
    # For now, log and skip — the boot test will catch real breakage.
    _emit(f"    [INFO] Interface contract fix deferred to boot test")
    return False


def _find_closest_name(target: str, candidates: set) -> Optional[str]:
    """Find the closest matching name from a set of candidates."""
    if not candidates:
        return None

    target_lower = target.lower()

    # Exact match (case-insensitive)
    for c in candidates:
        if c.lower() == target_lower:
            return c

    # Substring match
    for c in candidates:
        if target_lower in c.lower() or c.lower() in target_lower:
            return c

    # Word overlap
    import re as _re
    target_words = set(_re.findall(r'[a-z]+', target_lower))
    best = None
    best_score = 0
    for c in candidates:
        c_words = set(_re.findall(r'[a-z]+', c.lower()))
        overlap = len(target_words & c_words)
        if overlap > best_score:
            best_score = overlap
            best = c

    return best if best_score > 0 else None


async def _llm_fix_import(
    consumer_file: str,
    consumer_content: str,
    provider_file: str,
    missing_name: str,
    sandbox_base: str,
    sandbox_client: Any,
) -> bool:
    """Ask LLM to fix an import mismatch between two files."""
    try:
        from app.providers.registry import llm_call

        # Read provider file for context
        provider_content = ""
        if provider_file and os.path.isfile(provider_file):
            with open(provider_file, "r", encoding="utf-8", errors="replace") as f:
                provider_content = f.read(6000)
        elif sandbox_client:
            provider_content = _read_file_via_sandbox(
                sandbox_client, provider_file, sandbox_base
            ) or ""
            provider_content = provider_content[:6000]

        prompt = (
            f"Fix the import error in the CONSUMER file.\n\n"
            f"ERROR: '{missing_name}' is not defined in the provider file.\n\n"
            f"PROVIDER FILE ({os.path.basename(provider_file or 'unknown')}): "
            f"Shows what names are actually available:\n"
            f"```python\n{provider_content[:4000]}\n```\n\n"
            f"CONSUMER FILE ({os.path.basename(consumer_file)}): "
            f"Needs the import fixed:\n"
            f"```python\n{consumer_content[:4000]}\n```\n\n"
            f"Output ONLY the complete fixed consumer file. No explanations."
        )

        result = await llm_call(
            provider_id=os.getenv("ASTRA_FIX_PROVIDER", "anthropic"),
            model_id=os.getenv("ASTRA_FIX_MODEL", "claude-sonnet-4-5-20250929"),
            messages=[{"role": "user", "content": prompt}],
            system_prompt="Fix the import. Output only the complete file content.",
            max_tokens=8192,
            timeout_seconds=60,
        )

        fixed_content = result.content if result else None
        if fixed_content and len(fixed_content) > 50:
            return _write_file_to_sandbox(
                sandbox_client, consumer_file, fixed_content, sandbox_base
            )

        return False
    except Exception:
        return False


# =============================================================================
# CHECK 3: BOOT TEST WITH STRIKE TRACKING
# =============================================================================

async def _run_boot_with_strike_tracking(
    sandbox_base: str,
    state: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> Any:
    """
    Run boot test with StrikeTracker integration.

    The boot fix loop in phase_checkout_checks already implements retry
    logic. Here we wrap it with strike tracking for RAG recording.
    """
    _emit = emit or (lambda msg: None)

    boot = await run_boot_test_with_fix_loop(
        sandbox_base=sandbox_base,
        state=state,
        emit=_emit,
    )

    if boot.status == "pass":
        tracker.record_resolution()
    else:
        error_text = boot.error_summary or "Unknown boot failure"
        tracker.report_error(error_text)
        tracker.record_attempt(
            error_detail=error_text,
            fix_strategy="boot_fix_loop",
            fix_description=f"Boot fix loop exhausted (file: {boot.traceback_file or 'unknown'})",
            outcome=FixOutcome.SAME_ERROR,
        )
        tracker.record_hard_stop(error_text)

    return boot


# =============================================================================
# CHECK 4: AI REVIEW WITH AUTO-FIX
# =============================================================================

# Files to prioritise for review (highest risk, minimal token cost)
_REVIEW_PRIORITY_PATTERNS = [
    r"main\.py$",                    # Entry point
    r"__init__\.py$",                # Package init (import chains)
    r"auth|security|password",       # Security-sensitive
    r"api|router|endpoint|views",    # External-facing
    r"database|models|migration",    # Data integrity
    r"config|settings|env",          # Configuration
]

_MAX_REVIEW_FILES = 12       # Cap total files reviewed
_MAX_REVIEW_CHARS = 8000     # Cap per-file content in prompt
_REVIEW_MODEL = "claude-sonnet-4-5-20250929"  # Sonnet for cost efficiency


async def _run_ai_review_with_fixes(
    original_spec: Optional[str],
    file_scope: List[str],
    sandbox_base: str,
    sandbox_client: Any,
    tracker: StrikeTracker,
    emit: Optional[Callable] = None,
) -> AIReviewResult:
    """
    Run AI review. If critical issues are found, attempt targeted fixes.

    v3.0: After initial review, if score < 6/10 and critical issues exist,
    send each critical issue back to the LLM with the file content for
    a targeted fix. Then re-score. Uses strike rule for repeated failures.
    """
    _emit = emit or (lambda msg: None)

    if not original_spec:
        tracker.record_resolution()
        return AIReviewResult(
            status="skipped",
            notes="No original spec provided for review comparison",
        )

    # --- Initial review ---
    review = await _run_ai_review_pass(
        original_spec=original_spec,
        file_scope=file_scope,
        sandbox_base=sandbox_base,
        emit=_emit,
    )

    if review.status in ("skipped", "error"):
        tracker.record_resolution()
        return review

    if review.overall_score >= 6.0 or not review.critical_issues:
        tracker.record_resolution()
        return review

    # --- Score below threshold — attempt fixes ---
    _emit(f"  Score {review.overall_score}/10 with {len(review.critical_issues)} critical issues — attempting fixes...")

    for issue in review.critical_issues[:3]:  # Cap at 3 fixes per round
        error_text = f"ai_review_issue:{issue[:100]}"
        verdict = tracker.report_error(error_text)

        if verdict == StrikeVerdict.HARD_STOP:
            _emit(f"    [STRIKE 3] Hard stop on AI review issue")
            tracker.record_hard_stop(error_text)
            break

        strategy = "targeted_fix" if verdict == StrikeVerdict.PROCEED else "conservative_fix"
        _emit(f"    [{verdict.value}] Fixing: {issue[:80]}...")

        fix_start = time.time()
        fixed = await _apply_ai_review_fix(
            issue=issue,
            file_scope=file_scope,
            sandbox_base=sandbox_base,
            sandbox_client=sandbox_client,
            original_spec=original_spec,
            strategy=strategy,
            emit=_emit,
        )
        duration = int((time.time() - fix_start) * 1000)

        tracker.record_attempt(
            error_detail=error_text,
            fix_strategy=strategy,
            fix_description=f"AI review fix for: {issue[:100]}",
            outcome=FixOutcome.RESOLVED if fixed else FixOutcome.SAME_ERROR,
            duration_ms=duration,
        )

    # --- Re-score after fixes ---
    _emit("  Re-scoring after fixes...")
    review = await _run_ai_review_pass(
        original_spec=original_spec,
        file_scope=file_scope,
        sandbox_base=sandbox_base,
        emit=_emit,
    )

    if review.overall_score >= 6.0:
        tracker.record_resolution()
    else:
        tracker.record_hard_stop(f"AI review score still {review.overall_score}/10")

    return review


async def _run_ai_review_pass(
    original_spec: Optional[str],
    file_scope: List[str],
    sandbox_base: str,
    emit: Optional[Callable] = None,
) -> AIReviewResult:
    """Single pass of AI review (no fix attempts)."""
    _emit = emit or (lambda msg: None)

    # Select files for review
    priority_files: List[str] = []
    other_files: List[str] = []

    for rel_path in file_scope:
        if any(re.search(pat, rel_path, re.IGNORECASE) for pat in _REVIEW_PRIORITY_PATTERNS):
            priority_files.append(rel_path)
        else:
            other_files.append(rel_path)

    review_files = priority_files[:_MAX_REVIEW_FILES]
    remaining_slots = _MAX_REVIEW_FILES - len(review_files)
    if remaining_slots > 0:
        review_files.extend(other_files[:remaining_slots])

    if not review_files:
        return AIReviewResult(status="skipped", notes="No files available for review")

    # Read file contents
    file_contents: Dict[str, str] = {}
    for rel_path in review_files:
        abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(_MAX_REVIEW_CHARS)
                file_contents[rel_path] = content
            except Exception:
                pass

    if not file_contents:
        return AIReviewResult(status="skipped", notes="Could not read any files for review")

    _emit(f"  Reviewing {len(file_contents)} file(s)")

    # Build prompt and call LLM
    prompt = _build_ai_review_prompt(original_spec, file_contents)

    try:
        from app.providers.registry import llm_call

        provider_id = os.getenv("ASTRA_REVIEW_PROVIDER", "anthropic")
        model_id = os.getenv("ASTRA_REVIEW_MODEL", _REVIEW_MODEL)

        llm_result = await llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_REVIEW_SYSTEM_PROMPT,
            max_tokens=4096,
            timeout_seconds=120,
        )

        response = llm_result.content if llm_result else None
        if not response:
            return AIReviewResult(status="error", notes="Empty LLM response")

        return _parse_review_response(response, len(file_contents))

    except Exception as exc:
        logger.warning("[final_checkout] AI review failed: %s", exc)
        return AIReviewResult(status="error", notes=f"AI review failed: {exc}")


async def _apply_ai_review_fix(
    issue: str,
    file_scope: List[str],
    sandbox_base: str,
    sandbox_client: Any,
    original_spec: Optional[str],
    strategy: str,
    emit: Optional[Callable] = None,
) -> bool:
    """
    Apply a targeted fix for a single AI review issue.

    1. Parse the issue description to identify which file(s) are affected
    2. Read those files from sandbox/host
    3. Ask LLM for a targeted fix (only the affected section)
    4. Write the fix back to sandbox

    strategy='targeted_fix' (strike 1): Fix only the specific issue.
    strategy='conservative_fix' (strike 2): Broader context, safer changes.
    """
    _emit = emit or (lambda msg: None)

    # --- Step 1: Identify affected file(s) from the issue description ---
    affected_files = _extract_files_from_issue(issue, file_scope)
    if not affected_files:
        _emit(f"      Could not identify affected file for: {issue[:60]}")
        return False

    _emit(f"      Targeting {len(affected_files)} file(s): {', '.join(os.path.basename(f) for f in affected_files)}")

    # --- Step 2: Read file contents ---
    file_contents: Dict[str, str] = {}
    for rel_path in affected_files:
        content = None
        abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))

        if os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                pass

        if not content and sandbox_client:
            content = _read_file_via_sandbox(sandbox_client, abs_path, sandbox_base)

        if content:
            file_contents[rel_path] = content

    if not file_contents:
        _emit(f"      Could not read any affected files")
        return False

    # --- Step 3: Ask LLM for targeted fix ---
    try:
        from app.providers.registry import llm_call

        # Build prompt based on strategy
        if strategy == "conservative_fix":
            system = (
                "You are a careful code reviewer. Make the MINIMUM change needed "
                "to fix the issue. Do not refactor or restructure. If you are not "
                "confident in the fix, output the file unchanged."
            )
            context_cap = 6000
        else:
            system = (
                "You are a code fixer. Fix the specific issue described. "
                "Output ONLY the complete fixed file content. No explanations."
            )
            context_cap = 4000

        # Process each affected file
        any_fixed = False
        for rel_path, content in file_contents.items():
            prompt_parts = [
                f"Fix this issue in {rel_path}:\n",
                f"ISSUE: {issue}\n",
            ]

            if original_spec:
                prompt_parts.append(f"\nRELEVANT SPEC CONTEXT:\n{original_spec[:2000]}\n")

            prompt_parts.append(
                f"\nCURRENT FILE ({rel_path}):\n"
                f"```python\n{content[:context_cap]}\n```\n\n"
                f"Output ONLY the complete fixed file. No markdown fences. No explanations."
            )

            llm_result = await llm_call(
                provider_id=os.getenv("ASTRA_FIX_PROVIDER", "anthropic"),
                model_id=os.getenv("ASTRA_FIX_MODEL", "claude-sonnet-4-5-20250929"),
                messages=[{"role": "user", "content": "\n".join(prompt_parts)}],
                system_prompt=system,
                max_tokens=8192,
                timeout_seconds=60,
            )

            fixed_content = llm_result.content if llm_result else None

            # Strip markdown fences if LLM added them despite instructions
            if fixed_content:
                fixed_content = fixed_content.strip()
                if fixed_content.startswith("```"):
                    lines = fixed_content.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]
                    fixed_content = "\n".join(lines)

            # Sanity check: must be substantial and different
            if (
                fixed_content
                and len(fixed_content) > 50
                and fixed_content != content
                and abs(len(fixed_content) - len(content)) < len(content) * 0.5
            ):
                abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
                wrote = _write_file_to_sandbox(
                    sandbox_client, abs_path, fixed_content, sandbox_base
                )
                if wrote:
                    _emit(f"      Fixed {os.path.basename(rel_path)}")
                    any_fixed = True

        return any_fixed

    except Exception as exc:
        _emit(f"      LLM fix failed: {exc}")
        return False


def _extract_files_from_issue(
    issue: str,
    file_scope: List[str],
) -> List[str]:
    """
    Parse an AI review issue description to identify affected files.

    Looks for filenames, paths, and module references in the issue text,
    then matches them against the known file scope.
    """
    import re as _re

    affected = []
    issue_lower = issue.lower()

    # Direct filename mentions (e.g. "main.py", "executor.py")
    file_mentions = _re.findall(r'[\w/\\.-]+\.(?:py|ts|tsx|js|jsx)', issue)

    for mention in file_mentions:
        mention_base = os.path.basename(mention).lower()
        for scope_path in file_scope:
            if os.path.basename(scope_path).lower() == mention_base:
                if scope_path not in affected:
                    affected.append(scope_path)

    # Module references (e.g. "app.overwatcher.architecture_executor")
    module_refs = _re.findall(r'app\.[\w.]+', issue)
    for mod_ref in module_refs:
        expected_path = mod_ref.replace(".", "/") + ".py"
        for scope_path in file_scope:
            if scope_path.replace("\\", "/").endswith(expected_path):
                if scope_path not in affected:
                    affected.append(scope_path)

    # If no specific files found, try keyword matching
    if not affected:
        keywords = _re.findall(r'\b(?:auth|security|config|main|init|route|model|database)\b', issue_lower)
        for kw in keywords:
            for scope_path in file_scope:
                if kw in os.path.basename(scope_path).lower():
                    if scope_path not in affected:
                        affected.append(scope_path)
                    break  # One file per keyword

    return affected[:3]  # Cap at 3 files per issue


_REVIEW_SYSTEM_PROMPT = """\
You are a senior software reviewer performing a final quality gate assessment.
Review the code files against the original specification.

Focus on:
1. QUALITY: Code structure, error handling, naming consistency, dead code
2. SECURITY: Input validation, injection risks, hardcoded secrets, auth gaps
3. PERFORMANCE: Obvious bottlenecks, N+1 queries, unbounded loops, missing caching

Be concise and actionable. Only flag real issues, not style preferences.
Respond with valid JSON only.\
"""


def _build_ai_review_prompt(
    spec: str,
    file_contents: Dict[str, str],
) -> str:
    """Build the AI review prompt with spec + sampled files."""
    parts = [
        "# Final AI Review\n",
        "## Original Specification (summary)\n",
    ]

    spec_display = spec[:6000]
    if len(spec) > 6000:
        spec_display += "\n... [truncated]"
    parts.append(spec_display)
    parts.append("\n\n## Code Files for Review\n")

    for path, content in file_contents.items():
        parts.append(f"### `{path}`\n```python\n{content}\n```\n")

    parts.append("""
## Response Format

```json
{
  "overall_score": 7.5,
  "quality_score": 8.0,
  "security_score": 7.0,
  "performance_score": 7.5,
  "critical_issues": [
    "Brief description of critical issue 1",
    "Brief description of critical issue 2"
  ],
  "recommendations": [
    "Brief actionable recommendation 1",
    "Brief actionable recommendation 2"
  ],
  "notes": "Optional overall assessment"
}
```

Scores are 0-10. critical_issues are things that should be fixed before production.
recommendations are improvements. Keep both lists under 8 items each.
""")

    return "\n".join(parts)


def _parse_review_response(response: str, files_reviewed: int) -> AIReviewResult:
    """Parse the LLM review response."""
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning("[final_checkout] Failed to parse AI review: %s", e)
        return AIReviewResult(
            status="error",
            notes=f"Failed to parse review response: {e}",
            files_reviewed=files_reviewed,
        )

    overall = float(data.get("overall_score", 0))
    return AIReviewResult(
        status="pass" if overall >= 6.0 else "fail",
        overall_score=overall,
        quality_score=float(data.get("quality_score", 0)),
        security_score=float(data.get("security_score", 0)),
        performance_score=float(data.get("performance_score", 0)),
        critical_issues=data.get("critical_issues", [])[:8],
        recommendations=data.get("recommendations", [])[:8],
        files_reviewed=files_reviewed,
        notes=data.get("notes", ""),
    )


# =============================================================================
# HELPERS
# =============================================================================

def _file_exists_on_sandbox(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> bool:
    """Check if a file exists on the sandbox."""
    if client:
        try:
            normed = rel_path.replace("/", "\\")
            if not (normed.startswith("C:") or normed.startswith("D:")):
                abs_path = f"{sandbox_base}\\{normed}"
            else:
                abs_path = normed
            result = client.shell_run(
                f'Test-Path -Path "{abs_path}" -PathType Leaf',
                timeout_seconds=10,
            )
            return "True" in (result.stdout or "")
        except Exception:
            pass

    # Fallback to host filesystem
    abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
    return os.path.isfile(abs_path)


def _write_file_to_sandbox(
    client: Any,
    rel_path: str,
    content: str,
    sandbox_base: str,
) -> bool:
    """Write a file to the sandbox."""
    if not client:
        return False

    try:
        import base64
        normed = rel_path.replace("/", "\\")
        if not (normed.startswith("C:") or normed.startswith("D:")):
            abs_path = f"{sandbox_base}\\{normed}"
        else:
            abs_path = normed

        b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
        temp_path = abs_path + ".tmp_fc"

        # Ensure parent directory exists
        parent = "\\".join(abs_path.replace("/", "\\").split("\\")[:-1])
        client.shell_run(
            f'New-Item -ItemType Directory -Force -Path "{parent}" | Out-Null',
            timeout_seconds=10,
        )

        # Write via base64
        client.shell_run(
            f'Set-Content -Path "{temp_path}" -Value "{b64}" -NoNewline -Encoding ASCII',
            timeout_seconds=10,
        )
        client.shell_run(
            f'$b = [System.IO.File]::ReadAllText("{temp_path}"); '
            f'$bytes = [System.Convert]::FromBase64String($b); '
            f'[System.IO.File]::WriteAllBytes("{abs_path}", $bytes); '
            f'Remove-Item -Path "{temp_path}" -Force -ErrorAction SilentlyContinue; '
            f'"WRITE_OK"',
            timeout_seconds=15,
        )
        return True
    except Exception as exc:
        logger.warning("[final_checkout] write_file_to_sandbox failed: %s", exc)
        return False


def _write_file_to_sandbox_stub(
    client: Any,
    rel_path: str,
    sandbox_base: str,
) -> bool:
    """Write a minimal stub file to the sandbox."""
    if rel_path.endswith(".py"):
        stub = f'# STUB: Auto-generated by Final Checkout\n"""Stub for {rel_path}"""\n'
    elif rel_path.endswith("__init__.py"):
        stub = '# Auto-generated __init__.py\n'
    else:
        stub = f'// STUB: Auto-generated\n'
    return _write_file_to_sandbox(client, rel_path, stub, sandbox_base)


def _build_minimal_state_from_plan(
    plan: ConstructionPlan,
    sandbox_base: str,
) -> Any:
    """
    Build a minimal state-like object from a ConstructionPlan for boot fix mapping.
    """
    class _MinimalSegState:
        def __init__(self, output_files):
            self.output_files = output_files
            self.status = "complete"

    class _MinimalState:
        def __init__(self):
            self.segments = {}

    state = _MinimalState()
    for phase in plan.phases:
        state.segments[phase.phase_id] = _MinimalSegState(phase.file_scope)

    return state


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
