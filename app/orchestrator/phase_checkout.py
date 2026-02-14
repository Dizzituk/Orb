# FILE: app/orchestrator/phase_checkout.py
"""
Phase Checkout — Stage 9 Verification Orchestrator.

Runs all verification checks after a phase's segments complete, aggregates
results, determines failure routing, and saves the outcome. The actual
check implementations live in phase_checkout_checks.py.

v1.0 (2026-02-14): Initial implementation — Stage 9.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

from app.pot_spec.grounded.size_models import MAX_FILE_LINES
from .phase_checkout_models import (
    FailureRouting,
    PhaseCheckoutResult,
)
from .phase_checkout_checks import (
    check_output_file_sizes,
    check_skeleton_contracts,
    run_boot_test,
    map_file_to_segment,
)

logger = logging.getLogger(__name__)

PHASE_CHECKOUT_BUILD_ID = "2026-02-14-v1.0-initial"
print(f"[PHASE_CHECKOUT_LOADED] BUILD_ID={PHASE_CHECKOUT_BUILD_ID}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_phase_checkout(
    job_id: str,
    job_dir: str,
    state: Any,  # JobState
    manifest: Any,  # SegmentManifest
    skeleton: Any = None,  # SkeletonContractSet
    sandbox_base: str = r"D:\Orb",
    attempt: int = 1,
    emit: Optional[Any] = None,
) -> PhaseCheckoutResult:
    """
    Run full Phase Checkout (Stage 9) verification.

    Called from segment_loop.py after all segments complete.
    Runs: size validation → contract check → boot test.
    Returns PhaseCheckoutResult with pass/fail and routing decision.
    """
    start_time = time.time()
    _emit = emit or (lambda msg: None)

    result = PhaseCheckoutResult(job_id=job_id, attempt=attempt)

    _emit(f"\n{'='*50}")
    _emit(f"🏁 PHASE CHECKOUT — Stage 9 Verification (attempt {attempt}/3)")

    # --- Check 1: Output file size validation ---
    _emit("📏 Check 1: Output file size validation...")
    result.size_validation = check_output_file_sizes(state, sandbox_base)
    result.checks_run.append("size_validation")

    if result.size_validation.status == "fail":
        _emit(
            f"  ❌ SIZE FAIL: {len(result.size_validation.violations)} file(s) "
            f"exceed constraints"
        )
        for v in result.size_validation.violations:
            _emit(f"    • {v.file_path}: {v.line_count} lines / {v.kb_size} KB "
                  f"[{v.violation_type}] (seg: {v.produced_by_segment})")
    else:
        _emit(f"  ✅ Size check passed ({result.size_validation.files_checked} files)")

    # --- Check 2: Skeleton contract verification ---
    if skeleton:
        _emit("📋 Check 2: Skeleton contract verification...")
        result.contract_check = check_skeleton_contracts(state, skeleton, sandbox_base)
        result.checks_run.append("contract_check")

        if result.contract_check.status == "fail":
            _emit(
                f"  ❌ CONTRACT FAIL: {len(result.contract_check.violations)} violation(s)"
            )
            for v in result.contract_check.violations:
                _emit(f"    • [{v.segment_id}] {v.violation_type}: {v.detail}")
        else:
            _emit("  ✅ Contract check passed")
    else:
        _emit("📋 Check 2: Skeleton contract verification — SKIPPED (no skeleton)")

    # --- Check 3: Boot test ---
    _emit("🔧 Check 3: Application boot test...")
    result.boot_test = run_boot_test(sandbox_base)
    result.checks_run.append("boot_test")

    if result.boot_test.status == "pass":
        _emit("  ✅ Boot test passed — application starts cleanly")
    elif result.boot_test.status == "fail":
        _emit(f"  ❌ Boot test FAILED: {result.boot_test.error_summary}")
        if result.boot_test.traceback_file:
            _emit(f"    Failing file: {result.boot_test.traceback_file}")
            # Map to segment
            result.boot_test.traceback_segment = map_file_to_segment(
                result.boot_test.traceback_file, state
            )
            if result.boot_test.traceback_segment:
                _emit(f"    Produced by: {result.boot_test.traceback_segment}")
    else:
        _emit(f"  ⚠️ Boot test error: {result.boot_test.error_summary}")

    # --- Aggregate and route ---
    all_passed = (
        (result.size_validation and result.size_validation.status == "pass")
        and (not result.contract_check or result.contract_check.status == "pass")
        and (result.boot_test and result.boot_test.status == "pass")
    )

    if all_passed:
        result.status = "pass"
        _emit("\n✅ PHASE CHECKOUT PASSED — all checks green")
    else:
        result.status = "fail"
        result.routing = _determine_failure_routing(result, state)
        _emit(f"\n❌ PHASE CHECKOUT FAILED → route to {result.routing.target_stage}")
        if result.routing.target_segment:
            _emit(f"  Target segment: {result.routing.target_segment}")
        _emit(f"  Reason: {result.routing.reason}")

    elapsed_ms = int((time.time() - start_time) * 1000)
    result.duration_ms = elapsed_ms
    _emit(f"  Duration: {elapsed_ms}ms")

    _save_checkout_result(result, job_dir)
    return result


# =============================================================================
# FAILURE ROUTING
# =============================================================================

def _determine_failure_routing(
    result: PhaseCheckoutResult,
    state: Any,
) -> FailureRouting:
    """
    Diagnose what failed and decide where to route the retry.

    Priority: size violations → contract violations → boot failures.
    """
    # Size violations → re-decompose at SpecGate
    if result.size_validation and result.size_validation.violations:
        worst = result.size_validation.violations[0]
        return FailureRouting(
            target_stage="stage_4_specgate",
            target_segment=worst.produced_by_segment,
            target_file=worst.file_path,
            reason=(
                f"File '{worst.file_path}' is {worst.line_count} lines "
                f"(cap: {MAX_FILE_LINES}). Needs re-decomposition."
            ),
        )

    # Contract violations → re-run architecture
    if result.contract_check and result.contract_check.violations:
        worst = result.contract_check.violations[0]
        return FailureRouting(
            target_stage="stage_5_critical",
            target_segment=worst.segment_id,
            reason=(
                f"Contract violation in {worst.segment_id}: "
                f"{worst.violation_type} — {worst.detail}"
            ),
        )

    # Boot failures → route based on error type
    if result.boot_test and result.boot_test.status == "fail":
        err = result.boot_test.error_summary.lower()
        failing_seg = map_file_to_segment(
            result.boot_test.traceback_file, state
        )

        if "syntaxerror" in err:
            return FailureRouting(
                target_stage="stage_8_overwatcher",
                target_segment=failing_seg,
                target_file=result.boot_test.traceback_file,
                reason=f"Syntax error in {result.boot_test.traceback_file}",
            )

        return FailureRouting(
            target_stage="stage_5_critical",
            target_segment=failing_seg,
            target_file=result.boot_test.traceback_file,
            reason=f"Boot failure: {result.boot_test.error_summary[:200]}",
        )

    return FailureRouting(
        target_stage="stage_5_critical",
        reason="Unknown failure — re-run architecture generation",
    )


# =============================================================================
# PERSISTENCE
# =============================================================================

def _save_checkout_result(result: PhaseCheckoutResult, job_dir: str) -> None:
    """Save Phase Checkout result to job directory."""
    out_path = os.path.join(job_dir, "phase_checkout_result.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("[phase_checkout] Result saved to %s", out_path)
    except Exception as exc:
        logger.warning("[phase_checkout] Failed to save result: %s", exc)
