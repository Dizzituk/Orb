# FILE: app/orchestrator/phase_checkout.py
"""
Phase Checkout -- Stage 9 Verification Orchestrator.

Runs all verification checks after a phase's segments complete, aggregates
results, determines failure routing, and saves the outcome. The actual
check implementations live in phase_checkout_checks.py.

v2.0 (2026-02-15): Boot-fix loop. Size/contract checks become informational
    warnings -- only boot test determines pass/fail. Boot test now attempts
    deterministic fixes (bad imports, syntax) and retries up to 3 times
    before failing.
v1.0 (2026-02-14): Initial implementation -- Stage 9.
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
    run_boot_test_with_fix_loop,
    map_file_to_segment,
)

logger = logging.getLogger(__name__)

PHASE_CHECKOUT_BUILD_ID = "2026-02-16-v2.8-boot-fix-hardening"
print(f"[PHASE_CHECKOUT_LOADED] BUILD_ID={PHASE_CHECKOUT_BUILD_ID}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def run_phase_checkout(
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
    Runs: size validation (informational) -> contract check (informational)
          -> boot test with fix loop (pass/fail gate).
    Returns PhaseCheckoutResult with pass/fail and routing decision.
    """
    start_time = time.time()
    _emit = emit or (lambda msg: None)

    result = PhaseCheckoutResult(job_id=job_id, attempt=attempt)

    _emit(f"\n{'='*50}")
    _emit(f"[PHASE_CHECKOUT] Stage 9 Verification (attempt {attempt}/3)")

    # --- Collect segment output files for scoping ---
    segment_output_files = set()
    for seg_id, seg_state in state.segments.items():
        for f in (seg_state.output_files or []):
            segment_output_files.add(f)

    # --- Check 1: Output file size validation (INFORMATIONAL) ---
    _emit("[CHECK 1] Output file size validation (informational)...")

    # v6.1 FIX 3: Build baseline function sizes for deterministic refactor jobs
    _baseline_fn_sizes = None
    try:
        _manifest_path = os.path.join(job_dir, "segments", "manifest.json")
        if os.path.isfile(_manifest_path):
            import json as _json
            with open(_manifest_path, "r", encoding="utf-8") as _mf:
                _manifest_data = _json.load(_mf)
            _det_source = _manifest_data.get("deterministic_source")
            if _det_source:
                # Scan the source file for baseline function sizes
                _source_abs = os.path.join(sandbox_base, _det_source.replace("/", os.sep))
                if os.path.isfile(_source_abs):
                    import ast as _ast
                    with open(_source_abs, "r", encoding="utf-8") as _sf:
                        _src = _sf.read()
                    try:
                        _tree = _ast.parse(_src)
                        _baseline_fn_sizes = {}
                        for _n in _ast.walk(_tree):
                            if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                                if hasattr(_n, "end_lineno") and _n.end_lineno:
                                    _baseline_fn_sizes[_n.name] = _n.end_lineno - _n.lineno + 1
                        if _baseline_fn_sizes:
                            logger.info(
                                "[phase_checkout] v6.1 Loaded %d baseline function sizes from %s",
                                len(_baseline_fn_sizes), _det_source,
                            )
                    except SyntaxError:
                        pass
    except Exception as _bl_err:
        logger.debug("[phase_checkout] v6.1 Baseline sizes unavailable: %s", _bl_err)

    result.size_validation = check_output_file_sizes(state, sandbox_base, _baseline_fn_sizes)
    result.checks_run.append("size_validation")

    if result.size_validation.status == "fail":
        _emit(
            f"  [WARNING] {len(result.size_validation.violations)} file(s) "
            f"exceed size constraints (informational only)"
        )
        for v in result.size_validation.violations:
            _emit(f"    - {v.file_path}: {v.line_count} lines / {v.kb_size} KB "
                  f"[{v.violation_type}] (seg: {v.produced_by_segment})")
    else:
        _emit(f"  [OK] Size check passed ({result.size_validation.files_checked} files)")

    # --- Check 2: Skeleton contract verification (INFORMATIONAL) ---
    if skeleton:
        _emit("[CHECK 2] Skeleton contract verification (informational)...")
        result.contract_check = check_skeleton_contracts(state, skeleton, sandbox_base)
        result.checks_run.append("contract_check")

        if result.contract_check.status == "fail":
            _emit(
                f"  [WARNING] {len(result.contract_check.violations)} contract issue(s) "
                f"(informational only)"
            )
            for v in result.contract_check.violations:
                _emit(f"    - [{v.segment_id}] {v.violation_type}: {v.detail}")
        else:
            _emit("  [OK] Contract check passed")
    else:
        _emit("[CHECK 2] Skeleton contract verification -- SKIPPED (no skeleton)")

    # --- Check 3: Boot test with fix loop (PASS/FAIL GATE) ---
    _emit("[CHECK 3] Application boot test (with fix loop)...")
    result.boot_test = await run_boot_test_with_fix_loop(
        sandbox_base=sandbox_base,
        state=state,
        emit=_emit,
    )
    result.checks_run.append("boot_test")

    if result.boot_test.status == "pass":
        _emit("  [OK] Boot test passed -- application starts cleanly")
    elif result.boot_test.status == "fail":
        _emit(f"  [FAIL] Boot test FAILED after fix attempts: {result.boot_test.error_summary}")
        if result.boot_test.traceback_file:
            _emit(f"    Failing file: {result.boot_test.traceback_file}")
            result.boot_test.traceback_segment = map_file_to_segment(
                result.boot_test.traceback_file, state
            )
            if result.boot_test.traceback_segment:
                _emit(f"    Produced by: {result.boot_test.traceback_segment}")
    else:
        _emit(f"  [ERROR] Boot test error: {result.boot_test.error_summary}")

    # --- Aggregate and route ---
    # v2.0: Only the boot test determines pass/fail.
    # Size and contract checks are informational warnings -- earlier pipeline
    # stages (architecture, critique, cohesion) enforce those constraints.
    # Phase checkout's job is: does it boot? If not, can we fix it?
    boot_passed = (result.boot_test and result.boot_test.status == "pass")

    if boot_passed:
        result.status = "pass"
        warnings = []
        if result.size_validation and result.size_validation.status == "fail":
            warnings.append(f"{len(result.size_validation.violations)} size warning(s)")
        if result.contract_check and result.contract_check.status == "fail":
            warnings.append(f"{len(result.contract_check.violations)} contract warning(s)")
        if warnings:
            _emit(f"\n[PASS] PHASE CHECKOUT PASSED (boot OK) with warnings: {', '.join(warnings)}")
        else:
            _emit("\n[PASS] PHASE CHECKOUT PASSED -- all checks green")
    else:
        result.status = "fail"
        result.routing = _determine_failure_routing(result, state)
        _emit(f"\n[FAIL] PHASE CHECKOUT FAILED -> route to {result.routing.target_stage}")
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

    v2.0: Only boot failures reach here (size/contract are warnings now).
    """
    # Boot failures -- route based on error type
    if result.boot_test and result.boot_test.status == "fail":
        err = (result.boot_test.error_summary or "").lower()
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

        if "modulenotfounderror" in err or "importerror" in err:
            return FailureRouting(
                target_stage="stage_8_overwatcher",
                target_segment=failing_seg,
                target_file=result.boot_test.traceback_file,
                reason=f"Import error in {result.boot_test.traceback_file} "
                       f"(fix loop exhausted): {result.boot_test.error_summary[:200]}",
            )

        return FailureRouting(
            target_stage="stage_5_critical",
            target_segment=failing_seg,
            target_file=result.boot_test.traceback_file,
            reason=f"Boot failure: {result.boot_test.error_summary[:200]}",
        )

    return FailureRouting(
        target_stage="stage_5_critical",
        reason="Unknown failure -- re-run architecture generation",
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
