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
from app.sandbox_fs import sandbox_read_text as _sbx_read_text

logger = logging.getLogger(__name__)

PHASE_CHECKOUT_BUILD_ID = "2026-02-28-v3.2-demote-on-boot-fail"
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
            # v6.1 FIX 13: Support multi-file (deterministic_sources list)
            _det_sources = _manifest_data.get("deterministic_sources", [])
            if not _det_sources:
                _single = _manifest_data.get("deterministic_source")
                if _single:
                    _det_sources = [_single]
            if _det_sources:
                import ast as _ast
                _baseline_fn_sizes = {}
                for _det_source in _det_sources:
                    _source_abs = os.path.join(sandbox_base, _det_source.replace("/", os.sep))
                    if os.path.isfile(_source_abs):
                        _src = _sbx_read_text(_source_abs)
                        try:
                            _tree = _ast.parse(_src)
                            for _n in _ast.walk(_tree):
                                if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                                    if hasattr(_n, "end_lineno") and _n.end_lineno:
                                        _baseline_fn_sizes[_n.name] = _n.end_lineno - _n.lineno + 1
                        except SyntaxError:
                            pass
                if _baseline_fn_sizes:
                    logger.info(
                        "[phase_checkout] v6.1 Loaded %d baseline function sizes from %d source(s)",
                        len(_baseline_fn_sizes), len(_det_sources),
                    )
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

    # --- Check 3: Frontend syntax validation (INFORMATIONAL → FAIL if garbage) ---
    _has_frontend = any(
        f.endswith(('.ts', '.tsx', '.jsx'))
        for f in segment_output_files
    )
    if _has_frontend:
        _emit("[CHECK 3] Frontend TypeScript/TSX syntax validation...")
        from .phase_checkout_frontend import check_frontend_syntax
        result.frontend_check = check_frontend_syntax(
            state=state,
            sandbox_base=sandbox_base,
            emit=_emit,
        )
        result.checks_run.append("frontend_syntax")
        if result.frontend_check.get("status") == "fail":
            _failures = result.frontend_check.get("failures", [])
            _emit(
                f"  [FAIL] {len(_failures)} frontend file(s) contain "
                f"non-code content (SSE/markdown contamination)"
            )
            for _ff in _failures:
                _emit(f"    - {_ff['file']}: {_ff['reason']}")
        else:
            _checked = result.frontend_check.get("files_checked", 0)
            _emit(f"  [OK] Frontend syntax check passed ({_checked} files)")
    else:
        _emit("[CHECK 3] Frontend syntax validation -- SKIPPED (no TS/TSX files)")

    # --- Check 3B: Frontend BUILD check + deterministic fix (v1.0 HARD GATE) ---
    if _has_frontend:
        _emit("[CHECK 3B] Frontend TypeScript compilation (tsc --noEmit)...")
        try:
            from .frontend_fix_loop import run_frontend_fix_loop
            _frontend_fix = await run_frontend_fix_loop(
                segment_files=segment_output_files,
                emit=_emit,
            )
            result.frontend_build = _frontend_fix
            result.checks_run.append("frontend_build")
            if _frontend_fix.status == "pass":
                _emit("  [OK] TypeScript compilation passed")
            elif _frontend_fix.status == "fixed":
                _emit(
                    f"  [OK] TypeScript errors auto-fixed deterministically: "
                    f"{', '.join(_frontend_fix.fixes_applied[:5])}"
                )
            elif _frontend_fix.status == "fail":
                _emit(
                    f"  [FAIL] TypeScript compilation failed: "
                    f"{_frontend_fix.remaining_errors} error(s) remain"
                )
            else:
                _emit(f"  [ERROR] Frontend build check: {_frontend_fix.status}")
        except Exception as _fb_exc:
            logger.warning("[phase_checkout] Frontend build check failed: %s", _fb_exc)
            _emit(f"  [WARN] Frontend build check skipped: {_fb_exc}")
    else:
        _emit("[CHECK 3B] Frontend build check -- SKIPPED (no TS/TSX files)")

    # --- Check 4: Boot test with fix loop (PASS/FAIL GATE) ---
    _emit("[CHECK 4] Application boot test (with fix loop)...")
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

    # --- Check 4B: Frontend boot test (Vite build) ---
    if _has_frontend:
        _emit("[CHECK 4B] Frontend Vite build check...")
        try:
            from .frontend_boot_check import run_frontend_boot_check
            from app.overwatcher.sandbox_client import get_sandbox_client
            _fe_client = get_sandbox_client()
            _vite_result = run_frontend_boot_check(
                client=_fe_client,
                emit=_emit,
            )
            result.frontend_boot = _vite_result
            result.checks_run.append("frontend_boot")
            if _vite_result.status == "pass":
                _emit(f"  [OK] Vite build passed ({_vite_result.duration_ms}ms)")
            elif _vite_result.status == "fail":
                _emit(f"  [FAIL] Vite build FAILED: {_vite_result.error_summary[:150]}")
                for _ve in _vite_result.errors[:3]:
                    _emit(f"    {_ve.file}: {_ve.message[:100]}")
            else:
                _emit(f"  [WARN] Vite build check: {_vite_result.status}")
        except Exception as _vb_exc:
            logger.warning("[phase_checkout] Frontend boot check failed: %s", _vb_exc)
            _emit(f"  [WARN] Frontend boot check skipped: {_vb_exc}")
    else:
        _emit("[CHECK 4B] Frontend Vite build check -- SKIPPED (no frontend files)")

    # --- Pre-compute boot_passed for use in Check 5 and aggregation ---
    boot_passed = (result.boot_test and result.boot_test.status == "pass")

    # --- Check 5: Big-Model Verification (v8.0 — agentic pipeline enhancement) ---
    # When enabled, a big model reads all written files + spec + boot results
    # and produces a targeted diagnosis. Only runs if boot passed (no point
    # diagnosing if the app doesn't start).
    _model_verdict = None
    try:
        from app.agentic_pipeline.config import AGENTIC_PIPELINE_ENABLED
    except ImportError:
        AGENTIC_PIPELINE_ENABLED = False

    if AGENTIC_PIPELINE_ENABLED and boot_passed:
        _emit("[CHECK 5] Big-model verification (agentic pipeline)...")
        try:
            from app.agentic_pipeline.phase_checkout_model import run_phase_checkout as _run_model_checkout
            from app.llm.overwatcher_stream import create_overwatcher_llm_fn
            _llm_fn = create_overwatcher_llm_fn()
            if _llm_fn:
                # Read all written files from sandbox
                _written = {}
                for f in segment_output_files:
                    _content = _sbx_read_text(f)
                    if _content:
                        _written[f] = _content

                # Load spec summary
                _spec_path = os.path.join(job_dir, "spec.json")
                _spec_summary = "(not available)"
                if os.path.isfile(_spec_path):
                    with open(_spec_path, "r") as _sf:
                        _spec_data = json.load(_sf)
                    _spec_summary = _spec_data.get("summary", _spec_data.get("objective", ""))

                _boot_str = "PASS" if boot_passed else f"FAIL: {result.boot_test.error_summary[:200] if result.boot_test else 'unknown'}"
                _build_str = "PASS" if (hasattr(result, 'frontend_build') and result.frontend_build and result.frontend_build.status == 'pass') else "FAIL or N/A"

                _model_verdict = await _run_model_checkout(
                    spec_summary=_spec_summary,
                    written_files=_written,
                    boot_result=_boot_str,
                    build_result=_build_str,
                    llm_call_fn=_llm_fn,
                    on_progress=_emit,
                )
                result.checks_run.append("model_verification")

                if _model_verdict.passed:
                    _emit(f"  [OK] Big-model verification passed (confidence={_model_verdict.confidence:.2f})")
                else:
                    _emit(f"  [WARN] Big-model found {len(_model_verdict.fix_items)} issue(s):")
                    for _fi in _model_verdict.fix_items[:5]:
                        _emit(f"    - [{_fi.fix_type.value}] {_fi.file_path}: {_fi.description[:100]}")
            else:
                _emit("  [SKIP] No LLM provider available for big-model verification")
        except Exception as _mv_err:
            logger.warning("[phase_checkout] v8.0 Big-model verification failed (non-fatal): %s", _mv_err)
            _emit(f"  [WARN] Big-model verification skipped: {_mv_err}")
    elif not AGENTIC_PIPELINE_ENABLED:
        pass  # v8.0 not enabled — skip silently
    else:
        _emit("[CHECK 5] Big-model verification -- SKIPPED (boot failed)")

    # --- Aggregate and route ---
    # v2.0: Only the boot test determines pass/fail.
    # Size and contract checks are informational warnings -- earlier pipeline
    # stages (architecture, critique, cohesion) enforce those constraints.
    # Phase checkout's job is: does it boot? If not, can we fix it?
    # (boot_passed already computed above for Check 5)
    # v3.4-fix: Vite build failure is a hard gate
    frontend_boot_failed = (
        hasattr(result, 'frontend_boot')
        and result.frontend_boot
        and result.frontend_boot.status == "fail"
    )
    frontend_failed = (
        hasattr(result, 'frontend_check')
        and result.frontend_check
        and result.frontend_check.get('status') == 'fail'
    )

    if boot_passed and not frontend_failed and not frontend_boot_failed:
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
    elif boot_passed and frontend_failed:
        # v1.2 FIX: Distinguish scaffold-only failures from real contamination.
        # LLM_FILL scaffold markers are an expected intermediate state — the
        # implementer fills them. Routing to stage_8 for scaffold markers caused
        # regression in job sg-a798331a (overwatcher rewrote JobPage.tsx from
        # scratch, deleting all existing tab routing).
        # Rule: If boot passes, scaffold-only failures are WARNINGS not FAILs.
        _failures = result.frontend_check.get('failures', [])
        _scaffold_only = all(
            "scaffold marker" in f.get("reason", "").lower()
            for f in _failures
        )
        _real_contamination = [
            f for f in _failures
            if "scaffold marker" not in f.get("reason", "").lower()
        ]

        if _scaffold_only:
            # v3.4-fix: Scaffold markers surviving to phase checkout are now
            # boot-blocking errors. The implementer (v3.4) strips [LLM_FILL]
            # markers at write time. If any survive here, the write-time
            # sanitiser missed them — this is a genuine failure that will
            # crash Vite/tsc at runtime.
            # Previous behaviour (v1.2) treated these as warnings to avoid
            # the sg-a798331a regression. That regression is now prevented
            # by the write-time strip in run_implementer_task instead.
            result.status = "fail"
            result.routing = FailureRouting(
                target_stage="failed",
                target_segment=None,
                target_file=_failures[0]['file'] if _failures else None,
                reason=(
                    f"Scaffold markers survived write-time sanitiser: "
                    f"{len(_failures)} file(s) still contain [LLM_FILL] placeholders"
                ),
                scoped_files=[f['file'] for f in _failures],
                severity="major",
            )
            _emit(
                f"\n[FAIL] PHASE CHECKOUT FAILED -- "
                f"{len(_failures)} file(s) have unfilled scaffold markers "
                f"(should have been stripped at write time)"
            )
            for _ff in _failures:
                _emit(f"    ❌ {_ff['file']}: {_ff['reason'][:120]}")
        elif _real_contamination:
            # Real contamination (SSE garbage, markdown prose).
            # Severity: 1 file = minor (surgical fix), 2+ files = major (fail cleanly).
            _contam_severity = "minor" if len(_real_contamination) == 1 else "major"
            _contam_stage = "stage_8_overwatcher" if _contam_severity == "minor" else "failed"
            result.status = "fail"
            result.routing = FailureRouting(
                target_stage=_contam_stage,
                target_segment=None,
                target_file=_real_contamination[0]['file'],
                reason=(
                    f"Frontend syntax check failed: {len(_real_contamination)} "
                    f"file(s) contain non-code content (severity={_contam_severity})"
                ),
                scoped_files=[f['file'] for f in _real_contamination],
                severity=_contam_severity,
            )
            if _contam_severity == "minor":
                _emit(f"\n[FAIL] PHASE CHECKOUT FAILED -- 1 file has garbage content (surgical fix)")
                _emit(f"    ❌ {_real_contamination[0]['file']}: {_real_contamination[0]['reason'][:120]}")
            else:
                _emit(f"\n[FAIL] PHASE CHECKOUT FAILED -- {len(_real_contamination)} file(s) contaminated (major, not auto-fixable)")
                for _ff in _real_contamination:
                    _emit(f"    ❌ {_ff['file']}: {_ff['reason'][:120]}")
        else:
            result.status = "pass"
            _emit("\n[PASS] PHASE CHECKOUT PASSED (boot OK, no actionable frontend failures)")
    else:
        result.status = "fail"
        result.routing = _determine_failure_routing(result, state)
        _emit(f"\n[FAIL] PHASE CHECKOUT FAILED -> route to {result.routing.target_stage}")
        if result.routing.target_segment:
            _emit(f"  Target segment: {result.routing.target_segment}")
        _emit(f"  Reason: {result.routing.reason}")

        # v3.2: Demote causal segment from COMPLETE to FAILED so state
        # reflects the actual outcome. Without this, segments remain
        # COMPLETE despite the application not booting.
        _demote_failed_segments(result, state, job_dir, _emit)

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
    v3.0: Added severity classification (minor/major).
          - minor: single file, simple error → stage_8 can attempt surgical fix
          - major: multiple files, structural issues → fail cleanly, no LLM heroics

    Root cause for severity gate: job sg-a798331a regression. The overwatcher
    was given free rein to "fix" a failure and rewrote JobPage.tsx from scratch,
    destroying all existing tab routing. Major failures should not be handed
    to an unconstrained LLM — they should fail cleanly so the human can decide.
    """
    # Boot failures -- route based on error type and severity
    if result.boot_test and result.boot_test.status == "fail":
        err = (result.boot_test.error_summary or "").lower()
        traceback_file = result.boot_test.traceback_file
        failing_seg = map_file_to_segment(traceback_file, state)

        # ── Severity classification ──────────────────────────────────
        # Minor: single file identified, simple error type (syntax/import).
        # Major: no traceable file, multi-file issue, or complex error.
        _is_simple_error = (
            "syntaxerror" in err
            or "modulenotfounderror" in err
            or "importerror" in err
            or "nameerror" in err
            or "attributeerror" in err
        )
        _has_traceable_file = traceback_file is not None and traceback_file != ""
        _severity = "minor" if (_is_simple_error and _has_traceable_file) else "major"

        # ── Minor: surgical fix scoped to the one broken file ────────
        if _severity == "minor":
            _scoped = [traceback_file]

            if "syntaxerror" in err:
                _reason = f"Syntax error in {traceback_file}"
            elif "modulenotfounderror" in err or "importerror" in err:
                _reason = (
                    f"Import error in {traceback_file} "
                    f"(fix loop exhausted): {result.boot_test.error_summary[:200]}"
                )
            else:
                _reason = f"{traceback_file}: {result.boot_test.error_summary[:200]}"

            return FailureRouting(
                target_stage="stage_8_overwatcher",
                target_segment=failing_seg,
                target_file=traceback_file,
                reason=_reason,
                scoped_files=_scoped,
                severity="minor",
            )

        # ── Major: fail cleanly, no LLM heroics ─────────────────────
        return FailureRouting(
            target_stage="failed",
            target_segment=failing_seg,
            target_file=traceback_file,
            reason=f"Boot failure (major — not auto-fixable): {result.boot_test.error_summary[:200]}",
            severity="major",
        )

    # No boot test result or unknown state — fail cleanly
    return FailureRouting(
        target_stage="failed",
        reason="Unknown failure -- manual investigation required",
        severity="major",
    )


# =============================================================================
# v3.2: SEGMENT DEMOTION ON BOOT FAILURE
# =============================================================================

def _demote_failed_segments(
    result: PhaseCheckoutResult,
    state: Any,
    job_dir: str,
    emit: Any,
) -> None:
    """v3.2: When boot fails, demote the causal segment(s) from COMPLETE to FAILED.

    The boot test identified which file caused the failure (traceback_file)
    and which segment produced it (traceback_segment via map_file_to_segment).
    That segment should NOT remain COMPLETE — it produced code that breaks
    the application.

    If no specific segment can be identified, we do NOT demote all segments
    (that would be too aggressive). We log the situation for manual triage.
    """
    from app.orchestrator._segment_loop_utils_8 import update_segment_status

    try:
        from app.pot_spec.grounded.segment_schemas import SegmentStatus
    except ImportError:
        logger.warning("[phase_checkout] v3.2 Cannot import SegmentStatus for demotion")
        return

    causal_seg = None

    # 1. Use the traceback_segment if already identified
    if result.boot_test and result.boot_test.traceback_segment:
        causal_seg = result.boot_test.traceback_segment

    # 2. Fall back to routing target_segment
    if not causal_seg and result.routing and result.routing.target_segment:
        causal_seg = result.routing.target_segment

    # 3. If we know the failing file but not the segment, try to match
    if not causal_seg and result.boot_test and result.boot_test.traceback_file:
        causal_seg = map_file_to_segment(
            result.boot_test.traceback_file, state
        )

    if causal_seg:
        seg_state = state.segments.get(causal_seg)
        if seg_state and seg_state.status == SegmentStatus.COMPLETE.value:
            error_msg = (
                f"Boot check failed: {(result.boot_test.error_summary or '')[:200]}"
            )
            update_segment_status(
                state, causal_seg, SegmentStatus.FAILED, job_dir,
                error=error_msg,
            )
            emit(
                f"  ⬇️ v3.2 Demoted {causal_seg}: COMPLETE → FAILED "
                f"(produced code that fails boot)"
            )
            logger.info(
                "[phase_checkout] v3.2 Demoted %s from COMPLETE to FAILED: %s",
                causal_seg, error_msg[:150],
            )
        else:
            emit(
                f"  ℹ️ v3.2 Causal segment {causal_seg} is "
                f"{seg_state.status if seg_state else 'missing'} (no demotion needed)"
            )
    else:
        emit(
            "  ⚠️ v3.2 Could not identify causal segment for boot failure "
            "— segments remain COMPLETE (manual triage needed)"
        )
        logger.warning(
            "[phase_checkout] v3.2 Boot failed but no causal segment identified. "
            "traceback_file=%s",
            result.boot_test.traceback_file if result.boot_test else None,
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
