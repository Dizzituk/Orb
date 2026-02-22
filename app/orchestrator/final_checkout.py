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
from app.orchestrator._final_checkout_utils_2 import FINAL_CHECKOUT_BUILD_ID, _MAX_REVIEW_CHARS, _MAX_REVIEW_FILES, _REVIEW_PRIORITY_PATTERNS, _REVIEW_SYSTEM_PROMPT, _build_minimal_state_from_plan, _find_closest_name, _fix_interface_contract_issue
from app.orchestrator._final_checkout_utils_3 import _build_ai_review_prompt, _build_plan_from_manifest, _detect_and_fix_integration_issues, _extract_files_from_issue, _llm_fix_import, _run_boot_with_strike_tracking, _save_result, _write_file_to_sandbox_stub
from app.orchestrator._final_checkout_utils_4 import _apply_ai_review_fix, _fix_import_resolution_issue, _generate_missing_file, _parse_review_response
from app.orchestrator._final_checkout_utils_5 import _check_cross_phase_with_fixes, _check_spec_coverage_with_fixes, _run_ai_review_pass, _run_ai_review_with_fixes, _write_file_to_sandbox
from app.orchestrator._final_checkout_utils_6 import AIReviewResult, CrossPhaseResult, SpecCoverageResult, _file_exists_on_sandbox, run_final_checkout

logger = logging.getLogger(__name__)
print(f"[FINAL_CHECKOUT_LOADED] BUILD_ID={FINAL_CHECKOUT_BUILD_ID}")


# =============================================================================
# RESULT MODELS
# =============================================================================


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


# =============================================================================
# CHECK 1: SPEC COVERAGE WITH AUTO-GENERATION
# =============================================================================


# =============================================================================
# CHECK 2: CROSS-PHASE WITH AUTO-FIX
# =============================================================================


# =============================================================================
# CHECK 3: BOOT TEST WITH STRIKE TRACKING
# =============================================================================


# =============================================================================
# CHECK 4: AI REVIEW WITH AUTO-FIX
# =============================================================================

# Files to prioritise for review (highest risk, minimal token cost)


# =============================================================================
# HELPERS
# =============================================================================


# =============================================================================
# CHECK 6: PIPELINE LEARNING REPORT (RAG)
# =============================================================================


def compile_pipeline_learning_report(
    job_id: str,
    job_dir: str,
    state: Any,
    manifest: Any,
    final_result: FinalCheckoutResult,
    phase_checkout_result: Optional[Dict[str, Any]] = None,
    original_spec: Optional[str] = None,
    emit: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    v3.2: Compile the Pipeline Learning Report for RAG ingestion.

    This is the ABSOLUTE LAST thing written at the end of a job.
    It gathers every error-fix pair from every pipeline stage into
    a single JSON file that the RAG system can index.

    Data sources:
    - Phase Checkout boot fix logs (from state)
    - Final Checkout strike records (from final_result)
    - Cohesion check results (from job dir)
    - Integration check results (from state)
    - Execution traces per segment (from job dir)
    - Post-execution reconciliation fixes

    The report is written to:
      {job_dir}/pipeline_learning_report.json

    Returns the report dict.
    """
    _emit = emit or (lambda msg: None)
    _emit("\n[CHECK 6] Compiling Pipeline Learning Report...")

    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "job_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "outcome": final_result.status,  # "pass" or "fail"
        "total_segments": manifest.total_segments if manifest else 0,
        "total_files": len(final_result.spec_coverage.missing_files) + final_result.total_files_built if final_result.spec_coverage else 0,
        "duration_ms": final_result.duration_ms,
        "spec_summary": (original_spec or "")[:500],
        "pipeline_events": [],
        "strike_records": final_result.strike_records,
        "stage_summaries": {},
    }

    events = report["pipeline_events"]

    # --- 1. Cohesion check findings ---
    try:
        cohesion_path = os.path.join(job_dir, "cohesion_result.json")
        if os.path.isfile(cohesion_path):
            with open(cohesion_path, "r", encoding="utf-8") as f:
                cohesion_data = json.load(f)
            for issue in cohesion_data.get("issues", []):
                events.append({
                    "stage": "cohesion_check",
                    "event_type": "cohesion_finding",
                    "severity": issue.get("severity", "unknown"),
                    "category": issue.get("category", ""),
                    "description": issue.get("description", "")[:200],
                    "source_segment": issue.get("source_segment", ""),
                    "related_segment": issue.get("related_segment", ""),
                    "auto_fixed": issue.get("auto_fixed", False),
                    "auto_fix_tier": issue.get("auto_fix_tier"),
                    "auto_fix_note": issue.get("auto_fix_note", "")[:200],
                })
            report["stage_summaries"]["cohesion_check"] = {
                "status": cohesion_data.get("status"),
                "total_issues": len(cohesion_data.get("issues", [])),
                "auto_fixed": sum(1 for i in cohesion_data.get("issues", []) if i.get("auto_fixed")),
            }
    except Exception as exc:
        logger.debug("[learning_report] cohesion: %s", exc)

    # --- 2. Integration check results ---
    if state and hasattr(state, 'integration_check') and state.integration_check:
        ic = state.integration_check
        if isinstance(ic, dict):
            for issue in ic.get("tier1_issues", []):
                events.append({
                    "stage": "integration_check",
                    "event_type": "integration_finding",
                    "severity": issue.get("severity", "unknown"),
                    "check_type": issue.get("check_type", ""),
                    "message": issue.get("message", "")[:200],
                    "file_a": issue.get("file_a", ""),
                    "file_b": issue.get("file_b", ""),
                })
            report["stage_summaries"]["integration_check"] = {
                "status": ic.get("status"),
                "error_count": ic.get("error_count", 0),
                "warning_count": ic.get("warning_count", 0),
            }

    # --- 3. Phase checkout boot fix events ---
    if phase_checkout_result:
        pc = phase_checkout_result
        boot_info = pc.get("boot_test", {})
        if boot_info:
            report["stage_summaries"]["phase_checkout"] = {
                "boot_status": boot_info.get("status"),
                "boot_attempts": boot_info.get("attempts", 0),
                "error_summary": boot_info.get("error_summary", "")[:200],
            }
            # Fix attempts from phase checkout
            for fix in boot_info.get("fix_attempts", []):
                events.append({
                    "stage": "phase_checkout",
                    "event_type": "boot_fix",
                    "error": fix.get("error", "")[:200],
                    "strategy": fix.get("strategy", ""),
                    "resolved": fix.get("resolved", False),
                    "file": fix.get("file", ""),
                })

    # --- 4. Per-segment execution traces ---
    if manifest:
        segments_dir = os.path.join(job_dir, "segments")
        for seg in manifest.segments:
            trace_path = os.path.join(
                segments_dir, seg.segment_id, "execution_trace", "trace.json"
            )
            if os.path.isfile(trace_path):
                try:
                    with open(trace_path, "r", encoding="utf-8") as f:
                        trace_data = json.load(f)
                    events.append({
                        "stage": "execution",
                        "event_type": "execution_failure",
                        "segment_id": seg.segment_id,
                        "error": trace_data.get("error", "")[:200],
                        "success": trace_data.get("success", False),
                        "artifacts_written": len(trace_data.get("artifacts_written", [])),
                        "trace_event_count": len(trace_data.get("trace_events", [])),
                    })
                except Exception:
                    pass

    # --- 5. Post-execution reconciliation ---
    try:
        recon_path = os.path.join(job_dir, "reconciliation_result.json")
        if os.path.isfile(recon_path):
            with open(recon_path, "r", encoding="utf-8") as f:
                recon_data = json.load(f)
            for fix in recon_data.get("fixes_applied", []):
                events.append({
                    "stage": "post_execution_reconciliation",
                    "event_type": "import_fix",
                    "file": fix.get("file", ""),
                    "old_import": fix.get("old_import", "")[:100],
                    "new_import": fix.get("new_import", "")[:100],
                    "resolved": True,
                })
            report["stage_summaries"]["reconciliation"] = {
                "files_fixed": recon_data.get("files_fixed", 0),
                "total_fixes": len(recon_data.get("fixes_applied", [])),
            }
    except Exception as exc:
        logger.debug("[learning_report] reconciliation: %s", exc)

    # --- 6. Final checkout strike records (already in final_result) ---
    report["stage_summaries"]["final_checkout"] = {
        "status": final_result.status,
        "boot_test_status": final_result.boot_test_status,
        "spec_coverage_status": final_result.spec_coverage.status if final_result.spec_coverage else None,
        "ai_review_score": final_result.ai_review.overall_score if final_result.ai_review else None,
        "checks_run": final_result.checks_run,
        "files_generated_by_checkout": (
            final_result.spec_coverage.files_generated if final_result.spec_coverage else []
        ),
    }

    # --- 7. Segment-level summary ---
    segment_summaries = []
    if state and hasattr(state, 'segments'):
        for seg_id, seg_state in state.segments.items():
            seg_summary = {
                "segment_id": seg_id,
                "status": seg_state.status if hasattr(seg_state, 'status') else str(seg_state),
                "output_files": len(seg_state.output_files) if hasattr(seg_state, 'output_files') else 0,
            }
            if hasattr(seg_state, 'error') and seg_state.error:
                seg_summary["error"] = seg_state.error[:200]
            segment_summaries.append(seg_summary)
    report["segment_summaries"] = segment_summaries

    # --- Write report ---
    report_path = os.path.join(job_dir, "pipeline_learning_report.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        _emit(f"  [OK] Pipeline Learning Report written: {report_path}")
        _emit(f"       {len(events)} pipeline event(s), {len(report['strike_records'])} strike record(s)")
        logger.info(
            "[final_checkout] v3.2 Pipeline Learning Report: %s (%d events)",
            report_path, len(events),
        )
    except Exception as exc:
        _emit(f"  [ERROR] Failed to write report: {exc}")
        logger.warning("[final_checkout] v3.2 Report write failed: %s", exc)

    # Journal: emit job completion summary for distillation
    try:
        from app.experience.context import journal_emit
        journal_emit(
            job_id=job_id,
            job_dir=job_dir,
            stage="final_checkout",
            event_type="stage_exit",
            severity="info",
            description=f"Job {report.get('outcome', 'unknown')}: "
                        f"{len(events)} events, "
                        f"{len(report.get('strike_records', []))} strikes",
            details={
                "outcome": report.get("outcome"),
                "total_segments": report.get("total_segments"),
                "total_files": report.get("total_files"),
                "duration_ms": report.get("duration_ms"),
                "event_count": len(events),
                "strike_count": len(report.get("strike_records", [])),
            },
        )
    except Exception:
        pass

    # Trigger post-job distillation (journal → experience patterns)
    try:
        from app.experience.distillation import distill_job
        from app.db import get_db_session
        distill_db = get_db_session()
        patterns = distill_job(distill_db, job_id, job_dir)
        if patterns:
            _emit(f"  [OK] Distilled {len(patterns)} experience pattern(s) from journal")
            logger.info(
                "[final_checkout] Distilled %d patterns for job %s",
                len(patterns), job_id,
            )
        distill_db.close()
    except Exception as distill_err:
        logger.warning("[final_checkout] Distillation failed (non-fatal): %s", distill_err)

    # Calibrate confidence scores based on job outcome
    try:
        from app.experience.calibration import calibrate_from_job
        job_outcome = report.get("outcome", "unknown")
        cal_db = get_db_session()
        cal_stats = calibrate_from_job(cal_db, job_id, job_outcome)
        if cal_stats.get("boosted") or cal_stats.get("penalised"):
            _emit(
                f"  [OK] Calibrated {cal_stats['boosted']} boosted, "
                f"{cal_stats['penalised']} penalised patterns"
            )
        cal_db.close()
    except Exception as cal_err:
        logger.warning("[final_checkout] Calibration failed (non-fatal): %s", cal_err)

    return report


# =============================================================================
# PERSISTENCE
# =============================================================================
