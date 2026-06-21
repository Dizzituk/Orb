# FILE: app/pipeline_v2/orchestrator_report.py
# Purpose: Orchestrator build-report compilation + cost estimate + ledger summary — split from orchestrator.py.
# Called-by: app.pipeline_v2.orchestrator
# Depends-on: app.builds.models, app.builds.service, app.db, app.debug.project_service, app.pipeline_v2.models
# Last-renovated: 2026-06-21
from __future__ import annotations
import json
import logging
from typing import Any, Dict
from app.pipeline_v2.models import PipelineResult

logger = logging.getLogger(__name__)


def _compile_and_send_to_debug(
    result: PipelineResult,
    job_id: str,
    spec: Dict,
    emit: Any,
    profile: Any = None,
) -> None:
    """Compile the build report and create a debug project with it."""
    emit("\n📝 Compiling build report...")

    try:
        from app.db import SessionLocal
        from app.builds.service import compile_build_report
        from app.builds.models import BuildProject
        db = SessionLocal()
        project = db.query(BuildProject).order_by(BuildProject.updated_at.desc()).first()
        if project:
            report_json = compile_build_report(db, project.id)
            emit(f"   📝 Report compiled: {len(report_json or '')} chars")
        else:
            report_json = None
            emit("   ⚠️ No build project found for report")
        db.close()
    except Exception as e:
        emit(f"   ⚠️ Report compilation failed: {e}")
        report_json = None

    try:
        from app.debug.project_service import create_project as create_debug_project

        spec_summary = ""
        if isinstance(spec, dict):
            spec_summary = spec.get("summary", spec.get("title", ""))[:100]
        title = f"Build: {spec_summary or job_id}"

        build = result.build_result
        lines = [
            f"Job ID: {job_id}",
            f"Status: {'PASSED' if result.success else 'ISSUES'}",
            f"Files: {len(build.all_files_written) if build else 0}",
            f"Tool calls: {build.total_tool_calls if build else 0}",
            f"Duration: {result.total_duration_seconds:.0f}s",
            f"Est. cost: ${result.estimated_cost_usd:.2f}",
        ]
        if result.errors:
            lines.append(f"Errors: {'; '.join(result.errors[:3])}")

        # v1.3 (2026-04-18): Fold spec-reviewer summary into the debug report
        # so findings are visible in the build report without having to open
        # the raw pipeline result.
        review = getattr(result, "spec_review_report", None)
        if review is not None:
            lines.append("")
            lines.append(f"Spec review: {review.summary_line()}")
            if review.summary:
                lines.append(f"  {review.summary}")
            for finding in review.findings[:10]:
                lines.append(f"  - {finding.one_line()}")
            if len(review.findings) > 10:
                lines.append(f"  … and {len(review.findings) - 10} more")
            if review.requirements_unmet:
                lines.append("  Unmet requirements:")
                for req in review.requirements_unmet[:6]:
                    lines.append(f"    ✗ {req}")

        description = "\n".join(lines)
        if report_json:
            description += f"\n\n--- BUILD REPORT ---\n{report_json[:5000]}"

        debug_project = create_debug_project(
            title=title,
            description=description,
            error_summary="; ".join(result.errors[:3]) if result.errors else "",
            metadata_json=json.dumps({
                "job_id": job_id,
                "build_target_id": profile.project_id if profile else None,
                "build_target_name": profile.project_name if profile else None,
                "project_root": profile.project_root if profile else None,
                "language": profile.language if profile else None,
                "framework": profile.framework if profile else None,
                "success": result.success,
                "files_written": len(build.all_files_written) if build else 0,
            }),
        )
        emit(f"   🐛 Debug project created: {debug_project.get('id', '?')}")
    except Exception as e:
        emit(f"   ⚠️ Debug project creation failed: {e}")
        logger.warning("[orchestrator] Debug project creation failed: %s", e)


def _estimate_cost(result: PipelineResult) -> float:
    """Rough cost estimate based on token usage."""
    if not result.build_result:
        return 0.0

    input_cost = (result.build_result.total_input_tokens / 1_000_000) * 2.50
    output_cost = (result.build_result.total_output_tokens / 1_000_000) * 15.00
    verify_cost = len(result.verify_results) * 0.01

    return input_cost + output_cost + verify_cost


def _emit_ledger_summary(job_dir: str, emit: Any) -> None:
    """Emit a one-line summary of the decision ledger state, if present."""
    try:
        from app.pipeline_v2.config import LEDGER_ENABLED
        if not LEDGER_ENABLED:
            return
        from app.pipeline_v2.ledger import load_ledger
        ledger = load_ledger(job_dir)
        if ledger is None:
            return
        decisions = sum(1 for e in ledger.entries if e.type == "decision")
        corrections = sum(1 for e in ledger.entries if e.status == "corrected")
        emit(
            f"   📒 Ledger: {ledger.entry_count} entries "
            f"({decisions} decisions, {corrections} superseded)"
        )
    except Exception as e:
        logger.debug("[orchestrator] Ledger summary skipped: %s", e)
