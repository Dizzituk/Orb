# FILE: app/pipeline_v2/orchestrator_spec_review.py
# Purpose: Always-on spec-reviewer stage wrapper + BVL build-output extraction — split from orchestrator.py.
# Called-by: app.pipeline_v2.orchestrator
# Depends-on: app.pipeline_v2.models, app.pipeline_v2.orchestrator_notify, app.pipeline_v2.spec_review
# Last-renovated: 2026-06-21
from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from app.pipeline_v2.models import PipelineResult
from app.pipeline_v2.orchestrator_notify import _push_narrative

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# v1.3 (2026-04-18): Always-on spec reviewer wrapper.
#
# Thin orchestrator-side wrapper that: imports the reviewer lazily (avoids
# circular-import risk with models.py), captures any build output from
# the BVL report, invokes the reviewer, attaches the report to the
# PipelineResult, and folds the cost + LLM-call count into pipeline
# totals. Never raises — the reviewer itself is fail-safe.
# ═══════════════════════════════════════════════════════════════════

async def _run_spec_review_stage(
    result: PipelineResult,
    spec: Dict,
    intent_text: str,
    profile: Any,
    emit: Any,
    bvl_report: Optional[Any] = None,
    job_id: Optional[str] = None,
    manifest: Optional[Dict] = None,
) -> None:
    """Run the always-on spec reviewer and attach the report to result."""
    # Reviewer needs a BuildResult; if the builder didn't produce one,
    # there's nothing to review.
    if result.build_result is None or not result.build_result.all_files_written:
        emit("   📋 Spec review skipped: no builder output to review.")
        return

    # Try to grab a useful build-output string from the BVL report.
    build_output = _extract_build_output_from_bvl(bvl_report)

    from app.pipeline_v2.spec_review import run_spec_review

    review = await run_spec_review(
        spec=spec,
        build_result=result.build_result,
        profile=profile,
        intent_text=intent_text,
        bvl_report=bvl_report,
        build_output=build_output,
        emit=emit,
        job_id=job_id,
        manifest=manifest,
    )

    result.spec_review_report = review
    # Count the reviewer's single LLM call toward pipeline totals so the
    # final summary line includes it honestly.
    result.total_llm_calls += 1

    # Narrative row for the build project so the debug UI shows it.
    try:
        _push_narrative(
            stage="spec_review",
            title="Spec Review Complete",
            output_summary=review.summary_line(),
            sections=[
                {
                    "heading": "Summary",
                    "body": review.summary or "(no summary)",
                },
                {
                    "heading": "Findings",
                    "body": "\n".join(
                        f"- {f.one_line()}" for f in review.findings[:30]
                    ) or "(none)",
                },
                {
                    "heading": "Unmet requirements",
                    "body": "\n".join(
                        f"- {r}" for r in review.requirements_unmet[:20]
                    ) or "(none)",
                },
            ],
            model_used=review.model_used,
            duration_ms=int(review.duration_seconds * 1000),
            tokens_used=review.total_input_tokens + review.total_output_tokens,
            warnings=[f.title for f in review.findings if f.severity.value in ("critical", "major")][:5],
        )
    except Exception as _nerr:
        logger.debug("[orchestrator] Spec review narrative push failed: %s", _nerr)


def _extract_build_output_from_bvl(bvl_report: Optional[Any]) -> str:
    """Best-effort extraction of build/compile output from the BVL report.

    BVLReport shapes vary between tiers; we grab whatever free-text build
    output is available and cap it. Returns an empty string if nothing
    useful is present — the reviewer handles that gracefully.
    """
    if bvl_report is None:
        return ""
    try:
        components = getattr(bvl_report, "component_results", []) or []
        pieces = []
        for comp in components:
            for tier in (getattr(comp, "tier_results", []) or []):
                out = getattr(tier, "output", "") or getattr(tier, "stdout", "")
                if out:
                    pieces.append(str(out))
        return "\n---\n".join(pieces)[:8000]
    except Exception:
        return ""
