# FILE: app/optimize/executor.py
"""
Phase D: Executor.

Converts approved proposals into optimisation specs and feeds them
through the existing Agentic Builder → BVL verification pipeline.

The Executor does NOT have its own code modification engine. It
generates specs that the Builder executes. This keeps the optimisation
path safe and verified.

Post-execution, it re-profiles to measure actual vs predicted improvement.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.optimize.models import (
    BeforeAfterMetrics,
    ExecutionResult,
    Proposal,
    ProposalStatus,
)

logger = logging.getLogger(__name__)


async def execute_proposal(
    proposal: Proposal,
    target_root: str,
    profile_before: Optional[Dict[str, Any]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> ExecutionResult:
    """Execute a single approved proposal through the pipeline.

    Args:
        proposal: The approved Proposal to execute.
        target_root: Root directory of the target.
        profile_before: Pre-execution metrics for comparison.
        emit: Progress callback.

    Returns:
        ExecutionResult with before/after metrics.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()

    emit(f"🔧 Execute: {proposal.title}")
    proposal.status = ProposalStatus.EXECUTING

    # Convert proposal to an optimisation spec
    opt_spec = _proposal_to_spec(proposal)
    emit(f"   Generated optimisation spec ({len(json.dumps(opt_spec))} chars)")

    # Run through the Agentic Builder
    try:
        build_result, bvl_report = await _run_through_pipeline(
            opt_spec, target_root, emit,
        )
    except Exception as e:
        logger.error("[executor] Pipeline execution failed: %s", e)
        proposal.status = ProposalStatus.FAILED
        return ExecutionResult(
            proposal_id=proposal.proposal_id,
            success=False,
            error=f"Pipeline failed: {e}",
            duration_seconds=time.time() - t_start,
        )

    # Determine success
    success = build_result is not None and (
        bvl_report is None or getattr(bvl_report, "all_signed_off", False)
    )
    bvl_passed = bvl_report is not None and getattr(bvl_report, "all_signed_off", False)

    # Post-execution profiling (re-measure affected chunks)
    metrics = []
    if success and profile_before:
        metrics = await _measure_delta(
            proposal.target_chunks, target_root, profile_before,
        )

    # Build result
    duration = time.time() - t_start
    files_changed = []
    if build_result and hasattr(build_result, "all_files_written"):
        files_changed = build_result.all_files_written

    result = ExecutionResult(
        proposal_id=proposal.proposal_id,
        success=success,
        bvl_passed=bvl_passed,
        metrics=metrics,
        files_changed=files_changed,
        token_cost=proposal.estimated_token_cost,
        duration_seconds=duration,
        ready_for_promotion=success and bvl_passed,
    )

    if success:
        proposal.status = ProposalStatus.PASSED
        emit(f"   ✅ Proposal passed ({duration:.1f}s)")
    else:
        proposal.status = ProposalStatus.FAILED
        emit(f"   ❌ Proposal failed ({duration:.1f}s)")

    return result


async def execute_batch(
    proposals: List[Proposal],
    target_root: str,
    profile_before: Optional[Dict[str, Any]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> List[ExecutionResult]:
    """Execute multiple approved proposals sequentially.

    Only executes proposals with APPROVED status. Each runs one at
    a time for safety.
    """
    emit = emit or (lambda msg: None)
    results = []

    approved = [p for p in proposals if p.status == ProposalStatus.APPROVED]
    emit(f"🔧 Executing {len(approved)} approved proposals...")

    for i, proposal in enumerate(approved, 1):
        emit(f"\n--- Proposal {i}/{len(approved)} ---")
        result = await execute_proposal(
            proposal, target_root, profile_before, emit,
        )
        results.append(result)

        # Pause between executions for safety
        if i < len(approved):
            import asyncio
            await asyncio.sleep(5)

    passed = sum(1 for r in results if r.success)
    emit(f"\n✅ Batch complete: {passed}/{len(results)} passed")

    return results


# ═══════════════════════════════════════════════════════════════════
# Spec generation
# ═══════════════════════════════════════════════════════════════════

def _proposal_to_spec(proposal: Proposal) -> Dict[str, Any]:
    """Convert a Proposal into a pipeline-compatible spec.

    The spec format matches what the Agentic Builder expects from
    SpecGate, so it flows through the existing pipeline.
    """
    return {
        "title": f"Optimisation: {proposal.title}",
        "summary": proposal.description,
        "type": "optimisation",
        "category": proposal.category.value,
        "requirements": [
            proposal.description,
            f"Risk level: {proposal.risk.value}",
            f"Target files: {', '.join(proposal.target_chunks)}",
        ],
        "file_scope": proposal.target_chunks,
        "constraints": [
            "Make SURGICAL changes only — do not rewrite entire files",
            "Preserve all existing tests and functionality",
            "If splitting files, update all import references",
            "Run syntax check after every file modification",
        ],
        "acceptance_criteria": [
            proposal.predicted_improvement,
            "No regressions in existing functionality",
            "All existing imports still resolve",
        ],
        "segments": [
            {
                "segment_id": "opt-main",
                "file_scope": proposal.target_chunks,
                "requirements": [proposal.description],
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Pipeline integration
# ═══════════════════════════════════════════════════════════════════

async def _run_through_pipeline(
    spec: Dict[str, Any],
    target_root: str,
    emit: Callable,
) -> tuple:
    """Feed the optimisation spec through the Agentic Builder.

    Returns (build_result, bvl_report) or (None, None) on failure.
    """
    try:
        from app.pipeline_v2.target_registry import get_default_profile
        from app.pipeline_v2.agentic_builder import run_agentic_builder
        from app.pipeline_v2.models import ScaffoldResult

        profile = get_default_profile()
        empty_scaffold = ScaffoldResult()
        manifest = spec  # The spec IS the manifest for optimisation jobs

        build_result = await run_agentic_builder(
            spec=spec,
            manifest=manifest,
            scaffold=empty_scaffold,
            job_dir="",
            handover_context=(
                f"This is an OPTIMISATION job, not a build job.\n"
                f"Category: {spec.get('category', '')}\n"
                f"Target files: {spec.get('file_scope', [])}\n"
                f"Read the target files, make the specified changes, "
                f"and verify the changes compile correctly."
            ),
            on_progress=emit,
            profile=profile,
        )

        bvl_report = None  # BVL runs from orchestrator, not here directly
        return build_result, bvl_report

    except Exception as e:
        logger.error("[executor] Pipeline run failed: %s", e)
        raise


# ═══════════════════════════════════════════════════════════════════
# Post-execution measurement
# ═══════════════════════════════════════════════════════════════════

async def _measure_delta(
    target_chunks: List[str],
    target_root: str,
    profile_before: Dict[str, Any],
) -> List[BeforeAfterMetrics]:
    """Re-profile affected chunks and compute deltas."""
    from app.optimize.profiler import _profile_chunk
    from app.optimize.models import CodeChunk
    from pathlib import Path

    metrics = []

    for chunk_path in target_chunks:
        fpath = Path(target_root) / chunk_path
        if not fpath.exists():
            # File was removed (dead code removal)
            before_size = profile_before.get(chunk_path, {}).get("size_bytes", 0)
            metrics.append(BeforeAfterMetrics(
                metric_name=f"size_bytes:{chunk_path}",
                before=before_size,
                after=0,
            ))
            continue

        # Create a minimal chunk for profiling
        try:
            stat = fpath.stat()
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            chunk = CodeChunk(
                path=chunk_path,
                lines=text.count("\n") + 1,
                size_bytes=stat.st_size,
            )
            new_metrics = _profile_chunk(chunk, target_root)

            before = profile_before.get(chunk_path, {})
            metrics.append(BeforeAfterMetrics(
                metric_name=f"size_bytes:{chunk_path}",
                before=before.get("size_bytes", 0),
                after=new_metrics.size_bytes,
            ))
            metrics.append(BeforeAfterMetrics(
                metric_name=f"complexity:{chunk_path}",
                before=before.get("cyclomatic_complexity", 0),
                after=new_metrics.cyclomatic_complexity,
            ))
        except Exception:
            pass

    return metrics
