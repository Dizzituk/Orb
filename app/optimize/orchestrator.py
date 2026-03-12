# FILE: app/optimize/orchestrator.py
"""
Optimize Orchestrator — coordinates the four-phase cycle.

Single entry point for running an optimisation pass:
  Phase A: Decompose → ChunkManifest
  Phase B: Profile → ProfileResult
  Phase C: Propose → List[Proposal]
  Phase D: Execute → List[ExecutionResult] (user-approved only)

The orchestrator is not smart. It routes data between phases,
tracks progress, and assembles the final report.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.optimize.models import (
    ExecutionResult,
    OptimizeReport,
    Proposal,
    ProposalStatus,
)

logger = logging.getLogger(__name__)

# Target root paths
TARGET_ROOTS = {
    "astra-backend": "D:/Orb",
    "astra-frontend": "D:/orb-desktop",
    "driver-copilot": "D:/Astra Android Folder/AndroidDriverCopilot",
}


async def run_optimize_pass(
    target_id: str = "astra-backend",
    auto_approve_low_risk: bool = False,
    emit: Optional[Callable[[str], None]] = None,
) -> OptimizeReport:
    """Run a complete optimisation pass (phases A-C).

    Phase D (Execute) requires explicit approval per proposal.
    If auto_approve_low_risk is True, LOW risk proposals are
    auto-approved and executed.

    Args:
        target_id: Which target to optimise.
        auto_approve_low_risk: Auto-execute LOW risk proposals.
        emit: Progress callback.

    Returns:
        OptimizeReport with full results.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()
    target_root = TARGET_ROOTS.get(target_id, "D:/Orb")

    emit(f"\n{'='*60}")
    emit(f"⚡ OPTIMIZE PASS: {target_id}")
    emit(f"{'='*60}")

    report = OptimizeReport(target=target_id)

    # ── Phase A: Decompose ──
    emit(f"\n{'─'*40}")
    emit("🔍 PHASE A: DECOMPOSE")
    emit(f"{'─'*40}")

    from app.optimize.decomposer import decompose
    manifest = await decompose(target_root, target_id, emit)
    report.manifest = manifest

    # ── Phase B: Profile ──
    emit(f"\n{'─'*40}")
    emit("📊 PHASE B: PROFILE")
    emit(f"{'─'*40}")

    from app.optimize.profiler import profile
    profile_result = await profile(manifest, target_root, emit)
    report.profile = profile_result

    # ── Phase C: Propose ──
    emit(f"\n{'─'*40}")
    emit("💡 PHASE C: PROPOSE")
    emit(f"{'─'*40}")

    from app.optimize.proposer import propose
    proposals = await propose(manifest, profile_result, emit)
    report.proposals = proposals

    # ── Phase D: Execute (if auto-approved) ──
    if auto_approve_low_risk and proposals:
        emit(f"\n{'─'*40}")
        emit("🔧 PHASE D: EXECUTE (auto-approved LOW risk)")
        emit(f"{'─'*40}")

        for p in proposals:
            if p.risk.value == "low":
                p.status = ProposalStatus.APPROVED

        approved_count = sum(1 for p in proposals if p.status == ProposalStatus.APPROVED)
        if approved_count > 0:
            emit(f"   Auto-approved {approved_count} LOW-risk proposals")

            profile_snapshot = _snapshot_profile(profile_result)

            from app.optimize.executor import execute_batch
            results = await execute_batch(
                proposals, target_root, profile_snapshot, emit,
            )
            report.execution_results = results

            # Learn patterns from successful executions
            from app.optimize.pattern_learner import get_pattern_learner
            learner = get_pattern_learner()
            for p, r in zip(
                [p for p in proposals if p.status != ProposalStatus.PENDING],
                results,
            ):
                if r.success:
                    learner.learn_from_execution(p, r)

    # ── Summary ──
    report.total_duration_seconds = time.time() - t_start
    report.total_token_cost = sum(
        r.token_cost for r in report.execution_results
    )

    emit(f"\n{'='*60}")
    emit(f"⚡ OPTIMIZE PASS COMPLETE ({report.total_duration_seconds:.1f}s)")
    emit(f"   Chunks: {manifest.total_files}")
    emit(f"   Bottlenecks: {len(profile_result.bottlenecks)}")
    emit(f"   Proposals: {len(proposals)}")
    emit(f"   Executed: {report.executed_count}")
    emit(f"   Passed: {report.success_count}")
    emit(f"{'='*60}")

    return report


async def execute_approved(
    proposals: List[Proposal],
    target_id: str = "astra-backend",
    profile_snapshot: Optional[Dict[str, Any]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> List[ExecutionResult]:
    """Execute previously approved proposals.

    Called from the UI after the user reviews and approves proposals.
    """
    emit = emit or (lambda msg: None)
    target_root = TARGET_ROOTS.get(target_id, "D:/Orb")

    from app.optimize.executor import execute_batch
    results = await execute_batch(
        proposals, target_root, profile_snapshot, emit,
    )

    # Learn from successes
    from app.optimize.pattern_learner import get_pattern_learner
    learner = get_pattern_learner()
    approved = [p for p in proposals if p.status != ProposalStatus.PENDING]
    for p, r in zip(approved, results):
        if r.success:
            learner.learn_from_execution(p, r)

    return results


def _snapshot_profile(profile_result) -> Dict[str, Any]:
    """Create a lookup dict from profile results for before/after comparison."""
    snapshot = {}
    for m in profile_result.chunk_metrics:
        snapshot[m.path] = {
            "size_bytes": m.size_bytes,
            "line_count": m.line_count,
            "cyclomatic_complexity": m.cyclomatic_complexity,
            "coupling_in": m.coupling_in,
            "coupling_out": m.coupling_out,
        }
    return snapshot
