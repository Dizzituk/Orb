from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from app.optimize.models import ExecutionResult, OptimizeReport, Proposal, ProposalStatus
from app.optimize.target_registry import get_target_definition

logger = logging.getLogger(__name__)


async def run_optimize_pass(
    target_id: str = "astra-backend:optimize",
    auto_approve_low_risk: bool = False,
    emit: Optional[Callable[[str], None]] = None,
) -> OptimizeReport:
    """Run a complete optimisation pass (phases A-C)."""
    emit = emit or (lambda msg: None)
    t_start = time.time()
    target = get_target_definition(target_id)

    emit(f"\n{'=' * 60}")
    emit(f"⚡ OPTIMIZE PASS: {target.display_label}")
    emit(f"   Scope: {target.user_outcome}")
    emit(f"   Root: {target.root_path}")
    emit(f"{'=' * 60}")

    report = OptimizeReport(target=target.target_id)

    emit(f"\n{'─' * 40}")
    emit("🔍 PHASE A: DECOMPOSE")
    emit(f"{'─' * 40}")
    from app.optimize.decomposer import decompose
    manifest = await decompose(target, emit)
    report.manifest = manifest

    emit(f"\n{'─' * 40}")
    emit("📊 PHASE B: PROFILE")
    emit(f"{'─' * 40}")
    from app.optimize.profiler import profile
    profile_result = await profile(manifest, target.root_path, emit)
    report.profile = profile_result

    emit(f"\n{'─' * 40}")
    emit("💡 PHASE C: PROPOSE")
    emit(f"{'─' * 40}")
    from app.optimize.proposer import propose
    proposals = await propose(manifest, profile_result, emit)
    report.proposals = proposals

    if auto_approve_low_risk and proposals:
        emit(f"\n{'─' * 40}")
        emit("🔧 PHASE D: EXECUTE (auto-approved LOW risk)")
        emit(f"{'─' * 40}")
        for proposal in proposals:
            if proposal.risk.value == "low":
                proposal.status = ProposalStatus.APPROVED

        approved_count = sum(1 for proposal in proposals if proposal.status == ProposalStatus.APPROVED)
        if approved_count > 0:
            emit(f"   Auto-approved {approved_count} LOW-risk proposals")
            profile_snapshot = _snapshot_profile(profile_result)
            from app.optimize.executor import execute_batch
            results = await execute_batch(proposals, target.root_path, profile_snapshot, emit)
            report.execution_results = results

            from app.optimize.pattern_learner import get_pattern_learner
            learner = get_pattern_learner()
            for proposal, result in zip([p for p in proposals if p.status != ProposalStatus.PENDING], results):
                if result.success:
                    learner.learn_from_execution(proposal, result)

    report.total_duration_seconds = time.time() - t_start
    report.total_token_cost = sum(result.token_cost for result in report.execution_results)

    emit(f"\n{'=' * 60}")
    emit(f"⚡ OPTIMIZE PASS COMPLETE ({report.total_duration_seconds:.1f}s)")
    emit(f"   Chunks: {manifest.total_files}")
    emit(f"   Bottlenecks: {len(profile_result.bottlenecks)}")
    emit(f"   Proposals: {len(proposals)}")
    emit(f"   Executed: {report.executed_count}")
    emit(f"   Passed: {report.success_count}")
    emit(f"{'=' * 60}")
    return report


async def execute_approved(
    proposals: List[Proposal],
    target_id: str = "astra-backend:optimize",
    profile_snapshot: Optional[Dict[str, Any]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> List[ExecutionResult]:
    """Execute previously approved proposals."""
    emit = emit or (lambda msg: None)
    target = get_target_definition(target_id)

    from app.optimize.executor import execute_batch
    results = await execute_batch(proposals, target.root_path, profile_snapshot, emit)

    from app.optimize.pattern_learner import get_pattern_learner
    learner = get_pattern_learner()
    approved = [proposal for proposal in proposals if proposal.status != ProposalStatus.PENDING]
    for proposal, result in zip(approved, results):
        if result.success:
            learner.learn_from_execution(proposal, result)
    return results


def _snapshot_profile(profile_result) -> Dict[str, Any]:
    snapshot = {}
    for metric in profile_result.chunk_metrics:
        snapshot[metric.path] = {
            "size_bytes": metric.size_bytes,
            "line_count": metric.line_count,
            "cyclomatic_complexity": metric.cyclomatic_complexity,
            "coupling_in": metric.coupling_in,
            "coupling_out": metric.coupling_out,
        }
    return snapshot
