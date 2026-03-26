# FILE: app/optimize/orchestrator.py
"""
Optimizer Orchestrator — Single-pass and recursive loop modes.

Single pass: scan → profile → propose (used for preview/manual mode).
Recursive loop: scan → propose → execute → re-scan → compare → repeat
until no meaningful improvement remains or max passes reached.

v2.0 (2026-03-26): Added recursive loop with evidence-based continuation.
v1.0: Single-pass only.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.optimize.models import (
    ExecutionResult, OptimizeReport, Proposal, ProposalStatus,
)
from app.optimize.target_registry import get_target_definition

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════
# SINGLE PASS (existing behaviour — used by /optimize/run)
# ══════════════════════════════════════════════════════════

async def run_optimize_pass(
    target_id: str = "astra-backend:optimize",
    auto_approve_low_risk: bool = False,
    emit: Optional[Callable[[str], None]] = None,
) -> OptimizeReport:
    """Run a single optimisation pass (phases A-C, optionally D)."""
    emit = emit or (lambda msg: None)
    t_start = time.time()
    target = get_target_definition(target_id)

    emit(f"\n{'=' * 60}")
    emit(f"\u26a1 OPTIMIZE PASS: {target.display_label}")
    emit(f"   Scope: {target.user_outcome}")
    emit(f"   Root: {target.root_path}")
    emit(f"{'=' * 60}")

    report = OptimizeReport(target=target.target_id)

    # Phase A: Decompose
    emit(f"\n{'\u2500' * 40}")
    emit("\U0001f50d PHASE A: DECOMPOSE")
    emit(f"{'\u2500' * 40}")
    from app.optimize.decomposer import decompose
    manifest = await decompose(target, emit)
    report.manifest = manifest

    # Phase B: Profile
    emit(f"\n{'\u2500' * 40}")
    emit("\U0001f4ca PHASE B: PROFILE")
    emit(f"{'\u2500' * 40}")
    from app.optimize.profiler import profile
    profile_result = await profile(manifest, target.root_path, emit)
    report.profile = profile_result

    # Phase C: Propose
    emit(f"\n{'\u2500' * 40}")
    emit("\U0001f4a1 PHASE C: PROPOSE")
    emit(f"{'\u2500' * 40}")
    from app.optimize.proposer import propose
    proposals = await propose(manifest, profile_result, emit)
    report.proposals = proposals

    # Phase D: Auto-execute low risk (if requested)
    if auto_approve_low_risk and proposals:
        emit(f"\n{'\u2500' * 40}")
        emit("\U0001f527 PHASE D: EXECUTE (auto-approved LOW risk)")
        emit(f"{'\u2500' * 40}")
        for p in proposals:
            if p.risk.value == "low":
                p.status = ProposalStatus.APPROVED
        approved_count = sum(1 for p in proposals if p.status == ProposalStatus.APPROVED)
        if approved_count > 0:
            emit(f"   Auto-approved {approved_count} LOW-risk proposals")
            snapshot = _snapshot_profile(profile_result)
            from app.optimize.executor import execute_batch
            results = await execute_batch(proposals, target.root_path, snapshot, emit)
            report.execution_results = results
            _learn_from_results(proposals, results)

    report.total_duration_seconds = time.time() - t_start
    report.total_token_cost = sum(r.token_cost for r in report.execution_results)

    emit(f"\n{'=' * 60}")
    emit(f"\u26a1 OPTIMIZE PASS COMPLETE ({report.total_duration_seconds:.1f}s)")
    emit(f"   Chunks: {manifest.total_files}")
    emit(f"   Bottlenecks: {len(profile_result.bottlenecks)}")
    emit(f"   Proposals: {len(proposals)}")
    emit(f"   Executed: {report.executed_count}")
    emit(f"   Passed: {report.success_count}")
    emit(f"{'=' * 60}")
    return report


# ══════════════════════════════════════════════════════════
# RECURSIVE LOOP
# ══════════════════════════════════════════════════════════

MAX_PASSES = 5  # Hard ceiling to prevent runaway

@dataclass
class LoopPassSummary:
    """Summary of one pass within the recursive loop."""
    pass_number: int
    proposals_found: int
    proposals_executed: int
    proposals_passed: int
    proposals_failed: int
    complexity_before: float
    complexity_after: float
    duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_number": self.pass_number,
            "proposals_found": self.proposals_found,
            "proposals_executed": self.proposals_executed,
            "proposals_passed": self.proposals_passed,
            "proposals_failed": self.proposals_failed,
            "complexity_before": self.complexity_before,
            "complexity_after": self.complexity_after,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class RecursiveLoopResult:
    """Result of a full recursive optimisation loop."""
    target_id: str
    target_label: str
    total_passes: int
    total_proposals_found: int
    total_executed: int
    total_passed: int
    total_failed: int
    stop_reason: str
    passes: List[LoopPassSummary] = field(default_factory=list)
    total_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_id": self.target_id,
            "target_label": self.target_label,
            "total_passes": self.total_passes,
            "total_proposals_found": self.total_proposals_found,
            "total_executed": self.total_executed,
            "total_passed": self.total_passed,
            "total_failed": self.total_failed,
            "stop_reason": self.stop_reason,
            "passes": [p.to_dict() for p in self.passes],
            "total_duration_seconds": self.total_duration_seconds,
        }


async def run_recursive_optimize(
    target_id: str,
    max_passes: int = MAX_PASSES,
    emit: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> RecursiveLoopResult:
    """Run the full recursive optimisation loop.

    Scans, proposes, executes ALL proposals, re-scans, and repeats
    until one of these stop conditions:
    - No proposals found (system is clean)
    - No improvement between passes (at optimum)
    - A pass had more failures than successes (risk of degradation)
    - Max passes reached (safety ceiling)
    """
    emit = emit or (lambda msg: None)
    target = get_target_definition(target_id)
    t_start = time.time()

    result = RecursiveLoopResult(
        target_id=target_id,
        target_label=target.display_label,
        total_passes=0,
        total_proposals_found=0,
        total_executed=0,
        total_passed=0,
        total_failed=0,
        stop_reason="",
    )

    emit(f"\n{'=' * 60}")
    emit(f"\U0001f501 RECURSIVE OPTIMIZE: {target.display_label}")
    emit(f"   Max passes: {max_passes}")
    emit(f"   Scope: {target.user_outcome}")
    emit(f"{'=' * 60}")

    prev_proposal_count = None

    for pass_num in range(1, max_passes + 1):
        pass_start = time.time()
        emit(f"\n\U0001f504 PASS {pass_num}/{max_passes}")
        emit(f"{'\u2500' * 40}")

        # Phase A: Decompose
        from app.optimize.decomposer import decompose
        manifest = await decompose(target, emit)

        complexity_before = sum(c.complexity_estimate for c in manifest.chunks)

        # Phase B: Profile
        from app.optimize.profiler import profile
        profile_result = await profile(manifest, target.root_path, emit)

        # Phase C: Propose
        from app.optimize.proposer import propose
        proposals = await propose(manifest, profile_result, emit)

        result.total_proposals_found += len(proposals)

        # Stop condition: no proposals
        if not proposals:
            emit(f"\n\u2705 No proposals found — system is clean for this scope.")
            result.stop_reason = "No proposals — system is at optimum for this scope"
            result.passes.append(LoopPassSummary(
                pass_number=pass_num, proposals_found=0,
                proposals_executed=0, proposals_passed=0, proposals_failed=0,
                complexity_before=complexity_before, complexity_after=complexity_before,
                duration_seconds=time.time() - pass_start,
            ))
            result.total_passes = pass_num
            break

        # Stop condition: no improvement (same or more proposals as last pass)
        if prev_proposal_count is not None and len(proposals) >= prev_proposal_count:
            emit(f"\n\u26a0\ufe0f No improvement — {len(proposals)} proposals (was {prev_proposal_count}). Stopping.")
            result.stop_reason = f"No improvement — proposals unchanged at {len(proposals)}"
            result.passes.append(LoopPassSummary(
                pass_number=pass_num, proposals_found=len(proposals),
                proposals_executed=0, proposals_passed=0, proposals_failed=0,
                complexity_before=complexity_before, complexity_after=complexity_before,
                duration_seconds=time.time() - pass_start,
            ))
            result.total_passes = pass_num
            break

        prev_proposal_count = len(proposals)

        # Check for external stop request
        if should_stop and should_stop():
            emit("\n\u23f9\ufe0f Stop requested before execution.")
            result.stop_reason = "Stop requested by user"
            result.total_passes = pass_num
            break

        # Capture code snapshots BEFORE execution (for learning)
        all_target_chunks = []
        for p in proposals:
            all_target_chunks.extend(p.target_chunks)
        from app.optimize.code_learner import capture_before_snapshot
        before_code = capture_before_snapshot(all_target_chunks, target.root_path)

        # Phase D: Execute ALL proposals
        emit(f"\n\U0001f527 Executing {len(proposals)} proposals...")
        for p in proposals:
            p.status = ProposalStatus.APPROVED

        snapshot = _snapshot_profile(profile_result)
        from app.optimize.executor import execute_batch
        exec_results = await execute_batch(proposals, target.root_path, snapshot, emit)

        passed = sum(1 for r in exec_results if r.success)
        failed = sum(1 for r in exec_results if not r.success)
        result.total_executed += len(exec_results)
        result.total_passed += passed
        result.total_failed += failed

        _learn_from_results(proposals, exec_results)

        # Learn structural lessons from successful executions
        await _learn_code_lessons(
            proposals, exec_results, before_code, snapshot,
            target.root_path, emit,
        )

        # Re-scan to get updated complexity
        manifest_after = await decompose(target, emit)
        complexity_after = sum(c.complexity_estimate for c in manifest_after.chunks)

        pass_summary = LoopPassSummary(
            pass_number=pass_num,
            proposals_found=len(proposals),
            proposals_executed=len(exec_results),
            proposals_passed=passed,
            proposals_failed=failed,
            complexity_before=complexity_before,
            complexity_after=complexity_after,
            duration_seconds=time.time() - pass_start,
        )
        result.passes.append(pass_summary)
        result.total_passes = pass_num

        emit(f"   Pass {pass_num} complete: {passed}/{len(exec_results)} passed, "
             f"complexity {complexity_before:.0f} \u2192 {complexity_after:.0f}")

        # Stop condition: more failures than successes
        if failed > passed:
            emit(f"\n\u274c More failures ({failed}) than successes ({passed}). Stopping to avoid degradation.")
            result.stop_reason = f"Pass {pass_num} had more failures than successes — stopping to protect stability"
            break

        # Stop condition: complexity didn't improve
        if complexity_after >= complexity_before:
            emit(f"\n\u26a0\ufe0f Complexity did not decrease ({complexity_before:.0f} \u2192 {complexity_after:.0f}). Stopping.")
            result.stop_reason = f"Complexity unchanged at {complexity_after:.0f} — further passes unlikely to help"
            break

    else:
        # Exhausted max passes
        result.stop_reason = f"Reached maximum {max_passes} passes"

    result.total_duration_seconds = time.time() - t_start

    emit(f"\n{'=' * 60}")
    emit(f"\U0001f501 RECURSIVE OPTIMIZE COMPLETE")
    emit(f"   Passes: {result.total_passes}")
    emit(f"   Total proposals found: {result.total_proposals_found}")
    emit(f"   Executed: {result.total_executed}")
    emit(f"   Passed: {result.total_passed}")
    emit(f"   Failed: {result.total_failed}")
    emit(f"   Stop reason: {result.stop_reason}")
    emit(f"   Duration: {result.total_duration_seconds:.1f}s")
    emit(f"{'=' * 60}")

    return result


# ══════════════════════════════════════════════════════════
# EXECUTE APPROVED (existing — used by /optimize/execute)
# ══════════════════════════════════════════════════════════

async def execute_approved(
    proposals: List[Proposal],
    target_id: str = "astra-backend:optimize",
    profile_snapshot: Optional[Dict[str, Any]] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> List[ExecutionResult]:
    """Execute previously approved proposals."""
    emit = emit or (lambda msg: None)
    from app.optimize.executor import execute_batch
    results = await execute_batch(proposals, get_target_definition(target_id).root_path, profile_snapshot, emit)
    _learn_from_results(proposals, results)
    return results


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

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


def _learn_from_results(proposals: List[Proposal], results: List[ExecutionResult]) -> None:
    """Record successful executions in the pattern learner."""
    try:
        from app.optimize.pattern_learner import get_pattern_learner
        learner = get_pattern_learner()
        approved = [p for p in proposals if p.status != ProposalStatus.PENDING]
        for proposal, result in zip(approved, results):
            if result.success:
                learner.learn_from_execution(proposal, result)
    except Exception as e:
        logger.debug("[orchestrator] Pattern learning failed: %s", e)



async def _learn_code_lessons(
    proposals: List[Proposal],
    results: List[ExecutionResult],
    before_code: Dict[str, str],
    profile_snapshot: Dict[str, Any],
    root_path: str,
    emit: Optional[Callable[[str], None]] = None,
) -> None:
    """Capture after-snapshots and record structural lessons for successful proposals."""
    emit = emit or (lambda msg: None)
    try:
        from app.optimize.code_learner import capture_after_snapshot, get_lesson_store

        # Collect all target chunks from successful proposals
        successful_chunks = []
        approved = [p for p in proposals if p.status != ProposalStatus.PENDING]
        for proposal, result in zip(approved, results):
            if result.success:
                successful_chunks.extend(proposal.target_chunks)

        if not successful_chunks:
            return

        # Capture code AFTER execution
        after_code = capture_after_snapshot(successful_chunks, root_path)

        # Re-profile changed files for after-metrics
        from app.optimize.profiler import _profile_chunk
        from app.optimize.models import CodeChunk
        from pathlib import Path

        after_metrics = {}
        for chunk_path in successful_chunks:
            fpath = Path(root_path) / chunk_path
            if fpath.exists():
                try:
                    stat = fpath.stat()
                    text = fpath.read_text(encoding='utf-8', errors='ignore')
                    chunk = CodeChunk(
                        path=chunk_path,
                        lines=text.count(chr(10)) + 1,
                        size_bytes=stat.st_size,
                    )
                    m = _profile_chunk(chunk, root_path)
                    after_metrics[chunk_path] = {
                        "size_bytes": m.size_bytes,
                        "cyclomatic_complexity": m.cyclomatic_complexity,
                    }
                except Exception:
                    pass

        # Record a lesson for each successful proposal
        store = get_lesson_store()
        lessons_recorded = 0
        for proposal, result in zip(approved, results):
            if not result.success:
                continue
            try:
                lesson = await store.record_lesson(
                    proposal=proposal,
                    before_snapshots=before_code,
                    after_snapshots=after_code,
                    metrics_before=profile_snapshot,
                    metrics_after=after_metrics,
                )
                if lesson:
                    lessons_recorded += 1
            except Exception as e:
                logger.debug("[orchestrator] Lesson recording failed for %s: %s", proposal.proposal_id, e)

        if lessons_recorded > 0:
            emit(f"   \U0001f4da Recorded {lessons_recorded} code lesson(s)")

    except Exception as e:
        logger.debug("[orchestrator] Code lesson capture failed: %s", e)

