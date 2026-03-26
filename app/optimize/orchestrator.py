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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from app.optimize.models import (
    CodeChunk,
    ExecutionResult,
    OptimizeReport,
    Proposal,
    ProposalStatus,
)
from app.optimize.target_registry import get_target_definition

logger = logging.getLogger(__name__)

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


@dataclass
class _PassContext:
    pass_number: int
    pass_start: float
    profile_result: Any
    proposals: List[Proposal]
    complexity_before: float


@dataclass(frozen=True)
class _StopDecision:
    reason: str
    message: str


@dataclass(frozen=True)
class _ExecutionStats:
    executed: int
    passed: int
    failed: int


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
    emit(f"⚡ OPTIMIZE PASS: {target.display_label}")
    emit(f"   Scope: {target.user_outcome}")
    emit(f"   Root: {target.root_path}")
    emit(f"{'=' * 60}")

    report = OptimizeReport(target=target.target_id)

    emit(f"\n{'─' * 40}")
    emit("🔍 PHASE A: DECOMPOSE")
    emit(f"{'─' * 40}")
    manifest = await _decompose_target(target, emit)
    report.manifest = manifest

    emit(f"\n{'─' * 40}")
    emit("📊 PHASE B: PROFILE")
    emit(f"{'─' * 40}")
    profile_result = await _profile_manifest(manifest, target.root_path, emit)
    report.profile = profile_result

    emit(f"\n{'─' * 40}")
    emit("💡 PHASE C: PROPOSE")
    emit(f"{'─' * 40}")
    proposals = await _propose_changes(manifest, profile_result, emit)
    report.proposals = proposals

    if auto_approve_low_risk and proposals:
        emit(f"\n{'─' * 40}")
        emit("🔧 PHASE D: EXECUTE (auto-approved LOW risk)")
        emit(f"{'─' * 40}")
        approved_count = _approve_low_risk_proposals(proposals)
        if approved_count > 0:
            emit(f"   Auto-approved {approved_count} LOW-risk proposals")
            snapshot = _snapshot_profile(profile_result)
            results = await _execute_proposals(proposals, target.root_path, snapshot, emit)
            report.execution_results = results
            _learn_from_results(proposals, results)

    report.total_duration_seconds = time.time() - t_start
    report.total_token_cost = sum(r.token_cost for r in report.execution_results)

    emit(f"\n{'=' * 60}")
    emit(f"⚡ OPTIMIZE PASS COMPLETE ({report.total_duration_seconds:.1f}s)")
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

async def run_recursive_optimize(
    target_id: str,
    max_passes: int = MAX_PASSES,
    emit: Optional[Callable[[str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> RecursiveLoopResult:
    """Run the full recursive optimisation loop."""
    emit = emit or (lambda msg: None)
    target = get_target_definition(target_id)
    t_start = time.time()
    result = _create_recursive_result(target_id, target.display_label)

    emit(f"\n{'=' * 60}")
    emit(f"🔁 RECURSIVE OPTIMIZE: {target.display_label}")
    emit(f"   Max passes: {max_passes}")
    emit(f"   Scope: {target.user_outcome}")
    emit(f"{'=' * 60}")

    prev_proposal_count = None
    # Track all files created or modified across passes to protect from dead code detection
    protected_paths: set = set()

    for pass_num in range(1, max_passes + 1):
        emit(f"\n🔄 PASS {pass_num}/{max_passes}")
        emit(f"{'─' * 40}")
        pass_context = await _run_recursive_pass(target, pass_num, emit, protected_paths=protected_paths)
        result.total_proposals_found += len(pass_context.proposals)

        pre_execution_decision = _get_pre_execution_stop_decision(pass_context, prev_proposal_count)
        if pre_execution_decision is not None:
            _stop_with_summary(result, pass_context, pre_execution_decision, emit)
            break

        prev_proposal_count = len(pass_context.proposals)

        if should_stop and should_stop():
            emit("\n⏹️ Stop requested before execution.")
            result.stop_reason = "Stop requested by user"
            result.total_passes = pass_num
            break

        execution_data = await _execute_recursive_pass(target, pass_context, emit, protected_paths=protected_paths)
        _update_recursive_totals(result, execution_data["results"])
        _append_pass_summary(result, execution_data["summary"])
        _emit_pass_completion(emit, execution_data["summary"])

        # Track files touched in this pass so future passes don't flag them as dead
        _track_modified_files(pass_context.proposals, execution_data["results"], protected_paths)

        # Boot check: verify sandbox still boots after this pass
        boot_ok = await _run_sandbox_boot_check(target.root_path, emit)
        if not boot_ok:
            emit("\n❌ Sandbox boot check failed after this pass. Stopping to protect stability.")
            result.stop_reason = f"Boot check failed after pass {pass_num}"
            result.total_passes = pass_num
            break

        post_execution_decision = _get_post_execution_stop_decision(execution_data["summary"])
        if post_execution_decision is not None:
            emit(f"\n{post_execution_decision.message}")
            result.stop_reason = post_execution_decision.reason
            break
    else:
        result.stop_reason = f"Reached maximum {max_passes} passes"

    result.total_duration_seconds = time.time() - t_start
    _emit_recursive_completion(result, emit)
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
    results = await _execute_proposals(
        proposals,
        get_target_definition(target_id).root_path,
        profile_snapshot,
        emit,
    )
    _learn_from_results(proposals, results)
    return results


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def _create_recursive_result(target_id: str, target_label: str) -> RecursiveLoopResult:
    return RecursiveLoopResult(
        target_id=target_id,
        target_label=target_label,
        total_passes=0,
        total_proposals_found=0,
        total_executed=0,
        total_passed=0,
        total_failed=0,
        stop_reason="",
    )


async def _run_recursive_pass(target: Any, pass_number: int, emit: Callable[[str], None], protected_paths: Optional[set] = None) -> _PassContext:
    pass_start = time.time()
    manifest = await _decompose_target(target, emit, protected_paths=protected_paths)
    complexity_before = _sum_manifest_complexity(manifest)
    profile_result = await _profile_manifest(manifest, target.root_path, emit)
    proposals = await _propose_changes(manifest, profile_result, emit)
    return _PassContext(
        pass_number=pass_number,
        pass_start=pass_start,
        profile_result=profile_result,
        proposals=proposals,
        complexity_before=complexity_before,
    )


async def _decompose_target(target: Any, emit: Callable[[str], None], protected_paths: Optional[set] = None):
    from app.optimize.decomposer import decompose
    return await decompose(target, emit, protected_paths=protected_paths)


async def _profile_manifest(manifest: Any, root_path: str, emit: Callable[[str], None]):
    from app.optimize.profiler import profile
    return await profile(manifest, root_path, emit)


async def _propose_changes(manifest: Any, profile_result: Any, emit: Callable[[str], None]) -> List[Proposal]:
    from app.optimize.proposer import propose
    return await propose(manifest, profile_result, emit)


async def _execute_proposals(
    proposals: List[Proposal],
    root_path: str,
    profile_snapshot: Optional[Dict[str, Any]],
    emit: Callable[[str], None],
) -> List[ExecutionResult]:
    from app.optimize.executor import execute_batch
    return await execute_batch(proposals, root_path, profile_snapshot, emit)


def _sum_manifest_complexity(manifest: Any) -> float:
    return sum(chunk.complexity_estimate for chunk in manifest.chunks)


def _approve_low_risk_proposals(proposals: List[Proposal]) -> int:
    approved_count = 0
    for proposal in proposals:
        if proposal.risk.value == "low":
            proposal.status = ProposalStatus.APPROVED
            approved_count += 1
    return approved_count


def _get_pre_execution_stop_decision(
    pass_context: _PassContext,
    prev_proposal_count: Optional[int],
) -> Optional[_StopDecision]:
    proposals_found = len(pass_context.proposals)
    if proposals_found == 0:
        return _StopDecision(
            reason="No proposals — system is at optimum for this scope",
            message="\n✅ No proposals found — system is clean for this scope.",
        )
    if prev_proposal_count is not None and proposals_found >= prev_proposal_count:
        return _StopDecision(
            reason=f"No improvement — proposals unchanged at {proposals_found}",
            message=(
                f"\n⚠️ No improvement — {proposals_found} proposals "
                f"(was {prev_proposal_count}). Stopping."
            ),
        )
    return None


def _stop_with_summary(
    result: RecursiveLoopResult,
    pass_context: _PassContext,
    decision: _StopDecision,
    emit: Callable[[str], None],
) -> None:
    emit(decision.message)
    _append_pass_summary(result, _build_no_execution_summary(pass_context))
    result.stop_reason = decision.reason


def _build_no_execution_summary(pass_context: _PassContext) -> LoopPassSummary:
    return LoopPassSummary(
        pass_number=pass_context.pass_number,
        proposals_found=len(pass_context.proposals),
        proposals_executed=0,
        proposals_passed=0,
        proposals_failed=0,
        complexity_before=pass_context.complexity_before,
        complexity_after=pass_context.complexity_before,
        duration_seconds=time.time() - pass_context.pass_start,
    )


async def _execute_recursive_pass(target: Any, pass_context: _PassContext, emit: Callable[[str], None], protected_paths: Optional[set] = None) -> Dict[str, Any]:
    before_code = _capture_before_code(pass_context.proposals, target.root_path)
    emit(f"\n🔧 Executing {len(pass_context.proposals)} proposals...")
    _approve_all_proposals(pass_context.proposals)

    snapshot = _snapshot_profile(pass_context.profile_result)
    exec_results = await _execute_proposals(pass_context.proposals, target.root_path, snapshot, emit)
    _learn_from_results(pass_context.proposals, exec_results)
    await _learn_code_lessons(
        pass_context.proposals,
        exec_results,
        before_code,
        snapshot,
        target.root_path,
        emit,
    )

    manifest_after = await _decompose_target(target, emit, protected_paths=protected_paths)
    complexity_after = _sum_manifest_complexity(manifest_after)
    summary = _build_execution_summary(pass_context, exec_results, complexity_after)
    return {"results": exec_results, "summary": summary}


def _capture_before_code(proposals: List[Proposal], root_path: str) -> Dict[str, str]:
    from app.optimize.code_learner import capture_before_snapshot
    return capture_before_snapshot(_collect_target_chunks(proposals), root_path)


def _collect_target_chunks(proposals: List[Proposal]) -> List[str]:
    target_chunks: List[str] = []
    for proposal in proposals:
        target_chunks.extend(proposal.target_chunks)
    return target_chunks


def _approve_all_proposals(proposals: List[Proposal]) -> None:
    for proposal in proposals:
        proposal.status = ProposalStatus.APPROVED


def _build_execution_summary(
    pass_context: _PassContext,
    exec_results: List[ExecutionResult],
    complexity_after: float,
) -> LoopPassSummary:
    stats = _summarize_execution_results(exec_results)
    return LoopPassSummary(
        pass_number=pass_context.pass_number,
        proposals_found=len(pass_context.proposals),
        proposals_executed=stats.executed,
        proposals_passed=stats.passed,
        proposals_failed=stats.failed,
        complexity_before=pass_context.complexity_before,
        complexity_after=complexity_after,
        duration_seconds=time.time() - pass_context.pass_start,
    )


def _summarize_execution_results(exec_results: List[ExecutionResult]) -> _ExecutionStats:
    passed = sum(1 for result in exec_results if result.success)
    return _ExecutionStats(
        executed=len(exec_results),
        passed=passed,
        failed=len(exec_results) - passed,
    )


def _update_recursive_totals(result: RecursiveLoopResult, exec_results: List[ExecutionResult]) -> None:
    stats = _summarize_execution_results(exec_results)
    result.total_executed += stats.executed
    result.total_passed += stats.passed
    result.total_failed += stats.failed


def _append_pass_summary(result: RecursiveLoopResult, summary: LoopPassSummary) -> None:
    result.passes.append(summary)
    result.total_passes = summary.pass_number


def _emit_pass_completion(emit: Callable[[str], None], summary: LoopPassSummary) -> None:
    emit(
        f"   Pass {summary.pass_number} complete: {summary.proposals_passed}/"
        f"{summary.proposals_executed} passed, complexity "
        f"{summary.complexity_before:.0f} → {summary.complexity_after:.0f}"
    )



async def _run_sandbox_boot_check(target_root: str, emit: Callable[[str], None]) -> bool:
    """Run 'from main import app' in the sandbox to verify the codebase still boots.

    Uses the sandbox shell/run API directly. Returns True if boot succeeds.
    """
    import httpx

    controller_url = "http://192.168.250.2:8765"
    venv_python = f"{target_root}\\.venv\\Scripts\\python.exe"
    payload = {
        "cmd": [venv_python, "-c", "from main import app; print('BOOT_OK')"],
        "cwd": target_root,
        "timeout_ms": 25000,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{controller_url}/shell/run", json=payload)
            data = resp.json()
        stdout = data.get("stdout", "")
        returncode = data.get("returncode", -1)
        if returncode == 0 and "BOOT_OK" in stdout:
            emit("   ✅ Boot check passed")
            return True
        stderr = data.get("stderr", "")
        err_lines = [l for l in stderr.strip().split("\n") if l.strip()]
        last_err = err_lines[-1] if err_lines else "Unknown error"
        emit(f"   ❌ Boot check FAILED: {last_err}")
        return False
    except Exception as exc:
        emit(f"   ❌ Boot check error: {exc}")
        return False

def _get_post_execution_stop_decision(summary: LoopPassSummary) -> Optional[_StopDecision]:
    for decision in (
        _get_failure_stop_decision(summary),
        _get_complexity_stop_decision(summary),
    ):
        if decision is not None:
            return decision
    return None


def _get_failure_stop_decision(summary: LoopPassSummary) -> Optional[_StopDecision]:
    if summary.proposals_failed <= summary.proposals_passed:
        return None
    return _StopDecision(
        reason=(
            f"Pass {summary.pass_number} had more failures than successes — "
            "stopping to protect stability"
        ),
        message=(
            f"❌ More failures ({summary.proposals_failed}) than successes "
            f"({summary.proposals_passed}). Stopping to avoid degradation."
        ),
    )


def _get_complexity_stop_decision(summary: LoopPassSummary) -> Optional[_StopDecision]:
    if summary.complexity_after < summary.complexity_before:
        return None
    return _StopDecision(
        reason=(
            f"Complexity unchanged at {summary.complexity_after:.0f} — "
            "further passes unlikely to help"
        ),
        message=(
            f"⚠️ Complexity did not decrease ({summary.complexity_before:.0f} → "
            f"{summary.complexity_after:.0f}). Stopping."
        ),
    )



def _track_modified_files(
    proposals: List[Proposal],
    results: List[ExecutionResult],
    protected_paths: set,
) -> None:
    """Record all files touched by successful proposals so future passes don't flag them as dead."""
    for proposal, result in zip(proposals, results):
        if not result.success:
            continue
        for chunk_path in proposal.target_chunks:
            protected_paths.add(chunk_path)
        # Also protect any new files created by file splits
        for path in getattr(result, "files_changed", []):
            protected_paths.add(path)
    if protected_paths:
        logger.info("[orchestrator] Protected paths (%d): %s", len(protected_paths), sorted(protected_paths)[:10])
def _emit_recursive_completion(result: RecursiveLoopResult, emit: Callable[[str], None]) -> None:
    emit(f"\n{'=' * 60}")
    emit("🔁 RECURSIVE OPTIMIZE COMPLETE")
    emit(f"   Passes: {result.total_passes}")
    emit(f"   Total proposals found: {result.total_proposals_found}")
    emit(f"   Executed: {result.total_executed}")
    emit(f"   Passed: {result.total_passed}")
    emit(f"   Failed: {result.total_failed}")
    emit(f"   Stop reason: {result.stop_reason}")
    emit(f"   Duration: {result.total_duration_seconds:.1f}s")
    emit(f"{'=' * 60}")


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

        successful_chunks = _collect_successful_chunks(proposals, results)
        if not successful_chunks:
            return

        after_code = capture_after_snapshot(successful_chunks, root_path)
        after_metrics = _profile_successful_chunks(successful_chunks, root_path)

        store = get_lesson_store()
        lessons_recorded = 0
        approved = [p for p in proposals if p.status != ProposalStatus.PENDING]
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
            emit(f"   📚 Recorded {lessons_recorded} code lesson(s)")

    except Exception as e:
        logger.debug("[orchestrator] Code lesson capture failed: %s", e)


def _collect_successful_chunks(proposals: List[Proposal], results: List[ExecutionResult]) -> List[str]:
    successful_chunks: List[str] = []
    approved = [p for p in proposals if p.status != ProposalStatus.PENDING]
    for proposal, result in zip(approved, results):
        if result.success:
            successful_chunks.extend(proposal.target_chunks)
    return successful_chunks


def _profile_successful_chunks(successful_chunks: List[str], root_path: str) -> Dict[str, Dict[str, Any]]:
    from app.optimize.profiler import _profile_chunk

    after_metrics: Dict[str, Dict[str, Any]] = {}
    for chunk_path in successful_chunks:
        file_path = Path(root_path) / chunk_path
        if not file_path.exists():
            continue
        try:
            stat = file_path.stat()
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            chunk = CodeChunk(
                path=chunk_path,
                lines=text.count(chr(10)) + 1,
                size_bytes=stat.st_size,
            )
            metric = _profile_chunk(chunk, root_path)
            after_metrics[chunk_path] = {
                "size_bytes": metric.size_bytes,
                "cyclomatic_complexity": metric.cyclomatic_complexity,
            }
        except Exception:
            continue
    return after_metrics
