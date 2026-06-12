# FILE: app/debug/orchestrator/loop_controller.py
# Purpose: Loop controller — the outer investigate -> plan -> execute -> verify loop.
# Called-by: app.debug.orchestrator.chat_stream_bridge, app.debug.orchestrator.endpoint
# Depends-on: app.debug.orchestrator.behaviour_verifier, app.debug.orchestrator.decomposer, app.debug.orchestrator.planner, app.debug.orchestrator.schemas (+1 more)
# Last-renovated: 2026-06-11
"""
Loop controller — the outer investigate -> plan -> execute -> verify loop.

Responsibilities:
  - Drive the orchestration through each phase
  - Emit SSE-style events for live UI updates
  - Enforce iteration cap and detect regression loops (same failure twice)
  - Convert a DebugPlan into executor briefs + run them batch by batch
  - Produce a final DebugResolution

Never raises — all failures flow into DebugResolution with final_phase=FAILED.

v1.0 (2026-04-13): Initial implementation.
"""
from __future__ import annotations

import logging
import os
import time
from typing import AsyncGenerator, Callable, Dict, List, Optional

from app.debug.orchestrator.behaviour_verifier import (
    code_verification_check,
    run_behaviour_checks,
)
from app.debug.orchestrator.decomposer import decompose
from app.debug.orchestrator.planner import plan_fixes, topological_batches
from app.debug.orchestrator.schemas import (
    BehaviourCheck,
    DebugPlan,
    DebugResolution,
    DecompositionResult,
    FixStep,
    IterationRecord,
    OrchestrationEvent,
    OrchestrationPhase,
    OrchestrationRequest,
    StepStatus,
    SubagentBrief,
    SubagentReport,
    SubagentRole,
    VerificationResult,
)
from app.debug.orchestrator.subagent_runner import (
    run_subagent,
    run_subagents_parallel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model routing (matches the existing {STAGE}_PROVIDER/{STAGE}_MODEL pattern)
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v else default


DECOMPOSER_MODEL = _env("DEBUG_DECOMPOSER_MODEL", _env("OPENAI_DEFAULT_MODEL", "gpt-5.4"))
PLANNER_MODEL = _env("DEBUG_PLANNER_MODEL", _env("OPENAI_DEFAULT_MODEL", "gpt-5.4"))
SUBAGENT_MODEL = _env("DEBUG_SUBAGENT_MODEL", _env("OPENAI_DEFAULT_MODEL", "gpt-5.4-mini"))
EXECUTOR_MODEL = _env("DEBUG_EXECUTOR_MODEL", SUBAGENT_MODEL)
CODE_VERIFIER_MODEL = _env("DEBUG_CODE_VERIFIER_MODEL", SUBAGENT_MODEL)


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------

EventCallback = Callable[[OrchestrationEvent], None]


def _emit(cb: Optional[EventCallback], event_type: str, phase: OrchestrationPhase,
          iteration: int, message: str = "", data: Optional[Dict] = None) -> None:
    if cb is None:
        return
    try:
        cb(OrchestrationEvent(
            event_type=event_type,
            iteration=iteration,
            phase=phase,
            message=message,
            data=data or {},
        ))
    except Exception as e:
        logger.warning("[loop_controller] Event callback raised: %s", e)


# ---------------------------------------------------------------------------
# Plan -> executor briefs
# ---------------------------------------------------------------------------

def _executor_brief_for_step(step: FixStep, root_cause: str) -> SubagentBrief:
    task = (
        f"Apply this specific fix: {step.description}\n\n"
        f"Rationale (from investigation): {step.rationale}\n\n"
        f"Synthesised root cause (shared context): {root_cause}\n\n"
        f"Stay within these files: {step.target_files or '(none specified — be conservative)'}."
    )
    return SubagentBrief(
        brief_id=f"exec_{step.step_id}",
        role=SubagentRole.EXECUTOR,
        task=task,
        target_project=step.target_project,
        context_files=step.target_files,
        related_bugs=[],
        depends_on=[],  # Topological ordering already handled by batching
        max_tool_calls=200,
    )


# ---------------------------------------------------------------------------
# Code verification helper — runs declared or default code check
# ---------------------------------------------------------------------------

def _default_code_checks(plan: DebugPlan) -> List[BehaviourCheck]:
    """Sensible default code checks derived from the plan's target files."""
    checks: List[BehaviourCheck] = []
    py_files = sorted({
        f for s in plan.steps for f in s.target_files
        if f.endswith(".py")
    })
    if py_files:
        # Single quick syntax check across all modified python files
        joined = " ".join(f'"{p}"' for p in py_files[:20])
        checks.append(code_verification_check(
            check_id="py_syntax",
            command=f"python -m py_compile {joined}",
            timeout_seconds=30,
        ))
    # TS/JS/TSX syntax via node check is project-dependent — skip default for now
    return checks


# ---------------------------------------------------------------------------
# Regression detection
# ---------------------------------------------------------------------------

def _signatures_from_verifications(results: List[VerificationResult]) -> List[str]:
    return [r.failure_signature for r in results if r.failure_signature]


def _regression_detected(
    prev_failures: List[str], current_failures: List[str],
) -> bool:
    """Return True if current failures repeat prior ones (ping-pong loop)."""
    if not prev_failures or not current_failures:
        return False
    return bool(set(prev_failures).intersection(set(current_failures)))


# ---------------------------------------------------------------------------
# Iteration summary (seeded into next iteration's planner prompt)
# ---------------------------------------------------------------------------

def _build_iteration_summary(record: IterationRecord) -> str:
    parts = [
        f"Iteration {record.iteration} reached {record.phase_reached.value}.",
        f"- Investigation: {len(record.reports)} subagent(s)",
        f"- Plan steps: {len(record.plan.steps) if record.plan else 0}",
        f"- Execution reports: {len(record.execution_reports)}",
    ]
    code_fail = [v for v in record.code_verifications if v.status != StepStatus.PASSED]
    behav_fail = [v for v in record.behaviour_verifications if v.status != StepStatus.PASSED]
    if code_fail:
        parts.append(f"- Code verifications failed: {len(code_fail)}")
        for v in code_fail[:3]:
            parts.append(f"  - {v.check_id}: {v.message[:160]}")
    if behav_fail:
        parts.append(f"- Behaviour verifications failed: {len(behav_fail)}")
        for v in behav_fail[:3]:
            parts.append(f"  - {v.check_id}: {v.message[:160]}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

async def run_orchestration(
    request: OrchestrationRequest,
    on_event: Optional[EventCallback] = None,
    behaviour_checks_provider: Optional[Callable[[DebugPlan], List[BehaviourCheck]]] = None,
) -> DebugResolution:
    """Run the full investigate -> plan -> execute -> verify loop.

    Args:
        request: OrchestrationRequest from the frontend.
        on_event: Optional callback for live SSE-style events.
        behaviour_checks_provider: Optional function that returns behaviour
            checks for a given plan. Falls back to empty list (skip behaviour
            verification) if not provided. The endpoint wires this in from
            the project metadata.

    Returns:
        DebugResolution — the full record of what happened.
    """
    t_start = time.time()
    iterations: List[IterationRecord] = []
    prior_failure_signatures: List[str] = []
    prior_iteration_summary: Optional[str] = None
    final_phase = OrchestrationPhase.FAILED
    resolved = False
    total_tokens = 0
    iteration_no = 0

    try:
        for iteration_no in range(1, request.max_iterations + 1):
            iter_start = time.time()
            record = IterationRecord(
                iteration=iteration_no,
                phase_reached=OrchestrationPhase.DECOMPOSING,
            )

            # ---- 1. DECOMPOSE ------------------------------------------
            _emit(on_event, "phase_change", OrchestrationPhase.DECOMPOSING,
                  iteration_no, "Decomposing bug list")
            decomposition = await decompose(
                bug_list=request.bug_list,
                target_project=request.target_project,
                model=DECOMPOSER_MODEL,
            )
            record.decomposition = decomposition
            record.phase_reached = OrchestrationPhase.INVESTIGATING
            _emit(on_event, "decomposition_complete", OrchestrationPhase.DECOMPOSING,
                  iteration_no,
                  f"{len(decomposition.briefs)} investigation brief(s)",
                  {"brief_ids": [b.brief_id for b in decomposition.briefs]})

            # ---- 2. INVESTIGATE (parallel) -----------------------------
            _emit(on_event, "phase_change", OrchestrationPhase.INVESTIGATING,
                  iteration_no, "Running investigators in parallel")

            def _on_tool_call(name: str, args: Dict) -> None:
                _emit(on_event, "subagent_progress", OrchestrationPhase.INVESTIGATING,
                      iteration_no, f"tool: {name}", {"tool": name})

            reports = await run_subagents_parallel(
                briefs=decomposition.briefs,
                model=SUBAGENT_MODEL,
                max_parallel=request.max_subagents_parallel,
                extra_context=prior_iteration_summary,
                on_tool_call=_on_tool_call,
            )
            record.reports = reports
            total_tokens += sum(r.tokens_used for r in reports)
            _emit(on_event, "subagent_complete", OrchestrationPhase.INVESTIGATING,
                  iteration_no, f"{len(reports)} investigation report(s) in",
                  {"findings_total": sum(len(r.findings) for r in reports)})

            # ---- 3. PLAN -----------------------------------------------
            record.phase_reached = OrchestrationPhase.PLANNING
            _emit(on_event, "phase_change", OrchestrationPhase.PLANNING,
                  iteration_no, "Synthesising fix plan")
            plan = await plan_fixes(
                bug_list=request.bug_list,
                reports=reports,
                model=PLANNER_MODEL,
                prior_iteration_summary=prior_iteration_summary,
            )
            record.plan = plan
            _emit(on_event, "plan_complete", OrchestrationPhase.PLANNING,
                  iteration_no,
                  f"{len(plan.steps)} step(s), confidence={plan.confidence:.2f}",
                  {"steps": [s.step_id for s in plan.steps],
                   "contradictions": plan.contradictions})

            if not plan.steps:
                logger.warning("[loop_controller] Planner returned no steps — stopping")
                final_phase = OrchestrationPhase.FAILED
                record.elapsed_ms = int((time.time() - iter_start) * 1000)
                iterations.append(record)
                break

            # ---- 4. EXECUTE (batched by dependency) --------------------
            record.phase_reached = OrchestrationPhase.EXECUTING
            _emit(on_event, "phase_change", OrchestrationPhase.EXECUTING,
                  iteration_no, "Applying fixes")

            batches = topological_batches(plan)
            exec_reports: List[SubagentReport] = []
            for batch_idx, batch in enumerate(batches):
                briefs = [_executor_brief_for_step(s, plan.root_cause) for s in batch]
                # Only parallelise within batch if the steps are marked parallelisable
                can_parallel = (
                    len(briefs) > 1
                    and all(s.parallelisable_with for s in batch)
                )
                max_par = request.max_subagents_parallel if can_parallel else 1
                _emit(on_event, "execution_start", OrchestrationPhase.EXECUTING,
                      iteration_no,
                      f"batch {batch_idx+1}/{len(batches)}: {len(briefs)} step(s), parallel={can_parallel}",
                      {"batch": [b.brief_id for b in briefs]})
                batch_reports = await run_subagents_parallel(
                    briefs=briefs,
                    model=EXECUTOR_MODEL,
                    max_parallel=max_par,
                    extra_context=None,
                    on_tool_call=_on_tool_call,
                )
                exec_reports.extend(batch_reports)
                total_tokens += sum(r.tokens_used for r in batch_reports)

                # Abort batch progression if any executor hard-failed
                if any(r.status == StepStatus.FAILED and r.error for r in batch_reports):
                    logger.warning("[loop_controller] Batch %d had hard failures — stopping execution", batch_idx + 1)
                    break

            record.execution_reports = exec_reports
            _emit(on_event, "execution_complete", OrchestrationPhase.EXECUTING,
                  iteration_no, f"{len(exec_reports)} executor report(s)",
                  {"files_modified": sorted({f for r in exec_reports for f in r.files_modified})})

            # ---- 5. VERIFY (code) --------------------------------------
            record.phase_reached = OrchestrationPhase.VERIFYING_CODE
            _emit(on_event, "phase_change", OrchestrationPhase.VERIFYING_CODE,
                  iteration_no, "Running code verification")
            code_checks = _default_code_checks(plan)
            if code_checks:
                code_results = await run_behaviour_checks(code_checks)
                # Flip check_type since these are really code checks
                for r in code_results:
                    r.check_type = "code"
                record.code_verifications = code_results
                _emit(on_event, "verification_complete", OrchestrationPhase.VERIFYING_CODE,
                      iteration_no,
                      f"code: {sum(1 for r in code_results if r.status == StepStatus.PASSED)}/{len(code_results)} passed",
                      {"results": [r.model_dump(mode='json') for r in code_results]})

            code_passed = all(
                r.status == StepStatus.PASSED for r in record.code_verifications
            ) if record.code_verifications else True

            # ---- 6. VERIFY (behaviour) ---------------------------------
            behaviour_passed = True
            if request.enable_behaviour_verify and behaviour_checks_provider:
                record.phase_reached = OrchestrationPhase.VERIFYING_BEHAVIOUR
                _emit(on_event, "phase_change", OrchestrationPhase.VERIFYING_BEHAVIOUR,
                      iteration_no, "Running behaviour verification")
                try:
                    behav_checks = behaviour_checks_provider(plan) or []
                except Exception as e:
                    logger.warning("[loop_controller] behaviour_checks_provider raised: %s", e)
                    behav_checks = []
                if behav_checks:
                    behav_results = await run_behaviour_checks(behav_checks)
                    record.behaviour_verifications = behav_results
                    behaviour_passed = all(
                        r.status == StepStatus.PASSED for r in behav_results
                    )
                    _emit(on_event, "verification_complete", OrchestrationPhase.VERIFYING_BEHAVIOUR,
                          iteration_no,
                          f"behaviour: {sum(1 for r in behav_results if r.status == StepStatus.PASSED)}/{len(behav_results)} passed",
                          {"results": [r.model_dump(mode='json') for r in behav_results]})

            # ---- 7. DECIDE NEXT MOVE -----------------------------------
            record.passed = code_passed and behaviour_passed
            record.elapsed_ms = int((time.time() - iter_start) * 1000)
            iterations.append(record)

            _emit(on_event, "iteration_complete", record.phase_reached, iteration_no,
                  f"iteration {iteration_no} passed={record.passed}",
                  {"code_passed": code_passed, "behaviour_passed": behaviour_passed})

            if record.passed:
                final_phase = OrchestrationPhase.RESOLVED
                resolved = True
                break

            # Regression check
            current_failures = (
                _signatures_from_verifications(record.code_verifications)
                + _signatures_from_verifications(record.behaviour_verifications)
            )
            if _regression_detected(prior_failure_signatures, current_failures):
                logger.warning("[loop_controller] Regression detected — stopping loop")
                final_phase = OrchestrationPhase.FAILED
                break

            # Otherwise carry failures into next iteration's planner context
            prior_failure_signatures = current_failures
            prior_iteration_summary = _build_iteration_summary(record)

        else:
            # Loop exited because max_iterations was reached without success
            final_phase = OrchestrationPhase.MAX_ITERATIONS

    except Exception as e:
        logger.exception("[loop_controller] Unhandled error: %s", e)
        final_phase = OrchestrationPhase.FAILED
        _emit(on_event, "error", final_phase, iteration_no,
              f"orchestration crashed: {type(e).__name__}: {e}")

    # ---- Build resolution ------------------------------------------
    total_elapsed_ms = int((time.time() - t_start) * 1000)
    summary = _build_resolution_summary(iterations, final_phase, resolved)
    surfaced_contradictions: List[str] = []
    unresolved_bugs: List[str] = []
    for ir in iterations:
        if ir.plan:
            surfaced_contradictions.extend(ir.plan.contradictions)
            unresolved_bugs.extend(ir.plan.unresolved_bugs)
        if ir.decomposition:
            unresolved_bugs.extend(ir.decomposition.unaddressed_bugs)

    resolution = DebugResolution(
        project_id=request.project_id,
        final_phase=final_phase,
        resolved=resolved,
        iterations=iterations,
        total_elapsed_ms=total_elapsed_ms,
        total_tokens=total_tokens,
        summary=summary,
        surfaced_contradictions=sorted(set(surfaced_contradictions)),
        unresolved_bugs=sorted(set(unresolved_bugs)),
    )

    _emit(on_event, "resolution", final_phase, iteration_no,
          f"done: resolved={resolved} phase={final_phase.value}",
          {"total_elapsed_ms": total_elapsed_ms, "total_tokens": total_tokens})

    return resolution


def _build_resolution_summary(
    iterations: List[IterationRecord],
    final_phase: OrchestrationPhase,
    resolved: bool,
) -> str:
    if not iterations:
        return f"Orchestration ended at phase={final_phase.value} with no iterations."
    last = iterations[-1]
    parts = [
        f"Completed {len(iterations)} iteration(s). Final phase: {final_phase.value}.",
        f"Resolved: {resolved}",
    ]
    if last.plan:
        parts.append(f"Root cause: {last.plan.root_cause[:280]}")
        if last.plan.steps:
            parts.append(f"Fix steps applied: {len(last.plan.steps)}")
    return " ".join(parts)
