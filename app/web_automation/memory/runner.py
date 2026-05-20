# FILE: app/web_automation/memory/runner.py
"""
Flow runner — executes a Flow step-by-step with halt-on-first-failure
semantics and structured per-step results.

Failure isolation contract:
  * Every successfully-completed step is recorded in result.completed_steps
    with its postcondition observation. These are KNOWN GOOD.
  * The step that failed is recorded in result.failed_step with full
    detail of which phase failed (precondition / action / postcondition)
    and what was observed vs. expected.
  * Steps not yet attempted are listed in result.remaining_step_ids.
  * The runner does NOT auto-retry the whole flow. Repair work belongs
    in the agent layer above, focused on the failing step only.

The runner is tool-layer agnostic: it takes an `executor` callable and
dispatches Action.kind through it. This avoids a hard dependency on the
chat tool layer and makes the runner unit-testable with a fake executor.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, List, Optional

from app.web_automation.memory.checks import evaluate_check
from app.web_automation.memory.diagnostics import build_diagnostic_summary
from app.web_automation.memory.models import (
    Check,
    Flow,
    FlowResult,
    Step,
    StepResult,
)

logger = logging.getLogger(__name__)

ToolDispatcher = Callable[[str, dict], Awaitable[str]]

# Tool result strings can be very large (DOM snapshots run to 6KB+).
# Truncate before storing on a StepResult so a FlowResult JSON stays
# manageable as chat tool output.
_ACTION_RESULT_MAX_CHARS = 600


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

async def run_flow(
    flow: Flow,
    executor: ToolDispatcher,
    *,
    default_session: Optional[str] = None,
) -> FlowResult:
    """
    Execute a flow's steps in order, halting on the first failure.

    Args:
        flow            : the Flow to execute.
        executor        : async dispatcher for Action.kind -> tool call.
        default_session : session id used when a Step doesn't specify one.

    Returns:
        FlowResult — ok=True iff every step's postcondition passed.
    """
    completed: List[StepResult] = []
    total_start = time.monotonic()

    for index, step in enumerate(flow.steps):
        step_session = step.session or default_session
        step_result = await _run_step(step, step_session, executor)

        if not step_result.ok:
            return _build_failed_result(
                flow=flow,
                completed=completed,
                failed=step_result,
                failed_index=index,
                total_start=total_start,
            )

        completed.append(step_result)

    total_ms = int((time.monotonic() - total_start) * 1000)
    logger.info(
        "[flow_runner] %s/%s OK — %d steps in %dms",
        flow.platform, flow.task, len(completed), total_ms,
    )
    return FlowResult(
        ok=True,
        platform=flow.platform,
        task=flow.task,
        completed_steps=completed,
        failed_step=None,
        remaining_step_ids=[],
        total_duration_ms=total_ms,
        diagnostic_summary=(
            f"All {len(completed)} step(s) of {flow.platform}/{flow.task} "
            f"completed successfully in {total_ms}ms."
        ),
    )


# =============================================================================
# SINGLE-STEP EXECUTION
# =============================================================================

async def _run_step(
    step: Step,
    session: Optional[str],
    executor: ToolDispatcher,
) -> StepResult:
    """Run one step through its three phases. First failed phase wins."""
    started = time.monotonic()

    # Phase 1: precondition ──────────────────────────────────────────
    pre_result = None
    if step.precondition is not None:
        pre_result = await evaluate_check(step.precondition, session, executor)
        if not pre_result.ok:
            return StepResult(
                step_id=step.step_id,
                description=step.description,
                ok=False,
                phase="precondition",
                duration_ms=_ms(started),
                precondition_result=pre_result,
                error=(
                    "Precondition not met. The PRIOR step claimed completion "
                    "but the page is not in the expected state."
                ),
            )

    # Phase 2: action ────────────────────────────────────────────────
    action_str: Optional[str] = None
    try:
        action_str = await _dispatch_action(step, executor)
    except Exception as exc:
        return StepResult(
            step_id=step.step_id,
            description=step.description,
            ok=False,
            phase="action",
            duration_ms=_ms(started),
            precondition_result=pre_result,
            error=f"Action raised: {exc}",
        )

    # Phase 3: postcondition ─────────────────────────────────────────
    post_result = None
    if step.postcondition is not None:
        post_result = await evaluate_check(step.postcondition, session, executor)
        if not post_result.ok:
            return StepResult(
                step_id=step.step_id,
                description=step.description,
                ok=False,
                phase="postcondition",
                duration_ms=_ms(started),
                action_result=_truncate(action_str),
                precondition_result=pre_result,
                postcondition_result=post_result,
                error=(
                    "Action ran but expected outcome did not appear within "
                    "timeout. Either the action targeted the wrong element "
                    "or the platform UI signature has changed."
                ),
            )

    return StepResult(
        step_id=step.step_id,
        description=step.description,
        ok=True,
        phase="complete",
        duration_ms=_ms(started),
        action_result=_truncate(action_str),
        precondition_result=pre_result,
        postcondition_result=post_result,
    )


# =============================================================================
# ACTION DISPATCH
# =============================================================================

async def _dispatch_action(step: Step, executor: ToolDispatcher) -> str:
    """
    Translate an Action into a tool call, with one runner-internal kind:

        wait : pure delay, no tool call. params={'ms': N}.

    Every other kind is forwarded directly to the executor.
    """
    kind = step.action.kind
    params = dict(step.action.params or {})

    if kind == "wait":
        ms = int(params.get("ms", 1000))
        await asyncio.sleep(ms / 1000.0)
        return f"waited {ms}ms"

    # Inject session for browser tools when the step has one and the
    # action params didn't specify it (saves the recorder having to
    # repeat session on every step).
    if step.session and "session" not in params and kind.startswith("web_"):
        params["session"] = step.session

    return await executor(kind, params)


# =============================================================================
# RESULT BUILDING
# =============================================================================

def _build_failed_result(
    *,
    flow: Flow,
    completed: List[StepResult],
    failed: StepResult,
    failed_index: int,
    total_start: float,
) -> FlowResult:
    remaining = [s.step_id for s in flow.steps[failed_index + 1:]]
    total_ms = int((time.monotonic() - total_start) * 1000)
    summary = build_diagnostic_summary(
        flow=flow,
        completed_steps=completed,
        failed_step=failed,
        remaining_step_ids=remaining,
    )
    logger.info(
        "[flow_runner] %s/%s FAILED at step '%s' (%s) — %d of %d steps OK",
        flow.platform, flow.task, failed.step_id, failed.phase,
        len(completed), len(flow.steps),
    )
    return FlowResult(
        ok=False,
        platform=flow.platform,
        task=flow.task,
        completed_steps=completed,
        failed_step=failed,
        remaining_step_ids=remaining,
        total_duration_ms=total_ms,
        diagnostic_summary=summary,
    )


# =============================================================================
# UTILITIES
# =============================================================================

def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _truncate(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= _ACTION_RESULT_MAX_CHARS:
        return value
    head = _ACTION_RESULT_MAX_CHARS - 30
    return f"{value[:head]}… [+{len(value) - head} chars]"
