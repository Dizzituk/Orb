# FILE: app/debug/executors/flow_actions.py
# Purpose: Chat tool executors for the flow memory system.
# Called-by: app.debug.executors
# Depends-on: app.debug.action_executor, app.web_automation.memory
# Last-renovated: 2026-06-11
"""
Chat tool executors for the flow memory system.

Three tools:

    flow_run     — execute a saved flow by (platform, task). Halts on
                   first verification failure and returns a structured
                   diagnostic naming the failed step and listing every
                   step that confirmed working before it.

    flow_save    — persist a flow definition (create or update). Used
                   by the agent to RECORD a working interaction trace
                   so future runs can replay it. Versioned: each save
                   bumps the version number.

    flow_inspect — list saved flows or read a single flow definition
                   in full JSON form. Use before flow_save to see what
                   already exists; use after a failure to read the step
                   that needs editing.

The runner is tool-layer agnostic — it takes an executor callable. We
inject action_executor.execute_tool via a lazy import to avoid the
circular: action_executor imports executors/__init__ which imports this
module. By keeping the import inside the function bodies, the cycle is
broken at module-load time.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.web_automation.memory import (
    Flow,
    FlowResult,
    Step,
    list_flows,
    load_flow,
    run_flow,
    save_flow,
    update_flow_stats,
)

logger = logging.getLogger(__name__)


# =============================================================================
# flow_run
# =============================================================================

async def execute_flow_run(params: Dict[str, Any]) -> str:
    """
    Run a saved flow.

    Params:
        platform        (required) : flow platform key (e.g. 'meta_business')
        task            (required) : flow task key (e.g. 'reply_top_comment')
        default_session (optional) : web session id for steps without one
    """
    p = params or {}
    platform = p.get("platform")
    task = p.get("task")
    default_session = p.get("default_session")

    if not platform or not task:
        return "Error: 'platform' and 'task' are both required."

    flow = load_flow(platform, task)
    if flow is None:
        return (
            f"No flow saved for {platform}/{task}. "
            f"Use flow_save to record one first, then flow_run will replay it."
        )

    # Lazy import — avoids circular with action_executor at load time.
    from app.debug.action_executor import execute_tool

    result = await run_flow(
        flow,
        executor=execute_tool,
        default_session=default_session,
    )

    update_flow_stats(
        platform, task,
        success=result.ok,
        failure_reason=(
            f"step '{result.failed_step.step_id}' failed at "
            f"{result.failed_step.phase}"
            if (not result.ok and result.failed_step) else None
        ),
    )

    return _format_flow_result(result)


# =============================================================================
# flow_save
# =============================================================================

async def execute_flow_save(params: Dict[str, Any]) -> str:
    """
    Save (create or update) a flow definition.

    Params:
        platform    (required) : platform key
        task        (required) : task key
        steps       (required) : list of step dicts (see Step schema)
        description (optional) : human-readable summary
    """
    p = params or {}
    platform = p.get("platform")
    task = p.get("task")
    steps_raw = p.get("steps")

    if not platform or not task:
        return "Error: 'platform' and 'task' are both required."
    if not isinstance(steps_raw, list) or not steps_raw:
        return "Error: 'steps' must be a non-empty list."

    try:
        steps: List[Step] = [Step(**s) for s in steps_raw]
    except Exception as exc:
        return f"Invalid step in 'steps': {exc}"

    existing = load_flow(platform, task)
    new_version = (existing.version + 1) if existing else 1

    flow = Flow(
        platform=platform,
        task=task,
        version=new_version,
        description=p.get("description", "") or "",
        steps=steps,
        success_count=existing.success_count if existing else 0,
        failure_count=existing.failure_count if existing else 0,
        last_run_at=existing.last_run_at if existing else None,
    )

    try:
        path = save_flow(flow)
    except Exception as exc:
        return f"Failed to write flow file: {exc}"

    return (
        f"Saved flow {flow.platform}/{flow.task} v{flow.version} "
        f"with {len(flow.steps)} step(s) → {path}"
    )


# =============================================================================
# flow_inspect
# =============================================================================

async def execute_flow_inspect(params: Dict[str, Any]) -> str:
    """
    Read flow metadata.

    Params:
        platform (optional) : restrict to one platform; omit to list all
        task     (optional) : combined with platform, returns full JSON
    """
    p = params or {}
    platform = p.get("platform")
    task = p.get("task")

    if platform and task:
        flow = load_flow(platform, task)
        if flow is None:
            return f"No flow saved for {platform}/{task}."
        return flow.model_dump_json(indent=2)

    flows = list_flows(platform=platform)
    if not flows:
        scope = f"for platform '{platform}'" if platform else "(any platform)"
        return f"No flows saved {scope}."

    lines = [f"Saved flows ({len(flows)}):"]
    for plat, t in flows:
        f = load_flow(plat, t)
        if f is None:
            continue
        lines.append(
            f"  {plat}/{t}  v{f.version}  "
            f"steps={len(f.steps)} success={f.success_count} "
            f"fail={f.failure_count} last_run={f.last_run_at or 'never'}"
        )
    return "\n".join(lines)


# =============================================================================
# RESULT FORMATTING
# =============================================================================

def _format_flow_result(result: FlowResult) -> str:
    """Render a FlowResult for the chat LLM. Diagnostic_summary already
    contains the failure context; we just frame and add the OK header."""
    if result.ok:
        return (
            f"flow_run OK {result.platform}/{result.task} — "
            f"{len(result.completed_steps)} step(s) in "
            f"{result.total_duration_ms}ms\n"
            f"{result.diagnostic_summary}"
        )
    return (
        f"flow_run FAILED {result.platform}/{result.task} after "
        f"{result.total_duration_ms}ms\n\n"
        f"{result.diagnostic_summary}"
    )
