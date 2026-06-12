# FILE: app/web_automation/memory/diagnostics.py
# Purpose: Failure-context formatter.
# Called-by: app.web_automation.memory, app.web_automation.memory.runner
# Depends-on: app.web_automation.memory.models
# Last-renovated: 2026-06-11
"""
Failure-context formatter.

When a Flow halts, the agent reading the result must see, with zero
ambiguity:

  1. Which steps confirmed working — these are KNOWN GOOD, do not redo.
  2. Which step failed and in which phase (precondition / action /
     postcondition) — repair attention belongs HERE only.
  3. What was expected vs. what was observed — concrete signal for what
     to change.
  4. Phase-specific guidance — different failure phases mean different
     things and need different fixes:
        * precondition  : the PRIOR step's exit state is not what we
                          thought; investigate the previous step's
                          postcondition or look for external interference
                          (popup, redirect, network).
        * action        : the tool call itself raised; bad params, network,
                          or session not in expected mode.
        * postcondition : action ran but the expected outcome never
                          appeared; selector/coordinate is stale or the
                          platform UI has changed.

This is the "everything before this point worked" guarantee made legible.
"""
from __future__ import annotations

from typing import List

from app.web_automation.memory.models import (
    CheckResult,
    Flow,
    StepResult,
)


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def build_diagnostic_summary(
    *,
    flow: Flow,
    completed_steps: List[StepResult],
    failed_step: StepResult,
    remaining_step_ids: List[str],
) -> str:
    """Render a focused failure report for the chat agent to act on."""
    lines: List[str] = []

    # ─── Header ──────────────────────────────────────────────────────
    lines.append(f"FLOW HALTED: {flow.platform}/{flow.task} v{flow.version}")
    lines.append(
        f"Progress: {len(completed_steps)} of {len(flow.steps)} step(s) "
        f"completed before halt."
    )
    lines.append("")

    # ─── Confirmed good ─────────────────────────────────────────────
    lines.append(f"CONFIRMED GOOD ({len(completed_steps)}):")
    if completed_steps:
        for s in completed_steps:
            lines.append(
                f"  [OK] {s.step_id}: {s.description or '(no description)'} "
                f"({s.duration_ms}ms)"
            )
    else:
        lines.append("  (none — failure occurred at the very first step)")
    lines.append("")

    # ─── Failure detail ─────────────────────────────────────────────
    lines.append(f"FAILED STEP: {failed_step.step_id}")
    lines.append(f"  description: {failed_step.description or '(none)'}")
    lines.append(f"  phase failed: {failed_step.phase}")
    lines.append(f"  duration: {failed_step.duration_ms}ms")
    if failed_step.error:
        lines.append(f"  error: {failed_step.error}")

    if failed_step.precondition_result:
        lines.extend(
            _render_check("precondition", failed_step.precondition_result)
        )
    if failed_step.postcondition_result:
        lines.extend(
            _render_check("postcondition", failed_step.postcondition_result)
        )
    if failed_step.action_result:
        lines.append(f"  action result: {failed_step.action_result}")
    lines.append("")

    # ─── Remaining work ─────────────────────────────────────────────
    lines.append(f"NOT YET ATTEMPTED ({len(remaining_step_ids)}):")
    if remaining_step_ids:
        for sid in remaining_step_ids:
            lines.append(f"  - {sid}")
    else:
        lines.append("  (none — failure was on the final step)")
    lines.append("")

    # ─── Guidance ───────────────────────────────────────────────────
    lines.append("REPAIR GUIDANCE:")
    lines.extend(_phase_guidance(failed_step))
    lines.append("")
    lines.append(
        "Do NOT redo the CONFIRMED GOOD steps. The page is in the state "
        "that follows from those steps having succeeded. Focus on the "
        "FAILED STEP only."
    )

    return "\n".join(lines)


# =============================================================================
# RENDERERS
# =============================================================================

def _render_check(label: str, result: CheckResult) -> List[str]:
    """Format one CheckResult as 4-5 indented lines."""
    status = "PASSED" if result.ok else "FAILED"
    timeout_marker = " (timed out)" if result.timed_out else ""
    out = [
        f"  {label} ({result.kind}): {status}{timeout_marker} "
        f"after {result.elapsed_ms}ms",
    ]
    if result.expected:
        out.append(f"    expected: {result.expected}")
    if not result.ok:
        out.append(f"    observed: {result.observed_summary}")
    return out


def _phase_guidance(failed_step: StepResult) -> List[str]:
    """Return phase-specific repair advice."""
    phase = failed_step.phase

    if phase == "precondition":
        return [
            "  - Precondition for this step did not become true within timeout.",
            "  - The PRIOR step's postcondition passed, so it claimed success,",
            "    but the page is not actually in the state this step needs.",
            "  - Likely causes: an unexpected popup intervened (cookie banner,",
            "    WhatsApp prompt, session expiry); the prior postcondition",
            "    was too lenient (matched on stale state); or the page",
            "    structure changed between recording and replay.",
            "  - Fix path: inspect current DOM, identify the discrepancy,",
            "    update either THIS step's precondition (if it's overstrict)",
            "    or the PRIOR step's postcondition (if it's underspecified).",
        ]

    if phase == "action":
        return [
            "  - The tool call itself raised an exception.",
            "  - Likely causes: bad parameters for this run (e.g. coordinate",
            "    out of viewport because the layout shifted); session no",
            "    longer logged in; transient network failure on the tool side.",
            "  - Fix path: read the error message above, correct the action's",
            "    parameters or use an alternative tool kind for this step",
            "    (e.g. switch from coordinate-based web_click to dom-relative",
            "    click via web_type+selector, or vice versa).",
        ]

    if phase == "postcondition":
        return [
            "  - The action executed without error, but the expected page",
            "    state never appeared within the postcondition timeout.",
            "  - Likely causes: the action targeted the wrong element (click",
            "    landed off-target, type went into the wrong field); the",
            "    expected outcome signature is stale (Meta UI redesign,",
            "    A/B test variant); the action is correct but needs more",
            "    time (raise timeout_ms on the postcondition).",
            "  - Fix path: take a fresh dom_snapshot, see what the page",
            "    actually shows now, decide whether the action is wrong or",
            "    the expectation is wrong, then update THIS step only.",
        ]

    return ["  - No guidance available for this phase."]
