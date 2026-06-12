# FILE: app/pipeline_v2/bvl/tier2_behavioral.py
# Purpose: Tier 2 — Behavioral Gate.
# Called-by: app.pipeline_v2.bvl.bvl_orchestrator
# Depends-on: app.pipeline_v2.build_targets, app.pipeline_v2.bvl.bvl_models, app.pipeline_v2.bvl.emulator_bridge, app.pipeline_v2.bvl.tier2_test_generator
# Last-renovated: 2026-06-11
"""
Tier 2 — Behavioral Gate.

Executes generated test flows against the running app via ADB/UI Automator.
Each flow exercises an interactive path: setup → actions → assertions → teardown.

The executor:
  1. Takes TestFlow objects from tier2_test_generator
  2. Resolves action targets to screen coordinates via UI tree dumps
  3. Executes actions via ADB input commands
  4. Validates assertions by dumping and inspecting the UI tree
  5. Returns per-flow pass/fail with detailed failure context

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.bvl.bvl_models import (
    FlowCoverage,
    SanityCheck,
    TestAction,
    TestAssertion,
    TestFlow,
    TierName,
    TierResult,
    TierStatus,
)
from app.pipeline_v2.bvl.emulator_bridge import (
    dump_ui_tree,
    element_exists,
    find_element,
    force_stop,
    launch_app,
    parse_ui_nodes,
    press_back,
    screenshot_as_base64,
    tap,
    tap_element,
    type_text,
    swipe,
)
from app.pipeline_v2.bvl.tier2_test_generator import (
    build_coverage_matrix,
    update_coverage,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


async def run_tier2(
    flows: List[TestFlow],
    profile: "BuildTargetProfile",
    emit: Optional[Callable[[str], None]] = None,
) -> TierResult:
    """Run all behavioral test flows.

    Args:
        flows: TestFlow objects from the test generator.
        profile: Build target profile.
        emit: Progress callback.

    Returns:
        TierResult with per-flow check results.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()

    if not flows:
        emit("   ⏭️ T2 Behavioral Gate: No test flows to execute")
        return TierResult(
            tier=TierName.BEHAVIORAL,
            status=TierStatus.SKIPPED,
            error_context="No test flows generated",
            duration_ms=0,
        )

    emit(f"   🧪 T2 Behavioral Gate: Executing {len(flows)} flows...")

    coverage = build_coverage_matrix(flows)
    checks = []
    all_passed = True

    for i, flow in enumerate(flows, 1):
        emit(f"   [{i}/{len(flows)}] {flow.flow_id}: {flow.description[:50]}...")
        flow_result = await _execute_flow(flow, profile, emit)

        check = SanityCheck(
            name=flow.flow_id,
            passed=flow_result.passed,
            message=flow_result.error if not flow_result.passed else "All assertions passed",
            duration_ms=flow_result.duration_ms,
            details={
                "description": flow_result.description,
                "assertions_total": len(flow_result.assertions),
                "assertions_passed": sum(1 for a in flow_result.assertions if a.passed),
            },
        )
        checks.append(check)

        update_coverage(
            coverage, flow.flow_id,
            passed=flow_result.passed,
            failure_reason=flow_result.error,
        )

        if not flow_result.passed:
            all_passed = False
            emit(f"   [{i}/{len(flows)}] ❌ {flow.flow_id}: {flow_result.error[:100]}")
        else:
            emit(f"   [{i}/{len(flows)}] ✅ {flow.flow_id}")

    duration_ms = int((time.time() - t_start) * 1000)
    status = TierStatus.PASSED if all_passed else TierStatus.FAILED

    # Build error context from failures
    error_context = ""
    if not all_passed:
        failed = [c for c in checks if not c.passed]
        error_lines = [f"T2 failed: {len(failed)}/{len(checks)} flows"]
        for c in failed:
            error_lines.append(f"  ✗ {c.name}: {c.message[:150]}")
        error_context = "\n".join(error_lines)

    emit(f"   {'✅' if all_passed else '❌'} T2 Behavioral Gate: "
         f"{sum(1 for c in checks if c.passed)}/{len(checks)} flows passed ({duration_ms}ms)")

    return TierResult(
        tier=TierName.BEHAVIORAL,
        status=status,
        checks=checks,
        error_context=error_context,
        duration_ms=duration_ms,
    )


# ═══════════════════════════════════════════════════════════════════
# Flow execution
# ═══════════════════════════════════════════════════════════════════

async def _execute_flow(
    flow: TestFlow,
    profile: "BuildTargetProfile",
    emit: Callable,
) -> TestFlow:
    """Execute a single test flow and update its results in-place.

    Returns the flow with pass/fail and assertion results filled in.
    """
    t_start = time.time()

    try:
        # Setup
        for action in flow.setup_actions:
            await _execute_action(action, profile)

        # Main actions
        for action in flow.actions:
            await _execute_action(action, profile)

        # Assertions
        all_passed = True
        for assertion in flow.assertions:
            result = await _check_assertion(assertion)
            if not result:
                all_passed = False

        flow.passed = all_passed
        if not all_passed:
            failed = [a for a in flow.assertions if not a.passed]
            flow.error = "; ".join(
                f"{a.assertion_type}({a.target}): expected '{a.expected}', got '{a.actual}'"
                for a in failed[:3]
            )

    except Exception as e:
        flow.passed = False
        flow.error = f"Execution error: {e}"
        logger.warning("[tier2] Flow %s failed: %s", flow.flow_id, e)

    finally:
        # Teardown — always run
        for action in flow.teardown_actions:
            try:
                await _execute_action(action, profile)
            except Exception:
                pass

    flow.duration_ms = int((time.time() - t_start) * 1000)
    return flow


async def _execute_action(
    action: TestAction,
    profile: "BuildTargetProfile",
) -> None:
    """Execute a single UI action via ADB.

    Resolves text-based targets to coordinates using UI tree dumps.
    """
    if action.action_type == "wait":
        ms = int(action.value) if action.value.isdigit() else 2000
        await asyncio.sleep(ms / 1000)
        return

    if action.action_type == "back":
        await press_back()
        return

    if action.action_type == "tap":
        node = await _resolve_target(action.target)
        if node:
            await tap_element(node)
        else:
            logger.warning("[tier2] Could not find tap target: %s", action.target)
        return

    if action.action_type == "type":
        # First try to find and tap the target field
        if action.target:
            node = await _resolve_target(action.target)
            if node:
                await tap_element(node)
                await asyncio.sleep(0.3)
        await type_text(action.value)
        return

    if action.action_type == "swipe":
        # Value format: "up", "down", "left", "right" or "x1,y1,x2,y2"
        coords = _parse_swipe_direction(action.value)
        if coords:
            await swipe(*coords)
        return

    if action.action_type == "scroll":
        # Default scroll down
        await swipe(540, 1500, 540, 500, 300)
        return

    logger.warning("[tier2] Unknown action type: %s", action.action_type)


async def _resolve_target(target: str) -> Optional[object]:
    """Resolve a text-based target description to a UINode.

    Tries multiple strategies: exact text match, partial text,
    content description, resource ID substring.
    """
    if not target:
        return None

    # Try exact text match
    node = await find_element(text=target)
    if node:
        return node

    # Try content description
    node = await find_element(content_desc=target)
    if node:
        return node

    # Try as resource ID fragment
    node = await find_element(resource_id=target.lower().replace(" ", "_"))
    if node:
        return node

    # Fuzzy: dump tree and search for partial matches
    xml = await dump_ui_tree()
    nodes = parse_ui_nodes(xml)
    target_lower = target.lower()

    for n in nodes:
        if target_lower in n.text.lower():
            return n
        if target_lower in n.content_desc.lower():
            return n
        if target_lower in n.resource_id.lower():
            return n

    return None


async def _check_assertion(assertion: TestAssertion) -> bool:
    """Validate a single assertion against the current UI state.

    Updates the assertion object in-place with actual values and pass/fail.
    """
    if assertion.assertion_type == "element_visible":
        exists = await element_exists(
            text=assertion.target,
            content_desc=assertion.target,
        )
        assertion.passed = exists
        assertion.actual = "visible" if exists else "not found"
        assertion.message = (
            f"Element '{assertion.target}' {'found' if exists else 'not found'}"
        )
        return exists

    if assertion.assertion_type == "text_matches":
        node = await find_element(
            text=assertion.expected,
            resource_id=assertion.target,
        )
        if node:
            assertion.passed = assertion.expected in node.text
            assertion.actual = node.text
        else:
            assertion.passed = False
            assertion.actual = "element not found"
        assertion.message = (
            f"Expected '{assertion.expected}' in '{assertion.target}', "
            f"got '{assertion.actual}'"
        )
        return assertion.passed

    if assertion.assertion_type == "screen_changed":
        # Check that the current screen has the expected element
        exists = await element_exists(text=assertion.expected)
        assertion.passed = exists
        assertion.actual = "screen has element" if exists else "element not found"
        return exists

    if assertion.assertion_type == "data_persisted":
        # Check element still shows expected data after navigation
        node = await find_element(
            resource_id=assertion.target,
            text=assertion.expected,
        )
        assertion.passed = node is not None
        assertion.actual = node.text if node else "not found"
        return assertion.passed

    logger.warning("[tier2] Unknown assertion type: %s", assertion.assertion_type)
    assertion.passed = False
    assertion.message = f"Unknown assertion type: {assertion.assertion_type}"
    return False


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _parse_swipe_direction(
    value: str,
    screen_width: int = 1080,
    screen_height: int = 2400,
) -> Optional[tuple]:
    """Parse a swipe direction string into coordinates.

    Accepts "up", "down", "left", "right" or "x1,y1,x2,y2".
    """
    v = value.lower().strip()
    cx, cy = screen_width // 2, screen_height // 2

    if v == "up":
        return (cx, cy + 400, cx, cy - 400, 300)
    if v == "down":
        return (cx, cy - 400, cx, cy + 400, 300)
    if v == "left":
        return (cx + 300, cy, cx - 300, cy, 300)
    if v == "right":
        return (cx - 300, cy, cx + 300, cy, 300)

    # Try "x1,y1,x2,y2" format
    parts = v.split(",")
    if len(parts) == 4:
        try:
            return tuple(int(p.strip()) for p in parts) + (300,)
        except ValueError:
            pass

    return None
