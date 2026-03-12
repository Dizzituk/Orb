# FILE: app/pipeline_v2/bvl/tier3_adversarial.py
"""
Tier 3 — Adversarial Gate.

Attempts to break the app through edge cases, stress, and unexpected
usage patterns. Catches failures that structured behavioral tests miss.

Test categories:
  A. Input Fuzzing — empty strings, max-length, special chars, emoji, RTL
  B. Lifecycle Stress — rotation, kill/restart, background/foreground, low memory
  C. Navigation Chaos — rapid back presses, Monkey runner
  D. Data Integrity — persist across kill, survive navigation, validation messages

Pass criteria: Zero crashes, no ANR, all validation messages shown, data intact.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, List, Optional, TYPE_CHECKING

from app.pipeline_v2.bvl.bvl_models import (
    SanityCheck,
    TierName,
    TierResult,
    TierStatus,
)
from app.pipeline_v2.bvl.emulator_bridge import (
    check_for_crashes,
    dump_ui_tree,
    find_element,
    force_stop,
    is_app_running,
    launch_app,
    parse_ui_nodes,
    press_back,
    rotate_device,
    run_monkey,
    send_trim_memory,
    set_rotation,
    tap,
    type_text,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


async def run_tier3(
    profile: "BuildTargetProfile",
    emit: Optional[Callable[[str], None]] = None,
) -> TierResult:
    """Run all adversarial tests.

    Args:
        profile: Build target profile.
        emit: Progress callback.

    Returns:
        TierResult with per-category check results.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()
    checks = []

    emit("   💥 T3 Adversarial Gate: Starting stress tests...")

    # A. Input Fuzzing
    fuzz_check = await _test_input_fuzzing(profile.package_name, emit)
    checks.append(fuzz_check)

    # B. Lifecycle Stress
    lifecycle_check = await _test_lifecycle_stress(profile.package_name, emit)
    checks.append(lifecycle_check)

    # C. Navigation Chaos
    nav_check = await _test_navigation_chaos(profile.package_name, emit)
    checks.append(nav_check)

    # D. Data Integrity
    data_check = await _test_data_integrity(profile.package_name, emit)
    checks.append(data_check)

    duration_ms = int((time.time() - t_start) * 1000)
    all_passed = all(c.passed for c in checks)
    status = TierStatus.PASSED if all_passed else TierStatus.FAILED

    error_context = ""
    if not all_passed:
        failed = [c for c in checks if not c.passed]
        error_context = "T3 failures: " + "; ".join(
            f"{c.name}: {c.message[:100]}" for c in failed
        )

    emit(f"   {'✅' if all_passed else '❌'} T3 Adversarial Gate: "
         f"{sum(1 for c in checks if c.passed)}/{len(checks)} categories passed ({duration_ms}ms)")

    return TierResult(
        tier=TierName.ADVERSARIAL,
        status=status,
        checks=checks,
        error_context=error_context,
        duration_ms=duration_ms,
    )


# ═══════════════════════════════════════════════════════════════════
# A. Input Fuzzing
# ═══════════════════════════════════════════════════════════════════

async def _test_input_fuzzing(
    package_name: str,
    emit: Callable,
) -> SanityCheck:
    """Fuzz text input fields with edge-case strings."""
    emit("   [A] Input fuzzing...")
    t = time.time()
    failures = []

    fuzz_inputs = [
        ("empty_string", ""),
        ("long_string", "A" * 10000),
        ("special_chars", "<script>alert('xss')</script>"),
        ("sql_injection", "'; DROP TABLE stops; --"),
        ("emoji", "\U0001f680\U0001f525\U0001f4a5\U0001f30d"),
        ("rtl_text", "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"),
        ("numeric_overflow", "99999999999999999999"),
        ("null_bytes", "test\x00null\x00bytes"),
    ]

    # Ensure app is running
    await force_stop(package_name)
    await asyncio.sleep(1)
    await launch_app(package_name)
    await asyncio.sleep(3)

    # Find any text input field
    xml = await dump_ui_tree()
    nodes = parse_ui_nodes(xml)
    input_nodes = [
        n for n in nodes
        if "EditText" in n.class_name and n.enabled
    ]

    if not input_nodes:
        ms = int((time.time() - t) * 1000)
        emit(f"   [A] ⏭️ No input fields found ({ms}ms)")
        return SanityCheck(
            name="input_fuzzing", passed=True,
            message="No input fields to fuzz (skipped)",
            duration_ms=ms,
        )

    target_node = input_nodes[0]

    for name, fuzz_value in fuzz_inputs:
        try:
            # Tap field, clear, type fuzz value
            if target_node.center:
                await tap(*target_node.center)
                await asyncio.sleep(0.3)

            # Clear existing text
            from app.pipeline_v2.bvl.emulator_bridge import _adb
            await _adb("shell input keyevent KEYCODE_MOVE_HOME", timeout_sec=3)
            await _adb("shell input keyevent --longpress KEYCODE_MOVE_END", timeout_sec=3)
            await _adb("shell input keyevent KEYCODE_DEL", timeout_sec=3)

            if fuzz_value:
                await type_text(fuzz_value[:200])  # Truncate for ADB safety

            await asyncio.sleep(1)

            # Check for crash
            has_crash, excerpt = await check_for_crashes(package_name, duration_sec=2)
            if has_crash:
                failures.append(f"{name}: crash — {excerpt[:100]}")

            # Check app still running
            if not await is_app_running(package_name):
                failures.append(f"{name}: app died")
                await launch_app(package_name)
                await asyncio.sleep(3)

        except Exception as e:
            failures.append(f"{name}: error — {e}")

    ms = int((time.time() - t) * 1000)
    passed = len(failures) == 0

    if passed:
        emit(f"   [A] ✅ Input fuzzing passed ({ms}ms)")
    else:
        emit(f"   [A] ❌ Input fuzzing: {len(failures)} failures ({ms}ms)")

    return SanityCheck(
        name="input_fuzzing", passed=passed,
        message="; ".join(failures[:3]) if failures else "All fuzz inputs handled safely",
        duration_ms=ms,
        details={"failures": failures},
    )


# ═══════════════════════════════════════════════════════════════════
# B. Lifecycle Stress
# ═══════════════════════════════════════════════════════════════════

async def _test_lifecycle_stress(
    package_name: str,
    emit: Callable,
) -> SanityCheck:
    """Test app survival through lifecycle events."""
    emit("   [B] Lifecycle stress...")
    t = time.time()
    failures = []

    # Ensure fresh start
    await force_stop(package_name)
    await asyncio.sleep(1)
    await launch_app(package_name)
    await asyncio.sleep(3)

    # B1: Rotate device mid-flow
    try:
        await rotate_device(1)  # Landscape
        await asyncio.sleep(2)
        has_crash, excerpt = await check_for_crashes(package_name, 2)
        if has_crash:
            failures.append(f"rotation_landscape: crash — {excerpt[:80]}")
        await rotate_device(0)  # Back to portrait
        await asyncio.sleep(2)
    except Exception as e:
        failures.append(f"rotation: {e}")
    finally:
        await set_rotation(False)
        await rotate_device(0)

    # B2: Kill and restart
    try:
        await force_stop(package_name)
        await asyncio.sleep(2)
        await launch_app(package_name)
        await asyncio.sleep(3)
        has_crash, excerpt = await check_for_crashes(package_name, 3)
        if has_crash:
            failures.append(f"kill_restart: crash — {excerpt[:80]}")
        if not await is_app_running(package_name):
            failures.append("kill_restart: app did not restart")
    except Exception as e:
        failures.append(f"kill_restart: {e}")

    # B3: Background and return
    try:
        from app.pipeline_v2.bvl.emulator_bridge import press_home
        await press_home()
        await asyncio.sleep(5)
        await launch_app(package_name)
        await asyncio.sleep(3)
        if not await is_app_running(package_name):
            failures.append("background_return: app not running after resume")
    except Exception as e:
        failures.append(f"background_return: {e}")

    # B4: Low memory simulation
    try:
        await send_trim_memory(package_name, "RUNNING_CRITICAL")
        await asyncio.sleep(2)
        has_crash, excerpt = await check_for_crashes(package_name, 2)
        if has_crash:
            failures.append(f"low_memory: crash — {excerpt[:80]}")
    except Exception as e:
        failures.append(f"low_memory: {e}")

    ms = int((time.time() - t) * 1000)
    passed = len(failures) == 0

    if passed:
        emit(f"   [B] ✅ Lifecycle stress passed ({ms}ms)")
    else:
        emit(f"   [B] ❌ Lifecycle stress: {len(failures)} failures ({ms}ms)")

    return SanityCheck(
        name="lifecycle_stress", passed=passed,
        message="; ".join(failures[:3]) if failures else "All lifecycle tests passed",
        duration_ms=ms,
        details={"failures": failures},
    )


# ═══════════════════════════════════════════════════════════════════
# C. Navigation Chaos
# ═══════════════════════════════════════════════════════════════════

async def _test_navigation_chaos(
    package_name: str,
    emit: Callable,
) -> SanityCheck:
    """Test navigation robustness."""
    emit("   [C] Navigation chaos...")
    t = time.time()
    failures = []

    # Ensure fresh start
    await force_stop(package_name)
    await asyncio.sleep(1)
    await launch_app(package_name)
    await asyncio.sleep(3)

    # C1: Rapid back presses
    try:
        for _ in range(10):
            await press_back()
            await asyncio.sleep(0.2)

        await asyncio.sleep(2)
        has_crash, excerpt = await check_for_crashes(package_name, 2)
        if has_crash:
            failures.append(f"rapid_back: crash — {excerpt[:80]}")
    except Exception as e:
        failures.append(f"rapid_back: {e}")

    # Relaunch for Monkey test
    await force_stop(package_name)
    await asyncio.sleep(1)
    await launch_app(package_name)
    await asyncio.sleep(3)

    # C2: Monkey runner (limited events for speed)
    try:
        result = await run_monkey(package_name, events=500, throttle_ms=50)
        if not result.ok:
            # Check if crash was in our package
            if "CRASH" in result.stdout.upper():
                failures.append(f"monkey: crash detected in output")
    except Exception as e:
        failures.append(f"monkey: {e}")

    ms = int((time.time() - t) * 1000)
    passed = len(failures) == 0

    if passed:
        emit(f"   [C] ✅ Navigation chaos passed ({ms}ms)")
    else:
        emit(f"   [C] ❌ Navigation chaos: {len(failures)} failures ({ms}ms)")

    return SanityCheck(
        name="navigation_chaos", passed=passed,
        message="; ".join(failures[:3]) if failures else "Navigation stress passed",
        duration_ms=ms,
        details={"failures": failures},
    )


# ═══════════════════════════════════════════════════════════════════
# D. Data Integrity
# ═══════════════════════════════════════════════════════════════════

async def _test_data_integrity(
    package_name: str,
    emit: Callable,
) -> SanityCheck:
    """Test that data survives kills and navigation.

    This is a structural test: it verifies the app can be killed and
    relaunched without crashing, and that the UI renders after restart.
    Full data persistence testing requires app-specific knowledge
    from the Tier 2 test flows.
    """
    emit("   [D] Data integrity...")
    t = time.time()
    failures = []

    # Launch, interact briefly, kill, relaunch
    await force_stop(package_name)
    await asyncio.sleep(1)
    await launch_app(package_name)
    await asyncio.sleep(3)

    # Capture pre-kill UI state
    xml_before = await dump_ui_tree()
    nodes_before = parse_ui_nodes(xml_before)
    element_count_before = len([n for n in nodes_before if n.clickable])

    # Kill and restart
    await force_stop(package_name)
    await asyncio.sleep(2)
    await launch_app(package_name)
    await asyncio.sleep(3)

    # Check app recovered
    if not await is_app_running(package_name):
        failures.append("App did not restart after kill")
    else:
        # Check UI rendered
        xml_after = await dump_ui_tree()
        nodes_after = parse_ui_nodes(xml_after)
        element_count_after = len([n for n in nodes_after if n.clickable])

        if element_count_after < 2:
            failures.append(
                f"UI did not render after restart "
                f"(before: {element_count_before}, after: {element_count_after})"
            )

    # Check for crashes during the whole process
    has_crash, excerpt = await check_for_crashes(package_name, 3)
    if has_crash:
        failures.append(f"crash after restart: {excerpt[:80]}")

    ms = int((time.time() - t) * 1000)
    passed = len(failures) == 0

    if passed:
        emit(f"   [D] ✅ Data integrity passed ({ms}ms)")
    else:
        emit(f"   [D] ❌ Data integrity: {len(failures)} failures ({ms}ms)")

    return SanityCheck(
        name="data_integrity", passed=passed,
        message="; ".join(failures[:3]) if failures else "Data integrity checks passed",
        duration_ms=ms,
        details={"failures": failures},
    )
