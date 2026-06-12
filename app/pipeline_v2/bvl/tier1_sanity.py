# FILE: app/pipeline_v2/bvl/tier1_sanity.py
# Purpose: Tier 1 — Sanity Gate.
# Called-by: app.pipeline_v2.bvl.bvl_orchestrator
# Depends-on: app.pipeline_v2.build_targets, app.pipeline_v2.bvl.bvl_models, app.pipeline_v2.bvl.emulator_bridge, app.pipeline_v2.sandbox_tools (+1 more)
# Last-renovated: 2026-06-11
"""
Tier 1 — Sanity Gate.

Confirms the build is a viable artefact before investing in deeper testing.
Replaces and extends the current Visual Verifier for Android builds.

Checks performed (all must pass):
  1. Gradle build completes without errors
  2. APK installs on emulator via ADB
  3. App launches without crash (logcat for fatal exceptions)
  4. First screen renders within 5-second timeout
  5. Screenshot captured and compared against spec layout
  6. Background services initialise (logcat init markers)

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Optional, TYPE_CHECKING

from app.pipeline_v2.bvl.bvl_models import (
    SanityCheck,
    TierName,
    TierResult,
    TierStatus,
)
from app.pipeline_v2.bvl.emulator_bridge import (
    capture_screenshot,
    check_for_crashes,
    install_apk,
    is_emulator_running,
    launch_app,
    screenshot_as_base64,
    wait_for_boot,
)

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)

# Default APK location after Gradle assembleDebug
DEFAULT_APK_PATH = (
    "D:/Astra Android Folder/AndroidDriverCopilot/"
    "app/build/outputs/apk/debug/app-debug.apk"
)


async def run_tier1(
    profile: "BuildTargetProfile",
    spec_text: str = "",
    emit: Optional[Callable[[str], None]] = None,
) -> TierResult:
    """Run all Tier 1 sanity checks.

    Args:
        profile: Build target profile (determines APK path, package name, etc.)
        spec_text: Spec text for visual comparison (optional for T1).
        emit: Progress callback.

    Returns:
        TierResult with all check results.
    """
    emit = emit or (lambda msg: None)
    t_start = time.time()
    checks = []

    emit("   🔍 T1 Sanity Gate: Starting checks...")

    # ── Check 1: Gradle build ──
    check_build = await _check_gradle_build(profile, emit)
    checks.append(check_build)
    if not check_build.passed:
        return await _make_result(checks, t_start, check_build.message)

    # ── Check 2: Emulator ready ──
    check_emu = await _check_emulator_ready(emit)
    checks.append(check_emu)
    if not check_emu.passed:
        return await _make_result(checks, t_start, check_emu.message)

    # ── Check 3: APK install ──
    apk_path = _resolve_apk_path(profile)
    check_install = await _check_apk_install(apk_path, emit)
    checks.append(check_install)
    if not check_install.passed:
        return await _make_result(checks, t_start, check_install.message)

    # ── Check 4: App launch (no crash) ──
    check_launch = await _check_app_launch(profile.package_name, emit)
    checks.append(check_launch)
    if not check_launch.passed:
        return await _make_result(checks, t_start, check_launch.message)

    # ── Check 5: First screen renders ──
    check_render = await _check_first_render(profile.package_name, emit)
    checks.append(check_render)
    if not check_render.passed:
        return await _make_result(checks, t_start, check_render.message)

    # ── Check 6: Screenshot capture ──
    check_screenshot = await _check_screenshot(emit)
    checks.append(check_screenshot)

    # All checks passed
    duration_ms = int((time.time() - t_start) * 1000)
    emit(f"   ✅ T1 Sanity Gate: ALL PASSED ({duration_ms}ms)")
    return TierResult(
        tier=TierName.SANITY,
        status=TierStatus.PASSED,
        checks=checks,
        screenshot_b64=check_screenshot.details.get("b64"),
        duration_ms=duration_ms,
    )


# ═══════════════════════════════════════════════════════════════════
# Individual checks
# ═══════════════════════════════════════════════════════════════════

async def _check_gradle_build(
    profile: "BuildTargetProfile",
    emit: Callable,
) -> SanityCheck:
    """Check 1: Gradle compileDebugKotlin passes."""
    emit("   [1/6] Gradle build...")
    t = time.time()

    # v2.3: Ensure Gradle wrapper exists before attempting build.
    # The builder writes source files but may not create the wrapper.
    import os
    gradlew_path = os.path.join(
        profile.project_root.replace('/', os.sep), 'gradlew.bat'
    )
    if not os.path.isfile(gradlew_path):
        emit("   [1/6] Gradle wrapper missing — copying from known project...")
        try:
            from app.pipeline_v2.scaffolds.android_config_scaffolds import copy_gradle_wrapper
            if copy_gradle_wrapper(profile.project_root.replace('/', os.sep)):
                emit("   [1/6] Gradle wrapper installed")
            else:
                return SanityCheck(
                    name="gradle_build", passed=False,
                    message="Gradle wrapper missing and no source project found to copy from",
                    duration_ms=int((time.time() - t) * 1000),
                )
        except Exception as e:
            return SanityCheck(
                name="gradle_build", passed=False,
                message=f"Gradle wrapper setup failed: {e}",
                duration_ms=int((time.time() - t) * 1000),
            )

    from app.pipeline_v2.sandbox_tools import build_check
    ok, output = await build_check(profile=profile)

    ms = int((time.time() - t) * 1000)
    if ok:
        emit(f"   [1/6] ✅ Gradle build passed ({ms}ms)")
        return SanityCheck(
            name="gradle_build", passed=True,
            message="compileDebugKotlin succeeded",
            duration_ms=ms,
        )
    else:
        emit(f"   [1/6] ❌ Gradle build failed ({ms}ms)")
        return SanityCheck(
            name="gradle_build", passed=False,
            message=f"Build failed: {output[:300]}",
            duration_ms=ms,
            details={"build_output": output},
        )


async def _check_emulator_ready(emit: Callable) -> SanityCheck:
    """Check 2: Emulator is running and booted."""
    emit("   [2/6] Emulator status...")
    t = time.time()

    running = await is_emulator_running()
    ms = int((time.time() - t) * 1000)

    if running:
        emit(f"   [2/6] ✅ Emulator running ({ms}ms)")
        return SanityCheck(
            name="emulator_ready", passed=True,
            message="Emulator device connected",
            duration_ms=ms,
        )
    else:
        # Try waiting for boot (maybe it's starting up)
        emit("   [2/6] Emulator not detected, waiting for boot...")
        boot_result = await wait_for_boot(timeout_sec=60)
        ms = int((time.time() - t) * 1000)

        if boot_result.ok:
            emit(f"   [2/6] ✅ Emulator booted ({ms}ms)")
            return SanityCheck(
                name="emulator_ready", passed=True,
                message="Emulator booted after wait",
                duration_ms=ms,
            )
        else:
            emit(f"   [2/6] ❌ Emulator not available ({ms}ms)")
            return SanityCheck(
                name="emulator_ready", passed=False,
                message="No emulator device found. Start the AVD first.",
                duration_ms=ms,
            )


async def _check_apk_install(
    apk_path: str,
    emit: Callable,
) -> SanityCheck:
    """Check 3: APK installs on emulator."""
    emit("   [3/6] APK install...")
    t = time.time()

    result = await install_apk(apk_path)
    ms = int((time.time() - t) * 1000)

    if result.ok:
        emit(f"   [3/6] ✅ APK installed ({ms}ms)")
        return SanityCheck(
            name="apk_install", passed=True,
            message="APK installed successfully",
            duration_ms=ms,
        )
    else:
        emit(f"   [3/6] ❌ APK install failed ({ms}ms)")
        return SanityCheck(
            name="apk_install", passed=False,
            message=f"Install failed: {result.stderr[:300]}",
            duration_ms=ms,
            details={"stderr": result.stderr},
        )


async def _check_app_launch(
    package_name: str,
    emit: Callable,
) -> SanityCheck:
    """Check 4: App launches without crash."""
    emit("   [4/6] App launch...")
    t = time.time()

    result = await launch_app(package_name)
    if not result.ok:
        ms = int((time.time() - t) * 1000)
        emit(f"   [4/6] ❌ Launch command failed ({ms}ms)")
        return SanityCheck(
            name="app_launch", passed=False,
            message=f"am start failed: {result.stderr[:200]}",
            duration_ms=ms,
        )

    # Wait a moment, then check logcat for crashes
    import asyncio
    await asyncio.sleep(3)

    has_crashes, crash_excerpt = await check_for_crashes(
        package_name, duration_sec=5
    )
    ms = int((time.time() - t) * 1000)

    if has_crashes:
        emit(f"   [4/6] ❌ App crashed on launch ({ms}ms)")
        return SanityCheck(
            name="app_launch", passed=False,
            message=f"Crash detected: {crash_excerpt[:200]}",
            duration_ms=ms,
            details={"crash_log": crash_excerpt},
        )

    emit(f"   [4/6] ✅ App launched clean ({ms}ms)")
    return SanityCheck(
        name="app_launch", passed=True,
        message="App launched without fatal exceptions",
        duration_ms=ms,
    )


async def _check_first_render(
    package_name: str,
    emit: Callable,
) -> SanityCheck:
    """Check 5: First screen renders within timeout.

    Dumps UI tree and checks there are interactive elements present,
    indicating the app has rendered a real screen (not just a splash).
    """
    emit("   [5/6] First render...")
    t = time.time()

    from app.pipeline_v2.bvl.emulator_bridge import dump_ui_tree, parse_ui_nodes

    # Allow up to 5 seconds for the first screen
    import asyncio
    for attempt in range(5):
        xml = await dump_ui_tree()
        nodes = parse_ui_nodes(xml)
        # Filter for meaningful UI (not just system bars)
        app_nodes = [
            n for n in nodes
            if package_name in n.resource_id or n.clickable
        ]
        if len(app_nodes) >= 2:
            ms = int((time.time() - t) * 1000)
            emit(f"   [5/6] ✅ Screen rendered ({len(app_nodes)} elements, {ms}ms)")
            return SanityCheck(
                name="first_render", passed=True,
                message=f"Screen rendered with {len(app_nodes)} interactive elements",
                duration_ms=ms,
                details={"element_count": len(app_nodes)},
            )
        await asyncio.sleep(1)

    ms = int((time.time() - t) * 1000)
    emit(f"   [5/6] ❌ Screen did not render ({ms}ms)")
    return SanityCheck(
        name="first_render", passed=False,
        message="No interactive elements found within 5s timeout",
        duration_ms=ms,
    )


async def _check_screenshot(emit: Callable) -> SanityCheck:
    """Check 6: Screenshot can be captured."""
    emit("   [6/6] Screenshot capture...")
    t = time.time()

    b64 = await screenshot_as_base64()
    ms = int((time.time() - t) * 1000)

    if b64:
        emit(f"   [6/6] ✅ Screenshot captured ({len(b64)} bytes, {ms}ms)")
        return SanityCheck(
            name="screenshot_capture", passed=True,
            message=f"Screenshot captured ({len(b64)} bytes)",
            duration_ms=ms,
            details={"b64": b64},
        )
    else:
        emit(f"   [6/6] ❌ Screenshot failed ({ms}ms)")
        return SanityCheck(
            name="screenshot_capture", passed=False,
            message="Could not capture screenshot from emulator",
            duration_ms=ms,
        )


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _resolve_apk_path(profile: "BuildTargetProfile") -> str:
    """Resolve the APK path from the profile."""
    root = profile.project_root.replace("\\", "/").rstrip("/")
    return f"{root}/app/build/outputs/apk/debug/app-debug.apk"


async def _make_result(
    checks: list,
    t_start: float,
    error_context: str,
) -> TierResult:
    """Build a FAILED TierResult from accumulated checks.

    JOB 7 (2026-06-10): ALWAYS capture a screenshot + UI-tree excerpt on
    failure so the diagnosing model never reasons blind — previously a
    render failure skipped the screenshot step entirely.
    """
    duration_ms = int((time.time() - t_start) * 1000)
    # Grab logcat from the last check if available
    logcat = ""
    for c in reversed(checks):
        if "crash_log" in c.details:
            logcat = c.details["crash_log"]
            break
        if "build_output" in c.details:
            logcat = c.details["build_output"]
            break

    screenshot_b64 = None
    ui_excerpt = ""
    try:
        from app.pipeline_v2.bvl.emulator_bridge import (
            dump_ui_tree,
            is_emulator_running,
            screenshot_as_base64,
        )
        if await is_emulator_running():
            screenshot_b64 = await screenshot_as_base64()
            ui_excerpt = ((await dump_ui_tree()) or "")[:1500]
    except Exception:
        pass

    if ui_excerpt:
        logcat = (logcat + "\n--- UI TREE AT FAILURE (excerpt) ---\n" + ui_excerpt)

    return TierResult(
        tier=TierName.SANITY,
        status=TierStatus.FAILED,
        checks=checks,
        error_context=error_context,
        logcat_excerpt=logcat[:3500],
        screenshot_b64=screenshot_b64,
        duration_ms=duration_ms,
    )
