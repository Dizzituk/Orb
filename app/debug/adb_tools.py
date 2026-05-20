# FILE: app/debug/adb_tools.py
"""
ADB Test Harness for ASTRA Debug Agent.

Provides tool functions that the debug agent (GPT-5.4) can call to:
- Take screenshots of the emulator
- Dump UI hierarchy (for element inspection)
- Tap, type, swipe on the emulator
- Launch/restart the app
- Run Gradle builds
- Capture logcat output

These tools enable a test-iterate loop:
  Fix code → Build → Deploy → Screenshot → Verify → Iterate

v1.0 (2026-03-22): Initial — ADB bridge for debug agent.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ADB_PATH = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Android", "Sdk", "platform-tools", "adb.exe"
)
SCREENSHOT_DIR = Path("D:/Orb/data/emulator_screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

ASTRA_BRIDGE_PACKAGE = "com.astra.astrabridge"
ASTRA_BRIDGE_ACTIVITY = f"{ASTRA_BRIDGE_PACKAGE}/.MainActivity"
ASTRA_BRIDGE_ROOT = "D:/Astra Android Folder/Astra-Bridge"


async def _run_adb(*args: str, timeout: int = 15) -> str:
    """Run an ADB command and return stdout."""
    cmd = [ADB_PATH] + list(args)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        result = stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            result += f"\nSTDERR: {err}"
        return result.strip()
    except asyncio.TimeoutError:
        return "ERROR: ADB command timed out"
    except FileNotFoundError:
        return f"ERROR: ADB not found at {ADB_PATH}"
    except Exception as e:
        return f"ERROR: {e}"


async def _run_shell(cmd: str, timeout: int = 15) -> str:
    """Run a shell command on the emulator."""
    return await _run_adb("shell", cmd, timeout=timeout)


def _normalize_activity_name(package_name: str, activity_name: Optional[str]) -> str:
    if activity_name:
        activity_name = activity_name.strip()
        if activity_name.startswith(package_name + "/"):
            return activity_name
        if activity_name.startswith("."):
            return f"{package_name}/{activity_name}"
        if "/" in activity_name:
            return activity_name
        return f"{package_name}/.{activity_name}"
    return f"{package_name}/.MainActivity"


def _resolve_project_root(project_root: Optional[str]) -> str:
    return project_root.strip() if project_root and project_root.strip() else ASTRA_BRIDGE_ROOT


# ═══════════════════════════════════════════════════════════════
# SCREENSHOT & VISUAL INSPECTION
# ═══════════════════════════════════════════════════════════════

async def take_screenshot() -> dict:
    """Take a screenshot of the emulator and return the file path + base64 preview.

    Returns:
        {"path": str, "size_bytes": int, "base64_preview": str (first 500 chars)}
    """
    timestamp = int(time.time())
    filename = f"screen_{timestamp}.png"
    local_path = SCREENSHOT_DIR / filename

    # Capture via exec-out (binary safe)
    proc = await asyncio.create_subprocess_exec(
        ADB_PATH, "exec-out", "screencap", "-p",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)

    if not stdout or len(stdout) < 1000:
        return {"error": "Screenshot capture failed — emulator may not be running"}

    local_path.write_bytes(stdout)
    b64 = base64.b64encode(stdout[:2000]).decode("ascii")

    logger.info("[adb_tools] Screenshot: %s (%d bytes)", local_path, len(stdout))
    return {
        "path": str(local_path),
        "size_bytes": len(stdout),
        "base64_preview": b64[:500],
    }


async def dump_ui_hierarchy() -> str:
    """Dump the UI view hierarchy from the emulator.

    Returns XML describing all visible UI elements with their:
    - class, text, content-desc, resource-id
    - bounds (click coordinates)
    - enabled, clickable, focusable states

    Use this to find buttons, text fields, and verify layout.
    """
    await _run_shell("uiautomator dump /sdcard/ui_dump.xml")
    result = await _run_adb("shell", "cat /sdcard/ui_dump.xml")
    if len(result) < 50:
        return "ERROR: UI dump failed — app may be crashed or emulator not running"
    return result


async def get_focused_activity() -> str:
    """Get the currently focused activity on the emulator."""
    result = await _run_shell("dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'")
    return result


# ═══════════════════════════════════════════════════════════════
# USER INTERACTION
# ═══════════════════════════════════════════════════════════════

async def tap(x: int, y: int) -> str:
    """Tap at the given screen coordinates."""
    result = await _run_shell(f"input tap {x} {y}")
    return result or f"Tapped at ({x}, {y})"


async def type_text(text: str) -> str:
    """Type text into the currently focused field.

    Note: Spaces are encoded as %s for ADB input.
    """
    escaped = text.replace(" ", "%s").replace("&", "\\&").replace(";", "\\;")
    result = await _run_shell(f'input text "{escaped}"')
    return result or f"Typed: {text}"


async def press_key(keycode: str) -> str:
    """Press a key by keycode name (e.g. KEYCODE_ENTER, KEYCODE_BACK)."""
    result = await _run_shell(f"input keyevent {keycode}")
    return result or f"Pressed: {keycode}"


async def swipe(x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> str:
    """Swipe from (x1,y1) to (x2,y2)."""
    result = await _run_shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")
    return result or f"Swiped ({x1},{y1}) → ({x2},{y2})"


async def clear_field() -> str:
    """Clear the currently focused text field."""
    # Select all + delete
    await _run_shell("input keyevent KEYCODE_MOVE_HOME")
    await _run_shell("input keyevent --longpress KEYCODE_SHIFT_LEFT KEYCODE_MOVE_END")
    await _run_shell("input keyevent KEYCODE_DEL")
    return "Field cleared"


# ═══════════════════════════════════════════════════════════════
# APP LIFECYCLE
# ═══════════════════════════════════════════════════════════════

async def launch_app(package_name: Optional[str] = None, activity_name: Optional[str] = None) -> str:
    """Launch an Android app on the emulator."""
    package_name = package_name or ASTRA_BRIDGE_PACKAGE
    component = _normalize_activity_name(package_name, activity_name)
    result = await _run_shell(f"am start -n {component}")
    return result


async def force_stop_app(package_name: Optional[str] = None) -> str:
    """Force stop an Android app."""
    package_name = package_name or ASTRA_BRIDGE_PACKAGE
    result = await _run_shell(f"am force-stop {package_name}")
    return result or "App force stopped"


async def restart_app(package_name: Optional[str] = None, activity_name: Optional[str] = None) -> str:
    """Force stop and relaunch an Android app."""
    await force_stop_app(package_name=package_name)
    await asyncio.sleep(1)
    return await launch_app(package_name=package_name, activity_name=activity_name)


async def clear_app_data(package_name: Optional[str] = None) -> str:
    """Clear all app data (SharedPreferences, databases, cache)."""
    package_name = package_name or ASTRA_BRIDGE_PACKAGE
    result = await _run_shell(f"pm clear {package_name}")
    return result


async def install_apk(apk_path: str, project_root: Optional[str] = None, package_name: Optional[str] = None, activity_name: Optional[str] = None) -> str:
    """Install an APK on the emulator."""
    result = await _run_adb("install", "-r", apk_path, timeout=60)
    return result


# ═══════════════════════════════════════════════════════════════
# BUILD & DEPLOY
# ═══════════════════════════════════════════════════════════════

async def gradle_build(project_root: Optional[str] = None) -> str:
    """Run Gradle assembleDebug for an Android project.

    Returns the build output (last 30 lines for brevity).
    """
    proc = await asyncio.create_subprocess_shell(
        f'cd /d "{_resolve_project_root(project_root)}" && gradlew.bat assembleDebug',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=_resolve_project_root(project_root),
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    output = stdout.decode("utf-8", errors="replace")
    lines = output.strip().split("\n")

    # Return last 30 lines for brevity
    summary = "\n".join(lines[-30:])
    success = "BUILD SUCCESSFUL" in output
    return f"{'BUILD SUCCESSFUL' if success else 'BUILD FAILED'}\n\n{summary}"


async def gradle_install(project_root: Optional[str] = None, apk_path: Optional[str] = None, package_name: Optional[str] = None, activity_name: Optional[str] = None) -> str:
    """Build and install an Android debug APK on the emulator."""
    proc = await asyncio.create_subprocess_shell(
        f'cd /d "{_resolve_project_root(project_root)}" && gradlew.bat installDebug',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=_resolve_project_root(project_root),
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    output = stdout.decode("utf-8", errors="replace")
    lines = output.strip().split("\n")
    summary = "\n".join(lines[-30:])
    success = "BUILD SUCCESSFUL" in output
    if apk_path:
        install_result = await install_apk(apk_path, project_root=project_root, package_name=package_name, activity_name=activity_name)
        return f"{'BUILD SUCCESSFUL' if success else 'BUILD FAILED'}\n\n{summary}\n\nINSTALL RESULT:
{install_result}"
    return f"{'BUILD + INSTALL SUCCESSFUL' if success else 'BUILD/INSTALL FAILED'}\n\n{summary}"


# ═══════════════════════════════════════════════════════════════
# LOGCAT
# ═══════════════════════════════════════════════════════════════

async def get_logcat(lines: int = 50, filter_tag: str = "") -> str:
    """Get recent logcat output, optionally filtered by tag.

    Args:
        lines: Number of recent lines to return
        filter_tag: Optional tag filter (e.g. 'AndroidRuntime' for crashes)
    """
    if filter_tag:
        cmd = f"logcat -d -t {lines} {filter_tag}:E *:S"
    else:
        cmd = f"logcat -d -t {lines} {ASTRA_BRIDGE_PACKAGE}:V *:S"
    return await _run_shell(cmd)


async def get_crash_log() -> str:
    """Get the most recent crash log for AstraBridge."""
    return await _run_shell(
        f"logcat -d -t 100 AndroidRuntime:E *:S | grep -A 50 '{ASTRA_BRIDGE_PACKAGE}'"
    )