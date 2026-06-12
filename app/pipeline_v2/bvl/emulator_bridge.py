# FILE: app/pipeline_v2/bvl/emulator_bridge.py
# Purpose: Emulator Bridge — ADB abstraction layer for BVL.
# Called-by: app.pipeline_v2.bvl.bvl_orchestrator, app.pipeline_v2.bvl.failure_classifier, app.pipeline_v2.bvl.preflight, app.pipeline_v2.bvl.tier1_sanity (+3 more)
# Depends-on: app.pipeline_v2.sandbox_tools
# Last-renovated: 2026-06-11
"""
Emulator Bridge — ADB abstraction layer for BVL.

All Android emulator interactions go through this module. No other
BVL file calls ADB directly. This keeps the emulator coupling in
one place and makes it testable with a mock.

Responsibilities:
  - Emulator lifecycle (start, wait-for-boot, stop)
  - APK installation
  - App launch and force-stop
  - Screenshot capture via adb screencap
  - Logcat capture with filtering
  - UI tree dump via uiautomator
  - Input injection (tap, type, swipe, keyevent)
  - Device property queries (rotation, memory)

All commands execute in the sandbox via sandbox_tools.run_shell().

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-BVL-001.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.pipeline_v2.sandbox_tools import run_shell

logger = logging.getLogger(__name__)

# Timeout defaults (seconds)
BOOT_TIMEOUT = 120
INSTALL_TIMEOUT = 60
LAUNCH_TIMEOUT = 10
SCREENSHOT_TIMEOUT = 10
LOGCAT_TIMEOUT = 15
UI_DUMP_TIMEOUT = 10


# ═══════════════════════════════════════════════════════════════════
# Response types
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ADBResult:
    """Standardised result from any ADB command."""
    ok: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration_ms: int = 0


@dataclass
class UINode:
    """A single node from the UI Automator dump."""
    resource_id: str = ""
    class_name: str = ""
    text: str = ""
    content_desc: str = ""
    bounds: str = ""            # "[left,top][right,bottom]"
    clickable: bool = False
    enabled: bool = True
    focused: bool = False

    @property
    def center(self) -> Optional[Tuple[int, int]]:
        """Calculate the center point for tapping."""
        match = re.match(
            r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", self.bounds
        )
        if not match:
            return None
        left, top, right, bottom = (int(g) for g in match.groups())
        return ((left + right) // 2, (top + bottom) // 2)


# ═══════════════════════════════════════════════════════════════════
# Core ADB wrapper
# ═══════════════════════════════════════════════════════════════════

# Module-level profile reference. Set by the BVL orchestrator before
# running any tier so ADB commands route to the correct shell (host vs sandbox).
_active_profile: Optional["BuildTargetProfile"] = None

# Resolved ADB path — found once, cached for the session.
_adb_path: Optional[str] = None


def _resolve_adb() -> str:
    """Find the ADB executable. Checks common SDK locations."""
    global _adb_path
    if _adb_path:
        return _adb_path

    import os
    candidates = [
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Android", "Sdk", "platform-tools", "adb.exe"),
        os.path.join(os.environ.get("ANDROID_HOME", ""), "platform-tools", "adb.exe"),
        os.path.join(os.environ.get("ANDROID_SDK_ROOT", ""), "platform-tools", "adb.exe"),
        r"C:\Android\Sdk\platform-tools\adb.exe",
        r"D:\Android\Sdk\platform-tools\adb.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            _adb_path = path
            logger.info("[emulator] ADB found: %s", path)
            return path

    # Fallback — hope it's on PATH
    _adb_path = "adb"
    return "adb"


def set_active_profile(profile: Optional["BuildTargetProfile"]) -> None:
    """Set the profile for ADB command routing."""
    global _active_profile
    _active_profile = profile


async def _adb(cmd: str, timeout_sec: int = 30) -> ADBResult:
    """Run an ADB command through the appropriate shell.

    v2.3: Routes through host shell when an Android profile is active,
    or through the sandbox shell for ASTRA self-builds.
    All ADB commands funnel through here.
    """
    adb_exe = _resolve_adb()
    full_cmd = f'& "{adb_exe}" {cmd}'
    result = await run_shell(full_cmd, timeout_sec=timeout_sec, profile=_active_profile)
    return ADBResult(
        ok=result["returncode"] == 0,
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        returncode=result["returncode"],
    )


# ═══════════════════════════════════════════════════════════════════
# Emulator lifecycle
# ═══════════════════════════════════════════════════════════════════

async def start_emulator(avd_name: str = "Pixel_9_Pro") -> ADBResult:
    """Start the Android emulator in the background.

    Uses -no-window for headless operation in the pipeline.
    The emulator must be installed and the AVD must exist.
    """
    cmd = (
        f'Start-Process -FilePath "emulator" '
        f'-ArgumentList "-avd {avd_name} -no-window -no-audio -gpu swiftshader_indirect" '
        f'-PassThru | Select-Object -ExpandProperty Id'
    )
    result = await run_shell(cmd, timeout_sec=10)
    return ADBResult(
        ok=result["returncode"] == 0,
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        returncode=result["returncode"],
    )


async def wait_for_boot(timeout_sec: int = BOOT_TIMEOUT) -> ADBResult:
    """Wait for the emulator to finish booting.

    Polls `adb shell getprop sys.boot_completed` until it returns '1'.
    """
    cmd = (
        f'$timeout = {timeout_sec}; $elapsed = 0; '
        f'while ($elapsed -lt $timeout) {{ '
        f'  $result = adb shell getprop sys.boot_completed 2>$null; '
        f'  if ($result -and $result.Trim() -eq "1") {{ '
        f'    Write-Output "BOOT_COMPLETED"; exit 0 '
        f'  }}; '
        f'  Start-Sleep -Seconds 2; $elapsed += 2 '
        f'}}; '
        f'Write-Output "BOOT_TIMEOUT"; exit 1'
    )
    result = await run_shell(cmd, timeout_sec=timeout_sec + 10)
    return ADBResult(
        ok="BOOT_COMPLETED" in result.get("stdout", ""),
        stdout=result.get("stdout", ""),
        stderr=result.get("stderr", ""),
        returncode=result["returncode"],
    )


async def stop_emulator() -> ADBResult:
    """Kill the running emulator."""
    return await _adb("emu kill", timeout_sec=10)


async def is_emulator_running() -> bool:
    """Check if an emulator device is connected."""
    result = await _adb("devices", timeout_sec=5)
    if not result.ok:
        return False
    lines = result.stdout.strip().split("\n")
    for line in lines[1:]:  # Skip header
        if "emulator" in line and "device" in line:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
# APK management
# ═══════════════════════════════════════════════════════════════════

async def install_apk(apk_path: str) -> ADBResult:
    """Install an APK on the emulator. Replaces existing if present."""
    return await _adb(f'install -r "{apk_path}"', timeout_sec=INSTALL_TIMEOUT)


async def uninstall_package(package_name: str) -> ADBResult:
    """Uninstall a package from the emulator."""
    return await _adb(f"uninstall {package_name}", timeout_sec=15)


# ═══════════════════════════════════════════════════════════════════
# App lifecycle
# ═══════════════════════════════════════════════════════════════════

async def launch_app(
    package_name: str,
    activity: str = ".MainActivity",
) -> ADBResult:
    """Launch an app's main activity."""
    component = f"{package_name}/{package_name}{activity}"
    return await _adb(
        f"shell am start -n {component}",
        timeout_sec=LAUNCH_TIMEOUT,
    )


async def force_stop(package_name: str) -> ADBResult:
    """Force-stop an app."""
    return await _adb(f"shell am force-stop {package_name}", timeout_sec=5)


async def is_app_running(package_name: str) -> bool:
    """Check if an app's process is alive."""
    result = await _adb(
        f'shell pidof {package_name}', timeout_sec=5
    )
    return result.ok and result.stdout.strip().isdigit()


# ═══════════════════════════════════════════════════════════════════
# Screenshot
# ═══════════════════════════════════════════════════════════════════

async def capture_screenshot(
    device_path: str = "/sdcard/bvl_screenshot.png",
    local_path: str = "D:/Orb/jobs/bvl_screenshot.png",
) -> ADBResult:
    """Capture a screenshot from the emulator.

    Takes screenshot on device, pulls to host, returns result.
    """
    cap_result = await _adb(
        f"shell screencap -p {device_path}",
        timeout_sec=SCREENSHOT_TIMEOUT,
    )
    if not cap_result.ok:
        return cap_result

    pull_result = await _adb(
        f'pull {device_path} "{local_path}"',
        timeout_sec=SCREENSHOT_TIMEOUT,
    )
    return pull_result


async def screenshot_as_base64(
    device_path: str = "/sdcard/bvl_screenshot.png",
    local_path: str = "D:/Orb/jobs/bvl_screenshot.png",
) -> Optional[str]:
    """Capture screenshot and return as base64 PNG string."""
    result = await capture_screenshot(device_path, local_path)
    if not result.ok:
        logger.error("[emulator] Screenshot failed: %s", result.stderr)
        return None

    read_result = await run_shell(
        f"[Convert]::ToBase64String([IO.File]::ReadAllBytes('{local_path}'))",
        timeout_sec=10,
    )
    b64 = read_result.get("stdout", "").strip()
    if not b64 or len(b64) < 100:
        return None
    return b64


# ═══════════════════════════════════════════════════════════════════
# Logcat
# ═══════════════════════════════════════════════════════════════════

async def capture_logcat(
    package_name: str,
    duration_sec: int = 10,
    level: str = "E",
) -> str:
    """Capture logcat output for a package, filtered by level.

    Args:
        package_name: App package to filter for.
        duration_sec: How long to capture.
        level: Minimum log level (V, D, I, W, E, F).

    Returns:
        Filtered logcat text.
    """
    # Clear logcat first for a clean window
    await _adb("logcat -c", timeout_sec=5)

    # Capture with timeout
    cmd = (
        f'$proc = Start-Process -FilePath "adb" '
        f'-ArgumentList "logcat *:{level} -d" '
        f'-PassThru -NoNewWindow -RedirectStandardOutput "D:\\Orb\\jobs\\logcat_raw.txt"; '
        f'Start-Sleep -Seconds {duration_sec}; '
        f'Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue; '
        f'Get-Content "D:\\Orb\\jobs\\logcat_raw.txt" | '
        f'Select-String -Pattern "{package_name}|fatal|crash|exception" -CaseSensitive:$false | '
        f'Select-Object -First 100 | ForEach-Object {{ $_.Line }}'
    )
    result = await run_shell(cmd, timeout_sec=duration_sec + 15)
    return result.get("stdout", "")


async def check_for_crashes(
    package_name: str,
    duration_sec: int = 10,
) -> Tuple[bool, str]:
    """Check logcat for fatal exceptions after launch.

    Returns:
        (has_crashes, crash_excerpt)
    """
    logcat = await capture_logcat(package_name, duration_sec, level="E")
    crash_patterns = [
        r"FATAL EXCEPTION",
        r"java\.lang\.\w+Exception",
        r"Process.*has died",
        r"ANR in",
        r"Force finishing activity",
    ]
    crash_lines = []
    for line in logcat.split("\n"):
        for pattern in crash_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                crash_lines.append(line.strip())
                break

    has_crashes = len(crash_lines) > 0
    excerpt = "\n".join(crash_lines[:20]) if crash_lines else ""
    return has_crashes, excerpt


# ═══════════════════════════════════════════════════════════════════
# UI Automator
# ═══════════════════════════════════════════════════════════════════

async def dump_ui_tree() -> str:
    """Dump the current UI hierarchy as XML.

    Uses uiautomator dump. Returns raw XML string.
    """
    dump_result = await _adb(
        "shell uiautomator dump /sdcard/ui_dump.xml",
        timeout_sec=UI_DUMP_TIMEOUT,
    )
    if not dump_result.ok:
        return ""

    cat_result = await _adb(
        "shell cat /sdcard/ui_dump.xml",
        timeout_sec=UI_DUMP_TIMEOUT,
    )
    return cat_result.stdout if cat_result.ok else ""


def parse_ui_nodes(xml_text: str) -> List[UINode]:
    """Parse UI Automator XML dump into UINode objects.

    Uses regex parsing to avoid importing xml.etree in the
    sandbox path. Handles malformed XML gracefully.
    """
    nodes = []
    pattern = re.compile(
        r'<node\s+'
        r'.*?resource-id="([^"]*)"'
        r'.*?class="([^"]*)"'
        r'.*?text="([^"]*)"'
        r'.*?content-desc="([^"]*)"'
        r'.*?bounds="([^"]*)"'
        r'.*?clickable="([^"]*)"'
        r'.*?enabled="([^"]*)"'
        r'.*?focused="([^"]*)"',
        re.DOTALL,
    )
    for match in pattern.finditer(xml_text):
        nodes.append(UINode(
            resource_id=match.group(1),
            class_name=match.group(2),
            text=match.group(3),
            content_desc=match.group(4),
            bounds=match.group(5),
            clickable=match.group(6).lower() == "true",
            enabled=match.group(7).lower() == "true",
            focused=match.group(8).lower() == "true",
        ))
    return nodes


# v12.0: Filler words stripped during fuzzy matching
_FILLER_WORDS = {"the", "a", "an", "in", "on", "at", "of", "for", "to", "is", "it"}


def _normalise_for_match(s: str) -> str:
    """Lowercase, strip filler words, collapse whitespace."""
    words = s.lower().split()
    return " ".join(w for w in words if w not in _FILLER_WORDS)


def _fuzzy_match(query: str, target: str) -> bool:
    """Check if query matches target using normalised token overlap.

    Returns True if:
      - exact substring match (case-insensitive), OR
      - normalised query is a substring of normalised target, OR
      - normalised target is a substring of normalised query, OR
      - >70% of query tokens appear in target tokens
    """
    if not query or not target:
        return False
    q_lower = query.lower()
    t_lower = target.lower()
    # Exact substring (case-insensitive)
    if q_lower in t_lower or t_lower in q_lower:
        return True
    # Normalised substring
    q_norm = _normalise_for_match(query)
    t_norm = _normalise_for_match(target)
    if q_norm in t_norm or t_norm in q_norm:
        return True
    # Token overlap
    q_tokens = set(q_norm.split())
    t_tokens = set(t_norm.split())
    if not q_tokens:
        return False
    overlap = len(q_tokens & t_tokens) / len(q_tokens)
    return overlap >= 0.7


async def find_element(
    resource_id: str = "",
    text: str = "",
    content_desc: str = "",
) -> Optional[UINode]:
    """Find a UI element by resource ID, text, or content description.

    v12.0: Uses fuzzy matching with filler-word stripping and token overlap
    to handle minor wording differences between test expectations and
    actual content descriptions (e.g. 'in the top app bar' vs 'in top app bar').
    """
    xml = await dump_ui_tree()
    if not xml:
        return None

    nodes = parse_ui_nodes(xml)

    # Pass 1: exact/substring (fast)
    for node in nodes:
        if resource_id and resource_id in node.resource_id:
            return node
        if text and text == node.text:
            return node
        if content_desc and content_desc in node.content_desc:
            return node

    # Pass 2: fuzzy match (handles filler words, case, minor wording diffs)
    for node in nodes:
        if text and _fuzzy_match(text, node.text):
            return node
        if content_desc and _fuzzy_match(content_desc, node.content_desc):
            return node
        if text and _fuzzy_match(text, node.content_desc):
            return node
        if content_desc and _fuzzy_match(content_desc, node.text):
            return node

    return None


async def element_exists(
    resource_id: str = "",
    text: str = "",
    content_desc: str = "",
) -> bool:
    """Check if a UI element is present on screen."""
    node = await find_element(resource_id, text, content_desc)
    return node is not None


# ═══════════════════════════════════════════════════════════════════
# Input injection
# ═══════════════════════════════════════════════════════════════════

async def tap(x: int, y: int) -> ADBResult:
    """Tap at screen coordinates."""
    return await _adb(f"shell input tap {x} {y}", timeout_sec=5)


async def tap_element(node: UINode) -> ADBResult:
    """Tap the center of a UI element."""
    center = node.center
    if center is None:
        return ADBResult(ok=False, stderr=f"Cannot compute center for bounds: {node.bounds}")
    return await tap(center[0], center[1])


async def type_text(text: str) -> ADBResult:
    """Type text into the currently focused field.

    Escapes spaces as %s for ADB input text command.
    """
    escaped = text.replace(" ", "%s").replace("&", "\\&").replace("<", "\\<")
    return await _adb(f'shell input text "{escaped}"', timeout_sec=5)


async def swipe(
    x1: int, y1: int,
    x2: int, y2: int,
    duration_ms: int = 300,
) -> ADBResult:
    """Swipe from (x1,y1) to (x2,y2)."""
    return await _adb(
        f"shell input swipe {x1} {y1} {x2} {y2} {duration_ms}",
        timeout_sec=5,
    )


async def press_back() -> ADBResult:
    """Press the Android back button."""
    return await _adb("shell input keyevent KEYCODE_BACK", timeout_sec=5)


async def press_home() -> ADBResult:
    """Press the Android home button."""
    return await _adb("shell input keyevent KEYCODE_HOME", timeout_sec=5)


async def press_enter() -> ADBResult:
    """Press enter/return."""
    return await _adb("shell input keyevent KEYCODE_ENTER", timeout_sec=5)


# ═══════════════════════════════════════════════════════════════════
# Device state queries
# ═══════════════════════════════════════════════════════════════════

async def set_rotation(enabled: bool) -> ADBResult:
    """Enable or disable auto-rotation."""
    value = "1" if enabled else "0"
    return await _adb(
        f"shell settings put system accelerometer_rotation {value}",
        timeout_sec=5,
    )


async def rotate_device(orientation: int = 1) -> ADBResult:
    """Set device orientation. 0=portrait, 1=landscape, 2=reverse-portrait, 3=reverse-landscape."""
    await set_rotation(False)
    return await _adb(
        f"shell settings put system user_rotation {orientation}",
        timeout_sec=5,
    )


async def send_trim_memory(
    package_name: str,
    level: str = "RUNNING_CRITICAL",
) -> ADBResult:
    """Simulate low memory condition."""
    return await _adb(
        f"shell am send-trim-memory {package_name} {level}",
        timeout_sec=5,
    )


async def run_monkey(
    package_name: str,
    events: int = 1000,
    throttle_ms: int = 100,
) -> ADBResult:
    """Run Android Monkey stress tester."""
    return await _adb(
        f"shell monkey -p {package_name} --throttle {throttle_ms} -v {events}",
        timeout_sec=max(60, events // 5),
    )
