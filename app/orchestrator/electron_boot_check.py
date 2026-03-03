# FILE: app/orchestrator/electron_boot_check.py
"""
Electron Boot Check — Full runtime validation.

v1.0 (2026-03-03): Real Electron boot test via sandbox.
Instead of just running `vite build` (compile check), this boots the full
Electron app via `npm run electron:dev`, waits for it to come online,
then reads the backend log and electron output for errors.

This catches runtime errors that vite build misses:
- Missing imports that only fail at runtime
- Components that crash on mount
- API calls that 404
- Backend startup failures
- CSS loading issues

Used as the FINAL step of the pipeline after all segments are complete.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

ELECTRON_BOOT_CHECK_BUILD_ID = "2026-03-03-v1.0-electron-boot"

# How long to wait for the backend to start (seconds)
_BACKEND_STARTUP_TIMEOUT = 30

# How long to wait after backend is up for Electron to settle
_ELECTRON_SETTLE_TIME = 5

# Patterns that indicate a clean boot
_BOOT_SUCCESS_PATTERNS = [
    re.compile(r"Application startup complete", re.IGNORECASE),
    re.compile(r"Uvicorn running on", re.IGNORECASE),
    re.compile(r"Backend is ready", re.IGNORECASE),
]

# Patterns that indicate real errors (not warnings)
_BOOT_ERROR_PATTERNS = [
    re.compile(r"\[ERROR\]"),
    re.compile(r"Error:|error:", re.IGNORECASE),
    re.compile(r"FAILED|FATAL", re.IGNORECASE),
    re.compile(r"Cannot find module", re.IGNORECASE),
    re.compile(r"ModuleNotFoundError", re.IGNORECASE),
    re.compile(r"ImportError", re.IGNORECASE),
    re.compile(r"SyntaxError", re.IGNORECASE),
    re.compile(r"TypeError:", re.IGNORECASE),
    re.compile(r"TS\d{4}:"),  # TypeScript errors
    re.compile(r"Expression tree is too large"),
]

# Patterns to IGNORE (known non-blocking warnings)
_BOOT_IGNORE_PATTERNS = [
    re.compile(r"SilentlyContinue"),
    re.compile(r"ErrorAction"),
    re.compile(r"No module named 'apscheduler'"),
    re.compile(r"No SIGNATURES file found"),
    re.compile(r"SyntaxWarning.*test_"),  # Warnings in test files
    re.compile(r"CJS build of Vite.*deprecated"),
]


@dataclass
class BootError:
    """Single error found during boot."""
    source: str       # "backend" or "electron"
    message: str
    severity: str     # "error" or "warning"


@dataclass
class ElectronBootResult:
    """Result of a full Electron boot check."""
    success: bool = False
    backend_up: bool = False
    electron_up: bool = False
    boot_time_ms: int = 0
    errors: List[BootError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    raw_log: str = ""
    error_summary: str = ""


def _shell_run(client, cmd_str: str, timeout: int = 15) -> Optional[str]:
    """Run a PowerShell command via sandbox and return stdout."""
    try:
        result = client.shell_run(
            command=cmd_str,
            timeout_seconds=timeout,
        )
        return getattr(result, "stdout", "") or ""
    except Exception as exc:
        logger.warning("[electron_boot] shell_run failed: %s", exc)
        return None


def _kill_existing_electron(client) -> None:
    """Kill any existing Electron/Node processes."""
    _shell_run(client, (
        "Get-Process -Name electron -ErrorAction SilentlyContinue | "
        "Stop-Process -Force -ErrorAction SilentlyContinue; "
        "'cleaned'"
    ), timeout=10)


def _clear_log(client) -> None:
    """Clear the backend log for a clean read."""
    _shell_run(client, (
        r"if (Test-Path D:\Orb\logs\astra.log) "
        r"{ Clear-Content D:\Orb\logs\astra.log }; 'cleared'"
    ), timeout=10)


def _launch_electron(client) -> bool:
    """Launch Electron in background. Returns True if command accepted."""
    result = _shell_run(client, (
        "Start-Process -FilePath 'cmd.exe' "
        r"-ArgumentList '/c cd /d D:\orb-desktop && npm run electron:dev "
        r"> D:\Orb\logs\electron_boot.log 2>&1' "
        "-WindowStyle Hidden; 'Launched'"
    ), timeout=10)
    return result is not None and "Launched" in (result or "")


def _wait_for_backend(client, timeout: int = _BACKEND_STARTUP_TIMEOUT) -> bool:
    """Poll the log until the backend reports startup complete."""
    start = time.time()
    while time.time() - start < timeout:
        log = _shell_run(client, (
            r"Get-Content D:\Orb\logs\electron_boot.log -ErrorAction "
            "SilentlyContinue | Select-Object -Last 20"
        ), timeout=10)
        if log and "Application startup complete" in log:
            return True
        time.sleep(3)
    return False


def _read_full_boot_log(client) -> str:
    """Read the complete electron boot log."""
    return _shell_run(client, (
        r"Get-Content D:\Orb\logs\electron_boot.log -Raw "
        "-ErrorAction SilentlyContinue"
    ), timeout=15) or ""


def _analyse_boot_log(raw_log: str) -> ElectronBootResult:
    """Parse the boot log for errors and success indicators."""
    result = ElectronBootResult(raw_log=raw_log)

    if not raw_log:
        result.error_summary = "No boot log produced"
        return result

    # Check for success indicators
    for pat in _BOOT_SUCCESS_PATTERNS:
        if pat.search(raw_log):
            if "startup complete" in pat.pattern.lower():
                result.backend_up = True
            if "Backend is ready" in pat.pattern:
                result.electron_up = True

    # Scan each line for errors
    for line in raw_log.split("\n"):
        line_stripped = line.strip()
        if not line_stripped:
            continue

        # Skip known-safe patterns
        if any(ip.search(line_stripped) for ip in _BOOT_IGNORE_PATTERNS):
            continue

        # Check for errors
        for ep in _BOOT_ERROR_PATTERNS:
            if ep.search(line_stripped):
                source = "backend" if "[Backend]" in line else "electron"
                result.errors.append(BootError(
                    source=source,
                    message=line_stripped[:300],
                    severity="error",
                ))
                break

    result.success = result.backend_up and result.electron_up and len(result.errors) == 0

    if result.errors:
        first_three = result.errors[:3]
        result.error_summary = "; ".join(e.message[:100] for e in first_three)
    elif not result.backend_up:
        result.error_summary = "Backend did not reach 'Application startup complete'"
    elif not result.electron_up:
        result.error_summary = "Electron did not report 'Backend is ready'"
    else:
        result.error_summary = ""

    return result


def run_electron_boot_check(
    client,
    frontend_base: str = r"D:\orb-desktop",
    emit=None,
) -> ElectronBootResult:
    """Run a full Electron boot check.

    1. Kill existing processes
    2. Clear logs
    3. Launch electron:dev
    4. Wait for backend startup
    5. Read and analyse full boot log
    6. Return structured result

    Args:
        client: SandboxClient instance
        frontend_base: Path to orb-desktop
        emit: Progress callback

    Returns:
        ElectronBootResult with success/failure and error details
    """
    _emit = emit or (lambda msg: None)
    _emit("  🚀 Running full Electron boot check...")
    logger.info("[electron_boot] Starting full Electron boot check")
    start_time = time.time()

    # Step 1: Kill existing
    _emit("  🧹 Cleaning up existing processes...")
    _kill_existing_electron(client)
    time.sleep(2)

    # Step 2: Clear log
    _clear_log(client)

    # Step 3: Launch
    _emit("  ⚡ Launching Electron app...")
    launched = _launch_electron(client)
    if not launched:
        logger.error("[electron_boot] Failed to launch Electron")
        return ElectronBootResult(error_summary="Failed to launch Electron process")

    # Step 4: Wait for backend
    _emit("  ⏳ Waiting for backend startup...")
    backend_ok = _wait_for_backend(client, timeout=_BACKEND_STARTUP_TIMEOUT)

    # Step 5: Let Electron settle
    if backend_ok:
        _emit("  ⏳ Waiting for Electron to settle...")
        time.sleep(_ELECTRON_SETTLE_TIME)

    # Step 6: Read and analyse
    raw_log = _read_full_boot_log(client)
    elapsed_ms = int((time.time() - start_time) * 1000)

    result = _analyse_boot_log(raw_log)
    result.boot_time_ms = elapsed_ms

    if result.success:
        _emit(f"  ✅ Electron boot PASSED ({elapsed_ms}ms) — backend up, electron ready, 0 errors")
        logger.info("[electron_boot] Boot PASSED in %dms", elapsed_ms)
    else:
        _emit(f"  ❌ Electron boot FAILED ({elapsed_ms}ms) — {result.error_summary[:200]}")
        logger.warning(
            "[electron_boot] Boot FAILED in %dms: %s",
            elapsed_ms, result.error_summary[:200],
        )
        if result.errors:
            for i, err in enumerate(result.errors[:5]):
                _emit(f"    [{err.source}] {err.message[:150]}")

    # Step 7: Kill after check
    _kill_existing_electron(client)

    return result