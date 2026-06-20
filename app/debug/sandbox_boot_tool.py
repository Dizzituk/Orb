# FILE: app/debug/sandbox_boot_tool.py
# Purpose: Debug-agent tool to read how the SANDBOX CLONE's boot went -- health of
#          the clone backend + a scan of the clone's astra.log for boot errors.
#          Pairs with the start/stop/check sandbox tools so the Debug agent can run
#          the boot -> read -> (offer fix) -> reboot -> verify loop entirely on the
#          clone, never the host.
# Called-by: app.llm.chat_tool_loop (execute_chat_tool routing + write-tier schema)
# Depends-on: app.sandbox.manager, app.sandbox.client (read/call only -- PROTECTED, not modified)
# Last-renovated: 2026-06-17
"""read_sandbox_boot -- did the clone come up, and was the boot clean?

Reads the sandbox CLONE (never the host): polls the clone backend's health for up
to wait_seconds (the boot is slow), then reads the clone's astra.log and surfaces
any ERROR / CRITICAL / Traceback lines from the recent boot so the Debug agent can
analyse the outcome and offer a fix.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_WAIT = 180
_DEFAULT_LOG = "logs/astra.log"
_ERROR_MARKERS = (
    "ERROR", "CRITICAL", "Traceback", "Exception", "[X]",
    # frontend / Electron / Vite resolution failures, so a frontend error that DOES
    # reach a scanned log is surfaced too. (The primary frontend channel is the text
    # probe in app.debug.frontend_boot_check -- the Vite overlay never hits astra.log.)
    "Failed to resolve import", "vite:import-analysis", "Cannot find module",
    "ERR_MODULE_NOT_FOUND", "Pre-transform error",
)

READ_SANDBOX_BOOT_TOOL: Dict[str, Any] = {
    "name": "read_sandbox_boot",
    "description": (
        "Read how the SANDBOX CLONE's boot went: checks whether the clone's backend is "
        "up, then reads the clone's astra.log and reports any boot errors "
        "(ERROR/CRITICAL/Traceback). The boot is slow, so pass wait_seconds (e.g. 60) to "
        "let it finish booting before reading. Use after start_sandbox_clone, and again "
        "after a fix+reboot to verify a clean boot. Reads the CLONE only, never the host."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wait_seconds": {
                "type": "integer",
                "description": "Poll the clone backend up to this many seconds before reading the log (boot is slow; 60 is sensible). Default 0 = read immediately.",
            },
            "log_path": {
                "type": "string",
                "description": "Clone-relative log path to read. Default 'logs/astra.log'.",
            },
            "tail": {
                "type": "integer",
                "description": "How many recent log lines to scan for errors. Default 200.",
            },
        },
        "required": [],
    },
}


def _clone_log_abs_path(repo_root: str, log_path: str) -> str:
    """Absolute sandbox path of the clone log, from REPO_ROOT + a relative log path."""
    base = (repo_root or r"D:\Orb").rstrip("\\/")
    return base + "\\" + log_path.replace("/", "\\").lstrip("\\")


def _read_clone_log(log_path: str, tail: int, mgr) -> tuple[str, str]:
    """Return (log_text, source). Prefer repo_file (fast HTTP); on failure -- the
    controller DENIES any 'logs/' path, so repo_file 403s here today -- fall back to
    a VISIBLE Get-Content in the sandbox, which reads the actual running-clone log
    and lets Taz watch it. Never raises."""
    # Primary: repo_file (kept for any non-denied log path).
    try:
        fc = mgr.client.repo_file(log_path)
        text = getattr(fc, "content", "") or ""
        if text.strip():
            return text, "repo_file"
        primary_err = "empty"
    except Exception as e:
        primary_err = str(e)

    # Fallback: read the real clone log via a visible sandbox console command.
    try:
        try:
            repo_root = getattr(mgr.client.health(), "repo_root", "") or r"D:\Orb"
        except Exception:
            repo_root = r"D:\Orb"
        abs_path = _clone_log_abs_path(repo_root, log_path)
        from app.debug.sandbox_console import visible_shell_run
        cmd = (
            "if (Test-Path -LiteralPath '" + abs_path + "') { "
            "Get-Content -LiteralPath '" + abs_path + "' -Tail " + str(int(tail)) + " } "
            "else { Write-Output 'CLONE_LOG_MISSING " + abs_path + "' }"
        )
        res = visible_shell_run(cmd, cwd_target="REPO", timeout_seconds=30, client=mgr.client)
        out = (getattr(res, "stdout", "") or "").strip()
        if out and "CLONE_LOG_MISSING" not in out:
            return out, "console"
        if "CLONE_LOG_MISSING" in out:
            return "", f"not found at {abs_path} (repo_file: {primary_err})"
    except Exception as e:
        return "", f"repo_file: {primary_err}; console: {e}"
    return "", f"repo_file: {primary_err}; console returned nothing"


def read_sandbox_boot(params: Optional[Dict[str, Any]] = None) -> str:
    """Report clone health + recent boot errors. Sync (run via asyncio.to_thread).
    Never raises -- errors are returned as text for the model to read."""
    params = params or {}
    try:
        wait_seconds = max(0, min(int(params.get("wait_seconds") or 0), _MAX_WAIT))
    except Exception:
        wait_seconds = 0
    log_path = str(params.get("log_path") or _DEFAULT_LOG).strip() or _DEFAULT_LOG
    try:
        tail = max(20, min(int(params.get("tail") or 200), 1000))
    except Exception:
        tail = 200

    try:
        from app.sandbox.manager import get_sandbox_manager
    except Exception as e:
        return f"Sandbox unavailable: {e}"

    mgr = get_sandbox_manager()

    # Is the controller reachable at all?
    try:
        health = mgr.check_health()
    except Exception as e:
        return f"Sandbox controller check failed: {e}. Is the Windows Sandbox running?"

    if not getattr(health, "controller_ok", False):
        return (
            "Sandbox controller is UNREACHABLE -- start the Windows Sandbox first. "
            f"({getattr(health, 'message', '')})"
        )

    # Poll the clone backend up to wait_seconds (boot is slow).
    backend_ok = bool(getattr(health, "backend_ok", False))
    waited = 0
    if not backend_ok and wait_seconds > 0:
        for _ in range(wait_seconds):
            time.sleep(1)
            waited += 1
            try:
                ok, _msg = mgr.check_backend()
            except Exception:
                ok = False
            if ok:
                backend_ok = True
                break

    out = []
    out.append(
        f"CLONE STATUS: controller_ok=True backend_ok={backend_ok}"
        + (f" (waited {waited}s for backend)" if waited else "")
    )

    # Read the clone's boot log and scan for errors. The controller's /repo/file
    # endpoint DENIES any "logs/" path, so the repo_file read 403s here; the helper
    # falls back to a VISIBLE Get-Content of the real running-clone log.
    content, log_source = _read_clone_log(log_path, tail, mgr)
    if content:
        all_lines = content.splitlines()
        tail_lines = all_lines[-tail:]
        errors = [ln for ln in tail_lines if any(m in ln for m in _ERROR_MARKERS)]
        out.append(f"--- scanned last {len(tail_lines)} lines of {log_path} (via {log_source}): {len(errors)} error line(s) ---")
        if errors:
            out.append("BOOT ERRORS:")
            out.extend(errors[-40:])
        else:
            out.append("No ERROR/CRITICAL/Traceback in the recent boot log -- boot looks clean.")
        out.append("--- last lines ---")
        out.extend(tail_lines[-15:])
    else:
        out.append(f"(could not read clone {log_path}: {log_source})")

    if not backend_ok:
        out.append(
            "NOTE: clone backend is not responding yet. It may still be booting "
            "(call again with a larger wait_seconds), or the boot failed -- check the errors above."
        )

    return "\n".join(out)[:8000]


# ---------------------------------------------------------------------------
# inspect_sandbox_boot -- the FULL visual loop (backend readout + screenshot/OCR)
# ---------------------------------------------------------------------------

INSPECT_SANDBOX_BOOT_TOOL: Dict[str, Any] = {
    "name": "inspect_sandbox_boot",
    "description": (
        "Full VISUAL check of how the SANDBOX CLONE booted -- the complete self-heal loop. "
        "Waits for the clone backend (internal :8000, NOT the controller :8765), reads the "
        "clone's astra.log for backend errors, AND takes a screenshot of the clone's Electron "
        "app and vision/OCR-reads it for FRONTEND errors (error overlays, Vite/compiler errors, "
        "blank windows, crash dialogs -- the kind the backend log never shows). Returns ONE "
        "combined report: backend health + frontend visual state + concrete error text. Use "
        "this (rather than read_sandbox_boot) after start_sandbox_clone in FULL mode when you "
        "need to SEE how the app actually came up. The boot is SLOW, so it WAITS ~90s by default "
        "for the backend to finish coming up before it reads (override with wait_seconds). Reads "
        "the CLONE only, never the host. The PowerShell it runs in the "
        "sandbox is shown in the visible ASTRA console."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wait_seconds": {
                "type": "integer",
                "description": "Poll the clone backend up to this many seconds before reading. The boot is SLOW, so this defaults to 90 when omitted -- give it time; don't read an un-booted backend and mistake a slow start for a failed one.",
            },
            "log_path": {
                "type": "string",
                "description": "Clone-relative log path to read. Default 'logs/astra.log'.",
            },
            "tail": {
                "type": "integer",
                "description": "How many recent log lines to scan for errors. Default 200.",
            },
            "screenshot_delay_seconds": {
                "type": "integer",
                "description": "Extra settle time (s) after the backend read, before the screenshot, so the Electron window can finish rendering. Default 3, max 30.",
            },
        },
        "required": [],
    },
}


def inspect_sandbox_boot(params: Optional[Dict[str, Any]] = None) -> str:
    """Combined backend + visual report of the clone's boot. Sync (run via
    asyncio.to_thread). Never raises -- failures are returned as text."""
    # The clone boot is SLOW: wait generously by default so we poll the backend long
    # enough rather than reading an un-booted :8000 and calling a slow start a failure.
    params = dict(params or {})
    params.setdefault("wait_seconds", 90)

    # Backend half: controller/backend distinction, health, backend_ok (:8000) and
    # the clone-log error scan (now via the working console fallback).
    try:
        backend = read_sandbox_boot(params)
    except Exception as e:
        backend = f"(backend readout failed: {e})"

    # Give the Electron renderer a moment to paint before we look at it.
    try:
        delay = max(0, min(int(params.get("screenshot_delay_seconds") or 3), 30))
    except Exception:
        delay = 3
    if delay:
        time.sleep(delay)

    # Frontend TEXT probe (deps installed? Vite dev server up?) -- the reliable, primary
    # frontend-error channel: the Vite "Failed to resolve import" overlay never reaches
    # astra.log, and the screenshot below gets blocked by AV in the sandbox. Read as text.
    try:
        from app.debug.frontend_boot_check import probe_frontend_boot
        frontend_text = probe_frontend_boot()
    except Exception as e:
        frontend_text = f"FRONTEND (text probe): inspection failed: {e}"

    # Visual half: screenshot + vision/OCR of the clone's Electron window (WI-3) -- the
    # backup channel (may be AV-blocked; the text probe above is authoritative).
    try:
        from app.debug.sandbox_screenshot import capture_and_read_sandbox_screen
        visual = capture_and_read_sandbox_screen()
    except Exception as e:
        visual = f"VISUAL: inspection failed: {e}"

    return (
        "=== SANDBOX BOOT INSPECTION (backend + frontend) ===\n\n"
        "--- BACKEND (clone backend on internal :8000 + clone astra.log) ---\n"
        + backend
        + "\n\n--- FRONTEND (text probe: deps installed? + Vite dev server :5173) ---\n"
        + frontend_text
        + "\n\n--- FRONTEND (screenshot + vision of the clone's Electron window -- backup) ---\n"
        + visual
    )[:14000]


__all__ = [
    "READ_SANDBOX_BOOT_TOOL", "read_sandbox_boot",
    "INSPECT_SANDBOX_BOOT_TOOL", "inspect_sandbox_boot",
]
