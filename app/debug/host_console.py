# FILE: app/debug/host_console.py
# Purpose: Persistent visible "ASTRA console" for HOST PowerShell. Every host
#          run_command is echoed (command + output) into a live-tailed transcript
#          window so Taz can watch ASTRA work; the model still receives the captured
#          output unchanged (the wrapper is transparent to the result).
# Called-by: app.debug.action_executor (run_command handler -> run_command_visible)
# Depends-on: app.debug.executors.filesystem.execute_run_command (lazy import)
# Last-renovated: 2026-06-17 (created -- WI-2 sandbox-visual-selfheal)
"""Visible host console.

WI-2 decision (Taz, 2026-06-17): ONE persistent live-tail window (not a window
per command), and ALL host commands are shown. The implementation is a passive
live-tail viewer -- a visible PowerShell window running ``Get-Content -Wait`` on a
transcript file that each command appends to. Execution and the model's output
capture are untouched; we only ALSO append to the tailed log. This keeps the
window robust (no stdin/stdout multiplexing) and matches the brief: "commands /
output as printed, not key-by-key".

Best-effort by design: any console failure is swallowed and logged -- it must
never break the command the model actually asked for.
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import threading
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Non-protected, backend-writable location (the backend already writes under data/).
# NB: this path is written by BACKEND code, not via the model's run_command tool,
# so the host protected-dir gate in filesystem.py does not apply here.
_LOG_DIR = r"D:\Orb\data\debug_console"
_LOG_PATH = os.path.join(_LOG_DIR, "host_console.log")

# Windows: give the child its own visible console window.
_CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0x00000010)

_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None   # the live-tail window process
_session_started = False                   # truncate the log once per backend run


def _ensure_dir() -> None:
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[host_console] could not create %s: %s", _LOG_DIR, e)


def _format_block(command: str, result: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar = "=" * 72
    return f"\n{bar}\n[{ts}]  HOST PS> {command}\n{'-' * 72}\n{result}\n"


def _ensure_window_sync() -> None:
    """Idempotently make sure ONE visible tail window is open. Best-effort.

    Relaunches if the window was closed (poll() returns an exit code). On a
    backend restart the module globals reset, so a fresh window is opened; an
    orphaned window from the prior run, if any, can simply be closed by Taz.
    """
    global _proc, _session_started
    with _lock:
        _ensure_dir()
        if not _session_started:
            try:
                with open(_LOG_PATH, "w", encoding="utf-8") as f:
                    f.write(
                        "=== ASTRA host console -- session started "
                        f"{datetime.now():%Y-%m-%d %H:%M:%S} ===\n"
                    )
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("[host_console] could not init log: %s", e)
            _session_started = True
        # Window already alive?
        if _proc is not None and _proc.poll() is None:
            return
        tail_cmd = (
            "$Host.UI.RawUI.WindowTitle = 'ASTRA -- host console (live)'; "
            f"Get-Content -LiteralPath '{_LOG_PATH}' -Tail 40 -Wait"
        )
        try:
            _proc = subprocess.Popen(
                ["powershell.exe", "-NoProfile", "-NoExit", "-Command", tail_cmd],
                creationflags=_CREATE_NEW_CONSOLE,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[host_console] could not launch console window: %s", e)
            _proc = None


def _append_block_sync(command: str, result: str) -> None:
    with _lock:
        _ensure_dir()
        try:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(_format_block(command, result))
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("[host_console] append failed: %s", e)


async def run_command_visible(params: Dict[str, Any]) -> str:
    """run_command handler wrapper: run on the host AS BEFORE, but also echo the
    command + output into the persistent visible console. Returns the SAME string
    the model would otherwise get -- fully transparent to callers.

    The console side-channel is best-effort: ensure-window and append are wrapped
    so a console failure can never change the command's result or raise.
    """
    # Lazy import avoids any import cycle / import-time cost.
    from app.debug.executors.filesystem import execute_run_command

    try:
        await asyncio.to_thread(_ensure_window_sync)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[host_console] ensure window failed: %s", e)

    result = await execute_run_command(params)

    try:
        command = str((params or {}).get("command", ""))
        await asyncio.to_thread(_append_block_sync, command, result)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("[host_console] echo failed: %s", e)

    return result


__all__ = ["run_command_visible"]
