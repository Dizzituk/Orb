# FILE: app/pot_spec/grounded/_sbx_fs.py
"""
v11.0: Sandbox filesystem helpers with intelligent routing.

RULES:
1. Paths inside ASTRA's own repos (D:\Orb, D:\orb-desktop) MUST go through
   the sandbox. If sandbox is unreachable, return False / None — the caller
   must handle this as a hard stop, NOT silently fall back to host.

2. Paths OUTSIDE ASTRA's own repos (app projects like Astra-Bridge, any
   other D:\ folder) go straight to the host filesystem. No sandbox needed.
   ASTRA can read/write these directly — if it breaks an app, ASTRA itself
   is still operational to fix it.

3. sandbox_available() helper for callers that need to check upfront and
   surface a clear message to the user.

v11.0 (2026-04-05): Dual-path routing — sandbox for ASTRA repos, host for apps
v10.0: Sandbox-only, no fallbacks
v9.0:  Host aliases
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── ASTRA's own repos — these MUST go through sandbox ──
_ASTRA_REPO_ROOTS = [
    "D:/Orb",
    "D:\\Orb",
    "D:/orb-desktop",
    "D:\\orb-desktop",
]


def _normalise(path: str) -> str:
    """Normalise path separators for comparison."""
    return path.replace("\\", "/").rstrip("/")


def _is_astra_repo_path(path: str) -> bool:
    """Check if path falls inside one of ASTRA's own repos."""
    norm = _normalise(path)
    for root in _ASTRA_REPO_ROOTS:
        root_norm = _normalise(root)
        if norm == root_norm or norm.startswith(root_norm + "/"):
            return True
    return False


def sandbox_available() -> bool:
    """Check if the sandbox controller is reachable.

    Callers can use this to surface a clear hard-stop message before
    attempting any sandbox-dependent work.
    """
    try:
        from app.sandbox_fs import _post
        status, _, _ = _post("/health", {})
        return status == 200
    except Exception:
        return False


def launch_sandbox(wait_seconds: int = 60) -> bool:
    """Attempt to auto-start Windows Sandbox with the ASTRA controller.

    Launches the .wsb file from D:\\Orb\\sandbox_controller\\astra_sandbox.wsb,
    then polls for the controller to come online.

    Args:
        wait_seconds: Max time to wait for controller to respond.

    Returns:
        True if sandbox came online within the timeout.
    """
    import subprocess
    import time

    wsb_path = os.path.join("D:\\", "Orb", "sandbox_controller", "astra_sandbox.wsb")
    if not os.path.isfile(wsb_path):
        logger.error("[_sbx_fs] Cannot auto-start sandbox: %s not found", wsb_path)
        return False

    # Check if already running
    if sandbox_available():
        logger.info("[_sbx_fs] Sandbox already available — no launch needed")
        return True

    logger.info("[_sbx_fs] Launching Windows Sandbox from %s ...", wsb_path)
    try:
        subprocess.Popen(
            ["WindowsSandbox.exe", wsb_path],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error("[_sbx_fs] WindowsSandbox.exe not found — is Windows Sandbox enabled?")
        return False
    except Exception as e:
        logger.error("[_sbx_fs] Failed to launch sandbox: %s", e)
        return False

    # Poll for controller to come online
    logger.info("[_sbx_fs] Waiting up to %ds for sandbox controller ...", wait_seconds)
    for i in range(wait_seconds):
        time.sleep(1)
        if sandbox_available():
            logger.info("[_sbx_fs] Sandbox controller online after %ds", i + 1)
            return True
        if (i + 1) % 10 == 0:
            logger.info("[_sbx_fs] Still waiting for sandbox... (%d/%ds)", i + 1, wait_seconds)

    logger.error("[_sbx_fs] Sandbox did not come online within %ds", wait_seconds)
    return False


# ── Sandbox imports (lazy, tolerant of import failures) ──
try:
    from app.sandbox_fs import (
        sandbox_isfile as _sandbox_isfile,
        sandbox_isdir as _sandbox_isdir,
        sandbox_exists as _sandbox_exists,
        sandbox_listdir as _sandbox_listdir,
        sandbox_read_text as _sandbox_read_text,
    )
    _SANDBOX_IMPORTS_OK = True
except ImportError:
    _SANDBOX_IMPORTS_OK = False
    _sandbox_isfile = None
    _sandbox_isdir = None
    _sandbox_exists = None
    _sandbox_listdir = None
    _sandbox_read_text = None
    logger.warning("[_sbx_fs] sandbox_fs imports failed — sandbox path unavailable")


# =========================================================================
# Public helpers — route based on path
# =========================================================================

def _sbx_isdir(path: str) -> bool:
    """Check if path is a directory.

    ASTRA repo paths → sandbox.
    All other paths → host filesystem.
    """
    if _is_astra_repo_path(path):
        if not _SANDBOX_IMPORTS_OK or not _sandbox_isdir:
            logger.error(
                "[_sbx_fs] ASTRA repo path %s requires sandbox but sandbox_fs not importable",
                path,
            )
            return False
        result = _sandbox_isdir(path)
        if not result:
            logger.warning(
                "[_sbx_fs] sandbox_isdir(%s) returned False — sandbox may be offline",
                path,
            )
        return result

    # Non-ASTRA path → host filesystem
    return os.path.isdir(path)


def _sbx_isfile(path: str) -> bool:
    """Check if path is a file.

    ASTRA repo paths → sandbox.
    All other paths → host filesystem.
    """
    if _is_astra_repo_path(path):
        if not _SANDBOX_IMPORTS_OK or not _sandbox_isfile:
            return False
        return _sandbox_isfile(path)

    return os.path.isfile(path)


def _sbx_exists(path: str) -> bool:
    """Check if path exists (file or directory).

    ASTRA repo paths → sandbox.
    All other paths → host filesystem.
    """
    if _is_astra_repo_path(path):
        if not _SANDBOX_IMPORTS_OK or not _sandbox_exists:
            return False
        return _sandbox_exists(path)

    return os.path.exists(path)


def _sbx_ls(path: str) -> list:
    """List directory contents.

    ASTRA repo paths → sandbox.
    All other paths → host filesystem.
    Returns a list of filenames (strings).
    """
    if _is_astra_repo_path(path):
        if not _SANDBOX_IMPORTS_OK or not _sandbox_listdir:
            return []
        entries = _sandbox_listdir(path)
        return [e.get("name", "") for e in entries if e.get("name")]

    try:
        return os.listdir(path)
    except OSError as e:
        logger.warning("[_sbx_fs] host listdir failed for %s: %s", path, e)
        return []


def _sbx_read(path: str) -> Optional[str]:
    """Read file text content.

    ASTRA repo paths → sandbox.
    All other paths → host filesystem.
    Returns None if unreadable.
    """
    if _is_astra_repo_path(path):
        if not _SANDBOX_IMPORTS_OK or not _sandbox_read_text:
            return None
        return _sandbox_read_text(path)

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        logger.warning("[_sbx_fs] host read failed for %s: %s", path, e)
        return None


# Alias for consistency
_sbx_read_text = _sbx_read