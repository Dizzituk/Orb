# FILE: app/debug/executors/_paths.py
# Purpose: Path helpers and constants shared across all executor modules.
# Called-by: app.debug.action_executor, app.debug.executors.file_ops, app.debug.executors.filesystem, app.debug.executors.user_files
# Depends-on: app.sandbox.manager
# Last-renovated: 2026-06-11
"""
Path helpers and constants shared across all executor modules.

This is the single source of truth for:
  - Sandbox controller URL
  - Host write blocking (which paths must NEVER be written on the host)
  - Host-only routing (which paths bypass the sandbox controller for reads)
  - Path resolution (relative -> absolute, alias expansion)
  - Sandbox health checks
  - Direct host file reads (for architecture/logs/Android projects)

If this file approaches 30 KB, split the constants tables out into
_paths_constants.py and keep only helper functions here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Sandbox controller base URL
SANDBOX_CONTROLLER_URL = "http://192.168.250.2:8765"

# Paths that ASTRA must NEVER write to directly on the host.
# All code changes go through the sandbox controller and are promoted
# to host only after human review + git commit.
_HOST_WRITE_BLOCKED_PREFIXES = [
    "D:/Orb/",
    "D:\\Orb\\",
    "D:/Orb.architecture",
    "D:\\Orb.architecture",
    "D:/orb-desktop",
    "D:\\orb-desktop",
    "D:/orb-electron-data",
    "D:\\orb-electron-data",
    "D:/Zobie-Orb",
    "D:\\Zobie-Orb",
    # System paths — never touch
    "C:/Windows",
    "C:\\Windows",
    "C:/Program Files",
    "C:\\Program Files",
    "C:/ProgramData",
    "C:\\ProgramData",
]

# Carve-outs: paths under a blocked prefix that ARE writable.
_HOST_WRITE_ALLOWED_SUBPATHS = [
    "D:/Orb/output",
    "D:\\Orb\\output",
    "D:/Orb/data/debug_uploads",
    "D:\\Orb\\data\\debug_uploads",
]

# Paths served directly from the host filesystem (bypassing the sandbox
# controller). Everything else routes through the sandbox.
_HOST_ONLY_PREFIXES = [
    "D:/Orb/.architecture",
    "D:\\Orb\\.architecture",
    "D:/Orb/logs",
    "D:\\Orb\\logs",
    "D:/Astra Android Folder",
    "D:\\Astra Android Folder",
    "D:/Orb.architecture",
    "D:\\Orb.architecture",
]

# Append the current user's home directory so user-data folders
# (OneDrive, Pictures, Documents, Desktop, Downloads, etc.) route to
# host-direct iterdir instead of the sandbox controller, which has no
# knowledge of the user profile.
try:
    _USER_HOME = str(Path.home())
    _HOST_ONLY_PREFIXES.extend([
        _USER_HOME.replace('\\', '/'),
        _USER_HOME.replace('/', '\\'),
    ])
except Exception:
    pass

# Known path aliases: models often send relative paths.
_PATH_ALIASES = [
    ("orb-desktop/", "D:/orb-desktop/"),
    ("orb-desktop\\", "D:\\orb-desktop\\"),
    ("src/", "D:/orb-desktop/src/"),
    ("src\\", "D:\\orb-desktop\\src\\"),
    ("app/", "D:/Orb/app/"),
    ("app\\", "D:\\Orb\\app\\"),
    ("Orb/", "D:/Orb/"),
    ("Orb\\", "D:\\Orb\\"),
    # Android project aliases
    ("Astra-Bridge/", "D:/Astra Android Folder/Astra-Bridge/"),
    ("Astra-Bridge\\", "D:\\Astra Android Folder\\Astra-Bridge\\"),
    ("AstraBridge/", "D:/Astra Android Folder/Astra-Bridge/"),
    ("AstraBridge\\", "D:\\Astra Android Folder\\Astra-Bridge\\"),
    ("AndroidDriverCopilot/", "D:/Astra Android Folder/AndroidDriverCopilot/"),
    ("AndroidDriverCopilot\\", "D:\\Astra Android Folder\\AndroidDriverCopilot\\"),
    ("DriverCopilot/", "D:/Astra Android Folder/AndroidDriverCopilot/"),
    ("DriverCopilot\\", "D:\\Astra Android Folder\\AndroidDriverCopilot\\"),
    ("Astra Android Folder/", "D:/Astra Android Folder/"),
    ("Astra Android Folder\\", "D:\\Astra Android Folder\\"),
]


# =============================================================================
# PATH HELPERS
# =============================================================================

def is_host_write_blocked(path: str) -> bool:
    """Return True if the path must NEVER be written to on the host.

    Single source of truth for host write protection. Carve-outs
    (D:/Orb/output etc.) are allowed even when nested under a
    blocked prefix. Everything else on D: is writable.
    """
    norm = path.replace("\\", "/").rstrip("/")

    for allowed in _HOST_WRITE_ALLOWED_SUBPATHS:
        allowed_norm = allowed.replace("\\", "/").rstrip("/")
        if norm == allowed_norm or norm.startswith(allowed_norm + "/"):
            return False

    for prefix in _HOST_WRITE_BLOCKED_PREFIXES:
        prefix_norm = prefix.replace("\\", "/").rstrip("/")
        if norm == prefix_norm or norm.startswith(prefix_norm + "/"):
            return True
    return False


def sandbox_health_status() -> tuple[bool, str]:
    """Check sandbox controller is reachable for filesystem operations."""
    try:
        from app.sandbox.manager import get_sandbox_manager
        health = get_sandbox_manager().check_health()
        if not health.controller_ok:
            return False, (
                "BLOCKED: Sandbox controller is unreachable. "
                "Cannot write files - start the sandbox first."
            )
        if not health.backend_ok:
            logger.info(
                "[executors._paths] Sandbox controller OK, "
                "backend not running (not required for file ops)"
            )
        return True, "ok"
    except Exception as e:
        return False, f"BLOCKED: Sandbox status check failed: {e}"


def is_host_only(path: str) -> bool:
    """Check if path is host-only data (architecture, logs, Android, user home)."""
    for prefix in _HOST_ONLY_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def resolve_sandbox_path(path: str) -> str:
    """Resolve relative/aliased paths to absolute sandbox paths."""
    # Already absolute Windows path (D:\... or D:/...)
    if len(path) > 2 and path[1] == ":":
        return path

    # Strip leading slashes
    path = path.lstrip("/\\ ")

    # Bare empty or root -> default to D:/Orb
    if not path or path == ".":
        return "D:/Orb"

    # Try alias matching
    for prefix, replacement in _PATH_ALIASES:
        if path.startswith(prefix):
            resolved = replacement + path[len(prefix):]
            logger.debug("[executors._paths] alias: '%s' -> '%s'", path, resolved)
            return resolved

    # Default: assume it's relative to D:/Orb
    resolved = f"D:/Orb/{path}"
    logger.debug("[executors._paths] default: '%s' -> '%s'", path, resolved)
    return resolved


def read_host_file(path: str, head: Optional[int] = None, tail: Optional[int] = None) -> Optional[str]:
    """Read a file from the host filesystem. Returns None if not found."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()
        if head:
            lines = lines[:head]
        elif tail:
            lines = lines[-tail:]
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading host file: {e}"
