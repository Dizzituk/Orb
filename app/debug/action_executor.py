# FILE: app/debug/action_executor.py
"""
Action Executor: Translates LLM tool calls into real operations.

v9.5 (2026-03-30): Host write access — sandbox gate bypass for host-only paths.
  - execute_edit_file now bypasses sandbox health gate for host-only paths,
    reading and writing directly to host filesystem (fixes "sandbox controller
    unreachable" blocking Android project edits).
  - Added path aliases for Astra-Bridge, AndroidDriverCopilot, and
    Astra Android Folder so relative paths resolve correctly.

v9.4 (2026-03-30): Host write access overhaul.
  - Write protection narrowed to ONLY: D:/Orb/, D:/Orb.architecture,
    D:/orb-desktop, D:/orb-electron-data, and Windows system folders.
  - Everything else on D: is now fully writable (Android projects, tools, etc.).
  - Host-only paths (Android folder, etc.) now write directly to host filesystem
    instead of routing through sandbox controller.
  - execute_run_command regex patterns updated to match only protected dirs.

v9.3 (2026-03-28): Sandbox health gate fix.
  - Controller-only gate: sandbox writes only require the controller bridge,
    NOT the Orb backend running inside the sandbox. The controller handles
    all filesystem ops independently. The sandbox is ASTRA's playpen —
    full unrestricted access once the controller is reachable.

v9.2 (2026-03-28): Write-safety cleanup and hardening.
  - Removed dead code: _PROTECTED_HOST_WRITE_PREFIXES (undefined),
    _HOST_ONLY_READ_PREFIXES (unused), duplicate _is_protected_host_write.
  - Consolidated host write blocking into single _is_host_write_blocked().
  - Added write-blocking to execute_edit_file (was only in execute_write_file).
  - Hardened execute_run_command: block redirect/pipe writes to project dirs.
  - All write paths now log at WARNING level when blocked.

v9.1 (2026-03-04): Sandbox-first for ALL file operations.
The sandbox is a persistent Hyper-V clone of the host desktop with the
full codebase. All read/write/search/list operations go through the
sandbox controller API at 192.168.250.2:8765.

Host filesystem access:
- READ: everywhere, no restrictions.
- WRITE: everywhere EXCEPT four protected dirs (D:/Orb/, D:/Orb.architecture,
  D:/orb-desktop, D:/orb-electron-data) and Windows system folders.
- Host-only paths (Android projects, Orb architecture ref, logs) are served
  directly from the host filesystem, bypassing the sandbox controller.
- Pipeline state — in-memory on the host.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from app.debug.size_warning import add_size_warning as _size_warn

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
    # Orb core — ASTRA's own codebase, read-only on host.
    # Changes go through sandbox -> review -> git commit.
    "D:/Orb/",
    "D:\\Orb\\",
    "D:/Orb.architecture",
    "D:\\Orb.architecture",
    "D:/orb-desktop",
    "D:\\orb-desktop",
    "D:/orb-electron-data",
    "D:\\orb-electron-data",
    # System paths — never touch
    "C:/Windows",
    "C:\\Windows",
    "C:/Program Files",
    "C:\\Program Files",
    "C:/ProgramData",
    "C:\\ProgramData",
]

# Paths served from the host filesystem for reads (not in sandbox).
# Everything else goes through the sandbox controller.
# Paths served from the host filesystem (not in sandbox).
# These bypass the sandbox controller for reads. Write access is
# governed separately by _HOST_WRITE_BLOCKED_PREFIXES.
_HOST_ONLY_PREFIXES = [
    # Read-only reference data on host
    "D:/Orb/.architecture",
    "D:\\Orb\\.architecture",
    "D:/Orb/logs",
    "D:\\Orb\\logs",
    # Android projects — host only (not in sandbox), full read/write
    "D:/Astra Android Folder",
    "D:\\Astra Android Folder",
    # Architecture reference — host only, read-only (write-blocked above)
    "D:/Orb.architecture",
    "D:\\Orb.architecture",
]

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

def _is_host_write_blocked(path: str) -> bool:
    """Return True if the path must NEVER be written to on the host.

    This is the single source of truth for host write protection.
    Covers the four Orb directories and Windows system paths.
    Everything else on D: (Android projects, tools, etc.) is writable.
    """
    norm = path.replace("\\", "/").rstrip("/")
    for prefix in _HOST_WRITE_BLOCKED_PREFIXES:
        prefix_norm = prefix.replace("\\", "/").rstrip("/")
        # Exact match (e.g. "D:/Orb" == "D:/Orb") or
        # child path (e.g. "D:/Orb/app/foo.py".startswith("D:/Orb/"))
        if norm == prefix_norm or norm.startswith(prefix_norm + "/"):
            return True
    return False


def _sandbox_health_status() -> tuple[bool, str]:
    """Check sandbox controller is reachable for filesystem operations.
    
    Only requires the controller to be up — the controller handles all
    filesystem ops (read/write/tree) independently of the Orb backend.
    The sandbox is ASTRA's playpen: full unrestricted access once the
    controller bridge is reachable.
    """
    try:
        from app.sandbox.manager import get_sandbox_manager
        health = get_sandbox_manager().check_health()
        if not health.controller_ok:
            return False, (
                "BLOCKED: Sandbox controller is unreachable. "
                "Cannot write files — start the sandbox first."
            )
        # Controller is up — sandbox is available for all operations.
        # The Orb backend inside the sandbox is NOT required for file ops.
        if not health.backend_ok:
            logger.info("[action_executor] Sandbox controller OK, backend not running (not required for file ops)")
        return True, "ok"
    except Exception as e:
        return False, f"BLOCKED: Sandbox status check failed: {e}"


def _is_host_only(path: str) -> bool:
    """Check if path is host-only data (architecture maps, logs, frontend)."""
    for prefix in _HOST_ONLY_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def _resolve_sandbox_path(path: str) -> str:
    """Resolve relative/aliased paths to absolute sandbox paths.

    Models frequently send paths like 'orb-desktop/src/App.tsx' or
    'src/components/debug/DebugView.tsx'. These need to become absolute
    paths that the sandbox controller can find.
    """
    # Already absolute Windows path (D:\... or D:/...)
    if len(path) > 2 and path[1] == ":":
        return path

    # Strip leading slashes — models sometimes send Unix-style /app/...
    path = path.lstrip("/\\ ")

    # Bare empty or root -> default to D:/Orb
    if not path or path == ".":
        return "D:/Orb"

    # Try alias matching
    for prefix, replacement in _PATH_ALIASES:
        if path.startswith(prefix):
            resolved = replacement + path[len(prefix):]
            logger.debug("[action_executor] Path alias: '%s' -> '%s'", path, resolved)
            return resolved

    # Default: assume it's relative to D:/Orb
    resolved = f"D:/Orb/{path}"
    logger.debug("[action_executor] Path default: '%s' -> '%s'", path, resolved)
    return resolved


def _read_host_file(path: str, head: int = None, tail: int = None) -> Optional[str]:
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


# =============================================================================
# READ TOOLS
# =============================================================================

async def execute_read_file(params: Dict[str, Any]) -> str:
    """Read a file from the sandbox (or host for architecture/logs)."""
    path = params.get("path", "")
    head = params.get("head")
    tail = params.get("tail")

    if not path:
        return "Error: path is required."

    path = _resolve_sandbox_path(path)

    # Host-only data (architecture maps, logs)
    if _is_host_only(path):
        result = _read_host_file(path, head, tail)
        if result is not None:
            return _size_warn(result, path) if not head and not tail else result
        return f"File not found on host: {path}"

    # Everything else: sandbox controller
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/contents",
                json={"paths": [path], "max_file_size": 500000, "include_line_numbers": True},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if files:
                    if files[0].get("error"):
                        return f"Error: {files[0]['error']}"
                    content = files[0].get("content", "")
                    if head:
                        content = "\n".join(content.splitlines()[:head])
                    elif tail:
                        content = "\n".join(content.splitlines()[-tail:])
                    return _size_warn(content, path) if not head and not tail else content
                return f"File not found in sandbox: {path}"
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox read failed for {path}: {e}"


async def execute_list_files(params: Dict[str, Any]) -> str:
    """List directory contents via sandbox controller /fs/tree endpoint."""
    path = params.get("path", "")
    if not path:
        return "Error: path is required."

    path = _resolve_sandbox_path(path)

    # Host-only directories (architecture, logs)
    if _is_host_only(path):
        p = Path(path)
        if p.exists() and p.is_dir():
            try:
                entries = []
                for item in sorted(p.iterdir()):
                    prefix = "[DIR]" if item.is_dir() else "[FILE]"
                    entries.append(f"{prefix} {item.name}")
                return "\n".join(entries) if entries else "(empty directory)"
            except Exception as e:
                return f"Error listing directory: {e}"
        return f"Directory not found: {path}"

    # Sandbox controller
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/tree",
                json={"roots": [path], "max_depth": 1},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if files:
                    lines = []
                    for f in files:
                        name = f.get("name", "?")
                        ext = f.get("ext", "")
                        is_dir = not ext and f.get("size_bytes") is None
                        prefix = "[DIR]" if is_dir else "[FILE]"
                        lines.append(f"{prefix} {name}")
                    return "\n".join(lines)
                return f"(empty directory or not found: {path})"
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox list failed for {path}: {e}"


async def execute_search_files(params: Dict[str, Any]) -> str:
    """Search for files matching a pattern via sandbox /fs/tree + client-side filter."""
    import fnmatch

    root = params.get("path", "D:/Orb")
    pattern = params.get("pattern", "**/*")

    root = _resolve_sandbox_path(root)

    # Host-only directories (architecture)
    if _is_host_only(root):
        try:
            p = Path(root)
            if not p.exists():
                return f"Directory not found: {root}"
            matches = list(p.glob(pattern))
            skip_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
            filtered = [m for m in matches if not any(sd in m.parts for sd in skip_dirs)]
            if not filtered:
                return f"No files matching '{pattern}' in {root}"
            result_lines = [str(m) for m in filtered[:100]]
            suffix = f"\n... ({len(filtered)} total)" if len(filtered) > 100 else ""
            return "\n".join(result_lines) + suffix
        except Exception as e:
            return f"Search error: {e}"

    # Sandbox controller: deep scan, then filter client-side
    try:
        import httpx
        max_depth = 10 if "**/" in pattern or pattern.startswith("*") else 3
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/tree",
                json={"roots": [root], "max_depth": max_depth},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if not files:
                    return f"No files found in {root}"

                skip_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
                match_pattern = pattern.lstrip("*/") if pattern.startswith("**/") else pattern

                matched = []
                for f in files:
                    fpath = f.get("path", "")
                    fname = f.get("name", "")
                    if any(sd in fpath for sd in skip_dirs):
                        continue
                    if fnmatch.fnmatch(fname, match_pattern) or fnmatch.fnmatch(fpath, pattern):
                        matched.append(fpath)

                if not matched:
                    return f"No files matching '{pattern}' in {root}"
                result_lines = matched[:100]
                suffix = f"\n... ({len(matched)} total)" if len(matched) > 100 else ""
                return "\n".join(result_lines) + suffix
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox search failed for {root}: {e}"


async def execute_read_pipeline_state(params: Dict[str, Any]) -> str:
    """Get current pipeline state (host-side, in-memory)."""
    parts = []

    try:
        from app.llm.spec_flow_state import get_active_flow
        flow = get_active_flow()
        if flow:
            parts.append(f"Active flow stage: {flow.stage.value if hasattr(flow, 'stage') else str(flow)}")
        else:
            parts.append("No active pipeline flow.")
    except Exception as e:
        parts.append(f"Flow state unavailable: {e}")

    try:
        from app.llm.stage_trace import get_recent_traces
        traces = get_recent_traces(limit=10)
        if traces:
            parts.append("\nRecent stage traces:")
            for t in traces:
                parts.append(f"  {t}")
    except Exception:
        pass

    try:
        from app.llm.routing.handler_registry import get_latest_validated_spec
        spec = get_latest_validated_spec()
        if spec:
            parts.append(f"\nValidated spec: ID={spec.get('id', '?')}, hash={spec.get('hash', '?')}")
    except Exception:
        pass

    return "\n".join(parts) if parts else "No pipeline state data available."


async def execute_read_logs(params: Dict[str, Any]) -> str:
    """Read filtered log entries (host-side)."""
    level = params.get("level", "ALL").upper()
    limit = params.get("limit", 50)

    try:
        log_dir = Path("D:/Orb/logs")
        if not log_dir.exists():
            return "Log directory not found."

        log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not log_files:
            return "No log files found."

        text = log_files[0].read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        if level != "ALL":
            lines = [l for l in lines if level in l]

        recent = lines[-limit:]
        return f"Log file: {log_files[0].name}\n\n" + "\n".join(recent)
    except Exception as e:
        return f"Error reading logs: {e}"


# =============================================================================
# WRITE TOOLS (sandbox-only, host writes blocked)
# =============================================================================

async def execute_write_file(params: Dict[str, Any]) -> str:
    """Write a file. Host-only paths write directly; everything else via sandbox."""
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "Error: path is required."

    path = _resolve_sandbox_path(path)

    # HARD BLOCK: protected directories (Orb core + Windows system)
    if _is_host_write_blocked(path):
        logger.warning("[action_executor] BLOCKED host write attempt: %s", path)
        return (
            f"BLOCKED: Cannot write to {path} — this is a protected directory. "
            "Protected dirs: D:/Orb, D:/Orb.architecture, D:/orb-desktop, "
            "D:/orb-electron-data, and Windows system folders."
        )

    # Host-only paths: write directly to host filesystem
    if _is_host_only(path):
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            size_kb = len(content.encode("utf-8")) / 1024
            logger.info("[action_executor] Host write: %s (%.1f KB)", path, size_kb)
            return f"Successfully wrote {len(content)} chars ({size_kb:.1f} KB) to {path} (host)"
        except Exception as e:
            return f"Host write error for {path}: {e}"

    # Sandbox paths: write via sandbox controller
    sandbox_ok, sandbox_message = _sandbox_health_status()
    if not sandbox_ok:
        return sandbox_message

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/write",
                json={"path": path, "content": content, "overwrite": True},
            )
            if resp.status_code == 200:
                return f"Successfully wrote {len(content)} chars to {path} (sandbox)"
            return f"Write failed ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Write error: {e}"


async def execute_edit_file(params: Dict[str, Any]) -> str:
    """Edit a file via read-modify-write. Host-only paths bypass sandbox."""
    path = params.get("path", "")
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")

    if not path or not old_text:
        return "Error: path and old_text are required."

    resolved = _resolve_sandbox_path(path)

    # HARD BLOCK: check write safety before doing any work
    if _is_host_write_blocked(resolved):
        logger.warning("[action_executor] BLOCKED host edit attempt: %s", resolved)
        return (
            f"BLOCKED: Cannot edit {resolved} — this is a protected directory. "
            "Protected dirs: D:/Orb, D:/Orb.architecture, D:/orb-desktop, "
            "D:/orb-electron-data, and Windows system folders."
        )

    # Host-only paths: read and write directly on the host
    if _is_host_only(resolved):
        current = _read_host_file(resolved)
        if current is None:
            return f"File not found on host: {resolved}"

        if old_text not in current:
            return f"Error: old_text not found in {resolved}. The text to replace must exist exactly."

        count = current.count(old_text)
        if count > 1:
            return f"Error: old_text found {count} times in {resolved}. Must be unique."

        updated = current.replace(old_text, new_text, 1)
        return await execute_write_file({"path": resolved, "content": updated})

    # Sandbox paths: require sandbox controller
    sandbox_ok, sandbox_message = _sandbox_health_status()
    if not sandbox_ok:
        return sandbox_message

    # Read current content from sandbox
    current = await execute_read_file({"path": path})
    if current.startswith("Error") or current.startswith("Sandbox read failed"):
        return current

    if old_text not in current:
        return f"Error: old_text not found in {path}. The text to replace must exist exactly."

    count = current.count(old_text)
    if count > 1:
        return f"Error: old_text found {count} times in {path}. Must be unique."

    updated = current.replace(old_text, new_text, 1)
    return await execute_write_file({"path": resolved, "content": updated})


async def execute_run_command(params: Dict[str, Any]) -> str:
    """Run a command on the host via asyncio subprocess.

    Debug lock mode needs to run commands on the host (e.g. Gradle builds
    for Android projects, Python syntax checks). The sandbox may not have
    the required SDKs or project files.
    """
    import asyncio
    import re

    command = params.get("command", "")
    cwd = params.get("cwd", "D:\\Orb")
    timeout_sec = params.get("timeout_sec", 30)

    if not command:
        return "Error: command is required."

    cmd_lower = command.lower()

    # Block destructive system commands
    _BLOCKED_COMMANDS = [
        "remove-item c:\\", "del c:\\", "rd c:\\", "rmdir c:\\",
        "format-volume", "clear-disk", "stop-computer", "restart-computer",
        "set-executionpolicy", "new-service", "remove-service",
        "reg delete", "reg add", "net user", "net localgroup",
    ]
    for blocked in _BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            logger.warning("[action_executor] BLOCKED dangerous command: %s", command[:80])
            return f"BLOCKED: Command contains '{blocked}' — not allowed for safety."

    # Block file-writing commands that target project directories
    # Catches: echo > file, Set-Content, Out-File, >> redirect, etc.
    # Protected host folders: D:\Orb, D:\Orb.architecture, D:\orb-desktop, D:\orb-electron-data
    _PROTECTED_DIR_PATTERN = r'[dD]:[/\\](?:Orb[/\\]|Orb$|Orb\.architecture|orb-desktop|orb-electron-data)'
    _WRITE_PATTERNS = [
        r'[>|]\s*["\']?' + _PROTECTED_DIR_PATTERN,
        r'set-content\s+.*' + _PROTECTED_DIR_PATTERN,
        r'out-file\s+.*' + _PROTECTED_DIR_PATTERN,
        r'add-content\s+.*' + _PROTECTED_DIR_PATTERN,
        r'new-item\s+.*' + _PROTECTED_DIR_PATTERN,
        r'copy-item\s+.*' + _PROTECTED_DIR_PATTERN,
        r'move-item\s+.*' + _PROTECTED_DIR_PATTERN,
        r'remove-item\s+.*' + _PROTECTED_DIR_PATTERN,
    ]
    for pat in _WRITE_PATTERNS:
        if re.search(pat, cmd_lower):
            logger.warning("[action_executor] BLOCKED write-via-command: %s", command[:80])
            return (
                "BLOCKED: This command would write to a protected project directory. "
                "Use the sandbox for all file modifications."
            )

    # Block git commands — Taz handles all version control manually
    if re.search(r'\bgit\b', cmd_lower):
        logger.warning("[action_executor] BLOCKED git command: %s", command[:80])
        return "BLOCKED: Git commands are not allowed. Taz handles all version control."

    try:
        import base64
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        proc = await asyncio.create_subprocess_exec(
            "powershell.exe", "-NoProfile", "-EncodedCommand", encoded,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd if Path(cwd).exists() else None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)

        result = f"Exit code: {proc.returncode}\n"
        if stdout:
            result += f"\nSTDOUT:\n{stdout.decode('utf-8', errors='replace')[:5000]}"
        if stderr:
            result += f"\nSTDERR:\n{stderr.decode('utf-8', errors='replace')[:2000]}"
        return result
    except asyncio.TimeoutError:
        return f"ERROR: Command timed out after {timeout_sec} seconds"
    except Exception as e:
        return f"Command execution error: {e}"


# =============================================================================
# ADB EMULATOR TOOL HANDLERS
# =============================================================================

async def execute_emulator_screenshot(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import take_screenshot
    result = await take_screenshot()
    if "error" in result:
        return result["error"]
    return f"Screenshot saved: {result['path']} ({result['size_bytes']} bytes)"


async def execute_emulator_ui_dump(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import dump_ui_hierarchy
    return await dump_ui_hierarchy()


async def execute_emulator_tap(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import tap
    return await tap(int(params.get("x", 0)), int(params.get("y", 0)))


async def execute_emulator_type(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import type_text
    return await type_text(params.get("text", ""))


async def execute_emulator_key(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import press_key
    return await press_key(params.get("keycode", "KEYCODE_ENTER"))


async def execute_gradle_build(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import gradle_build
    return await gradle_build()


async def execute_gradle_install(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import gradle_install
    return await gradle_install()


async def execute_app_restart(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import restart_app
    return await restart_app()


async def execute_get_crash_log(params: Dict[str, Any]) -> str:
    from app.debug.adb_tools import get_crash_log
    return await get_crash_log()


# =============================================================================
# USER FILE TOOLS (v0.14.0)
# =============================================================================

async def execute_search_my_files(params: Dict[str, Any]) -> str:
    """Search the drive_file_manifest for user files by name/category/extension."""
    query = params.get("query", "").strip()
    category = params.get("category", "").strip().lower()
    extension = params.get("extension", "").strip().lower().lstrip(".")

    if not query and not category and not extension:
        return "Please provide a search query, category, or extension."

    if query:
        query = query.replace(" ", "%")

    try:
        from app.db import SessionLocal
        from app.drive.manifest_models import DriveFileManifest

        db = SessionLocal()
        try:
            q = db.query(DriveFileManifest)

            if query:
                from sqlalchemy import or_
                q = q.filter(or_(
                    DriveFileManifest.filename.ilike(f"%{query}%"),
                    DriveFileManifest.path.ilike(f"%{query}%"),
                ))
            if category:
                q = q.filter(DriveFileManifest.category == category)
            if extension:
                q = q.filter(DriveFileManifest.extension == extension)

            results = q.order_by(DriveFileManifest.filename).limit(30).all()

            if not results:
                return f"No files found matching: query={query!r}, category={category!r}, extension={extension!r}"

            lines = [f"Found {len(results)} file(s):"]
            for r in results:
                size_kb = r.size_bytes / 1024
                size_str = f"{size_kb:.0f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
                indexed = "indexed" if r.content_indexed else "not indexed"
                lines.append(
                    f"  [{r.category}] {r.filename} ({size_str}, {r.file_class}, {indexed})"
                )
                lines.append(f"    Path: {r.path}")
            return "\n".join(lines)
        finally:
            db.close()
    except Exception as e:
        return f"Search failed: {e}"


async def execute_read_user_file(params: Dict[str, Any]) -> str:
    """Read a user file by extracting its text content."""
    path = params.get("path", "").strip()
    if not path:
        return "Please provide a file path."

    import os
    if not os.path.isfile(path):
        return f"File not found: {path}"

    try:
        from app.drive.file_utils import get_category_paths
        allowed_roots = [str(p) for p in get_category_paths().values()]
        allowed_roots.append(os.path.join("D:", os.sep, "Orb", "output"))
        allowed_roots.append(os.path.join("D:", os.sep, "Orb", "data", "debug_uploads"))

        path_norm = os.path.normpath(path)
        if not any(path_norm.startswith(os.path.normpath(r)) for r in allowed_roots):
            return f"Access denied: {path} is outside allowed user file areas."
    except Exception:
        pass

    try:
        from app.llm.file_analyzer import extract_text
        text, err = extract_text(file_path=path, filename=os.path.basename(path))
        if text:
            if len(text) > 50000:
                return text[:50000] + f"\n\n... [TRUNCATED — {len(text)} chars total]"
            return _size_warn(text, path)
        elif err:
            return f"Could not extract text from {os.path.basename(path)}: {err}"
        else:
            return f"No readable content in {os.path.basename(path)}"
    except Exception as e:
        return f"Read failed: {e}"


# =============================================================================
# WEB SEARCH TOOL (universal — available to all models)
# =============================================================================

async def execute_web_search(params: Dict[str, Any]) -> str:
    """Search the web via ASTRA's existing Brave/DDG infrastructure."""
    query = str(params.get("query", "")).strip()
    if not query:
        return "Error: query is required."

    max_results = int(params.get("max_results", 5))
    max_results = max(1, min(10, max_results))

    try:
        from app.tools.registry import web_search_handler
        result = await web_search_handler(
            {"query": query, "max_results": max_results},
            context=None,
        )
        results = result.get("results", [])
        provider = result.get("provider", "unknown")
        if not results:
            return f"No results found for: {query} (provider: {provider})"

        lines = [f"Web search results for: {query} (via {provider})", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'Untitled')}")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('snippet', '')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        logger.error("[action_executor] web_search failed: %s", e)
        return f"Web search error: {e}"


# =============================================================================
# USER FILE WRITE TOOLS (v0.15.0) — write to personal folders
# =============================================================================

async def execute_write_user_file(params: Dict[str, Any]) -> str:
    """Write a file to the user's personal folders (Documents, Pictures, etc.).

    Security: validates path is within allowed user folder roots.
    """
    path = params.get("path", "").strip()
    content = params.get("content", "")

    if not path:
        return "Error: path is required."
    if not content:
        return "Error: content is required (file would be empty)."

    import os

    try:
        from app.drive.file_utils import get_category_paths, is_safe_path

        allowed_roots = list(get_category_paths().values())
        target = Path(path)

        if not is_safe_path(target, allowed_roots):
            return (
                f"Access denied: {path} is outside allowed user folders. "
                f"Use get_user_folders to see valid base paths."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        size_kb = len(content.encode("utf-8")) / 1024

        logger.info("[action_executor] User file write: %s (%.1f KB)", path, size_kb)

        try:
            from app.drive.manifest_scanner import index_single_file
            index_single_file(str(target))
        except Exception:
            pass

        return f"Successfully wrote {len(content)} chars ({size_kb:.1f} KB) to {path}"
    except PermissionError:
        return f"Permission denied writing to {path}"
    except Exception as e:
        return f"Write failed: {e}"


async def execute_get_user_folders(params: Dict[str, Any]) -> str:
    """Return resolved paths for all user personal folders."""
    try:
        from app.drive.file_utils import get_category_paths

        paths = get_category_paths()
        lines = ["User folders (use these as base paths for write_user_file):"]
        for category, folder_path in sorted(paths.items()):
            exists = folder_path.exists()
            lines.append(
                f"  {category}: {folder_path}  ({'OK' if exists else 'NOT FOUND'})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error resolving user folders: {e}"


# =============================================================================
# TOOL DISPATCHER
# =============================================================================

TOOL_HANDLERS = {
    # User file tools
    "search_my_files":     execute_search_my_files,
    "read_user_file":      execute_read_user_file,
    "write_user_file":     execute_write_user_file,
    "get_user_folders":    execute_get_user_folders,
    # Read (sandbox, except architecture/logs on host)
    "read_file":           execute_read_file,
    "list_files":          execute_list_files,
    "read_pipeline_state": execute_read_pipeline_state,
    "read_logs":           execute_read_logs,
    "search_files":        execute_search_files,
    # Write (sandbox only)
    "write_file":          execute_write_file,
    "edit_file":           execute_edit_file,
    "run_command":         execute_run_command,
    # ADB emulator tools
    "emulator_screenshot":  execute_emulator_screenshot,
    "emulator_ui_dump":     execute_emulator_ui_dump,
    "emulator_tap":         execute_emulator_tap,
    "emulator_type":        execute_emulator_type,
    "emulator_key":         execute_emulator_key,
    "gradle_build":         execute_gradle_build,
    "gradle_install":       execute_gradle_install,
    "app_restart":          execute_app_restart,
    "get_crash_log":        execute_get_crash_log,
    # Universal tools (all models)
    "web_search":           execute_web_search,
}


async def execute_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """Execute a tool call from the LLM."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Unknown tool: {tool_name}"

    logger.info("[action_executor] Executing tool: %s with params: %s", tool_name, params)
    try:
        result = await handler(params)
        logger.info("[action_executor] Tool %s completed (%d chars)", tool_name, len(result))
        return result
    except Exception as e:
        logger.error("[action_executor] Tool %s failed: %s", tool_name, e)
        return f"Tool execution error: {e}"
