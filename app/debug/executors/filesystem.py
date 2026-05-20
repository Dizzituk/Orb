# FILE: app/debug/executors/filesystem.py
"""
Filesystem executors: read, write, edit, list, search, run_command.

Routing rules:
  - Host-only paths (architecture, logs, Android, user home) -> direct host
  - Protected paths (Orb codebase) -> sandbox controller
  - Hard-blocked paths (Windows system) -> always refused
  - Everything else writable on host

run_command runs ON THE HOST via PowerShell, with extensive blocklists
to prevent file-write escape hatches that would bypass sandbox routing.

If this file approaches 30 KB, split run_command + its blocklists into
filesystem_run_command.py.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict

from app.debug.executors._paths import (
    SANDBOX_CONTROLLER_URL,
    is_host_only,
    is_host_write_blocked,
    read_host_file,
    resolve_sandbox_path,
    sandbox_health_status,
)
from app.debug.size_warning import add_size_warning as _size_warn

logger = logging.getLogger(__name__)


# =============================================================================
# READ
# =============================================================================

async def execute_read_file(params: Dict[str, Any]) -> str:
    """Read a file from the sandbox (or host for architecture/logs)."""
    path = params.get("path", "")
    head = params.get("head")
    tail = params.get("tail")

    if not path:
        return "Error: path is required."

    path = resolve_sandbox_path(path)

    # Host-only data (architecture maps, logs, user home)
    if is_host_only(path):
        result = read_host_file(path, head, tail)
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
                return f"File not found: {path}"
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Sandbox read failed for {path}: {e}"


async def execute_list_files(params: Dict[str, Any]) -> str:
    """List directory contents via sandbox controller, or host iterdir for host-only paths."""
    path = params.get("path", "")
    if not path:
        return "Error: path is required."

    path = resolve_sandbox_path(path)

    # Host-only directories (architecture, logs, user home)
    if is_host_only(path):
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

    root = resolve_sandbox_path(root)

    # Host-only directories (architecture)
    if is_host_only(root):
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

    # Sandbox controller
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/tree",
                json={"roots": [root], "max_depth": 10},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
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


# =============================================================================
# WRITE / EDIT
# =============================================================================

async def execute_write_file(params: Dict[str, Any]) -> str:
    """Write a file. Host-only paths write directly; everything else via sandbox."""
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "Error: path is required."

    path = resolve_sandbox_path(path)

    # TRUNCATION GUARD
    _existing_size = 0
    try:
        _check_path = Path(path)
        if _check_path.is_file():
            _existing_size = _check_path.stat().st_size
    except Exception:
        pass
    _new_size = len(content)
    if _existing_size > 500 and _new_size < _existing_size * 0.5:
        _pct = int((_new_size / _existing_size) * 100)
        logger.warning(
            "[executors.filesystem] TRUNCATION GUARD: write_file would shrink %s "
            "from %d to %d chars (%d%%). Blocked.",
            path, _existing_size, _new_size, _pct,
        )
        return (
            f"BLOCKED - TRUNCATION GUARD: write_file would reduce {path} from "
            f"{_existing_size} to {_new_size} chars ({_pct}% of original). "
            f"This would destroy content. Use edit_file for targeted changes, "
            f"or read the COMPLETE file first before using write_file."
        )

    # Hard-block Windows system paths (no sandbox route exists)
    _HARD_BLOCKED = ["C:/Windows", "C:\\Windows", "C:/Program Files",
                     "C:\\Program Files", "C:/ProgramData", "C:\\ProgramData"]
    norm_check = path.replace("\\", "/").rstrip("/")
    for hb in _HARD_BLOCKED:
        hb_norm = hb.replace("\\", "/").rstrip("/")
        if norm_check == hb_norm or norm_check.startswith(hb_norm + "/"):
            logger.warning("[executors.filesystem] HARD BLOCKED system write: %s", path)
            return f"BLOCKED: Cannot write to {path} - system directory."

    if is_host_write_blocked(path):
        # Route through sandbox instead of blocking
        logger.info("[executors.filesystem] Protected path %s - routing to sandbox", path)
        sandbox_ok, sandbox_message = sandbox_health_status()
        if not sandbox_ok:
            return (
                f"Cannot write to {path} on host (protected). "
                f"Sandbox is also unavailable: {sandbox_message}\n"
                "Start the sandbox to modify ASTRA's own code."
            )
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{SANDBOX_CONTROLLER_URL}/fs/write",
                    json={"path": path, "content": content, "overwrite": True},
                )
                if resp.status_code == 200:
                    return (
                        f"Successfully wrote {len(content)} chars to {path} (SANDBOX).\n"
                        "This change is in the sandbox clone, not the live host. "
                        "Say 'promote' or use git to push changes to the host repo."
                    )
                return f"Sandbox write failed ({resp.status_code}): {resp.text}"
        except Exception as e:
            return f"Sandbox write error: {e}"

    # Host-only paths: write directly to host filesystem
    if is_host_only(path):
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            size_kb = len(content.encode("utf-8")) / 1024
            logger.info("[executors.filesystem] Host write: %s (%.1f KB)", path, size_kb)
            return f"Successfully wrote {len(content)} chars ({size_kb:.1f} KB) to {path} (host)"
        except Exception as e:
            return f"Host write error for {path}: {e}"

    # Sandbox paths
    sandbox_ok, sandbox_message = sandbox_health_status()
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

    resolved = resolve_sandbox_path(path)

    _HARD_BLOCKED_EDIT = ["C:/Windows", "C:\\Windows", "C:/Program Files",
                          "C:\\Program Files", "C:/ProgramData", "C:\\ProgramData"]
    norm_edit = resolved.replace("\\", "/").rstrip("/")
    for hb in _HARD_BLOCKED_EDIT:
        hb_norm = hb.replace("\\", "/").rstrip("/")
        if norm_edit == hb_norm or norm_edit.startswith(hb_norm + "/"):
            return f"BLOCKED: Cannot edit {resolved} - system directory."

    if is_host_write_blocked(resolved):
        # Route edit through sandbox
        logger.info("[executors.filesystem] Protected edit %s - routing to sandbox", resolved)
        current = await execute_read_file({"path": path})
        if current and current.startswith("Error"):
            return current
        if not current:
            return f"Cannot read {resolved} from sandbox for editing."
        if old_text not in current:
            return f"Error: old_text not found in {resolved} (sandbox). The text must exist exactly."
        count = current.count(old_text)
        if count > 1:
            return f"Error: old_text found {count} times in {resolved}. Must be unique."
        updated = current.replace(old_text, new_text, 1)
        return await execute_write_file({"path": resolved, "content": updated})

    # Host-only paths
    if is_host_only(resolved):
        current = read_host_file(resolved)
        if current is None:
            return f"File not found on host: {resolved}"
        if old_text not in current:
            return f"Error: old_text not found in {resolved}. The text to replace must exist exactly."
        count = current.count(old_text)
        if count > 1:
            return f"Error: old_text found {count} times in {resolved}. Must be unique."
        updated = current.replace(old_text, new_text, 1)
        return await execute_write_file({"path": resolved, "content": updated})

    # Sandbox paths
    sandbox_ok, sandbox_message = sandbox_health_status()
    if not sandbox_ok:
        return sandbox_message

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


# =============================================================================
# RUN COMMAND (HOST-SIDE, with extensive escape-hatch blocklists)
# =============================================================================

# Protected directory regex used by all command-body checks below.
_PROTECTED_DIR_PATTERN = r'd:[/\\](?:orb[/\\]|orb$|orb\.architecture|orb-desktop|orb-electron-data)'

_BLOCKED_COMMANDS = [
    "remove-item c:\\", "del c:\\", "rd c:\\", "rmdir c:\\",
    "format-volume", "clear-disk", "stop-computer", "restart-computer",
    "set-executionpolicy", "new-service", "remove-service",
    "reg delete", "reg add", "net user", "net localgroup",
]

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

_INTERPRETER_ESCAPES = [
    r'\bpython(?:\d|\.exe)?\s+-c\b',
    r'\bpython(?:\d|\.exe)?\s+-m\b(?!\s+pytest\b)',
    r'\bpy(?:\.exe)?\s+-c\b',
    r'\bnode(?:\.exe)?\s+-e\b',
    r'\bnode(?:\.exe)?\s+--eval\b',
    r'\bnode(?:\.exe)?\s+-p\b',
    r'\bnode(?:\.exe)?\s+--print\b',
    r'\bpwsh(?:\.exe)?\s+-c(?:ommand)?\b',
    r'\bpowershell(?:\.exe)?\s+-c(?:ommand)?\b',
    r'\bpwsh(?:\.exe)?\s+-encodedcommand\b',
    r'\bpowershell(?:\.exe)?\s+-encodedcommand\b',
    r'\bcmd(?:\.exe)?\s+/c\b',
    r'\bcmd(?:\.exe)?\s+/k\b',
    r'\bbash\s+-c\b',
    r'\bsh\s+-c\b',
    r'\bperl\s+-e\b',
    r'\bruby\s+-e\b',
    r'\bdotnet\s+script\b',
]

_WRITE_API_PATTERNS = [
    r'\.write_text\s*\(',
    r'\.write_bytes\s*\(',
    r'\bopen\s*\([^)]*[\'"]w',
    r'\bopen\s*\([^)]*[\'"]a',
    r'\[io\.file\]::writeall',
    r'\[system\.io\.file\]::writeall',
    r'\[io\.file\]::appendall',
    r'\bfs\.writefile\b',
    r'\bfs\.appendfile\b',
    r'\bshutil\.(?:copy|copyfile|copy2|move|rmtree)\b',
    r'\bos\.(?:remove|unlink|rmdir|removedirs)\b',
    r'\bpathlib\b',
]


async def execute_run_command(params: Dict[str, Any]) -> str:
    """Run a command on the host via PowerShell.

    Heavily restricted: blocks any command that could write to protected
    project directories (D:/Orb etc.), invoke an interpreter as an
    escape hatch (python -c, node -e), or call a file-writing API.
    All file modifications must go through edit_file/write_file so they
    route via the sandbox.
    """
    command = params.get("command", "")
    cwd = params.get("cwd", "D:\\Orb")
    timeout_sec = params.get("timeout_sec", 30)

    if not command:
        return "Error: command is required."

    cmd_lower = command.lower()

    for blocked in _BLOCKED_COMMANDS:
        if blocked in cmd_lower:
            logger.warning("[executors.filesystem] BLOCKED dangerous command: %s", command[:80])
            return f"BLOCKED: Command contains '{blocked}' - not allowed for safety."

    for pat in _WRITE_PATTERNS:
        if re.search(pat, cmd_lower):
            logger.warning("[executors.filesystem] BLOCKED write-via-command: %s", command[:80])
            return (
                "BLOCKED: This command would write to a protected project directory. "
                "Use the sandbox for all file modifications."
            )

    for pat in _INTERPRETER_ESCAPES:
        if re.search(pat, cmd_lower):
            logger.warning("[executors.filesystem] BLOCKED interpreter escape: %s", command[:120])
            return (
                "BLOCKED: Inline interpreter execution (python -c, node -e, pwsh -c, "
                "cmd /c, etc.) is not allowed. These can bypass file-write protections. "
                "Run the script as a file via the sandbox, or use edit_file for changes."
            )

    for pat in _WRITE_API_PATTERNS:
        if re.search(pat, cmd_lower):
            logger.warning("[executors.filesystem] BLOCKED write-API mention: %s", command[:120])
            return (
                "BLOCKED: Command references a file-write API (write_text, IO.File, "
                "fs.writeFile, shutil, etc.). File modifications must go through "
                "edit_file or write_file so they are routed via the sandbox."
            )

    if re.search(_PROTECTED_DIR_PATTERN, cmd_lower):
        logger.warning("[executors.filesystem] BLOCKED protected-dir reference in command: %s", command[:120])
        return (
            "BLOCKED: Command references a protected project directory "
            "(D:/Orb, D:/Orb.architecture, D:/orb-desktop, D:/orb-electron-data). "
            "Use cwd to run from there; the command body itself must not name these paths. "
            "All file modifications must go through edit_file/write_file (sandbox-routed)."
        )

    if re.search(r'\bgit\b', cmd_lower):
        logger.warning("[executors.filesystem] BLOCKED git command: %s", command[:80])
        return "BLOCKED: Git commands are not allowed. Taz handles all version control."

    try:
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
