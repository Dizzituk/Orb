# FILE: app/debug/executors/filesystem_run_command.py
# Purpose: Host-side run_command executor + its escape-hatch blocklists (split from filesystem.py).
# Called-by: app.debug.executors.filesystem (re-export), app.debug.executors
# Depends-on: app.debug.executors._paths
# Last-renovated: 2026-06-20
"""
Host command execution (PowerShell) with extensive escape-hatch blocklists.

Extracted verbatim from filesystem.py on 2026-06-20 (split campaign, batch 2),
per that module's own "if approaching 30 KB, split run_command + its blocklists
into filesystem_run_command.py" guidance. Logic is byte-identical to the
pre-split module; filesystem.py re-exports execute_run_command so __init__,
host_console, and tests resolve it unchanged.

run_command runs ON THE HOST via PowerShell, with extensive blocklists to
prevent file-write escape hatches that would bypass sandbox routing.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict

from app.debug.executors._paths import is_host_write_blocked

logger = logging.getLogger(__name__)


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


_HOST_BOOT_MUTATE_PATTERN = re.compile(
    r"npm\s+(?:run|start|ci|install|exec)\b"
    r"|\bnpx\b"
    r"|(?:yarn|pnpm)\s+(?:run|start|dev|install|add)\b"
    r"|\belectron\b"
    r"|\buvicorn\b|python\s+-m\s+uvicorn\b|\bgunicorn\b|\bflask\s+run\b"
    r"|python\s+(?:-u\s+)?[\w./\\-]*(?:main|app|server|run)\.py\b"
    r"|node\s+[\w./\\-]*(?:server|main|index|app)(?:\.js)?\b"
    r"|dotnet\s+run\b|\bgradlew?\b"
    r"|\bstart-process\b|\bstart\s+[\"']"
    r"|&\s*[\"']?[\w:./\\ -]*\.(?:ps1|bat|cmd)\b|\.[/\\][\w./\\-]*\.(?:ps1|bat|cmd)\b"
    r"|\bpip\s+install\b"
    r"|\bset-content\b|\bout-file\b|\badd-content\b|\bnew-item\b"
    r"|\bremove-item\b|\bcopy-item\b|\bmove-item\b|\brmdir\b|\bdel\s",
    re.IGNORECASE,
)


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
            "Do NOT try to run from there via cwd either -- booting or mutating ASTRA's own "
            "code on the host is refused. Boot/run via the sandbox clone; change code via "
            "edit_file/write_file (auto-routed to the sandbox). Read-only diagnostics are allowed."
        )

    if re.search(r'\bgit\b', cmd_lower):
        logger.warning("[executors.filesystem] BLOCKED git command: %s", command[:80])
        return "BLOCKED: Git commands are not allowed. Taz handles all version control."

    # Host boot/mutation gate (2026-06-17): the blocklists above inspect only the
    # command BODY; the model can dodge them by putting a protected repo in `cwd`
    # (the incident: `npm run electron:dev` cwd=D:/orb-desktop launched a 2nd Electron
    # + backend ON THE HOST). Block boots/launches/mutations whose cwd is a protected
    # self-repo. Read-only diagnostics still run. NEVER falls back to host -- ASTRA's
    # own code is booted/changed only in the sandbox clone.
    _cwd_for_gate = str(cwd or "D:/Orb").replace("\\", "/")
    if is_host_write_blocked(_cwd_for_gate) and _HOST_BOOT_MUTATE_PATTERN.search(cmd_lower):
        logger.warning(
            "[executors.filesystem] BLOCKED host boot/mutate in protected cwd=%s: %s",
            _cwd_for_gate, command[:120],
        )
        return (
            "BLOCKED: refusing to launch/boot or modify ASTRA's own code on the HOST. "
            f"cwd '{_cwd_for_gate}' is a protected repo (D:/Orb, D:/orb-desktop), and launching "
            "there would spawn a second live instance / corrupt the running system. To BOOT or "
            "RUN ASTRA, boot the sandbox clone -- never the host. To CHANGE code, use "
            "edit_file/write_file (auto-routed to the sandbox). Read-only diagnostics "
            "(ripgrep, compile checks, listing) are fine here."
        )

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
