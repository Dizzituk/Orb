# FILE: app/debug/gemini_file_tools.py
# Purpose: Direct-host file/shell tools for the Gemini tool loop (write-safety, declarations, executors).
# Called-by: app.debug.gemini_tool_loop
# Depends-on: (stdlib only)
# Last-renovated: 2026-06-20
"""
Direct-host file/shell tools for the Gemini native function-calling loop.

Extracted verbatim from gemini_tool_loop.py on 2026-06-20 (split campaign,
batch 2). Holds the filesystem write-safety guard, the six file-tool
JSON-Schema declarations, and their async executors. Logic is byte-identical
to the pre-split module; gemini_tool_loop.py imports these names back so its
TOOL_DECLARATIONS / _TOOL_EXECUTORS composition is unchanged.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


ALLOWED_WRITE_ROOTS = [
    Path("D:/Orb"),
    Path("D:/orb-desktop"),
    Path("D:/Astra Android Folder"),
    Path("C:/Users/dizzi/Documents"),
    Path("C:/Users/dizzi/OneDrive/Documents"),
    Path("C:/Users/dizzi/OneDrive/Pictures"),
]


def _is_path_allowed(path_str: str) -> bool:
    """Check if a path falls within allowed write roots."""
    try:
        target = Path(path_str).resolve()
        for root in ALLOWED_WRITE_ROOTS:
            try:
                if target == root.resolve() or root.resolve() in target.resolve().parents:
                    return True
            except (OSError, ValueError):
                continue
        return False
    except Exception:
        return False


_FILE_TOOL_DECLARATIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file from the ASTRA codebase on the host. "
            "Use absolute paths starting with D:/Orb/ for backend or D:/orb-desktop/ for frontend. "
            "Returns the file contents as text. Host is READ ONLY."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path, e.g. D:/Orb/app/debug/debug_chat.py",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file on the filesystem. Use absolute paths "
            "(e.g. D:/Orb/..., D:/Astra Android Folder/..., C:/Users/dizzi/...). "
            "Can write any file type: .py, .kt, .tsx, .txt, .html, .md, .json, etc. "
            "Parent directories are created automatically if they don't exist. "
            "Do NOT use this to log nutrition or workouts — use log_nutrition / "
            "log_workout for those, and start_work_day / finish_work_day to log a "
            "delivery work day or shift (the 'work tab') - all of those write to "
            "the database, not a file. "
            "Use write_file for documents, dashboards, reports, code, and other "
            "non-domain files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path",
                },
                "content": {
                    "type": "string",
                    "description": "The full file content to write",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Apply targeted find-and-replace edits to a file. The old_text must "
            "match exactly and appear only once in the file. Use absolute paths. "
            "NEVER use edit_file to log a delivery work day or shift by editing an "
            "HTML or dashboard file - use start_work_day / finish_work_day, which "
            "write to the work-day database."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute file path to edit",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find (must be unique in the file)",
                },
                "new_text": {
                    "type": "string",
                    "description": "Text to replace it with",
                },
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
    {
        "name": "run_shell",
        "description": (
            "Run a PowerShell command on the host machine. This is Windows with PowerShell. "
            "NEVER use Linux commands (grep, ls, cat, sed, awk, bash). "
            "Use PowerShell equivalents: Select-String (not grep), Get-ChildItem (not ls), "
            "Get-Content (not cat), python (not python3). "
            "Returns stdout, stderr, and exit code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The PowerShell command to execute",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for files matching a pattern in the ASTRA host codebase. "
            "Returns a list of matching file paths. Use glob patterns like "
            "'*.py' or '**/*router*.py'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Base directory to search in, e.g. D:/Orb/app",
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. **/*debug*.py",
                },
            },
            "required": ["directory", "pattern"],
        },
    },
    {
        "name": "list_dir",
        "description": (
            "List files and directories at a path on the host codebase. "
            "Returns names with [FILE] or [DIR] prefixes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute directory path, e.g. D:/Orb/app/debug",
                },
            },
            "required": ["path"],
        },
    },
]


async def _exec_read_file(args: dict) -> str:
    path = args.get("path", "")
    try:
        p = Path(path)
        if not p.exists():
            return f"ERROR: File not found: {path}"
        if not p.is_file():
            return f"ERROR: Not a file: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > 15000:
            return content[:15000] + f"\n\n... (truncated, {len(content)} total chars)"
        return content
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_write_file(args: dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    if not _is_path_allowed(path):
        return f"ERROR: Write blocked — {path} is outside allowed directories. Allowed roots: {', '.join(str(r) for r in ALLOWED_WRITE_ROOTS)}"
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        logger.info("[gemini_tool_loop] write_file: %s (%d chars)", path, len(content))
        return f"OK: Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_edit_file(args: dict) -> str:
    path = args.get("path", "")
    old_text = args.get("old_text", "")
    new_text = args.get("new_text", "")
    if not _is_path_allowed(path):
        return f"ERROR: Edit blocked — {path} is outside allowed directories."
    try:
        p = Path(path)
        if not p.exists():
            return f"ERROR: File not found: {path}"
        content = p.read_text(encoding="utf-8", errors="replace")
        if old_text not in content:
            return f"ERROR: old_text not found in {path}. Check exact match."
        if content.count(old_text) > 1:
            return f"ERROR: old_text appears {content.count(old_text)} times in {path}. Must be unique."
        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        logger.info("[gemini_tool_loop] edit_file: %s (replaced %d chars with %d chars)", path, len(old_text), len(new_text))
        return f"OK: Edited {path} (replaced {len(old_text)} chars with {len(new_text)} chars)"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_run_shell(args: dict) -> str:
    cmd = args.get("command", "")
    cmd_lower = cmd.lower()
    _BLOCKED_PATTERNS = [
        "remove-item c:\\", "remove-item 'c:", 'remove-item "c:',
        "del c:\\", "rd c:\\", "rmdir c:\\",
        "format-volume", "clear-disk",
        "stop-computer", "restart-computer",
        "set-executionpolicy",
        "new-service", "remove-service",
        "reg delete", "reg add",
        "net user", "net localgroup",
    ]
    for blocked in _BLOCKED_PATTERNS:
        if blocked in cmd_lower:
            return f"ERROR: Command blocked for safety — contains '{blocked}'"
    try:
        import asyncio
        proc = await asyncio.create_subprocess_shell(
            f'powershell.exe -NoProfile -Command "{cmd}"',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        parts = []
        if stdout:
            parts.append(stdout.decode("utf-8", errors="replace")[:3000])
        if stderr:
            parts.append(f"STDERR: {stderr.decode('utf-8', errors='replace')[:1000]}")
        parts.append(f"exit_code={proc.returncode}")
        return "\n".join(parts)
    except asyncio.TimeoutError:
        return "ERROR: Command timed out after 30 seconds"
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_search_files(args: dict) -> str:
    directory = args.get("directory", "D:/Orb")
    pattern = args.get("pattern", "*.py")
    try:
        p = Path(directory)
        if not p.exists():
            return f"ERROR: Directory not found: {directory}"
        matches = sorted(str(m) for m in p.glob(pattern))[:30]
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        return "\n".join(matches)
    except Exception as e:
        return f"ERROR: {e}"


async def _exec_list_dir(args: dict) -> str:
    path = args.get("path", "")
    try:
        p = Path(path)
        if not p.exists():
            return f"ERROR: Not found: {path}"
        if not p.is_dir():
            return f"ERROR: Not a directory: {path}"
        entries = []
        for item in sorted(p.iterdir()):
            prefix = "[DIR]" if item.is_dir() else "[FILE]"
            entries.append(f"{prefix} {item.name}")
        return "\n".join(entries[:50]) if entries else "(empty directory)"
    except Exception as e:
        return f"ERROR: {e}"
