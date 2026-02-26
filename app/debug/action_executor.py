# FILE: app/debug/action_executor.py
"""
Action Executor: Translates LLM tool calls into real operations.

Phase 1: Read-only operations (local filesystem + sandbox controller).
Phase 2: Write operations via sandbox bridge.

All operations go through the existing sandbox controller where applicable.
Host filesystem access is strictly read-only and limited to scan output files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# HOST ACCESS CONTROL
# =============================================================================

# Files/directories the debug assistant can read on the host
HOST_READABLE_PATHS = {
    Path("D:/Orb/.architecture"),
    Path("D:/Orb/logs"),
    Path("D:/Orb/config"),
    Path("D:/Orb/app"),
    Path("D:/Orb/main.py"),
    Path("D:/Orb/requirements.txt"),
    Path("D:/Orb/TECH_DEBT.md"),
}

# Sandbox controller base URL
SANDBOX_CONTROLLER_URL = "http://192.168.250.2:8765"


def _is_host_readable(path: str) -> bool:
    """Check if a path is in the host-readable allow list."""
    p = Path(path).resolve()
    for allowed in HOST_READABLE_PATHS:
        try:
            p.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue
    return False


# =============================================================================
# TOOL EXECUTORS
# =============================================================================

async def execute_read_file(params: Dict[str, Any]) -> str:
    """Read a file from the host filesystem or sandbox."""
    path = params.get("path", "")
    head = params.get("head")
    tail = params.get("tail")

    if not path:
        return "Error: path is required."

    # Try host first (for development codebase access)
    p = Path(path)
    if p.exists() and _is_host_readable(path):
        try:
            content = p.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            if head:
                lines = lines[:head]
            elif tail:
                lines = lines[-tail:]
            return "\n".join(lines)
        except Exception as e:
            return f"Error reading file: {e}"

    # Try via sandbox controller
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/contents",
                json={"paths": [path], "max_file_size": 500000, "include_line_numbers": True},
            )
            if resp.status_code == 200:
                data = resp.json()
                files = data.get("files", [])
                if files:
                    content = files[0].get("content", "")
                    if files[0].get("error"):
                        return f"Error: {files[0]['error']}"
                    if head:
                        content = "\n".join(content.splitlines()[:head])
                    elif tail:
                        content = "\n".join(content.splitlines()[-tail:])
                    return content
                return f"File not found: {path}"
            return f"Sandbox error ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Both host and sandbox read failed for {path}: {e}"


async def execute_list_files(params: Dict[str, Any]) -> str:
    """List directory contents."""
    path = params.get("path", "")
    if not path:
        return "Error: path is required."

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

    return f"Directory not found or not accessible: {path}"


async def execute_read_pipeline_state(params: Dict[str, Any]) -> str:
    """Get current pipeline state."""
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
    """Read filtered log entries."""
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


async def execute_search_files(params: Dict[str, Any]) -> str:
    """Search for files matching a pattern."""
    root = params.get("path", "D:/Orb")
    pattern = params.get("pattern", "**/*")

    try:
        p = Path(root)
        if not p.exists():
            return f"Directory not found: {root}"

        matches = list(p.glob(pattern))
        # Filter out common noise
        skip_dirs = {".git", "node_modules", ".venv", "__pycache__", "dist", "build"}
        filtered = [
            m for m in matches
            if not any(sd in m.parts for sd in skip_dirs)
        ]

        if not filtered:
            return f"No files matching '{pattern}' in {root}"

        result_lines = [str(m) for m in filtered[:100]]
        suffix = f"\n... ({len(filtered)} total)" if len(filtered) > 100 else ""
        return "\n".join(result_lines) + suffix
    except Exception as e:
        return f"Search error: {e}"


# =============================================================================
# PHASE 2: WRITE TOOLS (sandbox-only)
# =============================================================================

async def execute_write_file(params: Dict[str, Any]) -> str:
    """Write a file via the sandbox controller."""
    path = params.get("path", "")
    content = params.get("content", "")
    if not path:
        return "Error: path is required."

    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/fs/write",
                json={"path": path, "content": content, "overwrite": True},
            )
            if resp.status_code == 200:
                return f"Successfully wrote {len(content)} chars to {path}"
            return f"Write failed ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Write error: {e}"


async def execute_edit_file(params: Dict[str, Any]) -> str:
    """Edit a file via read-modify-write through sandbox controller."""
    path = params.get("path", "")
    old_text = params.get("old_text", "")
    new_text = params.get("new_text", "")

    if not path or not old_text:
        return "Error: path and old_text are required."

    # Read current content
    current = await execute_read_file({"path": path})
    if current.startswith("Error"):
        return current

    if old_text not in current:
        return f"Error: old_text not found in {path}. The text to replace must exist exactly."

    count = current.count(old_text)
    if count > 1:
        return f"Error: old_text found {count} times in {path}. Must be unique."

    updated = current.replace(old_text, new_text, 1)
    return await execute_write_file({"path": path, "content": updated})


async def execute_run_command(params: Dict[str, Any]) -> str:
    """Run a command via the sandbox controller shell endpoint."""
    command = params.get("command", "")
    cwd = params.get("cwd", "D:\\Orb")
    timeout_sec = params.get("timeout_sec", 30)

    if not command:
        return "Error: command is required."

    try:
        import httpx
        async with httpx.AsyncClient(timeout=float(timeout_sec + 5)) as client:
            resp = await client.post(
                f"{SANDBOX_CONTROLLER_URL}/shell/run",
                json={
                    "cmd": ["powershell", "-Command", command],
                    "cwd": cwd,
                    "timeout_sec": timeout_sec,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                stdout = data.get("stdout", "")
                stderr = data.get("stderr", "")
                rc = data.get("returncode", -1)
                result = f"Exit code: {rc}\n"
                if stdout:
                    result += f"\nSTDOUT:\n{stdout}"
                if stderr:
                    result += f"\nSTDERR:\n{stderr}"
                return result
            return f"Command failed ({resp.status_code}): {resp.text}"
    except Exception as e:
        return f"Command execution error: {e}"


# =============================================================================
# DISPATCHER
# =============================================================================

TOOL_HANDLERS = {
    # Phase 1: read-only
    "read_file":           execute_read_file,
    "list_files":          execute_list_files,
    "read_pipeline_state": execute_read_pipeline_state,
    "read_logs":           execute_read_logs,
    "search_files":        execute_search_files,
    # Phase 2: write access
    "write_file":          execute_write_file,
    "edit_file":           execute_edit_file,
    "run_command":         execute_run_command,
}


async def execute_tool(tool_name: str, params: Dict[str, Any]) -> str:
    """
    Execute a tool call from the LLM.

    Args:
        tool_name: Name of the tool to execute.
        params: Tool parameters from the LLM.

    Returns:
        String result to feed back to the LLM.
    """
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
