# FILE: app/debug/executors/emulator.py
"""
Thin wrappers around external systems:
  - ADB / Android emulator (screenshot, tap, type, key, build, install)
  - Gradle (build, install)
  - Pipeline state (read-only inspection of in-memory flow)
  - Log file reading

These are all single-call delegations to other modules. If any of them
grow beyond ~20 lines of logic, factor them out into their own module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


# =============================================================================
# ADB / EMULATOR
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
# PIPELINE STATE / LOGS
# =============================================================================

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
