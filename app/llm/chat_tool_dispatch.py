# FILE: app/llm/chat_tool_dispatch.py
# Purpose: Chat tool dispatcher — route a tool name+params to its executor.
# Called-by: app.llm.chat_tool_loop (re-export shim; the streaming loops call it)
# Depends-on: app.debug.action_executor, app.debug.gemini_finance_tools, app.debug.gemini_expense_tools, app.tools.chat_bridge, app.sandbox.tools (+more, all lazy)
# Last-renovated: 2026-06-21
"""
Chat tool dispatcher — split out of chat_tool_loop.py (SPLIT BATCH 9, 2026-06-21).

Single responsibility: execute_chat_tool(name, params) — the name->executor
routing table (sandbox boot tools, finance/expense adapters, central-registry
tools, then the debug action_executor) with result-size capping. Moved VERBATIM
from chat_tool_loop.py; that module re-exports it so all importers resolve
unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict

logger = logging.getLogger(__name__)


async def execute_chat_tool(name: str, params: Dict) -> str:
    """Execute a tool call and return the result string."""
    # spawn_agents: non-streaming fallback (the Debug OpenAI loop special-cases it
    # to stream live sub-agent activity; this path serves any other caller).
    if name == "spawn_agents":
        from app.debug.orchestrator.spawn_tool import run_spawn_agents_blocking
        return await run_spawn_agents_blocking(params or {})
    # Sandbox clone control (start/stop/status + read boot log). Sync + can block
    # (start waits for boot; read_sandbox_boot can poll), so run off the event loop.
    if name in ("start_sandbox_clone", "stop_sandbox_clone", "check_sandbox_status"):
        from app.sandbox.tools import execute_sandbox_tool
        return await asyncio.to_thread(execute_sandbox_tool, name, params or {})
    if name == "read_sandbox_boot":
        from app.debug.sandbox_boot_tool import read_sandbox_boot
        return await asyncio.to_thread(read_sandbox_boot, params or {})
    # inspect_sandbox_boot: full visual loop (backend readout + sandbox screenshot +
    # vision/OCR). Sync + blocks (polls boot, runs a sandbox screenshot, calls vision),
    # so run it off the event loop just like read_sandbox_boot.
    if name == "inspect_sandbox_boot":
        from app.debug.sandbox_boot_tool import inspect_sandbox_boot
        return await asyncio.to_thread(inspect_sandbox_boot, params or {})
    # selfheal_sandbox_boot: autonomous fix->reboot->re-verify loop; sync + long-blocking.
    if name == "selfheal_sandbox_boot":
        from app.debug.sandbox_selfheal import selfheal_sandbox_boot
        return await asyncio.to_thread(selfheal_sandbox_boot, params or {})
    # Work-day ledger tools run via their own adapters (central-registry
    # handlers), not the debug action_executor - same path as the image loop.
    try:
        from app.debug.gemini_finance_tools import FINANCE_TOOL_EXECUTORS
        if name in FINANCE_TOOL_EXECUTORS:
            return await FINANCE_TOOL_EXECUTORS[name](params or {})
    except Exception as exc:
        logger.error("[chat_tool_loop] finance tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"
    try:
        from app.debug.gemini_expense_tools import EXPENSE_TOOL_EXECUTORS
        if name in EXPENSE_TOOL_EXECUTORS:
            return await EXPENSE_TOOL_EXECUTORS[name](params or {})
    except Exception as exc:
        logger.error("[chat_tool_loop] expense tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"
    # Central-registry tools (documents/editor, podcasts, email, streams,
    # movies, displays, ...). action_executor keeps precedence for every
    # name it owns; only names it does NOT know are tried against the
    # registry, so read_file/web_search/etc. behave exactly as before.
    try:
        from app.debug.action_executor import TOOL_HANDLERS as _AE_HANDLERS
    except Exception:
        _AE_HANDLERS = {}
    if name not in _AE_HANDLERS:
        try:
            from app.tools.chat_bridge import (
                is_registry_chat_tool, execute_registry_chat_tool,
            )
            if is_registry_chat_tool(name):
                return await execute_registry_chat_tool(name, params or {})
        except Exception as exc:
            logger.error("[chat_tool_loop] registry tool %s failed: %s", name, exc)
            return f"Tool error: {exc}"
    from app.debug.action_executor import execute_tool
    try:
        result = await execute_tool(name, params)
        cap = 128000 if name == "read_file" else 30000
        if len(result) > cap:
            result = result[:cap] + f"\n\n... [TRUNCATED — {len(result)} chars total]"
        return result
    except Exception as exc:
        logger.error("[chat_tool_loop] Tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"
