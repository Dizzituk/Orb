# FILE: app/tools/chat_bridge.py
# Purpose: Expose central-registry tools (app.tools.registry) to the chat
#          tool loop — declarations in Anthropic chat format + an execution
#          adapter that serialises ToolResponse for the LLM.
# Called-by: app.llm.chat_tool_loop, app.debug.tool_definitions
# Depends-on: app.tools.registry
# Last-renovated: 2026-06-12
"""
Registry -> chat bridge (v1, 2026-06-12).

WHY THIS EXISTS: ASTRA has two tool worlds. The chat tool loop
(app.llm.chat_tool_loop) builds its tool list from app.debug.tool_definitions
plus finance/expense declarations, and executes via app.debug.action_executor.
The central registry (app.tools.registry) holds the Session-7 families —
documents/editor, podcasts, email, streams, movie night, displays — plus
lifestyle, finance, sentinel, vehicle. Deterministic command dispatch reads
the registry; the chat LLM never did. Net effect: with a document open in
the Univer pane, chat answered "upload it here" because it had no editor
tools and no idea the pane existed.

This bridge closes that gap at the chokepoint:

  get_registry_chat_declarations()  registry ToolDefinitions as chat decls
  is_registry_chat_tool()           membership check for dispatch
  execute_registry_chat_tool()      run via execute_tool_async, JSON out

PRECEDENCE RULE: action_executor keeps every name it owns (read_file,
web_search, write_user_file, ...). The chat loop only routes here for names
action_executor does NOT know. _CHAT_EXCLUDED additionally hard-blocks
registry tools that duplicate chat-world tools, so they are never declared
twice even if dispatch order changes.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Registry tools that must never surface in chat through this bridge —
# the chat world already has its own versions with richer behaviour.
_CHAT_EXCLUDED: Set[str] = {"http_fetch", "web_search", "weather"}

# Editor-pane tools every chat model needs in front of it (the user talks
# to whatever model sits next to the open document; see get_universal_tools
# in app.debug.tool_definitions for the universality precedent).
EDITOR_TOOL_NAMES: Set[str] = {
    "editor_status", "read_document", "edit_document_text",
    "read_sheet_range", "set_sheet_range", "list_sheet_names",
    "save_document",
}


def get_registry_chat_declarations(
    exclude_names: Optional[Set[str]] = None,
    only_names: Optional[Set[str]] = None,
) -> List[Dict]:
    """Central-registry tools as Anthropic-format chat declarations.

    exclude_names: names already declared by the caller (skip duplicates).
    only_names:    restrict to a subset (e.g. EDITOR_TOOL_NAMES for the
                   universal tier).
    """
    from app.tools.registry import list_tool_definitions

    skip = _CHAT_EXCLUDED | (exclude_names or set())
    out: List[Dict] = []
    for td in list_tool_definitions():
        if td.name in skip:
            continue
        if only_names is not None and td.name not in only_names:
            continue
        out.append({
            "name": td.name,
            "description": td.description,
            "input_schema": td.input_schema or {"type": "object", "properties": {}},
        })
    return out


def is_registry_chat_tool(name: str) -> bool:
    """True when the central registry owns this tool AND chat may use it."""
    if not name or name in _CHAT_EXCLUDED:
        return False
    try:
        from app.tools.registry import get_tool
        get_tool(name, "v1")
        return True
    except KeyError:
        return False
    except Exception as exc:  # pragma: no cover — registry import failure
        logger.warning("[chat_bridge] registry lookup failed for %s: %s", name, exc)
        return False


async def execute_registry_chat_tool(name: str, params: Optional[Dict]) -> str:
    """Run a registry tool and serialise the ToolResponse for the chat loop.

    The chat loop expects a plain string result; tools downstream return
    the executor-style {ok, result?, error?}, so JSON keeps it parseable.
    """
    from app.tools.registry import execute_tool_async

    resp = await execute_tool_async(name, "v1", params or {})
    payload: Dict = {"ok": bool(getattr(resp, "ok", False))}
    result = getattr(resp, "result", None)
    if result is not None:
        payload["result"] = result
    err = getattr(resp, "error_message", "") or ""
    if err:
        payload["error"] = err
    try:
        return json.dumps(payload, default=str)
    except Exception:
        return str(payload)
