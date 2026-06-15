# FILE: app/documents/editor_client.py
# Purpose: Python wrapper over the editor action queue so intent handlers /
#          the agentic pipeline can read and edit the OPEN document.
# Called-by: app.tools.document_tools
# Depends-on: app.documents.editor_actions
# Last-renovated: 2026-06-12
"""
Editor client.

Thin async wrappers — the pane's univerBridge.ts implements the other end
of each action through Univer's Facade API. Everything returns the
executor-style {ok, result?, error?} dict and never raises.
"""
from __future__ import annotations

from typing import List, Optional

from app.documents import editor_actions


def state() -> dict:
    """What the editor pane currently has open (no round trip)."""
    return editor_actions.editor_state()


async def doc_get_text() -> dict:
    return await editor_actions.execute("doc_get_text", {})


async def doc_insert_text(text: str, position: str = "end") -> dict:
    """position: 'end' | 'start' | integer character offset (as string)."""
    return await editor_actions.execute(
        "doc_insert_text", {"text": text, "position": position})


async def doc_replace_range(start: int, end: int, text: str) -> dict:
    return await editor_actions.execute(
        "doc_replace_range", {"start": start, "end": end, "text": text})


async def sheet_get_range(a1: str) -> dict:
    return await editor_actions.execute("sheet_get_range", {"a1": a1})


async def sheet_set_range(a1: str, values: List[List]) -> dict:
    return await editor_actions.execute(
        "sheet_set_range", {"a1": a1, "values": values})


async def sheet_get_names() -> dict:
    return await editor_actions.execute("sheet_get_names", {})


async def save(path: Optional[str] = None) -> dict:
    """Ask the pane to run its normal save path (snapshot -> backend -> disk)."""
    return await editor_actions.execute("editor_save", {"path": path},
                                        timeout_seconds=20.0)
