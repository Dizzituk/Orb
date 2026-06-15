# FILE: app/documents/editor_actions.py
# Purpose: In-memory action queue between backend tools and the open Univer
#          editor pane (enqueue+await / long-poll deliver / post result).
# Called-by: app.documents.router, app.documents.editor_client
# Depends-on: stdlib asyncio
# Last-renovated: 2026-06-12
"""
Editor action queue.

Same shape as web_automation's action loop, scoped to ONE editor session
and kept in memory (a backend restart simply drops pending commands —
they're sub-second interactive actions, not jobs).

  tools  -> execute("sheet_set_range", {...})        (enqueue + await)
  pane   -> GET  /documents/editor/pending-action    (long-poll, pops one)
  pane   -> POST /documents/editor/action-result/{id}

The pane also posts open/close status here so tools can answer "is
anything open?" without a round trip.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Optional

_queue: deque = deque()
_arrival = asyncio.Event()
_futures: dict[str, asyncio.Future] = {}

# Last status the editor pane reported: {open, path, kind, name, at}.
_editor_state: dict = {"open": False, "path": None, "kind": None,
                       "name": None, "at": 0.0}


def editor_state() -> dict:
    return dict(_editor_state)


def set_editor_state(open_: bool, path: Optional[str] = None,
                     kind: Optional[str] = None, name: Optional[str] = None) -> dict:
    _editor_state.update(open=open_, path=path, kind=kind, name=name,
                         at=time.time())
    return editor_state()


async def execute(action_type: str, payload: Optional[dict] = None,
                  timeout_seconds: float = 10.0) -> dict:
    """Enqueue an action for the pane and await its result."""
    if not _editor_state["open"]:
        return {"ok": False, "error": "no document is open in the editor pane"}
    action_id = uuid.uuid4().hex
    future: asyncio.Future = asyncio.get_running_loop().create_future()
    _futures[action_id] = future
    _queue.append({"action_id": action_id, "action_type": action_type,
                   "payload": payload or {}})
    _arrival.set()
    try:
        return await asyncio.wait_for(future, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"editor didn't answer '{action_type}' in "
                                      f"{timeout_seconds:.0f}s — is the pane open?"}
    finally:
        _futures.pop(action_id, None)


async def next_action(wait_seconds: float = 25.0) -> Optional[dict]:
    """Long-poll pop for the pane. None when nothing arrived in the window."""
    deadline = time.monotonic() + max(0.0, min(wait_seconds, 60.0))
    while True:
        try:
            return _queue.popleft()
        except IndexError:
            pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        _arrival.clear()
        try:
            await asyncio.wait_for(_arrival.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            return None


def post_result(action_id: str, body: dict) -> bool:
    """Resolve the awaiting execute() call. False when it already timed out."""
    future = _futures.get(action_id)
    if future is None or future.done():
        return False
    future.set_result({
        "ok": bool(body.get("ok")),
        "result": body.get("result"),
        "error": body.get("error"),
    })
    return True
