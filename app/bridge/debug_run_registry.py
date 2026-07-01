# FILE: app/bridge/debug_run_registry.py
# Purpose: In-memory registry of live phone-initiated debug runs, keyed by run_id.
#          Decouples a debug run's asyncio lifetime from the HTTP request/response
#          that started it (§2.2 of the live-session plan) so a dropped phone
#          connection does not kill the run, and so a separate cancel call has
#          something to reach. Also the single place that arms the run_context
#          contextvars (cancel_event + phone-initiated flag) for a background run.
# Called-by: app.bridge.router (POST /bridge/debug-and-speak, POST /bridge/debug/cancel)
# Depends-on: app.debug.run_context, asyncio
# Last-renovated: 2026-07-01
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Dict, Optional

from app.debug.run_context import CANCEL_EVENT, PHONE_INITIATED

logger = logging.getLogger(__name__)


@dataclass
class RunState:
    # int, not a uuid string: tts_cache.py / tts_audio.py hard-cast their cache
    # key to int(message_id) (they were built for real DB message ids), and this
    # run_id doubles as that key so /bridge/debug-and-speak can reuse both
    # unmodified. See new_run_id() for the collision-avoidance scheme.
    run_id: int
    project_id: int
    task: Optional[asyncio.Task] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    status: str = "running"  # running | done | cancelled | error
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    # Optional reference to the active TtsCacheWriter, kept for observability only
    # (the growing-file transport reads tts_cache from disk, not via this object).
    cache_writer: Optional[object] = None


_RUNS: Dict[int, RunState] = {}
_MAX_RUNS = 200  # unbounded-growth guard; pruned lazily on new-run start


def new_run_id() -> int:
    """Millisecond timestamp + 3 random digits: a big int, naturally increasing,
    and astronomically unlikely to collide with a real (small, autoincrement)
    Message.id -- so it's safe to share the tts_cache/tts_audio keyspace."""
    return int(time.time() * 1000) * 1000 + random.randint(0, 999)


def reserve(project_id: int) -> RunState:
    """Create and register a RunState with a run_id already assigned, but no
    task yet. Split from launch() so a caller whose coro_factory needs to know
    the run_id up front (e.g. to open a same-keyed TtsCacheWriter before the
    task starts) can build that closure with a real id instead of a promise."""
    _prune()
    run_id = new_run_id()
    state = RunState(run_id=run_id, project_id=project_id)
    _RUNS[run_id] = state
    return state


def launch(state: RunState, coro_factory: Callable[[], Awaitable[None]]) -> None:
    """Arm CANCEL_EVENT + PHONE_INITIATED and start coro_factory() as a
    background asyncio.Task detached from the caller's request lifetime. Both
    contextvars are set BEFORE the task is created, so every await inside
    coro_factory() (including nested asyncio.create_task calls, e.g. the
    spawn_agents fan-out) inherits both signals — no parameter threading
    needed through the intervening call chain."""
    CANCEL_EVENT.set(state.cancel_event)
    PHONE_INITIATED.set(True)

    async def _wrapper() -> None:
        try:
            await coro_factory()
            if state.status == "running":
                state.status = "done"
        except asyncio.CancelledError:
            state.status = "cancelled"
            raise
        except Exception as e:  # never let a debug run crash silently out of the registry
            logger.exception("[debug_run_registry] run=%s crashed: %s", state.run_id, e)
            state.status = "error"
            state.error = f"{type(e).__name__}: {e}"

    state.task = asyncio.create_task(_wrapper())


def start_background_run(
    project_id: int,
    coro_factory: Callable[[], Awaitable[None]],
) -> RunState:
    """Convenience one-shot for callers that don't need the run_id before
    launch: reserve() + launch() together."""
    state = reserve(project_id)
    launch(state, coro_factory)
    return state


def get(run_id: int) -> Optional[RunState]:
    return _RUNS.get(run_id)


def get_active_for_project(project_id: int) -> Optional[RunState]:
    for state in _RUNS.values():
        if state.project_id == project_id and state.status == "running":
            return state
    return None


def cancel(run_id: int) -> bool:
    """Idempotent — safe to call repeatedly (the phone retries a cancel until it
    gets an ack, per the plan's flaky-link requirement). Sets the cooperative
    cancel_event AND hard-cancels the asyncio task; cooperative checkpoints in
    loop_controller/spawn_tool make it stop promptly, task.cancel() is the backstop."""
    state = _RUNS.get(run_id)
    if not state:
        return False
    state.cancel_event.set()
    if state.task and not state.task.done():
        state.task.cancel()
    if state.status == "running":
        state.status = "cancelled"
    return True


def cancel_for_project(project_id: int) -> Optional[int]:
    state = get_active_for_project(project_id)
    if not state:
        return None
    cancel(state.run_id)
    return state.run_id


def _prune(max_age_s: float = 3600.0) -> None:
    if len(_RUNS) < _MAX_RUNS:
        return
    now = time.time()
    stale = [
        rid for rid, s in _RUNS.items()
        if s.status != "running" and (now - s.created_at) > max_age_s
    ]
    for rid in stale:
        _RUNS.pop(rid, None)
    if stale:
        logger.info("[debug_run_registry] pruned %d stale run(s)", len(stale))
