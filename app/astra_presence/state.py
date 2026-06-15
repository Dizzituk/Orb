# FILE: app/astra_presence/state.py
# Purpose: In-memory ASTRA presence state (one of 8 orb states) + asyncio subscriber
#          registry so the Room orb and any surface can reflect ASTRA's live state.
# Called-by: app.astra_presence.router, app.scene_director.voice, tests
# Depends-on: stdlib only
# Last-renovated: 2026-06-13
"""ASTRA presence state (v2 — deliberately no DB).

Mirrors the scene_director.state subscriber-registry pattern: ONE current
value + a set of asyncio.Queue subscribers; set_state() fans the new value out
to every queue (put_nowait; a stalled queue is dropped, never blocks). Single
FastAPI event loop, so no locking. State resets to idle on backend restart.

The eight states are the orb's emotional/processing states (see the Room
OrbController). 'deep_research' is accepted as an alias of 'deep research'.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)

# Canonical orb states (match the Room OrbController STATES keys exactly).
VALID_STATES = (
    "idle", "listening", "thinking", "speaking",
    "deep research", "message", "error", "wake",
)
_MAX_QUEUE = 8  # per-subscriber buffer; a stuck client is dropped, never blocks


def normalise_state(name: str) -> Optional[str]:
    """Return the canonical state string, or None if unknown.
    Accepts 'deep_research' as an alias for 'deep research'."""
    if not name:
        return None
    s = str(name).strip().lower().replace("_", " ")
    return s if s in VALID_STATES else None


class PresenceState:
    """Holder for ASTRA's single current presence state + subscriber registry."""

    def __init__(self) -> None:
        self._state = "idle"
        self._version = 0
        self._subscribers: Set[asyncio.Queue] = set()

    def get_state(self) -> str:
        return self._state

    @property
    def version(self) -> int:
        return self._version

    def set_state(self, name: str) -> int:
        """Set the current state (raises ValueError on unknown), stamp a
        monotonic version, fan out to subscribers. Returns the version."""
        norm = normalise_state(name)
        if norm is None:
            raise ValueError(f"unknown state '{name}'")
        self._version += 1
        self._state = norm
        payload = {"type": "astra_state", "state": norm, "version": self._version}
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.discard(q)
            logger.warning("[presence] dropped a stalled subscriber")
        logger.info("[presence] state v%d -> %s (%d subscriber(s))",
                    self._version, norm, len(self._subscribers))
        return self._version

    # ── subscribers (websocket layer) ────────────────────────────────────
    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=_MAX_QUEUE)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def current_payload(self) -> dict:
        return {"type": "astra_state", "state": self._state, "version": self._version}

    def reset(self) -> None:
        """Test helper: clear state, version and subscribers."""
        self._state = "idle"
        self._version = 0
        self._subscribers.clear()


presence_state = PresenceState()
