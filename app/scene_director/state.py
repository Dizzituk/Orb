# FILE: app/scene_director/state.py
# Purpose: In-memory current-scene store — monotonic versioning, asyncio subscriber
#          fan-out for the websocket layer, and the renderer status breadcrumb feed.
# Called-by: app.scene_director.router
# Depends-on: app.scene_director.schemas
# Last-renovated: 2026-06-12
"""Scene state (v1 — deliberately no DB).

One current scene at a time, held in memory. Subscribers (websocket
connections) each get an asyncio.Queue; set_scene() fans the new doc out to
every queue. Everything runs on FastAPI's single event loop, so no locking
is needed — put_nowait from the serving loop is safe. State resets on
backend restart by design; the Room tab simply composes again.

Also keeps a bounded deque of status events reported by the renderer
({"type":"status", ...} over the websocket) — these are the debug
breadcrumbs surfaced by GET /scene/events and the Room tab feed.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.scene_director.schemas import SceneDoc

logger = logging.getLogger(__name__)

_MAX_QUEUE = 16          # per-subscriber buffer; a stuck client drops, never blocks
_MAX_STATUS_EVENTS = 200


class SceneState:
    """Holder for the single current scene + subscriber registry."""

    def __init__(self) -> None:
        self._doc: Optional[SceneDoc] = None
        self._version: int = 0
        self._subscribers: set[asyncio.Queue] = set()
        self._status_events: deque[Dict[str, Any]] = deque(maxlen=_MAX_STATUS_EVENTS)

    # ── scene ────────────────────────────────────────────────────────────
    def set_scene(self, doc: SceneDoc) -> int:
        """Store doc as current, stamp a monotonic version, fan out to
        subscribers. Returns the stamped version."""
        self._version += 1
        doc.version = self._version
        self._doc = doc
        dead: List[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(doc)
            except asyncio.QueueFull:
                dead.append(q)  # renderer stopped draining — drop it
        for q in dead:
            self._subscribers.discard(q)
            logger.warning("[scene] dropped a stalled websocket subscriber")
        logger.info("[scene] scene v%d set: %s (%d subscriber(s) notified)",
                    self._version, doc.title, len(self._subscribers))
        return self._version

    def get_scene(self) -> Optional[SceneDoc]:
        return self._doc

    @property
    def version(self) -> int:
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

    # ── renderer status breadcrumbs ──────────────────────────────────────
    def add_status_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Record a status message from the renderer (scene_loaded,
        actor_done, error, …). Stamps received_at; returns the stored item."""
        item = dict(payload)
        item["received_at"] = datetime.now(timezone.utc).isoformat()
        self._status_events.append(item)
        logger.info("[scene] renderer status: %s", item.get("event") or item)
        return item

    def status_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Most recent renderer status events, newest first."""
        items = list(self._status_events)
        items.reverse()
        return items[: max(1, min(limit, _MAX_STATUS_EVENTS))]

    # ── tests ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Test helper: clear scene, version, subscribers and breadcrumbs."""
        self._doc = None
        self._version = 0
        self._subscribers.clear()
        self._status_events.clear()


scene_state = SceneState()
