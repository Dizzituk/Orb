# FILE: app/content/sse_manager.py
# Purpose: Content SSE manager — per-project event streams.
# Called-by: app.content.stream_router, app.content.style_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Content SSE manager — per-project event streams.
Follows existing streaming patterns from app/llm/streaming.py.
"""

import asyncio
import json
import logging
from typing import Dict, Set, AsyncGenerator
from collections import defaultdict

logger = logging.getLogger(__name__)


class ContentSSEManager:
    def __init__(self):
        self._queues: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    def subscribe(self, project_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._queues[project_id].add(q)
        return q

    def unsubscribe(self, project_id: str, q: asyncio.Queue):
        self._queues[project_id].discard(q)
        if not self._queues[project_id]:
            del self._queues[project_id]

    async def publish(self, project_id: str, event: dict):
        for q in self._queues.get(project_id, set()):
            try:
                await q.put(event)
            except Exception:
                pass

    async def event_generator(self, project_id: str) -> AsyncGenerator[str, None]:
        q = self.subscribe(project_id)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            self.unsubscribe(project_id, q)


sse_manager = ContentSSEManager()
