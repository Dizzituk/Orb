# FILE: app/voice_ambient/conversation_memory.py
# Purpose: In-memory conversation history for ambient voice sessions.
# Called-by: app.invocation.router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
In-memory conversation history for ambient voice sessions.

Keyed by session_id (UUID from the client). Stores the last N user+assistant
turns per session, with a TTL so stale sessions drop. Pure process-memory -
resets on backend restart. Persistence to the chat database happens in a
separate module.

This is deliberately small and lock-free where possible. Concurrent reads are
fine (worst case: stale but consistent history); writes use a single lock.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

MAX_TURNS_PER_SESSION = 8        # keep last N (user+assistant = 1 turn)
TTL_SECONDS = 30 * 60            # drop session after 30 min idle


@dataclass
class Turn:
    role: str        # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class _Session:
    turns: Deque[Turn] = field(default_factory=lambda: deque(maxlen=MAX_TURNS_PER_SESSION * 2))
    last_touch: float = field(default_factory=time.monotonic)


class ConversationMemory:
    _instance: Optional["ConversationMemory"] = None
    _cls_lock = threading.Lock()

    def __init__(self):
        self._sessions: Dict[str, _Session] = {}
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ConversationMemory":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ─── Public API ──────────────────────────────────────────────
    def get_history(self, session_id: str) -> List[Turn]:
        """Get the turns for a session. Empty if unknown or expired."""
        if not session_id:
            return []
        self._sweep_expired()
        with self._lock:
            sess = self._sessions.get(session_id)
            if not sess:
                return []
            sess.last_touch = time.monotonic()
            return list(sess.turns)

    def append_user(self, session_id: str, text: str) -> None:
        if not session_id or not text:
            return
        self._append(session_id, Turn(role="user", content=text))

    def append_assistant(self, session_id: str, text: str) -> None:
        if not session_id or not text:
            return
        self._append(session_id, Turn(role="assistant", content=text))

    def clear(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

    def as_llm_messages(self, session_id: str) -> List[dict]:
        """Return history in OpenAI-compatible role/content dicts."""
        return [
            {"role": t.role, "content": t.content}
            for t in self.get_history(session_id)
        ]

    # ─── Internals ───────────────────────────────────────────────
    def _append(self, session_id: str, turn: Turn) -> None:
        self._sweep_expired()
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = _Session()
                self._sessions[session_id] = sess
            sess.turns.append(turn)
            sess.last_touch = time.monotonic()

    def _sweep_expired(self) -> None:
        now = time.monotonic()
        with self._lock:
            drop = [sid for sid, s in self._sessions.items()
                    if now - s.last_touch > TTL_SECONDS]
            for sid in drop:
                self._sessions.pop(sid, None)


def get_conversation_memory() -> ConversationMemory:
    return ConversationMemory.get_instance()