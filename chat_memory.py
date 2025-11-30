from dataclasses import dataclass
from datetime import datetime
from threading import Lock
from typing import Dict, List


@dataclass
class ChatMessage:
    role: str          # "system", "user", or "assistant"
    content: str
    timestamp: datetime


class InMemoryChatStore:
    def __init__(self, max_messages: int = 20) -> None:
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._lock = Lock()
        self._max_messages = max_messages

    def get_history(self, session_id: str) -> List[ChatMessage]:
        """Return a copy of the history list for this session (may be empty)."""
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def append_message(self, session_id: str, message: ChatMessage) -> None:
        """Append a message and trim to the configured max length."""
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append(message)

            if len(history) > self._max_messages:
                # Keep only the last N messages
                self._sessions[session_id] = history[-self._max_messages :]

    def clear(self, session_id: str) -> None:
        """Remove all messages for a session (not used yet, but handy later)."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        """List all session IDs currently in memory."""
        with self._lock:
            return list(self._sessions.keys())
