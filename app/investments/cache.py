# FILE: app/investments/cache.py
# Purpose: Simple in-memory TTL cache for rate-limit protection.
# Called-by: app.investments.service
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Simple in-memory TTL cache for rate-limit protection.

Stores cached values with expiry timestamps.
Thread-safe via a simple lock.
"""
import time
import threading
from typing import Any, Optional


class TTLCache:
    """Key-value cache with per-entry TTL (seconds)."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if not expired, else None."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if time.time() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any, ttl: int) -> None:
        """Store value with TTL in seconds."""
        with self._lock:
            self._store[key] = (value, time.time() + ttl)

    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()


# Module-level singleton
_cache = TTLCache()


def get_cache() -> TTLCache:
    """Return the shared cache instance."""
    return _cache
