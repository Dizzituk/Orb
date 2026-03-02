# FILE: app/pot_spec/pattern_cache.py
"""
SpecGate Pattern Cache — Job 9.

Caches discovered codebase patterns (component structures, routing
conventions, CSS approaches, file contents) so SpecGate doesn't
re-discover the same patterns on every job.

Cache key = normalised file path.
Cache invalidation = file hash comparison via sandbox /fs/tree.

Storage: data/pattern_cache.json (auto-rotated, TTL-based expiry).

v1.0 (2026-03-01): Initial implementation with TTL and hash-based invalidation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PATTERN_CACHE_BUILD_ID = "2026-03-01-v1.0-specgate-pattern-cache"

CACHE_DIR = os.path.join("data")
CACHE_FILE = os.path.join(CACHE_DIR, "pattern_cache.json")
MAX_ENTRIES = 200
DEFAULT_TTL_SECONDS = 86400  # 24 hours
MAX_CONTENT_SIZE = 10000     # Max chars of file content to cache


@dataclass
class CachedPattern:
    """A cached codebase pattern entry."""
    cache_key: str              # Normalised file path
    content: str                # File content (truncated to MAX_CONTENT_SIZE)
    content_hash: str           # SHA-256 of full file content
    file_size: int              # Original file size in bytes
    pattern_type: str           # "component", "router", "model", "type_def", "css", "config"
    area: str                   # Codebase area: "components/investments", "app/content"
    created_at: float           # time.time() when cached
    ttl_seconds: int            # Time-to-live
    hit_count: int = 0
    last_hit_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CachedPattern:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ─── Cache storage ───────────────────────────────────────────────────

class PatternCache:
    """In-memory + disk cache for discovered codebase patterns.

    Thread-safe for single-process use (ASTRA runs single-threaded).
    """

    def __init__(self, cache_path: str = CACHE_FILE):
        self._path = cache_path
        self._entries: Dict[str, CachedPattern] = {}
        self._dirty = False
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not os.path.isfile(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key, entry_data in data.items():
                pattern = CachedPattern.from_dict(entry_data)
                if not pattern.is_expired:
                    self._entries[key] = pattern
            logger.info(
                "[pattern_cache] Loaded %d entries (%d expired, dropped)",
                len(self._entries), len(data) - len(self._entries),
            )
        except Exception as exc:
            logger.warning("[pattern_cache] Load failed: %s", exc)

    def _save(self) -> None:
        """Persist cache to disk."""
        if not self._dirty:
            return
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            # Evict oldest if over limit
            if len(self._entries) > MAX_ENTRIES:
                sorted_keys = sorted(
                    self._entries,
                    key=lambda k: self._entries[k].last_hit_at or self._entries[k].created_at,
                )
                for k in sorted_keys[: len(self._entries) - MAX_ENTRIES]:
                    del self._entries[k]

            with open(self._path, "w", encoding="utf-8") as fh:
                json.dump(
                    {k: v.to_dict() for k, v in self._entries.items()},
                    fh, indent=1,
                )
            self._dirty = False
        except Exception as exc:
            logger.warning("[pattern_cache] Save failed: %s", exc)

    def get(
        self,
        file_path: str,
        current_hash: Optional[str] = None,
    ) -> Optional[CachedPattern]:
        """Look up a cached pattern by file path.

        Args:
            file_path: File path (normalised internally).
            current_hash: If provided, verify the cached content
                         is still valid (file hasn't changed).

        Returns:
            CachedPattern if found and valid, None otherwise.
        """
        key = _normalise_key(file_path)
        entry = self._entries.get(key)

        if entry is None:
            return None

        if entry.is_expired:
            del self._entries[key]
            self._dirty = True
            logger.debug("[pattern_cache] Expired: %s", key)
            return None

        if current_hash and entry.content_hash != current_hash:
            del self._entries[key]
            self._dirty = True
            logger.info(
                "[pattern_cache] Invalidated (hash mismatch): %s", key,
            )
            return None

        # Cache hit
        entry.hit_count += 1
        entry.last_hit_at = time.time()
        self._dirty = True
        logger.debug("[pattern_cache] HIT: %s (hits=%d)", key, entry.hit_count)
        return entry

    def put(
        self,
        file_path: str,
        content: str,
        pattern_type: str = "component",
        area: str = "",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CachedPattern:
        """Cache a discovered pattern.

        Args:
            file_path: File path.
            content: File content (truncated if needed).
            pattern_type: Type of pattern.
            area: Codebase area.
            ttl_seconds: Time-to-live.
            metadata: Extra data.

        Returns:
            The created CachedPattern entry.
        """
        key = _normalise_key(file_path)
        truncated = content[:MAX_CONTENT_SIZE]
        full_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        entry = CachedPattern(
            cache_key=key,
            content=truncated,
            content_hash=full_hash,
            file_size=len(content),
            pattern_type=pattern_type,
            area=area or _extract_area(file_path),
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )
        self._entries[key] = entry
        self._dirty = True
        self._save()

        logger.info(
            "[pattern_cache] STORED: %s (%s, %d chars, TTL=%ds)",
            key, pattern_type, len(truncated), ttl_seconds,
        )
        return entry

    def invalidate_area(self, area: str) -> int:
        """Invalidate all cached patterns in a codebase area.

        Used when files in an area change (e.g. after implementation).

        Args:
            area: Codebase area to invalidate (e.g. "components/investments").

        Returns:
            Number of entries invalidated.
        """
        norm_area = area.replace("\\", "/").lower()
        to_remove = [
            k for k, v in self._entries.items()
            if v.area.lower().startswith(norm_area)
            or k.startswith(norm_area)
        ]
        for k in to_remove:
            del self._entries[k]
        if to_remove:
            self._dirty = True
            self._save()
            logger.info(
                "[pattern_cache] Invalidated %d entries in area '%s'",
                len(to_remove), area,
            )
        return len(to_remove)

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = len(self._entries)
        expired = sum(1 for e in self._entries.values() if e.is_expired)
        total_hits = sum(e.hit_count for e in self._entries.values())
        areas = set(e.area for e in self._entries.values())
        return {
            "total_entries": total,
            "expired": expired,
            "total_hits": total_hits,
            "unique_areas": len(areas),
            "areas": sorted(areas),
        }

    def flush(self) -> None:
        """Force-save dirty state."""
        self._save()


# ─── Module-level singleton ──────────────────────────────────────────

_instance: Optional[PatternCache] = None


def get_pattern_cache() -> PatternCache:
    """Get or create the singleton PatternCache."""
    global _instance
    if _instance is None:
        _instance = PatternCache()
    return _instance


# ─── Helpers ─────────────────────────────────────────────────────────

def _normalise_key(file_path: str) -> str:
    """Normalise a file path to a consistent cache key."""
    return file_path.replace("\\", "/").lower()


def _extract_area(file_path: str) -> str:
    """Extract the codebase area from a file path.

    src/components/investments/InvestmentsView.tsx → components/investments
    app/content/router.py → app/content
    """
    norm = file_path.replace("\\", "/")
    # Remove common prefixes
    for prefix in ("src/", "app/", "D:/Orb/", "D:/orb-desktop/"):
        if norm.lower().startswith(prefix.lower()):
            norm = norm[len(prefix):]
            break

    parts = norm.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if parts else ""
