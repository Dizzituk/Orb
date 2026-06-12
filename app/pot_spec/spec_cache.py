# FILE: app/pot_spec/spec_cache.py
# Purpose: v2.2: Spec similarity cache — avoid regenerating specs for similar tasks.
# Called-by: app.endpoints.cost_dashboard, app.pot_spec.grounded._spec_runner_result
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
v2.2: Spec similarity cache — avoid regenerating specs for similar tasks.

After a spec is validated, hash its key parameters (goal, in_scope files,
operation type). Before generating a new spec, check if a similar spec
exists (hash match). If found, return the cached spec for user review.

Does NOT auto-reuse — returns the candidate for confirmation.

Storage: data/spec_cache.json (compact JSON, rotated at 500 entries)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

CACHE_DIR = os.path.join("data")
CACHE_FILE = os.path.join(CACHE_DIR, "spec_cache.json")
MAX_CACHE_ENTRIES = 500


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class CachedSpec:
    """A cached spec entry."""
    spec_hash: str
    goal: str
    operation_type: str  # create, modify, refactor, etc.
    in_scope_files: List[str]
    spec_id: str
    spec_json: str  # Full spec JSON for reuse
    created_at: str = ""
    hit_count: int = 0
    last_hit_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CachedSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheMatchResult:
    """Result of a cache lookup."""
    found: bool = False
    cached_spec: Optional[CachedSpec] = None
    match_type: str = "none"  # "exact_hash", "none"
    confidence: float = 0.0


# =========================================================================
# Hash computation
# =========================================================================

def compute_spec_hash(
    goal: str,
    in_scope_files: List[str],
    operation_type: str = "",
) -> str:
    """Compute a deterministic hash for a spec's key parameters.

    Hash is based on: normalised goal + sorted file list + operation type.
    """
    normalised_goal = goal.strip().lower()
    sorted_files = sorted(set(f.strip().lower().replace("\\", "/") for f in in_scope_files if f.strip()))
    payload = json.dumps({
        "goal": normalised_goal,
        "files": sorted_files,
        "op": operation_type.strip().lower(),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# =========================================================================
# Cache I/O
# =========================================================================

def _load_cache() -> List[Dict[str, Any]]:
    """Load cache from disk."""
    if not os.path.isfile(CACHE_FILE):
        return []
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("[spec_cache] Failed to load cache: %s", e)
        return []


def _save_cache(entries: List[Dict[str, Any]]) -> None:
    """Save cache to disk, rotating if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Rotate: keep most recent entries
    if len(entries) > MAX_CACHE_ENTRIES:
        entries = sorted(entries, key=lambda e: e.get("last_hit_at", e.get("created_at", "")))
        entries = entries[-MAX_CACHE_ENTRIES:]
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1)
    except Exception as e:
        logger.warning("[spec_cache] Failed to save cache: %s", e)


# =========================================================================
# Public API
# =========================================================================

def store_spec(
    goal: str,
    in_scope_files: List[str],
    operation_type: str,
    spec_id: str,
    spec_json: str,
) -> str:
    """Store a validated spec in the cache. Returns the spec_hash."""
    spec_hash = compute_spec_hash(goal, in_scope_files, operation_type)
    now = datetime.now(timezone.utc).isoformat()

    entries = _load_cache()

    # Check if hash already exists — update rather than duplicate
    for entry in entries:
        if entry.get("spec_hash") == spec_hash:
            entry["spec_id"] = spec_id
            entry["spec_json"] = spec_json
            entry["last_hit_at"] = now
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            _save_cache(entries)
            logger.info("[spec_cache] Updated existing entry: %s", spec_hash)
            return spec_hash

    cached = CachedSpec(
        spec_hash=spec_hash,
        goal=goal,
        operation_type=operation_type,
        in_scope_files=in_scope_files,
        spec_id=spec_id,
        spec_json=spec_json,
        created_at=now,
        hit_count=0,
        last_hit_at=now,
    )
    entries.append(cached.to_dict())
    _save_cache(entries)
    logger.info("[spec_cache] Stored new spec: hash=%s, goal='%s'", spec_hash, goal[:80])
    return spec_hash


def lookup_spec(
    goal: str,
    in_scope_files: List[str],
    operation_type: str = "",
) -> CacheMatchResult:
    """Look up a cached spec by hash match.

    Returns CacheMatchResult with found=True if an exact hash match exists.
    Does NOT auto-reuse — caller must present to user for confirmation.
    """
    spec_hash = compute_spec_hash(goal, in_scope_files, operation_type)
    entries = _load_cache()

    for entry in entries:
        if entry.get("spec_hash") == spec_hash:
            now = datetime.now(timezone.utc).isoformat()
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            entry["last_hit_at"] = now
            _save_cache(entries)

            cached = CachedSpec.from_dict(entry)
            logger.info(
                "[spec_cache] Cache HIT: hash=%s, hits=%d",
                spec_hash, cached.hit_count,
            )
            return CacheMatchResult(
                found=True,
                cached_spec=cached,
                match_type="exact_hash",
                confidence=1.0,
            )

    logger.debug("[spec_cache] Cache MISS: hash=%s", spec_hash)
    return CacheMatchResult(found=False)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    entries = _load_cache()
    total_hits = sum(e.get("hit_count", 0) for e in entries)
    return {
        "total_entries": len(entries),
        "total_hits": total_hits,
        "cache_file": CACHE_FILE,
        "max_entries": MAX_CACHE_ENTRIES,
    }


__all__ = [
    "CachedSpec",
    "CacheMatchResult",
    "compute_spec_hash",
    "store_spec",
    "lookup_spec",
    "get_cache_stats",
]
