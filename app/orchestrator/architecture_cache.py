# FILE: app/orchestrator/architecture_cache.py
"""
v2.2: Architecture template cache — reuse successful architectures for repeat patterns.

After a successful architecture generation, stores: task_type + file_targets → arch document.
For repeat patterns (e.g. "add a new endpoint"), retrieve and adapt rather than regenerate.

Particularly valuable for the delivery driver app build where many features share
structural patterns (new screen + new API endpoint + new model).

Storage: data/architecture_cache.json (compact JSON, rotated at 200 entries)
Does NOT auto-apply — returns candidate for review/adaptation.
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
CACHE_FILE = os.path.join(CACHE_DIR, "architecture_cache.json")
MAX_CACHE_ENTRIES = 200


# =========================================================================
# Data structures
# =========================================================================

@dataclass
class CachedArchitecture:
    """A cached architecture entry."""
    arch_hash: str
    task_pattern: str  # e.g. "add_endpoint", "new_screen", "add_model"
    file_targets: List[str]  # Files the architecture touches
    arch_content: str  # The architecture document
    spec_hash: str = ""  # Hash of the original spec
    model_used: str = ""  # Which model generated this
    created_at: str = ""
    hit_count: int = 0
    last_hit_at: str = ""
    critique_passed: bool = False  # Did this pass critique?

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CachedArchitecture":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ArchCacheMatchResult:
    """Result of an architecture cache lookup."""
    found: bool = False
    cached_arch: Optional[CachedArchitecture] = None
    match_type: str = "none"  # "exact_hash", "pattern_match", "none"
    confidence: float = 0.0
    differences: List[str] = field(default_factory=list)


# =========================================================================
# Pattern detection
# =========================================================================

def detect_task_pattern(
    goal: str,
    file_targets: List[str],
) -> str:
    """Detect the structural pattern of a task.

    Returns a normalised pattern string like 'add_endpoint', 'new_screen', etc.
    """
    goal_lower = goal.lower()

    # Detect common patterns
    if any(kw in goal_lower for kw in ("add endpoint", "new endpoint", "create endpoint", "api route")):
        return "add_endpoint"
    if any(kw in goal_lower for kw in ("new screen", "add screen", "create screen", "new page", "add page")):
        return "new_screen"
    if any(kw in goal_lower for kw in ("add model", "new model", "create model", "database model")):
        return "add_model"
    if any(kw in goal_lower for kw in ("refactor", "split", "decompose", "extract")):
        return "refactor"
    if any(kw in goal_lower for kw in ("fix bug", "bugfix", "fix error", "fix issue")):
        return "bugfix"
    if any(kw in goal_lower for kw in ("add test", "write test", "unit test")):
        return "add_test"
    if any(kw in goal_lower for kw in ("integrate", "wire", "connect", "hook up")):
        return "integration"

    # File-based patterns
    extensions = set(os.path.splitext(f)[1].lower() for f in file_targets if f)
    if ".kt" in extensions or ".java" in extensions:
        return "android_feature"
    if ".tsx" in extensions or ".jsx" in extensions:
        return "react_component"

    return "general"


# =========================================================================
# Hash computation
# =========================================================================

def compute_arch_hash(
    task_pattern: str,
    file_targets: List[str],
) -> str:
    """Compute a deterministic hash for an architecture's key parameters."""
    sorted_files = sorted(set(
        f.strip().lower().replace("\\", "/") for f in file_targets if f.strip()
    ))
    payload = json.dumps({
        "pattern": task_pattern,
        "files": sorted_files,
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
        logger.warning("[arch_cache] Failed to load cache: %s", e)
        return []


def _save_cache(entries: List[Dict[str, Any]]) -> None:
    """Save cache to disk, rotating if needed."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    if len(entries) > MAX_CACHE_ENTRIES:
        # Keep highest-hit, most-recent entries
        entries = sorted(
            entries,
            key=lambda e: (e.get("critique_passed", False), e.get("hit_count", 0)),
        )
        entries = entries[-MAX_CACHE_ENTRIES:]
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=1)
    except Exception as e:
        logger.warning("[arch_cache] Failed to save cache: %s", e)


# =========================================================================
# Public API
# =========================================================================

def store_architecture(
    goal: str,
    file_targets: List[str],
    arch_content: str,
    spec_hash: str = "",
    model_used: str = "",
    critique_passed: bool = False,
) -> str:
    """Store a successful architecture in the cache. Returns the arch_hash."""
    task_pattern = detect_task_pattern(goal, file_targets)
    arch_hash = compute_arch_hash(task_pattern, file_targets)
    now = datetime.now(timezone.utc).isoformat()

    entries = _load_cache()

    # Update if hash exists, only if new one passed critique
    for entry in entries:
        if entry.get("arch_hash") == arch_hash:
            if critique_passed or not entry.get("critique_passed"):
                entry["arch_content"] = arch_content
                entry["spec_hash"] = spec_hash
                entry["model_used"] = model_used
                entry["critique_passed"] = critique_passed
                entry["last_hit_at"] = now
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            _save_cache(entries)
            logger.info("[arch_cache] Updated entry: %s (%s)", arch_hash, task_pattern)
            return arch_hash

    cached = CachedArchitecture(
        arch_hash=arch_hash,
        task_pattern=task_pattern,
        file_targets=file_targets,
        arch_content=arch_content,
        spec_hash=spec_hash,
        model_used=model_used,
        created_at=now,
        hit_count=0,
        last_hit_at=now,
        critique_passed=critique_passed,
    )
    entries.append(cached.to_dict())
    _save_cache(entries)
    logger.info(
        "[arch_cache] Stored: hash=%s, pattern=%s, files=%d, passed=%s",
        arch_hash, task_pattern, len(file_targets), critique_passed,
    )
    return arch_hash


def lookup_architecture(
    goal: str,
    file_targets: List[str],
) -> ArchCacheMatchResult:
    """Look up a cached architecture by pattern + file hash.

    Returns ArchCacheMatchResult. Does NOT auto-apply — caller should
    present to user for adaptation/confirmation.
    """
    task_pattern = detect_task_pattern(goal, file_targets)
    arch_hash = compute_arch_hash(task_pattern, file_targets)
    entries = _load_cache()

    # Exact hash match
    for entry in entries:
        if entry.get("arch_hash") == arch_hash:
            now = datetime.now(timezone.utc).isoformat()
            entry["hit_count"] = entry.get("hit_count", 0) + 1
            entry["last_hit_at"] = now
            _save_cache(entries)

            cached = CachedArchitecture.from_dict(entry)
            logger.info(
                "[arch_cache] Cache HIT (exact): hash=%s, pattern=%s, hits=%d",
                arch_hash, task_pattern, cached.hit_count,
            )
            return ArchCacheMatchResult(
                found=True,
                cached_arch=cached,
                match_type="exact_hash",
                confidence=1.0,
            )

    # Pattern match: same pattern, different files — useful as template
    for entry in entries:
        if (
            entry.get("task_pattern") == task_pattern
            and entry.get("critique_passed", False)
        ):
            cached = CachedArchitecture.from_dict(entry)
            # Compute file overlap
            cached_files = set(f.lower().replace("\\", "/") for f in cached.file_targets)
            target_files = set(f.lower().replace("\\", "/") for f in file_targets)
            overlap = len(cached_files & target_files)
            total = max(len(cached_files | target_files), 1)
            confidence = overlap / total

            if confidence >= 0.3:  # At least 30% file overlap
                diffs = []
                added = target_files - cached_files
                removed = cached_files - target_files
                if added:
                    diffs.append(f"New files: {', '.join(sorted(added)[:5])}")
                if removed:
                    diffs.append(f"Removed files: {', '.join(sorted(removed)[:5])}")

                logger.info(
                    "[arch_cache] Cache HIT (pattern): pattern=%s, confidence=%.2f",
                    task_pattern, confidence,
                )
                return ArchCacheMatchResult(
                    found=True,
                    cached_arch=cached,
                    match_type="pattern_match",
                    confidence=confidence,
                    differences=diffs,
                )

    logger.debug("[arch_cache] Cache MISS: pattern=%s, hash=%s", task_pattern, arch_hash)
    return ArchCacheMatchResult(found=False)


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    entries = _load_cache()
    patterns = {}
    for e in entries:
        p = e.get("task_pattern", "unknown")
        patterns[p] = patterns.get(p, 0) + 1
    return {
        "total_entries": len(entries),
        "total_hits": sum(e.get("hit_count", 0) for e in entries),
        "passed_critique": sum(1 for e in entries if e.get("critique_passed")),
        "patterns": patterns,
        "cache_file": CACHE_FILE,
    }


__all__ = [
    "CachedArchitecture",
    "ArchCacheMatchResult",
    "detect_task_pattern",
    "compute_arch_hash",
    "store_architecture",
    "lookup_architecture",
    "get_cache_stats",
]
