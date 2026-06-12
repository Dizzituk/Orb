# FILE: app/translation/rule_suppression.py
# Purpose: Per-rule suppression based on user rejection feedback.
# Called-by: app.translation.translator
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Per-rule suppression based on user rejection feedback.

When the user rejects a confirmation gate (clicks "Not what I meant"),
the rejected event is logged via confirmation_log.log_confirmation_event
with the matching rule_name. Over time, a rule that consistently triggers
on messages the user did not intend as that intent will accumulate
rejections.

This module reads those events and tells the translator which rules to
silently skip. Suppressed rules behave as if they never matched —
the message falls through to the next classifier (or to CHAT_ONLY).

Suppression rule:
- Window: last 30 days.
- Net rejections (rejects - confirms) >= 3 within the window => suppress.
- Net is used (not raw count) so a rule that's correct most of the time
  but has a few rejections does NOT get suppressed.

Self-healing:
- Rolling 30-day window. Old rejections age out automatically.
- If the underlying rule is fixed (e.g. by tightening the heuristic),
  future events will confirm well, net drops below threshold, and the
  rule resumes firing.

Caching:
- The confirmation_events.jsonl file is re-read only when its mtime
  changes. Between reads, lookups are constant-time dict access.
- Suppression check is called on every classifier match, so this
  matters for hot-path latency.

v1.0 (2026-05-01): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# Window for counting rejections — older events don't contribute to suppression.
WINDOW_DAYS = 30

# Suppression threshold: net rejections (rejects - confirms) at or above this
# value within the window suppresses the rule.
NET_REJECTION_THRESHOLD = 3

# Path to the confirmation events log (same file confidence_graduation reads).
_LOG_DIR = Path(os.getenv("ORB_METRICS_DIR", "D:/Orb/metrics"))
_LOG_FILE = _LOG_DIR / "confirmation_events.jsonl"


# =========================================================================
# Cache state
# =========================================================================

_cache_lock = threading.Lock()
_cache: Dict[str, object] = {
    # Mtime of _LOG_FILE at last refresh; -1 means never refreshed.
    "mtime": -1.0,
    # rule_name -> {"rejects": int, "confirms": int, "net": int}
    "counts": {},
}


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp from the log; return None on failure."""
    try:
        # Python's fromisoformat handles the format we write.
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _refresh_cache_if_stale() -> None:
    """Re-read the log file if its mtime has changed since last refresh.

    Cheap when called repeatedly with no new events: a single stat() call.
    """
    if not _LOG_FILE.exists():
        # Reset to empty if the file went away.
        with _cache_lock:
            _cache["mtime"] = -1.0
            _cache["counts"] = {}
        return

    try:
        current_mtime = _LOG_FILE.stat().st_mtime
    except OSError:
        return

    with _cache_lock:
        cached_mtime = _cache["mtime"]
        if isinstance(cached_mtime, float) and current_mtime <= cached_mtime:
            return  # Cache is fresh.

    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"rejects": 0, "confirms": 0, "net": 0}
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)

    try:
        with open(_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                rule_name = event.get("rule_name")
                if not rule_name:
                    # Older events logged before rule_name was added to the
                    # schema have no rule_name. They cannot be attributed
                    # to a rule, so they don't contribute to suppression.
                    continue

                ts_raw = event.get("timestamp")
                ts = _parse_timestamp(ts_raw) if isinstance(ts_raw, str) else None
                if ts is None:
                    continue
                # Make timestamp tz-aware UTC for comparison.
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue  # Outside window.

                bucket = counts[rule_name]
                if event.get("confirmed"):
                    bucket["confirms"] += 1
                else:
                    bucket["rejects"] += 1

    except OSError as e:
        logger.warning(
            "[rule_suppression] Failed to read confirmation log: %s", e
        )
        return

    # Compute net per rule.
    for rule_name, bucket in counts.items():
        bucket["net"] = bucket["rejects"] - bucket["confirms"]

    with _cache_lock:
        _cache["mtime"] = current_mtime
        _cache["counts"] = dict(counts)


def is_rule_suppressed(rule_name: Optional[str]) -> bool:
    """Return True if the rule should be silently skipped.

    Hot-path safe: stats a small file, returns from a dict on cache hit.
    Returns False for empty / None rule_name.
    """
    if not rule_name:
        return False

    _refresh_cache_if_stale()

    with _cache_lock:
        counts = _cache["counts"]
        bucket = counts.get(rule_name) if isinstance(counts, dict) else None

    if not bucket:
        return False

    return bucket.get("net", 0) >= NET_REJECTION_THRESHOLD


def get_rule_stats(rule_name: str) -> Dict[str, int]:
    """Return the current cached counts for a rule. Diagnostic helper.

    Returns a dict with keys: rejects, confirms, net.
    Empty stats (all zeros) if the rule has no events in the window.
    """
    if not rule_name:
        return {"rejects": 0, "confirms": 0, "net": 0}

    _refresh_cache_if_stale()

    with _cache_lock:
        counts = _cache["counts"]
        bucket = counts.get(rule_name) if isinstance(counts, dict) else None

    if not bucket:
        return {"rejects": 0, "confirms": 0, "net": 0}

    return dict(bucket)


def list_suppressed_rules() -> Dict[str, Dict[str, int]]:
    """Return all currently suppressed rules with their stats.

    Diagnostic helper for an Astra "what rules have you silenced?" query.
    """
    _refresh_cache_if_stale()

    with _cache_lock:
        counts = _cache["counts"]
        if not isinstance(counts, dict):
            return {}
        suppressed = {
            rule: dict(bucket)
            for rule, bucket in counts.items()
            if bucket.get("net", 0) >= NET_REJECTION_THRESHOLD
        }

    return suppressed


def force_refresh() -> None:
    """Drop the cache and re-read on next call. For tests."""
    with _cache_lock:
        _cache["mtime"] = -1.0
        _cache["counts"] = {}


__all__ = [
    "WINDOW_DAYS",
    "NET_REJECTION_THRESHOLD",
    "is_rule_suppressed",
    "get_rule_stats",
    "list_suppressed_rules",
    "force_refresh",
]
