# FILE: app/pot_spec/ledger_core.py
"""Core ledger operations - base module with no internal imports.

This file exists to avoid circular imports between ledger modules.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _utc_iso() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        ts = ts_str.replace("Z", "+00:00")
        if "+" not in ts and "-" not in ts[10:]:
            ts = ts + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None


# =============================================================================
# Write Operations
# =============================================================================

def append_event(job_artifact_root: str, job_id: str, event: dict[str, Any]) -> str:
    """Append event to jobs/<job_id>/ledger/events.ndjson.

    Returns the absolute path written.
    """
    ledger_dir = os.path.join(job_artifact_root, "jobs", job_id, "ledger")
    os.makedirs(ledger_dir, exist_ok=True)
    path = os.path.join(ledger_dir, "events.ndjson")

    record = dict(event)
    record.setdefault("ts", _utc_iso())
    record.setdefault("job_id", job_id)

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

    return path


# =============================================================================
# Read Operations
# =============================================================================

def read_events(job_artifact_root: str, job_id: str) -> list[dict[str, Any]]:
    """Read all events from a job's ledger.
    
    Returns list of event dicts, or empty list if ledger doesn't exist.
    """
    ledger_path = os.path.join(job_artifact_root, "jobs", job_id, "ledger", "events.ndjson")
    
    if not os.path.exists(ledger_path):
        return []
    
    events = []
    try:
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"[ledger] Invalid JSON in {job_id}: {e}")
                    continue
    except Exception as e:
        logger.warning(f"[ledger] Failed to read ledger for {job_id}: {e}")
    
    return events


def read_events_in_range(
    job_artifact_root: str,
    job_id: str,
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    """Read events from a job's ledger within a time range.
    
    Args:
        job_artifact_root: Root folder for job artifacts
        job_id: Job UUID
        start: Start of time window (inclusive)
        end: End of time window (inclusive)
    
    Returns list of event dicts within the time range.
    """
    all_events = read_events(job_artifact_root, job_id)
    
    filtered = []
    for event in all_events:
        ts = _parse_timestamp(event.get("ts", ""))
        if ts and start <= ts <= end:
            filtered.append(event)
    
    return filtered


__all__ = [
    "append_event",
    "read_events",
    "read_events_in_range",
    "_utc_iso",
    "_parse_timestamp",
]
