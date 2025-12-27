# FILE: app/pot_spec/ledger.py
"""Append-only ledger for deterministic replay.

Each event is a single JSON object written as one line (ndjson).

New in v2:
- read_events(): Read all events from a job's ledger
- read_events_in_range(): Read events within a time window
- emit_spec_hash_computed(): Emit STAGE_SPEC_HASH_COMPUTED event
- emit_spec_hash_verified(): Emit STAGE_SPEC_HASH_VERIFIED event
- emit_spec_hash_mismatch(): Emit STAGE_SPEC_HASH_MISMATCH event
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
# Write Operations (existing)
# =============================================================================

def append_event(job_artifact_root: str, job_id: str, event: dict[str, Any]) -> str:
    """Append event to jobs/<job_id>/ledger/events.ndjson.

    Returns the absolute path written.
    """
    ledger_dir = os.path.join(job_artifact_root, job_id, "ledger")
    os.makedirs(ledger_dir, exist_ok=True)
    path = os.path.join(ledger_dir, "events.ndjson")

    record = dict(event)
    record.setdefault("ts", _utc_iso())

    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")

    return path


# =============================================================================
# Read Operations (new)
# =============================================================================

def read_events(job_artifact_root: str, job_id: str) -> list[dict[str, Any]]:
    """Read all events from a job's ledger.
    
    Returns list of event dicts, or empty list if ledger doesn't exist.
    """
    ledger_path = os.path.join(job_artifact_root, job_id, "ledger", "events.ndjson")
    
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


# =============================================================================
# Spec-Gate Hash Events (new)
# =============================================================================

def emit_spec_hash_computed(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
) -> str:
    """Emit STAGE_SPEC_HASH_COMPUTED event.
    
    Emitted when a spec hash is computed before a stage runs.
    
    Returns the ledger file path.
    """
    event = {
        "event": "STAGE_SPEC_HASH_COMPUTED",
        "job_id": job_id,
        "stage_name": stage_name,
        "spec_id": spec_id,
        "expected_spec_hash": expected_spec_hash,
        "status": "ok",
        "ts": _utc_iso(),
    }
    return append_event(job_artifact_root, job_id, event)


def emit_spec_hash_verified(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
    observed_spec_hash: str,
) -> str:
    """Emit STAGE_SPEC_HASH_VERIFIED event.
    
    Emitted when spec hash verification succeeds (hashes match).
    
    Returns the ledger file path.
    """
    event = {
        "event": "STAGE_SPEC_HASH_VERIFIED",
        "job_id": job_id,
        "stage_name": stage_name,
        "spec_id": spec_id,
        "expected_spec_hash": expected_spec_hash,
        "observed_spec_hash": observed_spec_hash,
        "verified": True,
        "status": "ok",
        "ts": _utc_iso(),
    }
    return append_event(job_artifact_root, job_id, event)


def emit_spec_hash_mismatch(
    job_artifact_root: str,
    job_id: str,
    stage_name: str,
    spec_id: str,
    expected_spec_hash: str,
    observed_spec_hash: Optional[str],
    reason: Optional[str] = None,
) -> str:
    """Emit STAGE_SPEC_HASH_MISMATCH event.
    
    Emitted when spec hash verification fails (hashes don't match).
    This is a WARNING/ERROR severity event.
    
    Returns the ledger file path.
    """
    message = "spec hash mismatch — pipeline aborted before applying output"
    if reason:
        message = f"{message} ({reason})"
    
    event = {
        "event": "STAGE_SPEC_HASH_MISMATCH",
        "job_id": job_id,
        "stage_name": stage_name,
        "spec_id": spec_id,
        "expected_spec_hash": expected_spec_hash,
        "observed_spec_hash": observed_spec_hash,
        "verified": False,
        "severity": "ERROR",
        "message": message,
        "status": "error",
        "ts": _utc_iso(),
    }
    
    # Also log to Python logger for visibility
    logger.warning(
        f"[ledger] SPEC_HASH_MISMATCH job={job_id} stage={stage_name} "
        f"expected={expected_spec_hash[:16]}... observed={observed_spec_hash[:16] if observed_spec_hash else 'None'}..."
    )
    
    return append_event(job_artifact_root, job_id, event)


__all__ = [
    # Write
    "append_event",
    # Read
    "read_events",
    "read_events_in_range",
    # Spec-hash events
    "emit_spec_hash_computed",
    "emit_spec_hash_verified",
    "emit_spec_hash_mismatch",
]
