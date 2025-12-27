# FILE: app/introspection/service.py
"""
Log Query Service - Read-Only Introspection

Provides helpers for querying job logs:
- get_last_job_logs() - Most recent completed job
- get_jobs_in_range(start, end) - Jobs in time window
- get_job_logs(job_id) - Specific job by ID

All operations are strictly read-only.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.introspection.schemas import (
    JobLogBundle,
    LogEvent,
    LogQueryResponse,
    LogRequestType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _get_job_artifact_root() -> str:
    """Get root folder for job artifacts."""
    try:
        from app.pot_spec.service import get_job_artifact_root
        return get_job_artifact_root()
    except Exception:
        return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs").strip() or "jobs")


def _parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        # Handle various ISO formats
        ts = ts_str.replace("Z", "+00:00")
        if "+" not in ts and "-" not in ts[10:]:
            ts = ts + "+00:00"
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            # Fallback: try parsing without timezone
            return datetime.strptime(ts_str[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None


def _read_ledger_events(job_id: str) -> list[dict[str, Any]]:
    """Read all events from a job's ledger file."""
    root = _get_job_artifact_root()
    ledger_path = Path(root) / job_id / "ledger" / "events.ndjson"
    
    if not ledger_path.exists():
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
                    logger.warning(f"[introspection] Invalid JSON in ledger {job_id}: {e}")
                    continue
    except Exception as e:
        logger.warning(f"[introspection] Failed to read ledger for {job_id}: {e}")
    
    return events


def _event_to_log_event(raw: dict[str, Any]) -> LogEvent:
    """Convert raw ledger event dict to LogEvent model."""
    ts = _parse_timestamp(raw.get("ts", "")) or datetime.now(timezone.utc)
    
    # Extract known fields
    event_type = raw.get("event", "UNKNOWN")
    job_id = raw.get("job_id")
    stage_name = raw.get("stage") or raw.get("stage_name")
    status = raw.get("status")
    spec_id = raw.get("spec_id")
    expected_hash = raw.get("expected_spec_hash")
    observed_hash = raw.get("observed_spec_hash") or raw.get("returned_spec_hash")
    verified = raw.get("verified")
    error = raw.get("error")
    
    # Remaining fields go to metadata
    known_keys = {
        "ts", "event", "job_id", "stage", "stage_name", "status",
        "spec_id", "expected_spec_hash", "observed_spec_hash", "returned_spec_hash",
        "verified", "error"
    }
    metadata = {k: v for k, v in raw.items() if k not in known_keys}
    
    return LogEvent(
        timestamp=ts,
        event_type=event_type,
        job_id=job_id,
        stage_name=stage_name,
        status=status,
        spec_id=spec_id,
        expected_spec_hash=expected_hash,
        observed_spec_hash=observed_hash,
        verified=verified,
        error=error,
        metadata=metadata,
    )


def _build_job_bundle(job_id: str, events: list[LogEvent], db_job: Optional[Any] = None) -> JobLogBundle:
    """Build a JobLogBundle from events and optional DB job record."""
    bundle = JobLogBundle(
        job_id=job_id,
        events=events,
    )
    
    # Extract job metadata from DB if available
    if db_job:
        bundle.job_type = getattr(db_job, "job_type", None)
        bundle.state = getattr(db_job, "state", None)
        bundle.created_at = getattr(db_job, "created_at", None)
        bundle.completed_at = getattr(db_job, "completed_at", None)
    
    # Check for spec-hash events
    for ev in events:
        if ev.event_type == "STAGE_SPEC_HASH_COMPUTED":
            bundle.spec_hash_computed = True
        elif ev.event_type == "STAGE_SPEC_HASH_VERIFIED":
            bundle.spec_hash_verified = True
        elif ev.event_type == "STAGE_SPEC_HASH_MISMATCH":
            bundle.spec_hash_mismatch = True
    
    return bundle


# =============================================================================
# Result Container
# =============================================================================

@dataclass
class LogQueryResult:
    """Internal result container for log queries."""
    bundles: list[JobLogBundle]
    total_events: int
    error: Optional[str] = None


# =============================================================================
# Public Query Functions
# =============================================================================

def get_last_job_logs(db: Session) -> LogQueryResult:
    """
    Get logs for the most recently completed job.
    
    Returns:
        LogQueryResult with single job bundle or error
    """
    try:
        from app.jobs.models import Job
        from app.jobs.schemas import JobState
        
        # Find most recent completed job
        completed_states = [
            JobState.SUCCEEDED.value,
            JobState.FAILED.value,
            JobState.CANCELLED.value,
        ]
        
        job = (
            db.query(Job)
            .filter(Job.state.in_(completed_states))
            .order_by(Job.completed_at.desc())
            .first()
        )
        
        if not job:
            return LogQueryResult(bundles=[], total_events=0, error="No completed jobs found.")
        
        # Read ledger events
        raw_events = _read_ledger_events(job.id)
        events = [_event_to_log_event(e) for e in raw_events]
        
        bundle = _build_job_bundle(job.id, events, job)
        
        return LogQueryResult(bundles=[bundle], total_events=len(events))
        
    except ImportError:
        return LogQueryResult(bundles=[], total_events=0, error="Job system not available.")
    except Exception as e:
        logger.exception("[introspection] Error in get_last_job_logs")
        return LogQueryResult(bundles=[], total_events=0, error=str(e))


def get_jobs_in_range(
    db: Session,
    start: datetime,
    end: datetime,
) -> LogQueryResult:
    """
    Get logs for all jobs whose events fall within the time window.
    
    Args:
        db: Database session
        start: Start of time window (inclusive)
        end: End of time window (inclusive)
    
    Returns:
        LogQueryResult with job bundles in the range
    """
    try:
        from app.jobs.models import Job
        
        # Query jobs that overlap with the time window
        jobs = (
            db.query(Job)
            .filter(
                (Job.created_at <= end) &
                ((Job.completed_at >= start) | (Job.completed_at.is_(None)))
            )
            .order_by(Job.created_at.desc())
            .limit(100)  # Safety limit
            .all()
        )
        
        if not jobs:
            return LogQueryResult(bundles=[], total_events=0, error="No jobs found in the specified time range.")
        
        bundles = []
        total_events = 0
        
        for job in jobs:
            raw_events = _read_ledger_events(job.id)
            
            # Filter events to those within the time window
            filtered_events = []
            for raw in raw_events:
                ts = _parse_timestamp(raw.get("ts", ""))
                if ts and start <= ts <= end:
                    filtered_events.append(_event_to_log_event(raw))
            
            if filtered_events:
                bundle = _build_job_bundle(job.id, filtered_events, job)
                bundles.append(bundle)
                total_events += len(filtered_events)
        
        if not bundles:
            return LogQueryResult(bundles=[], total_events=0, error="No log events found in the specified time range.")
        
        return LogQueryResult(bundles=bundles, total_events=total_events)
        
    except ImportError:
        return LogQueryResult(bundles=[], total_events=0, error="Job system not available.")
    except Exception as e:
        logger.exception("[introspection] Error in get_jobs_in_range")
        return LogQueryResult(bundles=[], total_events=0, error=str(e))


def get_job_logs(db: Session, job_id: str) -> LogQueryResult:
    """
    Get logs for a specific job by ID.
    
    Args:
        db: Database session
        job_id: Job UUID
    
    Returns:
        LogQueryResult with single job bundle or error
    """
    try:
        from app.jobs.models import Job
        
        # Look up job in DB
        job = db.query(Job).filter(Job.id == job_id).first()
        
        if not job:
            # Job not in DB, but ledger might exist
            raw_events = _read_ledger_events(job_id)
            if not raw_events:
                return LogQueryResult(bundles=[], total_events=0, error=f"Job not found: {job_id}")
            
            events = [_event_to_log_event(e) for e in raw_events]
            bundle = _build_job_bundle(job_id, events, None)
            return LogQueryResult(bundles=[bundle], total_events=len(events))
        
        # Read ledger events
        raw_events = _read_ledger_events(job.id)
        events = [_event_to_log_event(e) for e in raw_events]
        
        bundle = _build_job_bundle(job.id, events, job)
        
        return LogQueryResult(bundles=[bundle], total_events=len(events))
        
    except ImportError:
        return LogQueryResult(bundles=[], total_events=0, error="Job system not available.")
    except Exception as e:
        logger.exception("[introspection] Error in get_job_logs")
        return LogQueryResult(bundles=[], total_events=0, error=str(e))


# =============================================================================
# Time Window Helpers
# =============================================================================

def get_time_window_for_description(description: str) -> tuple[datetime, datetime]:
    """
    Parse a natural language time description into start/end datetimes.
    
    Supports:
    - "last hour" / "past hour"
    - "today"
    - "yesterday"
    - "this week" / "past week"
    
    Returns:
        (start, end) tuple of datetimes
    """
    now = datetime.now(timezone.utc)
    desc = description.lower().strip()
    
    if "hour" in desc:
        return (now - timedelta(hours=1), now)
    
    if "today" in desc:
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start_of_day, now)
    
    if "yesterday" in desc:
        yesterday = now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return (start, end)
    
    if "week" in desc:
        return (now - timedelta(days=7), now)
    
    # Default: last 24 hours
    return (now - timedelta(days=1), now)


__all__ = [
    "get_last_job_logs",
    "get_jobs_in_range",
    "get_job_logs",
    "get_time_window_for_description",
    "LogQueryResult",
]
