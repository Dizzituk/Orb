# FILE: app/astra_memory/service.py
"""
ASTRA Memory Service (AstraJob 5)

Core principle:
1. Write to NDJSON ledger FIRST (source of truth)
2. Project key facts to SQLite (queryable index)

This service handles:
- AstraJob lifecycle (create, update status, complete)
- File tracking (link to Atlas)
- Event projection (ledger → SQLite)
- Overwatcher state persistence
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Import models
try:
    from app.astra_memory.models import (
        AstraJob,
        JobFile,
        JobEvent,
        JobChunk,
        OverwatchSummary,
        GlobalPref,
        OverwatchPattern,
    )
    _MODELS_AVAILABLE = True
except ImportError:
    _MODELS_AVAILABLE = False
    logger.warning("[astra_memory] Models not available")

# Import ledger
try:
    from app.pot_spec.ledger_core import append_event as ledger_append
    _LEDGER_AVAILABLE = True
except ImportError:
    _LEDGER_AVAILABLE = False
    logger.warning("[astra_memory] Ledger not available")


# =============================================================================
# Helpers
# =============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_ts() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _artifact_root() -> str:
    return os.path.abspath(os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs"))


def _hash_content(content: str) -> str:
    """SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# AstraJob LIFECYCLE
# =============================================================================

def create_job(
    db: Session,
    job_id: Optional[str] = None,
    user_intent: Optional[str] = None,
    repo_root: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> AstraJob:
    """
    Create a new AstraJob record.
    
    1. Write JOB_CREATED event to ledger
    2. Create AstraJob row in SQLite
    """
    if not _MODELS_AVAILABLE:
        raise RuntimeError("astra_memory models not available")
    
    job_id = job_id or str(uuid4())
    now = _utc_now()
    
    # 1. Write to ledger
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "JOB_CREATED",
                "job_id": job_id,
                "user_intent": user_intent[:500] if user_intent else None,
                "repo_root": repo_root,
                "provider": provider,
                "model": model,
                "status": "created",
                "ts": _utc_ts(),
            },
        )
    
    # 2. Create SQLite row
    AstraJob = AstraJob(
        job_id=job_id,
        user_intent=user_intent,
        repo_root=repo_root,
        status="created",
        primary_provider=provider,
        primary_model=model,
        created_at=now,
        updated_at=now,
    )
    db.add(AstraJob)
    db.commit()
    db.refresh(AstraJob)
    
    logger.info(f"[astra_memory] Created AstraJob {job_id}")
    return AstraJob


def update_job_status(
    db: Session,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
) -> Optional[AstraJob]:
    """
    Update AstraJob status.
    
    Valid statuses: created, spec_gate, planning, executing, verifying, completed, failed, aborted
    """
    if not _MODELS_AVAILABLE:
        return None
    
    AstraJob = db.query(AstraJob).filter(AstraJob.job_id == job_id).first()
    if not AstraJob:
        logger.warning(f"[astra_memory] AstraJob not found: {job_id}")
        return None
    
    # Write to ledger
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "JOB_STATUS_CHANGED",
                "job_id": job_id,
                "old_status": AstraJob.status,
                "new_status": status,
                "error_message": error_message,
                "ts": _utc_ts(),
            },
        )
    
    # Update SQLite
    AstraJob.status = status
    AstraJob.updated_at = _utc_now()
    
    if status in ("completed", "failed", "aborted"):
        AstraJob.completed_at = _utc_now()
    
    db.commit()
    return AstraJob


def link_spec_to_job(
    db: Session,
    job_id: str,
    spec_id: str,
    spec_hash: str,
    spec_version: int,
) -> Optional[AstraJob]:
    """Link a PoT spec to a AstraJob."""
    if not _MODELS_AVAILABLE:
        return None
    
    AstraJob = db.query(AstraJob).filter(AstraJob.job_id == job_id).first()
    if not AstraJob:
        return None
    
    AstraJob.spec_id = spec_id
    AstraJob.spec_hash = spec_hash
    AstraJob.spec_version = spec_version
    AstraJob.updated_at = _utc_now()
    
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "JOB_SPEC_LINKED",
                "job_id": job_id,
                "spec_id": spec_id,
                "spec_hash": spec_hash,
                "spec_version": spec_version,
                "ts": _utc_ts(),
            },
        )
    
    db.commit()
    return AstraJob


def link_arch_to_job(
    db: Session,
    job_id: str,
    arch_id: str,
    arch_hash: str,
    arch_version: int,
) -> Optional[AstraJob]:
    """Link an architecture snapshot to a AstraJob."""
    if not _MODELS_AVAILABLE:
        return None
    
    AstraJob = db.query(AstraJob).filter(AstraJob.job_id == job_id).first()
    if not AstraJob:
        return None
    
    AstraJob.arch_id = arch_id
    AstraJob.arch_hash = arch_hash
    AstraJob.arch_version = arch_version
    AstraJob.updated_at = _utc_now()
    
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "JOB_ARCH_LINKED",
                "job_id": job_id,
                "arch_id": arch_id,
                "arch_hash": arch_hash,
                "arch_version": arch_version,
                "ts": _utc_ts(),
            },
        )
    
    db.commit()
    return AstraJob


# =============================================================================
# FILE TRACKING
# =============================================================================

def record_file_touch(
    db: Session,
    job_id: str,
    path: str,
    action: str,
    hash_before: Optional[str] = None,
    hash_after: Optional[str] = None,
    chunk_id: Optional[str] = None,
    arch_id: Optional[str] = None,
    symbol_name: Optional[str] = None,
) -> Optional[JobFile]:
    """
    Record a file being touched by a AstraJob.
    
    Actions: read, create, modify, delete
    """
    if not _MODELS_AVAILABLE:
        return None
    
    # Write to ledger
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "FILE_TOUCHED",
                "job_id": job_id,
                "path": path,
                "action": action,
                "hash_before": hash_before,
                "hash_after": hash_after,
                "chunk_id": chunk_id,
                "arch_id": arch_id,
                "ts": _utc_ts(),
            },
        )
    
    # Create SQLite row
    file_record = JobFile(
        job_id=job_id,
        arch_id=arch_id,
        path=path,
        symbol_name=symbol_name,
        action=action,
        hash_before=hash_before,
        hash_after=hash_after,
        chunk_id=chunk_id,
        touched_at=_utc_now(),
    )
    db.add(file_record)
    db.commit()
    
    return file_record


def get_files_for_job(db: Session, job_id: str) -> List[JobFile]:
    """Get all files touched by a AstraJob."""
    if not _MODELS_AVAILABLE:
        return []
    return db.query(JobFile).filter(JobFile.job_id == job_id).all()


def get_jobs_for_file(db: Session, path: str) -> List[Tuple[str, str, datetime]]:
    """Get all jobs that touched a specific file.
    
    Returns: List of (job_id, action, touched_at)
    """
    if not _MODELS_AVAILABLE:
        return []
    
    results = (
        db.query(JobFile.job_id, JobFile.action, JobFile.touched_at)
        .filter(JobFile.path == path)
        .order_by(JobFile.touched_at.desc())
        .all()
    )
    return results


# =============================================================================
# EVENT PROJECTION
# =============================================================================

def project_event_to_db(
    db: Session,
    job_id: str,
    event_type: str,
    stage: Optional[str] = None,
    severity: str = "info",
    status: Optional[str] = None,
    spec_id: Optional[str] = None,
    chunk_id: Optional[str] = None,
    error_message: Optional[str] = None,
    ledger_line: Optional[int] = None,
) -> Optional[JobEvent]:
    """
    Project a ledger event to SQLite for querying.
    
    Call this after writing to the NDJSON ledger.
    """
    if not _MODELS_AVAILABLE:
        return None
    
    event = JobEvent(
        job_id=job_id,
        event_type=event_type,
        stage=stage,
        severity=severity,
        status=status,
        spec_id=spec_id,
        chunk_id=chunk_id,
        error_message=error_message,
        ledger_line=ledger_line,
        ts=_utc_now(),
        created_at=_utc_now(),
    )
    db.add(event)
    db.commit()
    
    return event


def get_events_for_job(
    db: Session,
    job_id: str,
    event_type: Optional[str] = None,
    severity: Optional[str] = None,
) -> List[JobEvent]:
    """Query events for a AstraJob with optional filters."""
    if not _MODELS_AVAILABLE:
        return []
    
    query = db.query(JobEvent).filter(JobEvent.job_id == job_id)
    
    if event_type:
        query = query.filter(JobEvent.event_type == event_type)
    if severity:
        query = query.filter(JobEvent.severity == severity)
    
    return query.order_by(JobEvent.ts).all()


# =============================================================================
# CHUNK TRACKING
# =============================================================================

def create_chunk(
    db: Session,
    job_id: str,
    chunk_id: Optional[str] = None,
    sequence: int = 0,
    target_path: Optional[str] = None,
    target_symbol: Optional[str] = None,
    description: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[JobChunk]:
    """Create a chunk record for execution planning."""
    if not _MODELS_AVAILABLE:
        return None
    
    chunk_id = chunk_id or str(uuid4())
    
    # Write to ledger
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "CHUNK_PLANNED",
                "job_id": job_id,
                "chunk_id": chunk_id,
                "sequence": sequence,
                "target_path": target_path,
                "target_symbol": target_symbol,
                "description": description[:200] if description else None,
                "ts": _utc_ts(),
            },
        )
    
    chunk = JobChunk(
        job_id=job_id,
        chunk_id=chunk_id,
        sequence=sequence,
        target_path=target_path,
        target_symbol=target_symbol,
        description=description,
        status="pending",
        provider=provider,
        model=model,
        planned_at=_utc_now(),
    )
    db.add(chunk)
    db.commit()
    
    return chunk


def update_chunk_status(
    db: Session,
    chunk_id: str,
    status: str,
    diff_summary: Optional[str] = None,
    lines_added: Optional[int] = None,
    lines_removed: Optional[int] = None,
    tests_run: Optional[int] = None,
    tests_passed: Optional[int] = None,
    tests_failed: Optional[int] = None,
) -> Optional[JobChunk]:
    """Update chunk execution status."""
    if not _MODELS_AVAILABLE:
        return None
    
    chunk = db.query(JobChunk).filter(JobChunk.chunk_id == chunk_id).first()
    if not chunk:
        return None
    
    chunk.status = status
    
    if status == "executing" and not chunk.started_at:
        chunk.started_at = _utc_now()
    
    if status in ("completed", "failed", "blocked"):
        chunk.completed_at = _utc_now()
    
    if diff_summary:
        chunk.diff_summary = diff_summary
    if lines_added is not None:
        chunk.lines_added = lines_added
    if lines_removed is not None:
        chunk.lines_removed = lines_removed
    if tests_run is not None:
        chunk.tests_run = tests_run
    if tests_passed is not None:
        chunk.tests_passed = tests_passed
    if tests_failed is not None:
        chunk.tests_failed = tests_failed
    
    # Write to ledger
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=chunk.job_id,
            event={
                "event": "CHUNK_STATUS_CHANGED",
                "job_id": chunk.job_id,
                "chunk_id": chunk_id,
                "status": status,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "ts": _utc_ts(),
            },
        )
    
    db.commit()
    return chunk


# =============================================================================
# OVERWATCHER SUMMARY
# =============================================================================

def get_or_create_overwatch_summary(
    db: Session,
    job_id: str,
) -> Optional[OverwatchSummary]:
    """Get or create Overwatcher summary for a AstraJob."""
    if not _MODELS_AVAILABLE:
        return None
    
    summary = db.query(OverwatchSummary).filter(OverwatchSummary.job_id == job_id).first()
    
    if not summary:
        summary = OverwatchSummary(
            job_id=job_id,
            risk_level="low",
            risk_score=0.0,
            total_interventions=0,
            warnings_count=0,
            blocks_count=0,
            escalated=False,
            hard_stopped=False,
            current_strikes=0,
            max_strikes_hit=False,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
    
    return summary


def record_overwatch_intervention(
    db: Session,
    job_id: str,
    intervention_type: str,  # warning, block, escalate
    reason: str,
    error_signature: Optional[str] = None,
) -> Optional[OverwatchSummary]:
    """Record an Overwatcher intervention."""
    if not _MODELS_AVAILABLE:
        return None
    
    summary = get_or_create_overwatch_summary(db, job_id)
    if not summary:
        return None
    
    summary.total_interventions += 1
    
    if intervention_type == "warning":
        summary.warnings_count += 1
    elif intervention_type == "block":
        summary.blocks_count += 1
        summary.current_strikes += 1
        
        # Track strike signatures
        if error_signature:
            sigs = summary.strike_signatures or []
            sigs.append(error_signature)
            summary.strike_signatures = sigs
        
        # Check for max strikes
        if summary.current_strikes >= 3:
            summary.max_strikes_hit = True
            summary.hard_stopped = True
            summary.risk_level = "critical"
    elif intervention_type == "escalate":
        summary.escalated = True
        summary.escalation_reason = reason
        summary.risk_level = "high"
    
    # Update issue types
    issue_types = summary.issue_types or {}
    issue_types[intervention_type] = issue_types.get(intervention_type, 0) + 1
    summary.issue_types = issue_types
    
    summary.updated_at = _utc_now()
    
    # Write to ledger
    if _LEDGER_AVAILABLE:
        ledger_append(
            job_artifact_root=_artifact_root(),
            job_id=job_id,
            event={
                "event": "OVERWATCH_INTERVENTION",
                "job_id": job_id,
                "intervention_type": intervention_type,
                "reason": reason,
                "error_signature": error_signature,
                "current_strikes": summary.current_strikes,
                "hard_stopped": summary.hard_stopped,
                "ts": _utc_ts(),
            },
        )
    
    db.commit()
    return summary


# =============================================================================
# GLOBAL PREFERENCES
# =============================================================================

def set_global_pref(
    db: Session,
    key: str,
    value: str,
    category: str = "preference",
    source: str = "user_declared",
    applies_to: Optional[str] = None,
) -> Optional[GlobalPref]:
    """Set or update a global preference."""
    if not _MODELS_AVAILABLE:
        return None
    
    pref = db.query(GlobalPref).filter(GlobalPref.key == key).first()
    
    if pref:
        pref.value = value
        pref.category = category
        pref.source = source
        pref.applies_to = applies_to
        pref.updated_at = _utc_now()
    else:
        pref = GlobalPref(
            key=key,
            value=value,
            category=category,
            source=source,
            applies_to=applies_to,
            active=True,
        )
        db.add(pref)
    
    db.commit()
    return pref


def get_global_pref(db: Session, key: str) -> Optional[str]:
    """Get a global preference value."""
    if not _MODELS_AVAILABLE:
        return None
    
    pref = db.query(GlobalPref).filter(GlobalPref.key == key, GlobalPref.active == True).first()
    return pref.value if pref else None


def get_prefs_for_component(db: Session, component: str) -> List[GlobalPref]:
    """Get all active preferences that apply to a component."""
    if not _MODELS_AVAILABLE:
        return []
    
    return (
        db.query(GlobalPref)
        .filter(
            GlobalPref.active == True,
            (GlobalPref.applies_to == component) | (GlobalPref.applies_to == "all") | (GlobalPref.applies_to == None),
        )
        .all()
    )


# =============================================================================
# OVERWATCHER PATTERNS (Cross-AstraJob)
# =============================================================================

def record_overwatch_pattern(
    db: Session,
    pattern_type: str,
    job_id: str,
    target_path: Optional[str] = None,
    target_model: Optional[str] = None,
    error_signature: Optional[str] = None,
    severity: str = "info",
) -> Optional[OverwatchPattern]:
    """Record or update an Overwatcher pattern."""
    if not _MODELS_AVAILABLE:
        return None
    
    # Find existing pattern
    query = db.query(OverwatchPattern).filter(OverwatchPattern.pattern_type == pattern_type)
    
    if target_path:
        query = query.filter(OverwatchPattern.target_path == target_path)
    if target_model:
        query = query.filter(OverwatchPattern.target_model == target_model)
    if error_signature:
        query = query.filter(OverwatchPattern.error_signature == error_signature)
    
    pattern = query.first()
    
    if pattern:
        pattern.occurrence_count += 1
        pattern.last_occurrence = _utc_now()
        
        # Add AstraJob to list
        job_ids = pattern.job_ids or []
        if job_id not in job_ids:
            job_ids.append(job_id)
        pattern.job_ids = job_ids
        
        # Escalate severity if repeated
        if pattern.occurrence_count >= 3 and pattern.severity == "info":
            pattern.severity = "warn"
        if pattern.occurrence_count >= 5 and pattern.severity == "warn":
            pattern.severity = "error"
            pattern.action = "require_review"
    else:
        pattern = OverwatchPattern(
            pattern_type=pattern_type,
            target_path=target_path,
            target_model=target_model,
            error_signature=error_signature,
            occurrence_count=1,
            last_occurrence=_utc_now(),
            job_ids=[job_id],
            severity=severity,
            first_seen=_utc_now(),
        )
        db.add(pattern)
    
    db.commit()
    return pattern


def get_patterns_for_file(db: Session, path: str) -> List[OverwatchPattern]:
    """Get all patterns for a file path."""
    if not _MODELS_AVAILABLE:
        return []
    
    return (
        db.query(OverwatchPattern)
        .filter(OverwatchPattern.target_path == path)
        .order_by(OverwatchPattern.occurrence_count.desc())
        .all()
    )


# =============================================================================
# QUERY HELPERS
# =============================================================================

def get_job(db: Session, job_id: str) -> Optional[AstraJob]:
    """Get a AstraJob by ID."""
    if not _MODELS_AVAILABLE:
        return None
    return db.query(AstraJob).filter(AstraJob.job_id == job_id).first()


def get_jobs_by_status(db: Session, status: str, limit: int = 100) -> List[AstraJob]:
    """Get jobs by status."""
    if not _MODELS_AVAILABLE:
        return []
    return (
        db.query(AstraJob)
        .filter(AstraJob.status == status)
        .order_by(AstraJob.created_at.desc())
        .limit(limit)
        .all()
    )


def get_escalated_jobs(db: Session, limit: int = 100) -> List[AstraJob]:
    """Get jobs where Overwatcher escalated."""
    if not _MODELS_AVAILABLE:
        return []
    
    return (
        db.query(AstraJob)
        .join(OverwatchSummary)
        .filter(OverwatchSummary.escalated == True)
        .order_by(AstraJob.created_at.desc())
        .limit(limit)
        .all()
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # AstraJob lifecycle
    "create_job",
    "update_job_status",
    "link_spec_to_job",
    "link_arch_to_job",
    # File tracking
    "record_file_touch",
    "get_files_for_job",
    "get_jobs_for_file",
    # Event projection
    "project_event_to_db",
    "get_events_for_job",
    # Chunk tracking
    "create_chunk",
    "update_chunk_status",
    # Overwatcher
    "get_or_create_overwatch_summary",
    "record_overwatch_intervention",
    "record_overwatch_pattern",
    "get_patterns_for_file",
    # Global prefs
    "set_global_pref",
    "get_global_pref",
    "get_prefs_for_component",
    # Queries
    "get_job",
    "get_jobs_by_status",
    "get_escalated_jobs",
]
