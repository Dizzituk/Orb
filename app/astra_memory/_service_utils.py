import hashlib
from app.astra_memory.models import AstraJob, GlobalPref, JobFile, OverwatchPattern, OverwatchSummary
from sqlalchemy.orm import Session
from typing import List, Optional
from app.astra_memory.models import AstraJob, GlobalPref, JobEvent, JobFile, OverwatchPattern, OverwatchSummary
from app.pot_spec.ledger_core import append_event as ledger_append
from datetime import datetime
from typing import List, Optional, Tuple


def _hash_content(content: str) -> str:
    """SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()

def get_files_for_job(db: Session, job_id: str) -> List[JobFile]:
    """Get all files touched by a job."""
    if not _MODELS_AVAILABLE:
        return []
    return db.query(JobFile).filter(JobFile.job_id == job_id).all()

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

def get_job(db: Session, job_id: str) -> Optional[AstraJob]:
    """Get a job by ID."""
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

def link_spec_to_job(
    db: Session,
    job_id: str,
    spec_id: str,
    spec_hash: str,
    spec_version: int,
) -> Optional[AstraJob]:
    """Link a PoT spec to a job."""
    if not _MODELS_AVAILABLE:
        return None

    job = db.query(AstraJob).filter(AstraJob.job_id == job_id).first()
    if not job:
        return None

    job.spec_id = spec_id
    job.spec_hash = spec_hash
    job.spec_version = spec_version
    job.updated_at = _utc_now()

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
    return job

def link_arch_to_job(
    db: Session,
    job_id: str,
    arch_id: str,
    arch_hash: str,
    arch_version: int,
) -> Optional[AstraJob]:
    """Link an architecture snapshot to a job."""
    if not _MODELS_AVAILABLE:
        return None

    job = db.query(AstraJob).filter(AstraJob.job_id == job_id).first()
    if not job:
        return None

    job.arch_id = arch_id
    job.arch_hash = arch_hash
    job.arch_version = arch_version
    job.updated_at = _utc_now()

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
    return job

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
    """Query events for a job with optional filters."""
    if not _MODELS_AVAILABLE:
        return []

    query = db.query(JobEvent).filter(JobEvent.job_id == job_id)

    if event_type:
        query = query.filter(JobEvent.event_type == event_type)
    if severity:
        query = query.filter(JobEvent.severity == severity)

    return query.order_by(JobEvent.ts).all()

def get_or_create_overwatch_summary(
    db: Session,
    job_id: str,
) -> Optional[OverwatchSummary]:
    """Get or create Overwatcher summary for a job."""
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

        # Add job to list
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
