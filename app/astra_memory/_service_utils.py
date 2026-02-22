from __future__ import annotations
import hashlib
from sqlalchemy.orm import Session
from typing import List, Optional
_MODELS_AVAILABLE = True


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
