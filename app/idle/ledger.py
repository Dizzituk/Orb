# FILE: app/idle/ledger.py
# Purpose: CRUD + lifecycle transitions for the persistent idle-task ledger.
# Called-by: app.idle.governor, app.idle.router, app.idle.tools_registration
# Depends-on: app.idle.models
# Last-renovated: 2026-07-01
"""
Every write commits immediately — the ledger must survive the nightly PC
shutdown at any point mid-task. All functions take an explicit Session and
never raise on lookup misses (they return None / empty lists).
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy.orm import Session

from app.idle.models import (
    IdleTaskRecord,
    OPEN_STATUSES,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SKIPPED,
)

logger = logging.getLogger(__name__)


def task_key(task_type: str, params: Optional[dict]) -> str:
    """Canonical identity for a (type, params) pair; hashed if oversized."""
    blob = json.dumps(params or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    key = f"{task_type}:{blob}"
    if len(key) > 512:
        key = f"{task_type}:sha256:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"
    return key


def enqueue(
    db: Session,
    task_type: str,
    params: Optional[dict] = None,
    dedupe: bool = True,
) -> Optional[IdleTaskRecord]:
    """Queue a task. With dedupe (default), an already-open row with the same
    task_key absorbs the request and None is returned."""
    key = task_key(task_type, params)
    if dedupe:
        existing = (
            db.query(IdleTaskRecord)
            .filter(IdleTaskRecord.task_key == key, IdleTaskRecord.status.in_(OPEN_STATUSES))
            .first()
        )
        if existing is not None:
            return None
    rec = IdleTaskRecord(
        task_type=task_type,
        task_key=key,
        params_json=json.dumps(params or {}, ensure_ascii=False),
        status=STATUS_QUEUED,
        queued_at=datetime.utcnow(),
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    logger.info("[idle.ledger] queued %s (#%d)", key, rec.id)
    return rec


def next_runnable(db: Session) -> Optional[IdleTaskRecord]:
    """Paused work resumes before new work starts; FIFO within each."""
    rec = (
        db.query(IdleTaskRecord)
        .filter(IdleTaskRecord.status == STATUS_PAUSED)
        .order_by(IdleTaskRecord.queued_at.asc())
        .first()
    )
    if rec is not None:
        return rec
    return (
        db.query(IdleTaskRecord)
        .filter(IdleTaskRecord.status == STATUS_QUEUED)
        .order_by(IdleTaskRecord.queued_at.asc())
        .first()
    )


def last_terminal_run(db: Session, key: str) -> Optional[IdleTaskRecord]:
    """Most recent completed OR skipped run — both count as 'ran' for cadence
    (a skipped run verified its inputs were unchanged)."""
    return (
        db.query(IdleTaskRecord)
        .filter(
            IdleTaskRecord.task_key == key,
            IdleTaskRecord.status.in_((STATUS_COMPLETED, STATUS_SKIPPED)),
        )
        .order_by(IdleTaskRecord.completed_at.desc())
        .first()
    )


def last_completed_fingerprint(db: Session, key: str) -> Optional[str]:
    rec = (
        db.query(IdleTaskRecord)
        .filter(
            IdleTaskRecord.task_key == key,
            IdleTaskRecord.status.in_((STATUS_COMPLETED, STATUS_SKIPPED)),
            IdleTaskRecord.input_fingerprint.isnot(None),
        )
        .order_by(IdleTaskRecord.completed_at.desc())
        .first()
    )
    return rec.input_fingerprint if rec is not None else None


def mark_running(db: Session, rec: IdleTaskRecord) -> None:
    rec.status = STATUS_RUNNING
    if rec.started_at is None:
        rec.started_at = datetime.utcnow()
    db.commit()


def mark_paused(db: Session, rec: IdleTaskRecord) -> None:
    rec.status = STATUS_PAUSED
    db.commit()
    logger.info("[idle.ledger] paused %s (#%d) — user activity checkpoint", rec.task_key, rec.id)


def save_progress(db: Session, rec: IdleTaskRecord, progress: dict) -> None:
    rec.progress_json = json.dumps(progress or {}, ensure_ascii=False)
    db.commit()


def mark_completed(
    db: Session,
    rec: IdleTaskRecord,
    fingerprint: Optional[str] = None,
    coverage: str = "",
    summary: str = "",
) -> None:
    rec.status = STATUS_COMPLETED
    rec.input_fingerprint = fingerprint
    rec.coverage = coverage or rec.coverage
    rec.result_summary = summary
    rec.completed_at = datetime.utcnow()
    db.commit()
    logger.info("[idle.ledger] completed %s (#%d): %s", rec.task_key, rec.id, summary[:160])


def mark_skipped(db: Session, rec: IdleTaskRecord, fingerprint: Optional[str], note: str) -> None:
    rec.status = STATUS_SKIPPED
    rec.input_fingerprint = fingerprint
    rec.result_summary = note
    rec.completed_at = datetime.utcnow()
    db.commit()
    logger.info("[idle.ledger] skipped %s (#%d): %s", rec.task_key, rec.id, note)


def mark_failed(db: Session, rec: IdleTaskRecord, error: str) -> None:
    rec.status = STATUS_FAILED
    rec.error = (error or "")[:2000]
    rec.completed_at = datetime.utcnow()
    db.commit()
    logger.warning("[idle.ledger] FAILED %s (#%d): %s", rec.task_key, rec.id, (error or "")[:200])


def recover_stale_running(db: Session) -> int:
    """Boot recovery: rows left 'running' by a shutdown resume as 'paused'
    (progress_json already holds their last checkpoint)."""
    stale = db.query(IdleTaskRecord).filter(IdleTaskRecord.status == STATUS_RUNNING).all()
    for rec in stale:
        rec.status = STATUS_PAUSED
    if stale:
        db.commit()
        logger.info("[idle.ledger] boot recovery: %d running task(s) -> paused", len(stale))
    return len(stale)


def _retry_cooldown_minutes() -> float:
    import os

    try:
        return float(os.getenv("ASTRA_IDLE_RETRY_COOLDOWN_MINUTES", "60"))
    except Exception:
        return 60.0


def is_due(db: Session, key: str, cadence_hours: float) -> bool:
    last = last_terminal_run(db, key)
    if last is not None and last.completed_at is not None:
        if datetime.utcnow() - last.completed_at < timedelta(hours=cadence_hours):
            return False
    # Failed runs don't satisfy the cadence, but they do impose a cooldown —
    # otherwise a task whose dependency is down (e.g. sandbox offline all
    # night) gets re-queued and re-failed every idle window.
    last_fail = (
        db.query(IdleTaskRecord)
        .filter(IdleTaskRecord.task_key == key, IdleTaskRecord.status == STATUS_FAILED)
        .order_by(IdleTaskRecord.completed_at.desc())
        .first()
    )
    if last_fail is not None and last_fail.completed_at is not None:
        if datetime.utcnow() - last_fail.completed_at < timedelta(minutes=_retry_cooldown_minutes()):
            return False
    return True


def recent_terminal(db: Session, limit: int = 10) -> List[IdleTaskRecord]:
    return (
        db.query(IdleTaskRecord)
        .filter(IdleTaskRecord.status.in_((STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED)))
        .order_by(IdleTaskRecord.completed_at.desc())
        .limit(max(1, min(int(limit), 50)))
        .all()
    )


def open_tasks(db: Session, limit: int = 20) -> List[IdleTaskRecord]:
    return (
        db.query(IdleTaskRecord)
        .filter(IdleTaskRecord.status.in_(OPEN_STATUSES))
        .order_by(IdleTaskRecord.queued_at.asc())
        .limit(max(1, min(int(limit), 50)))
        .all()
    )
