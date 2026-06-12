# FILE: app/sentinel/baseline.py
# Purpose: Learned-normality store + sentinel key/value state + learn-mode logic
#          (14 CALENDAR days from first boot, wall-clock dates, not runtime).
# Called-by: app.sentinel.collector, app.sentinel.router, app.sentinel.tools, app.sentinel.scheduler
# Depends-on: app.sentinel.models, app.db
# Last-renovated: 2026-06-12
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional, Set, Tuple

from sqlalchemy.orm import Session

from app.sentinel.models import SentinelBaseline, SentinelState

logger = logging.getLogger(__name__)

LEARN_MODE_DAYS = 14
STALE_BASELINE_DAYS = 90  # untrusted pairs unseen this long get pruned by maintenance

_KEY_LEARN_STARTED = "learn_mode_started_at"


# =============================================================================
# Key/value state
# =============================================================================

def get_state(db: Session, key: str) -> Optional[str]:
    row = db.query(SentinelState).filter(SentinelState.key == key).first()
    return row.value if row else None


def set_state(db: Session, key: str, value: str) -> None:
    row = db.query(SentinelState).filter(SentinelState.key == key).first()
    if row:
        row.value = value
        row.updated_at = datetime.utcnow()
    else:
        db.add(SentinelState(key=key, value=value))
    db.commit()


# =============================================================================
# Learn mode — calendar days, stamped once
# =============================================================================

def _today_local() -> date:
    return datetime.now().astimezone().date()


def ensure_learn_mode_started(db: Session) -> date:
    """Stamp learn_mode_started_at exactly once; return the start date."""
    raw = get_state(db, _KEY_LEARN_STARTED)
    if raw:
        try:
            return date.fromisoformat(raw.strip())
        except ValueError:
            logger.warning("[sentinel] bad %s value %r — restamping", _KEY_LEARN_STARTED, raw)
    started = _today_local()
    set_state(db, _KEY_LEARN_STARTED, started.isoformat())
    logger.info("[sentinel] learn mode started %s (%s calendar days)", started, LEARN_MODE_DAYS)
    return started


def learn_mode_started_at(db: Session) -> Optional[date]:
    raw = get_state(db, _KEY_LEARN_STARTED)
    if not raw:
        return None
    try:
        return date.fromisoformat(raw.strip())
    except ValueError:
        return None


def learn_mode_days_remaining(db: Session) -> int:
    """Whole days of learn mode left; 0 once normal operation begins."""
    started = learn_mode_started_at(db)
    if started is None:
        return LEARN_MODE_DAYS
    elapsed = (_today_local() - started).days
    return max(0, LEARN_MODE_DAYS - elapsed)


def learn_mode_active(db: Session) -> bool:
    return learn_mode_days_remaining(db) > 0


# =============================================================================
# Baseline observe / query / trust
# =============================================================================

def observe(
    db: Session,
    process_name: str,
    remote_key: str,
    remote_port: int,
    kind: str = "outbound",
    country: Optional[str] = None,
) -> Tuple[SentinelBaseline, bool]:
    """Upsert one observed pair. Returns (row, created_now)."""
    row = (
        db.query(SentinelBaseline)
        .filter(
            SentinelBaseline.process_name == process_name,
            SentinelBaseline.remote_key == (remote_key or ""),
            SentinelBaseline.remote_port == int(remote_port or 0),
            SentinelBaseline.kind == kind,
        )
        .first()
    )
    if row:
        row.hit_count = (row.hit_count or 0) + 1
        row.last_seen = datetime.utcnow()
        if country and not row.country:
            row.country = country
        db.commit()
        return row, False
    row = SentinelBaseline(
        process_name=process_name,
        remote_key=remote_key or "",
        remote_port=int(remote_port or 0),
        kind=kind,
        country=country,
    )
    db.add(row)
    db.commit()
    return row, True


def process_known(db: Session, process_name: str) -> bool:
    return (
        db.query(SentinelBaseline.id)
        .filter(
            SentinelBaseline.process_name == process_name,
            SentinelBaseline.kind == "outbound",
        )
        .first()
        is not None
    )


def known_ports_for_process(db: Session, process_name: str) -> Set[int]:
    rows = (
        db.query(SentinelBaseline.remote_port)
        .filter(
            SentinelBaseline.process_name == process_name,
            SentinelBaseline.kind == "outbound",
        )
        .all()
    )
    return {r[0] for r in rows if r[0]}


def known_countries_for_process(db: Session, process_name: str) -> Set[str]:
    rows = (
        db.query(SentinelBaseline.country)
        .filter(
            SentinelBaseline.process_name == process_name,
            SentinelBaseline.kind == "outbound",
            SentinelBaseline.country.isnot(None),
        )
        .all()
    )
    return {r[0] for r in rows if r[0]}


def is_trusted(
    db: Session,
    process_name: str,
    remote_key: Optional[str] = None,
    remote_port: Optional[int] = None,
) -> bool:
    """True if the whole process is trusted, or this exact pair is."""
    q = db.query(SentinelBaseline.id).filter(
        SentinelBaseline.process_name == process_name,
        SentinelBaseline.trusted.is_(True),
    )
    whole = q.filter(SentinelBaseline.remote_key == "*").first()
    if whole is not None:
        return True
    if remote_key is None:
        return False
    return (
        q.filter(
            SentinelBaseline.remote_key == remote_key,
            SentinelBaseline.remote_port == int(remote_port or 0),
        ).first()
        is not None
    )


def trust_process(db: Session, process_name: str) -> None:
    """Trust everything a process does — stored as a '*' wildcard row."""
    row = (
        db.query(SentinelBaseline)
        .filter(
            SentinelBaseline.process_name == process_name,
            SentinelBaseline.remote_key == "*",
        )
        .first()
    )
    if row:
        row.trusted = True
        row.last_seen = datetime.utcnow()
    else:
        db.add(
            SentinelBaseline(
                process_name=process_name, remote_key="*", remote_port=0,
                kind="trust", trusted=True,
            )
        )
    db.commit()
    logger.info("[sentinel] process trusted: %s", process_name)


def trust_pair(db: Session, process_name: str, remote_key: str, remote_port: int) -> None:
    row, _ = observe(db, process_name, remote_key, remote_port, kind="outbound")
    row.trusted = True
    db.commit()
    logger.info("[sentinel] pair trusted: %s -> %s:%s", process_name, remote_key, remote_port)


def baseline_size(db: Session) -> int:
    return db.query(SentinelBaseline).count()


def maintenance(db: Session) -> int:
    """Daily tidy-up: drop untrusted pairs unseen for STALE_BASELINE_DAYS."""
    cutoff = datetime.utcnow() - timedelta(days=STALE_BASELINE_DAYS)
    stale = (
        db.query(SentinelBaseline)
        .filter(
            SentinelBaseline.trusted.is_(False),
            SentinelBaseline.last_seen < cutoff,
        )
        .delete(synchronize_session=False)
    )
    db.commit()
    if stale:
        logger.info("[sentinel] baseline maintenance pruned %d stale pairs", stale)
    return int(stale or 0)
