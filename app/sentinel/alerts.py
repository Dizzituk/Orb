# FILE: app/sentinel/alerts.py
# Purpose: Sentinel alert store + lifecycle — DB rows, nudges-style active-alerts JSON
#          mirror with injection cooldown, dedupe so one anomaly never spams.
# Called-by: app.sentinel.collector, app.sentinel.router, app.sentinel.tools, app.llm.routing.memory_injection
# Depends-on: app.sentinel.models, app.db (nudges reference: app/lifestyle/nudges.py)
# Last-renovated: 2026-07-01
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.sentinel.models import SentinelAlert

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "sentinel"
_ACTIVE_FILE = _DATA_DIR / "alerts_active.json"

_INJECTION_COOLDOWN_HOURS = float(os.getenv("ASTRA_SENTINEL_INJECTION_COOLDOWN_HOURS", "2"))
_DEDUPE_WINDOW_HOURS = 6
_SEVERITY_ORDER = {"severe": 0, "medium": 1, "low": 2}


def _now() -> datetime:
    return datetime.now().astimezone()


def alert_to_dict(alert: SentinelAlert) -> Dict[str, Any]:
    return {
        "id": alert.id,
        "created_at": alert.created_at.isoformat() if alert.created_at else "",
        "severity": alert.severity,
        "rule_key": alert.rule_key,
        "title": alert.title,
        "explanation": alert.explanation,
        "process": alert.process,
        "remote": alert.remote,
        "recommended_action": alert.recommended_action,
        "confidence": alert.confidence,
        "acknowledged": bool(alert.acknowledged),
        "action_taken": alert.action_taken,
        "suppressed": bool(alert.suppressed),
    }


# =============================================================================
# Create / lifecycle
# =============================================================================

def create_alert(
    db: Session,
    *,
    severity: str,
    rule_key: str,
    title: str,
    explanation: str,
    process: str = "",
    remote: str = "",
    recommended_action: str = "watch",
    confidence: Optional[float] = None,
    suppressed: bool = False,
) -> Optional[SentinelAlert]:
    """Insert an alert unless the same (rule_key, process, remote) fired recently."""
    cutoff = datetime.utcnow() - timedelta(hours=_DEDUPE_WINDOW_HOURS)
    duplicate = (
        db.query(SentinelAlert.id)
        .filter(
            SentinelAlert.rule_key == rule_key,
            SentinelAlert.process == (process or ""),
            SentinelAlert.remote == (remote or ""),
            SentinelAlert.created_at >= cutoff,
        )
        .first()
    )
    if duplicate is not None:
        logger.debug("[sentinel] alert deduped: %s %s %s", rule_key, process, remote)
        return None
    alert = SentinelAlert(
        severity=severity if severity in _SEVERITY_ORDER else "low",
        rule_key=rule_key,
        title=title[:300],
        explanation=explanation,
        process=process or "",
        remote=remote or "",
        recommended_action=recommended_action,
        confidence=confidence,
        suppressed=suppressed,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    if not suppressed:
        refresh_active_file(db)
        logger.info("[sentinel] ALERT [%s] %s", severity.upper(), title)
    return alert


def ack_alert(db: Session, alert_id: int) -> Optional[SentinelAlert]:
    alert = db.query(SentinelAlert).filter(SentinelAlert.id == alert_id).first()
    if alert is None:
        return None
    alert.acknowledged = True
    db.commit()
    refresh_active_file(db)
    return alert


def set_action_taken(db: Session, alert_id: int, action: str) -> Optional[SentinelAlert]:
    alert = db.query(SentinelAlert).filter(SentinelAlert.id == alert_id).first()
    if alert is None:
        return None
    alert.action_taken = action[:300]
    db.commit()
    refresh_active_file(db)
    return alert


def mark_action_for_remote(db: Session, ip: str, action: str) -> int:
    """Stamp action_taken on every unacked alert whose remote involves this IP."""
    rows = (
        db.query(SentinelAlert)
        .filter(SentinelAlert.acknowledged.is_(False), SentinelAlert.remote.like(f"{ip}%"))
        .all()
    )
    for row in rows:
        row.action_taken = action[:300]
    if rows:
        db.commit()
        refresh_active_file(db)
    return len(rows)


# =============================================================================
# Queries
# =============================================================================

def list_alerts(
    db: Session,
    *,
    unacked_only: bool = False,
    include_suppressed: bool = False,
    min_severity: Optional[str] = None,
    limit: int = 100,
) -> List[SentinelAlert]:
    q = db.query(SentinelAlert)
    if unacked_only:
        q = q.filter(SentinelAlert.acknowledged.is_(False))
    if not include_suppressed:
        q = q.filter(SentinelAlert.suppressed.is_(False))
    if min_severity in _SEVERITY_ORDER:
        floor = _SEVERITY_ORDER[min_severity]
        allowed = [sev for sev, order in _SEVERITY_ORDER.items() if order <= floor]
        q = q.filter(SentinelAlert.severity.in_(allowed))
    return list(q.order_by(SentinelAlert.created_at.desc()).limit(max(1, min(limit, 500))))


def get_alert(db: Session, alert_id: int) -> Optional[SentinelAlert]:
    return db.query(SentinelAlert).filter(SentinelAlert.id == alert_id).first()


def alert_counts(db: Session) -> Dict[str, int]:
    base = db.query(SentinelAlert)
    unacked = base.filter(SentinelAlert.acknowledged.is_(False), SentinelAlert.suppressed.is_(False))
    return {
        "total": base.count(),
        "unacked": unacked.count(),
        "unacked_severe": unacked.filter(SentinelAlert.severity == "severe").count(),
        "suppressed": base.filter(SentinelAlert.suppressed.is_(True)).count(),
    }


# =============================================================================
# Active-alerts mirror + chat injection (nudges pattern)
# =============================================================================

def _load_active() -> Dict[str, Any]:
    try:
        if _ACTIVE_FILE.exists():
            return json.loads(_ACTIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def refresh_active_file(db: Session) -> None:
    """Mirror unacked, unsuppressed alerts to JSON, preserving the injection stamp."""
    previous = _load_active()
    items = [
        {
            "id": a.id,
            "severity": a.severity,
            "title": a.title,
            "explanation": (a.explanation or "")[:400],
            "created_at": a.created_at.isoformat() if a.created_at else "",
        }
        for a in list_alerts(db, unacked_only=True, limit=20)
    ]
    payload = {
        "generated_at": _now().isoformat(),
        "items": items,
        "last_injected_at": previous.get("last_injected_at"),
    }
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _ACTIVE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("[sentinel] active-alerts mirror write failed: %s", exc)


def get_alert_for_injection() -> Optional[str]:
    """Highest-severity active alert for prompt injection, honouring a cooldown so
    it appears in one conversation turn, not every message. Stamps the read.
    Mirrors app/lifestyle/nudges.get_nudge_for_injection."""
    data = _load_active()
    items = data.get("items") or []
    if not items:
        return None
    last = data.get("last_injected_at")
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            if _now() - last_dt < timedelta(hours=_INJECTION_COOLDOWN_HOURS):
                return None
        except ValueError:
            pass
    items.sort(key=lambda i: (_SEVERITY_ORDER.get(i.get("severity"), 3), -(i.get("id") or 0)))
    top = items[0]
    data["last_injected_at"] = _now().isoformat()
    try:
        _ACTIVE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
    extra = len(items) - 1
    text = (
        f"{str(top.get('severity', '')).upper()} alert #{top.get('id')}: {top.get('title')} "
        f"— {top.get('explanation')}"
    )
    if extra > 0:
        text += f" (+{extra} more unacknowledged in the Security tab)"
    return text
