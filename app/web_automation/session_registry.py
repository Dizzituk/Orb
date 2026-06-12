# FILE: app/web_automation/session_registry.py
# Purpose: CRUD helpers for WebSession.
# Called-by: app.content.distribution.browser_analytics.recon, app.content.distribution.browser_analytics.scrape, app.web_automation.bridge, app.web_automation.router (+2 more)
# Depends-on: app.web_automation.models
# Last-renovated: 2026-06-11
"""
CRUD helpers for WebSession.

Kept deliberately thin — this module knows nothing about actions or
the Electron bridge. It just owns session rows in the database.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from app.web_automation.models import WebSession, SessionStatus

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def list_sessions(db: Session, platform: Optional[str] = None) -> List[WebSession]:
    q = db.query(WebSession)
    if platform:
        q = q.filter(WebSession.platform == platform)
    return q.order_by(WebSession.platform.asc()).all()


def get_session(db: Session, session_id: str) -> Optional[WebSession]:
    return db.query(WebSession).filter(WebSession.id == session_id).one_or_none()


def get_session_by_platform(db: Session, platform: str) -> Optional[WebSession]:
    return db.query(WebSession).filter(WebSession.platform == platform).one_or_none()


def get_session_by_partition(db: Session, partition: str) -> Optional[WebSession]:
    return db.query(WebSession).filter(WebSession.partition == partition).one_or_none()


def create_session(
    db: Session,
    *,
    platform: str,
    label: str,
    partition: str,
    landing_url: str,
    purpose: Optional[str] = None,
) -> WebSession:
    """Idempotent — if a session with this platform exists, return it."""
    existing = get_session_by_platform(db, platform)
    if existing:
        return existing
    session = WebSession(
        platform=platform,
        label=label,
        partition=partition,
        landing_url=landing_url,
        purpose=purpose,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def touch_last_used(db: Session, session_id: str) -> None:
    sess = get_session(db, session_id)
    if not sess:
        return
    sess.last_used_at = _now()
    db.commit()


def set_status(
    db: Session,
    session_id: str,
    *,
    status: Optional[SessionStatus] = None,
    current_url: Optional[str] = None,
    current_title: Optional[str] = None,
    last_error: Optional[str] = None,
) -> Optional[WebSession]:
    sess = get_session(db, session_id)
    if not sess:
        return None
    if status is not None:
        sess.status = status
    if current_url is not None:
        sess.current_url = current_url
    if current_title is not None:
        sess.current_title = current_title
    if last_error is not None:
        sess.last_error = last_error or None
    db.commit()
    db.refresh(sess)
    return sess


def update_session(
    db: Session,
    session_id: str,
    *,
    label: Optional[str] = None,
    landing_url: Optional[str] = None,
    purpose: Optional[str] = None,
) -> Optional[WebSession]:
    sess = get_session(db, session_id)
    if not sess:
        return None
    if label is not None:
        sess.label = label
    if landing_url is not None:
        sess.landing_url = landing_url
    if purpose is not None:
        sess.purpose = purpose
    db.commit()
    db.refresh(sess)
    return sess


def delete_session(db: Session, session_id: str) -> bool:
    sess = get_session(db, session_id)
    if not sess:
        return False
    db.delete(sess)
    db.commit()
    return True
