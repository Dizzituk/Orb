# FILE: app/memory/conversation_service.py
"""
Service layer for the Conversational Memory Layer.

Manages ConversationSession lifecycle and ConversationSummary CRUD.
All DB operations go through this module — no direct ORM access elsewhere.

Key design decisions:
  - One active session per project at a time.
  - Session auto-creates on first message if none exists.
  - Activity bump is a separate call (not implicit on every read).
  - Summary versioning is append-only; latest version wins.

Part of CONV-MEMORY-001: Phase 1 (Session Model + Storage).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, List

from sqlalchemy.orm import Session as DbSession

from app.memory.conversation_models import (
    ConversationSession,
    ConversationSummary,
)
from app.memory.conversation_schemas import (
    SessionCreate,
    SessionOut,
    SessionUpdate,
    SummaryCreate,
    SummaryOut,
    SessionWithSummary,
)

logger = logging.getLogger(__name__)

# =========================================================================
# Constants
# =========================================================================

# Session transitions to 'idle' after this much inactivity
IDLE_THRESHOLD = timedelta(hours=2)

# Session transitions to 'archived' after this much inactivity
ARCHIVE_THRESHOLD = timedelta(hours=48)


# =========================================================================
# Session: create / get / update
# =========================================================================

def create_session(db: DbSession, data: SessionCreate) -> ConversationSession:
    """
    Create a new active session for a project.

    Does NOT check for existing active sessions — the caller
    (get_or_create_active_session) handles that.
    """
    session = ConversationSession(
        project_id=data.project_id,
        status="active",
        message_count=0,
        models_used="",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    logger.info(
        "[conv_service] Created session %d for project %d",
        session.id, data.project_id,
    )
    return session


def get_active_session(
    db: DbSession, project_id: int,
) -> Optional[ConversationSession]:
    """
    Return the active session for a project, or None.

    If the session's last_activity_at exceeds IDLE_THRESHOLD, it is NOT
    automatically transitioned here — that's the lifecycle job's
    responsibility (Phase 5). This keeps reads fast and side-effect-free.
    """
    return (
        db.query(ConversationSession)
        .filter(
            ConversationSession.project_id == project_id,
            ConversationSession.status == "active",
        )
        .order_by(ConversationSession.last_activity_at.desc())
        .first()
    )


def get_or_create_active_session(
    db: DbSession, project_id: int,
) -> ConversationSession:
    """
    Get the current active session, or create one.

    If an active session exists but is stale (past IDLE_THRESHOLD),
    transition it to 'idle' and create a fresh one. This provides
    inline staleness detection without waiting for the background job.
    """
    existing = get_active_session(db, project_id)

    if existing is not None:
        age = datetime.utcnow() - existing.last_activity_at
        if age < IDLE_THRESHOLD:
            return existing

        # Session is stale — transition and create new
        logger.info(
            "[conv_service] Session %d stale (%.1f hrs), transitioning to idle",
            existing.id, age.total_seconds() / 3600,
        )
        existing.status = "idle"
        db.commit()

    # Create fresh session
    return create_session(db, SessionCreate(project_id=project_id))


def get_session(db: DbSession, session_id: int) -> Optional[ConversationSession]:
    """Get a session by ID (any status)."""
    return (
        db.query(ConversationSession)
        .filter(ConversationSession.id == session_id)
        .first()
    )


def bump_session_activity(
    db: DbSession,
    session: ConversationSession,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Update session after a new message.

    Increments message_count, refreshes last_activity_at,
    and appends the model identifier if not already tracked.
    """
    session.message_count += 1
    session.last_activity_at = datetime.utcnow()

    # Track model usage
    if provider and model:
        model_key = f"{provider}:{model}"
        existing = set(
            m.strip() for m in session.models_used.split(",") if m.strip()
        )
        if model_key not in existing:
            existing.add(model_key)
            session.models_used = ",".join(sorted(existing))

    db.commit()


def update_session_status(
    db: DbSession, session_id: int, new_status: str,
) -> Optional[ConversationSession]:
    """
    Transition a session to a new status.

    Valid transitions:
        active  → idle
        active  → archived  (skip idle, e.g. explicit close)
        idle    → archived
        idle    → active    (user resumes after short break)
    """
    valid_transitions = {
        "active": {"idle", "archived"},
        "idle": {"archived", "active"},
    }

    session = get_session(db, session_id)
    if session is None:
        logger.warning("[conv_service] Session %d not found", session_id)
        return None

    allowed = valid_transitions.get(session.status, set())
    if new_status not in allowed:
        logger.warning(
            "[conv_service] Invalid transition: %s → %s for session %d",
            session.status, new_status, session_id,
        )
        return None

    session.status = new_status
    db.commit()
    db.refresh(session)
    logger.info(
        "[conv_service] Session %d transitioned to '%s'",
        session_id, new_status,
    )
    return session


def list_sessions(
    db: DbSession,
    project_id: int,
    status: Optional[str] = None,
    limit: int = 20,
) -> List[ConversationSession]:
    """List sessions for a project, optionally filtered by status."""
    query = db.query(ConversationSession).filter(
        ConversationSession.project_id == project_id,
    )
    if status:
        query = query.filter(ConversationSession.status == status)
    return (
        query.order_by(ConversationSession.last_activity_at.desc())
        .limit(limit)
        .all()
    )


# =========================================================================
# Summary: create / get / list
# =========================================================================

def create_summary(db: DbSession, data: SummaryCreate) -> ConversationSummary:
    """
    Create a new summary version for a session.

    Automatically sets version = max(existing) + 1.
    """
    # Get the latest version number
    latest = get_latest_summary(db, data.session_id)
    next_version = (latest.version + 1) if latest else 1

    summary = ConversationSummary(
        session_id=data.session_id,
        project_id=data.project_id,
        version=next_version,
        summarised_through_message_id=data.summarised_through_message_id,
        summary_json=data.summary_json,
        token_estimate=data.token_estimate,
    )
    db.add(summary)
    db.commit()
    db.refresh(summary)
    logger.info(
        "[conv_service] Created summary v%d for session %d (through msg %s)",
        next_version, data.session_id, data.summarised_through_message_id,
    )
    return summary


def get_latest_summary(
    db: DbSession, session_id: int,
) -> Optional[ConversationSummary]:
    """Get the highest-version summary for a session."""
    return (
        db.query(ConversationSummary)
        .filter(ConversationSummary.session_id == session_id)
        .order_by(ConversationSummary.version.desc())
        .first()
    )


def get_latest_summary_for_project(
    db: DbSession, project_id: int,
) -> Optional[ConversationSummary]:
    """
    Get the most recent summary across all active/idle sessions
    for a project. Used by prompt_builders for context injection.
    """
    return (
        db.query(ConversationSummary)
        .join(ConversationSession)
        .filter(
            ConversationSummary.project_id == project_id,
            ConversationSession.status.in_(["active", "idle"]),
        )
        .order_by(ConversationSummary.updated_at.desc())
        .first()
    )


def list_summaries(
    db: DbSession, session_id: int,
) -> List[ConversationSummary]:
    """List all summary versions for a session, newest first."""
    return (
        db.query(ConversationSummary)
        .filter(ConversationSummary.session_id == session_id)
        .order_by(ConversationSummary.version.desc())
        .all()
    )


# =========================================================================
# Composite: session + summary together
# =========================================================================

def get_session_with_summary(
    db: DbSession, project_id: int,
) -> Optional[SessionWithSummary]:
    """
    Convenience method: get the active session and its latest summary.
    Returns None if no active session exists.
    """
    session = get_active_session(db, project_id)
    if session is None:
        return None

    latest = get_latest_summary(db, session.id)

    return SessionWithSummary(
        session=SessionOut.model_validate(session),
        latest_summary=(
            SummaryOut.model_validate(latest) if latest else None
        ),
    )
