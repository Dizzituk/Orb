# FILE: app/voice_ambient/persistence.py
# Purpose: Persist ambient voice turns to the chat messages DB so they appear in
# Called-by: app.invocation.router
# Depends-on: app.memory, app.memory.conversation_models, app.memory.schemas, app.memory.service
# Last-renovated: 2026-06-11
"""
Persist ambient voice turns to the chat messages DB so they appear in
the user's chat history and survive backend restart.

Strategy:
- A single persistent project called "Voice" holds every voice turn.
- On the first turn of a new session, we prepend a visible separator
  message so distinct sessions are still readable when scrolling.
- User + assistant turns get written via memory_service.create_message,
  exactly like a normal /chat call does.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional, Set

from sqlalchemy.orm import Session

from app.memory import service as memory_service
from app.memory import schemas as memory_schemas
# Eagerly import the conversation_sessions model so its table is registered with
# SQLAlchemy metadata before we attempt to write to messages (which has a FK to it)
from app.memory import conversation_models  # noqa: F401

logger = logging.getLogger(__name__)

VOICE_PROJECT_NAME = "Voice"
VOICE_PROJECT_TYPE = "voice"

# Cached project id - populated on first use, invalidated if DB wiped
_cached_project_id: Optional[int] = None
_cache_lock = threading.Lock()

# Which sessions have had their "start of session" marker written
_marked_sessions: Set[str] = set()
_marked_lock = threading.Lock()


def _get_or_create_voice_project(db: Session) -> Optional[int]:
    """Return the single persistent Voice project's id, creating it if missing."""
    global _cached_project_id

    with _cache_lock:
        if _cached_project_id is not None:
            if memory_service.get_project(db, _cached_project_id):
                return _cached_project_id
            _cached_project_id = None

    # Look up by name first - might already exist from a previous backend run
    try:
        existing = memory_service.get_project_by_name(db, VOICE_PROJECT_NAME)
        if existing:
            with _cache_lock:
                _cached_project_id = existing.id
            return existing.id
    except Exception:
        logger.exception("[voice_persistence] get_project_by_name failed")

    # Create it
    try:
        data = memory_schemas.ProjectCreate(
            name=VOICE_PROJECT_NAME,
            description="Ambient voice conversations with Astra.",
            type=VOICE_PROJECT_TYPE,
        )
        project = memory_service.create_project(db, data)
        with _cache_lock:
            _cached_project_id = project.id
        logger.info("[voice_persistence] created 'Voice' project id=%s", project.id)
        return project.id
    except Exception:
        logger.exception("[voice_persistence] create_project failed")
        return None


def _ensure_session_marker(
    db: Session, project_id: int, session_id: str
) -> None:
    """Write a session-start separator if this is the first turn we've seen."""
    with _marked_lock:
        if session_id in _marked_sessions:
            return
        _marked_sessions.add(session_id)

    try:
        marker = (
            f"--- New voice session started "
            f"{datetime.now().strftime('%d %b %Y %I:%M %p')} ---"
        )
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id,
            role="system",
            content=marker,
            provider="voice",
        ))
    except Exception:
        logger.exception("[voice_persistence] session marker write failed")


def record_turn(
    db: Session,
    session_id: str,
    user_text: str,
    assistant_text: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[int]:
    """Write a user+assistant turn pair to the Voice project. Returns project_id."""
    if not (user_text or assistant_text):
        return None

    project_id = _get_or_create_voice_project(db)
    if project_id is None:
        return None

    if session_id:
        _ensure_session_marker(db, project_id, session_id)

    try:
        if user_text:
            memory_service.create_message(db, memory_schemas.MessageCreate(
                project_id=project_id,
                role="user",
                content=user_text,
                provider="voice",
            ))
        if assistant_text:
            memory_service.create_message(db, memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=assistant_text,
                provider=provider or "voice",
                model=model,
            ))
        logger.info(
            "[voice_persistence] wrote turn to 'Voice' project=%s session=%s",
            project_id, (session_id or "-")[:8]
        )
        return project_id
    except Exception:
        logger.exception("[voice_persistence] create_message failed")
        return None


# Backwards-compat alias used elsewhere in the codebase (if any)
def get_or_create_project_id(db: Session, session_id: str) -> Optional[int]:
    return _get_or_create_voice_project(db)