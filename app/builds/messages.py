# FILE: app/builds/messages.py
"""
Per-project pipeline message log.

Stores the pipeline output stream per build project so each
project's workspace can show its own chat/output history
independent of the main chat window.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy import Column, String, Text, DateTime, Integer
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db import Base

logger = logging.getLogger(__name__)


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


# ── DB Model ──

class BuildProjectMessage(Base):
    """A single message in a build project's pipeline log."""
    __tablename__ = "build_project_messages"

    id = Column(String, primary_key=True, default=_uuid)
    build_project_id = Column(String, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "system", "assistant", "pipeline"
    stage = Column(String(50), nullable=True)  # "weaver", "spec_gate", etc.
    content = Column(Text, nullable=False)
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)


# ── Pydantic Schemas ──

class BuildMessageResponse(BaseModel):
    id: str
    build_project_id: str
    role: str
    stage: Optional[str] = None
    content: str
    provider: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Service Functions ──

def add_message(
    db: Session,
    build_project_id: str,
    role: str,
    content: str,
    stage: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BuildProjectMessage:
    """Add a message to a build project's log."""
    msg = BuildProjectMessage(
        build_project_id=build_project_id,
        role=role,
        stage=stage,
        content=content,
        provider=provider,
        model=model,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


def get_messages(
    db: Session,
    build_project_id: str,
    limit: int = 200,
) -> List[BuildProjectMessage]:
    """Get all messages for a build project, ordered chronologically."""
    return (
        db.query(BuildProjectMessage)
        .filter(BuildProjectMessage.build_project_id == build_project_id)
        .order_by(BuildProjectMessage.created_at.asc())
        .limit(limit)
        .all()
    )
