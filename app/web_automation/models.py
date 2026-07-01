# FILE: app/web_automation/models.py
# Purpose: Web Automation ORM models.
# Called-by: app.web_automation.action_queue, app.web_automation.bridge, app.web_automation.router, app.web_automation.session_registry (+2 more)
# Depends-on: app.db
# Last-renovated: 2026-07-01
"""
Web Automation ORM models.

WebSession — persistent identity for a logged-in browser session
  (e.g. 'facebook_page', 'coursera', 'instagram_astraukai').

WebAction — audit trail of every action dispatched to Electron.
  Persistent because composite actions and debugging benefit from
  being able to replay what happened.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Enum as SAEnum, ForeignKey, String, Text, JSON
from sqlalchemy.orm import relationship

from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SessionStatus(str, enum.Enum):
    idle = "idle"            # Configured but no live Electron view
    opening = "opening"      # Electron was asked to spawn, not yet confirmed
    live = "live"            # Electron has the view mounted
    error = "error"          # Last open attempt failed


class ActionStatus(str, enum.Enum):
    pending = "pending"
    in_flight = "in_flight"   # Delivered to Electron, awaiting result
    completed = "completed"
    failed = "failed"
    timed_out = "timed_out"


class WebSession(Base):
    __tablename__ = "web_sessions"

    id = Column(String, primary_key=True, default=_uuid)
    platform = Column(String, nullable=False, index=True, unique=True)
    label = Column(String, nullable=False)
    # Electron session partition. NOT unique: the media sessions (displays /
    # soundcloud / mixcloud / youtube_watch / streaming) all share
    # "persist:media" by design — one login per site survives restarts.
    # Legacy DBs carry an inline UNIQUE(partition); app.web_automation
    # .migrations rebuilds them at startup.
    partition = Column(String, nullable=False)
    landing_url = Column(String, nullable=False)
    purpose = Column(Text, nullable=True)                     # What ASTRA should use it for
    status = Column(SAEnum(SessionStatus), nullable=False, default=SessionStatus.idle)
    current_url = Column(String, nullable=True)
    current_title = Column(String, nullable=True)
    last_error = Column(Text, nullable=True)
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now, onupdate=_now)

    actions = relationship("WebAction", back_populates="session", cascade="all, delete-orphan")


class WebAction(Base):
    __tablename__ = "web_actions"

    id = Column(String, primary_key=True, default=_uuid)
    session_id = Column(String, ForeignKey("web_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    action_type = Column(String, nullable=False)      # navigate / click / type / screenshot / ...
    payload = Column(JSON, nullable=False, default=dict)
    status = Column(SAEnum(ActionStatus), nullable=False, default=ActionStatus.pending, index=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_now)
    delivered_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    session = relationship("WebSession", back_populates="actions")
