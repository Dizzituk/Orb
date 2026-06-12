# FILE: app/transparency/models.py
# Purpose: SQLAlchemy models for Pipeline Transparency system.
# Called-by: app.builds.service, app.db, app.transparency.collector, app.transparency.corrections (+1 more)
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
SQLAlchemy models for Pipeline Transparency system.

Tables:
- reasoning_events: Full reasoning traces from pipeline stages
- user_corrections: User feedback pinned to specific reasoning events

v1.0 (2026-02): Initial implementation
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, Integer, Float, DateTime, JSON
from app.db import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class ReasoningEventModel(Base):
    """Persisted reasoning trace from a pipeline stage."""
    __tablename__ = "reasoning_events"

    event_id = Column(String, primary_key=True, default=_uuid)
    job_id = Column(String, nullable=True, index=True)
    run_id = Column(String, nullable=True, index=True)
    build_project_id = Column(String, nullable=True, index=True)

    # Stage info
    stage_name = Column(String(50), nullable=False, index=True)
    stage_index = Column(Integer, default=0)

    # Status
    status = Column(String(20), nullable=False, default="running")
    confidence_score = Column(Float, default=0.0)

    # Reasoning content
    reasoning_summary = Column(Text, nullable=True)
    reasoning_detail = Column(Text, nullable=True)

    # Data operations (stored as JSON)
    evidence_sources = Column(JSON, nullable=True, default=list)
    decisions = Column(JSON, nullable=True, default=list)

    # Cost/performance
    model_used = Column(String(100), nullable=True)
    token_cost_usd = Column(Float, default=0.0)
    duration_ms = Column(Integer, default=0)

    # Extra metadata
    metadata_json = Column(JSON, nullable=True, default=dict)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=_now)


class UserCorrectionModel(Base):
    """User feedback pinned to a reasoning event."""
    __tablename__ = "user_corrections"

    correction_id = Column(String, primary_key=True, default=_uuid)
    reasoning_event_id = Column(String, nullable=False, index=True)
    job_id = Column(String, nullable=True, index=True)
    run_id = Column(String, nullable=True)
    build_project_id = Column(String, nullable=True, index=True)

    # What was corrected
    stage_name = Column(String(50), nullable=False, index=True)
    stage_index = Column(Integer, default=0)
    decision_index = Column(Integer, nullable=True)

    # The correction
    user_comment = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, default="note")
    correction_type = Column(String(30), nullable=False, default="wrong_decision")

    # For matching future similar contexts
    context_keywords = Column(JSON, nullable=True, default=list)

    # Timestamps
    created_at = Column(DateTime, nullable=False, default=_now)
