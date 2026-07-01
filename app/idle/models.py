# FILE: app/idle/models.py
# Purpose: SQLAlchemy model for the persistent idle-task ledger (survives restarts).
# Called-by: app.idle.ledger, app.db (init_db model import for create_all)
# Depends-on: app.db.Base
# Last-renovated: 2026-07-01
"""
One row per background task run. The ledger is the idle system's memory:
what ran, what it covered, the input fingerprint it saw (unchanged inputs
mean the next run is skipped), and resumable progress for checkpointed work.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.db import Base

# Status lifecycle: queued -> running -> (paused -> running ...) ->
#                   completed | failed | skipped
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

OPEN_STATUSES = (STATUS_QUEUED, STATUS_RUNNING, STATUS_PAUSED)
TERMINAL_STATUSES = (STATUS_COMPLETED, STATUS_FAILED, STATUS_SKIPPED)


class IdleTaskRecord(Base):
    __tablename__ = "idle_task_ledger"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(String(64), nullable=False, index=True)
    # Canonical identity: task_type + sorted params JSON (hashed when long).
    # Cadence checks and duplicate-suppression key off this.
    task_key = Column(String(512), nullable=False, index=True)
    params_json = Column(Text, nullable=False, default="{}")
    # Digest of the task's inputs at run time; equal digest on the last
    # completed run of the same task_key => this run is recorded as skipped.
    input_fingerprint = Column(String(128), nullable=True)
    # Human-readable statement of what the run covered.
    coverage = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default=STATUS_QUEUED, index=True)
    # Checkpointed unit progress for resumable tasks (JSON dict).
    progress_json = Column(Text, nullable=True)
    result_summary = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    queued_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
