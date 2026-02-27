# FILE: app/builds/models.py
"""
Build project DB models.

Tracks projects that go through the ASTRA pipeline:
Weaver → SpecGate → Critical Pipeline → Overwatcher → Implementer.

Each project stores its current pipeline stage and per-stage status
so the UI can visualise progress.
"""

import uuid
import enum
from datetime import datetime, timezone

from sqlalchemy import Column, String, Text, DateTime, Enum as SAEnum, Integer, JSON
from app.db import Base


def _uuid():
    return str(uuid.uuid4())


def _now():
    return datetime.now(timezone.utc)


class BuildStatus(str, enum.Enum):
    """Overall project lifecycle status."""
    active = "active"
    completed = "completed"
    failed = "failed"
    archived = "archived"


class PipelineStage(str, enum.Enum):
    """Which pipeline stage the project is currently at."""
    idle = "idle"                   # Not started — just created / ramble captured
    weaver = "weaver"               # Weaver is building a spec from the ramble
    spec_gate = "spec_gate"         # Spec Gate validation + clarification
    critical_pipeline = "critical_pipeline"  # Architecture + segmentation
    overwatcher = "overwatcher"     # Overwatcher supervision
    implementer = "implementer"     # Implementation / code generation
    complete = "complete"           # Pipeline finished successfully


class StageStatus(str, enum.Enum):
    """Status of an individual pipeline stage."""
    pending = "pending"
    running = "running"
    awaiting_input = "awaiting_input"   # e.g. Spec Gate needs clarification answers
    passed = "passed"
    failed = "failed"
    skipped = "skipped"


class BuildProject(Base):
    """A project being built through the ASTRA pipeline."""
    __tablename__ = "build_projects"

    id = Column(String, primary_key=True, default=_uuid)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(SAEnum(BuildStatus), default=BuildStatus.active, nullable=False)

    # Pipeline tracking
    current_stage = Column(SAEnum(PipelineStage), default=PipelineStage.idle, nullable=False)
    weaver_status = Column(SAEnum(StageStatus), default=StageStatus.pending, nullable=False)
    spec_gate_status = Column(SAEnum(StageStatus), default=StageStatus.pending, nullable=False)
    critical_pipeline_status = Column(SAEnum(StageStatus), default=StageStatus.pending, nullable=False)
    overwatcher_status = Column(SAEnum(StageStatus), default=StageStatus.pending, nullable=False)
    implementer_status = Column(SAEnum(StageStatus), default=StageStatus.pending, nullable=False)

    # Links to pipeline artefacts (spec IDs, job IDs, etc.)
    spec_id = Column(String, nullable=True)
    job_id = Column(String, nullable=True)
    target_path = Column(String, nullable=True)   # Where the output project lives (sandbox path)

    # The original ramble / brief that kicked this off
    original_brief = Column(Text, nullable=True)

    # Stage logs — JSON array of {stage, event, timestamp, detail}
    stage_log = Column(JSON, nullable=True, default=list)

    # Link to the chat project ID used for conversation context
    chat_project_id = Column(Integer, nullable=True)

    # Segment tracking
    total_segments = Column(Integer, default=0)
    completed_segments = Column(Integer, default=0)

    created_at = Column(DateTime, nullable=False, default=_now)
    updated_at = Column(DateTime, nullable=False, default=_now, onupdate=_now)
