# FILE: app/astra_memory/models.py
"""
ASTRA Memory System - SQLAlchemy Models (Job 5)

Three layers of memory:
1. Job-local: Per-job state, execution timeline, Overwatcher verdicts
2. Global brain: Cross-job preferences, patterns, lessons
3. Overwatcher state: Risk scores, intervention history, cross-job patterns

Design:
- SQLite = queryable index (fast filtering, aggregation)
- JSON/NDJSON = append-only ledger (audit, replay, ground truth)
- Atlas links via arch_id and file hashes (no duplication)
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    ForeignKey, DateTime, JSON, Index, UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.db import Base


# =============================================================================
# JOB REGISTRY
# =============================================================================

class AstraJob(Base):
    """
    Central job registry.
    
    Each job has:
    - A PoT spec (spec_id, spec_hash)
    - An architecture snapshot (arch_id, arch_hash)
    - Status tracking
    - Links to execution and Overwatcher data
    """
    __tablename__ = "astra_jobs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), unique=True, nullable=False, index=True)
    
    # Spec reference (from spec_gate)
    spec_id = Column(String(36), nullable=True, index=True)
    spec_hash = Column(String(64), nullable=True)
    spec_version = Column(Integer, nullable=True)
    
    # Architecture reference (from Atlas)
    arch_id = Column(String(36), nullable=True, index=True)
    arch_hash = Column(String(64), nullable=True)
    arch_version = Column(Integer, nullable=True)
    
    # Job metadata
    repo_root = Column(String(512), nullable=True)
    user_intent = Column(Text, nullable=True)  # Original request
    
    # Status tracking
    status = Column(String(32), default="created", nullable=False, index=True)
    # created -> spec_gate -> planning -> executing -> verifying -> completed/failed/aborted
    
    # Model info
    primary_provider = Column(String(50), nullable=True)
    primary_model = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    files = relationship("JobFile", back_populates="job", cascade="all, delete-orphan")
    events = relationship("JobEvent", back_populates="job", cascade="all, delete-orphan")
    overwatch = relationship("OverwatchSummary", back_populates="job", uselist=False, cascade="all, delete-orphan")
    chunks = relationship("JobChunk", back_populates="job", cascade="all, delete-orphan")


# =============================================================================
# JOB FILES (Atlas Link)
# =============================================================================

class JobFile(Base):
    """
    Files touched by a job.
    
    Links job execution to Atlas architecture:
    - Which files were read/modified
    - Content hashes before/after (drift detection)
    - Which chunk made the change
    """
    __tablename__ = "astra_job_files"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), ForeignKey("astra_jobs.job_id"), nullable=False, index=True)
    
    # Atlas reference
    arch_id = Column(String(36), nullable=True, index=True)
    
    # File info
    path = Column(String(512), nullable=False, index=True)
    symbol_name = Column(String(256), nullable=True)  # function/class if known
    
    # Change tracking
    action = Column(String(20), nullable=False)  # read, create, modify, delete
    hash_before = Column(String(64), nullable=True)
    hash_after = Column(String(64), nullable=True)
    
    # Which chunk made this change
    chunk_id = Column(String(36), nullable=True, index=True)
    
    # Timestamps
    touched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("AstraJob", back_populates="files")
    
    __table_args__ = (
        Index("ix_job_files_path_job", "path", "job_id"),
    )


# =============================================================================
# JOB EVENTS (Ledger Index)
# =============================================================================

class JobEvent(Base):
    """
    Queryable index of ledger events.
    
    The NDJSON ledger is the source of truth.
    This table projects key fields for fast querying.
    """
    __tablename__ = "astra_job_events"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), ForeignKey("astra_jobs.job_id"), nullable=False, index=True)
    
    # Event identity
    event_type = Column(String(64), nullable=False, index=True)
    stage = Column(String(32), nullable=True, index=True)  # spec_gate, planning, execution, verify
    
    # Severity/status
    severity = Column(String(16), default="info", nullable=False)  # info, warn, error, critical
    status = Column(String(16), nullable=True)  # ok, failed, blocked, etc.
    
    # Reference to full event in ledger
    ledger_line = Column(Integer, nullable=True)  # Line number in events.ndjson
    
    # Key payload fields (denormalized for querying)
    spec_id = Column(String(36), nullable=True)
    chunk_id = Column(String(36), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    ts = Column(DateTime, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    job = relationship("AstraJob", back_populates="events")
    
    __table_args__ = (
        Index("ix_job_events_type_ts", "event_type", "ts"),
    )


# =============================================================================
# JOB CHUNKS (Execution Timeline)
# =============================================================================

class JobChunk(Base):
    """
    Execution chunks for a job.
    
    Tracks what was planned and what happened:
    - Chunk definition (what to do)
    - Execution result (what happened)
    - Test results
    """
    __tablename__ = "astra_job_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), ForeignKey("astra_jobs.job_id"), nullable=False, index=True)
    chunk_id = Column(String(36), unique=True, nullable=False, index=True)
    
    # Chunk definition
    sequence = Column(Integer, nullable=False)  # Execution order
    target_path = Column(String(512), nullable=True)
    target_symbol = Column(String(256), nullable=True)
    description = Column(Text, nullable=True)
    
    # Execution result
    status = Column(String(32), default="pending", nullable=False)
    # pending -> executing -> completed/failed/blocked
    
    diff_summary = Column(Text, nullable=True)  # Summary of changes
    lines_added = Column(Integer, nullable=True)
    lines_removed = Column(Integer, nullable=True)
    
    # Test results
    tests_run = Column(Integer, nullable=True)
    tests_passed = Column(Integer, nullable=True)
    tests_failed = Column(Integer, nullable=True)
    
    # Model that executed this chunk
    provider = Column(String(50), nullable=True)
    model = Column(String(100), nullable=True)
    
    # Timestamps
    planned_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    job = relationship("AstraJob", back_populates="chunks")


# =============================================================================
# OVERWATCHER SUMMARY (Per-Job)
# =============================================================================

class OverwatchSummary(Base):
    """
    Overwatcher summary for a job.
    
    Aggregates:
    - Risk assessment
    - Intervention count
    - Issue types
    - Escalation status
    """
    __tablename__ = "astra_overwatch_summary"
    
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(36), ForeignKey("astra_jobs.job_id"), unique=True, nullable=False, index=True)
    
    # Risk assessment
    risk_level = Column(String(16), default="low", nullable=False)  # low, medium, high, critical
    risk_score = Column(Float, default=0.0, nullable=False)
    
    # Intervention counts
    total_interventions = Column(Integer, default=0, nullable=False)
    warnings_count = Column(Integer, default=0, nullable=False)
    blocks_count = Column(Integer, default=0, nullable=False)
    
    # Issue breakdown (JSON for flexibility)
    issue_types = Column(JSON, nullable=True)
    # e.g. {"drift": 2, "safety_gap": 1, "test_failure": 3}
    
    # Escalation
    escalated = Column(Boolean, default=False, nullable=False, index=True)
    escalation_reason = Column(Text, nullable=True)
    hard_stopped = Column(Boolean, default=False, nullable=False)
    
    # Strike tracking
    current_strikes = Column(Integer, default=0, nullable=False)
    max_strikes_hit = Column(Boolean, default=False, nullable=False)
    strike_signatures = Column(JSON, nullable=True)  # List of error signatures
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    job = relationship("AstraJob", back_populates="overwatch")


# =============================================================================
# GLOBAL PREFERENCES (Cross-Job)
# =============================================================================

class GlobalPref(Base):
    """
    Cross-job preferences and policies.
    
    Stores:
    - User-declared preferences ("Never use destructive commands")
    - Learned patterns ("Sandbox writes go through patch executor")
    - Quarantined behaviors
    """
    __tablename__ = "astra_global_prefs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Key-value
    key = Column(String(128), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    
    # Metadata
    category = Column(String(32), nullable=False, index=True)
    # preference, policy, pattern, quarantine
    
    source = Column(String(64), nullable=True)
    # user_declared, learned, system_default
    
    # Scope
    applies_to = Column(String(128), nullable=True)
    # e.g. "overwatcher", "spec_gate", "all"
    
    # Active state
    active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =============================================================================
# OVERWATCHER CROSS-JOB PATTERNS
# =============================================================================

class OverwatchPattern(Base):
    """
    Cross-job patterns detected by Overwatcher.
    
    Tracks:
    - Repeated failures on same file/module
    - Model-specific error patterns
    - Historical fragility data
    """
    __tablename__ = "astra_overwatch_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Pattern identity
    pattern_type = Column(String(32), nullable=False, index=True)
    # file_fragility, model_error, repeated_failure
    
    # Target (what the pattern is about)
    target_path = Column(String(512), nullable=True, index=True)
    target_model = Column(String(100), nullable=True, index=True)
    error_signature = Column(String(64), nullable=True, index=True)
    
    # Pattern data
    occurrence_count = Column(Integer, default=1, nullable=False)
    last_occurrence = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Jobs involved
    job_ids = Column(JSON, nullable=True)  # List of job_ids
    
    # Severity
    severity = Column(String(16), default="info", nullable=False)
    
    # Action taken
    action = Column(String(32), nullable=True)
    # warn, require_review, block
    
    # Timestamps
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_overwatch_patterns_target", "pattern_type", "target_path"),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "AstraJob",
    "JobFile",
    "JobEvent",
    "JobChunk",
    "OverwatchSummary",
    "GlobalPref",
    "OverwatchPattern",
]
