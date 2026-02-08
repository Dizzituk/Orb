# FILE: app/jobs/models.py
"""
Phase 4 Job System - Database Models

SQLAlchemy ORM models for jobs, sessions, artefacts, and scheduled jobs.
Uses EncryptedText/EncryptedJSON for sensitive fields.

Compatible with existing memory system (Projects, Notes, Tasks, Files, Messages).
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean, Float, JSON
from sqlalchemy.orm import relationship
from app.db import Base

# Import encrypted column types for sensitive data
from app.crypto import EncryptedText, EncryptedJSON


class Session(Base):
    """
    Conversation session containing multiple jobs.
    
    Sessions provide context continuity across multiple job executions.
    All jobs in a session can reference shared artefacts and memory.
    """
    __tablename__ = "sessions"
    
    id = Column(String(36), primary_key=True)  # UUID
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Metadata (not encrypted - for search/filtering)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, nullable=False)
    job_count = Column(Integer, default=0, nullable=False)
    total_cost_estimate = Column(Float, default=0.0, nullable=False)
    
    # Session name/description (not encrypted - for display)
    name = Column(String(255), nullable=True)
    
    # Relationships
    project = relationship("Project", backref="sessions")
    jobs = relationship("Job", back_populates="session", cascade="all, delete-orphan", passive_deletes=True)


class Job(Base):
    """
    Single unit of work in Phase 4 job system.
    
    Each job has:
    - Complete envelope specification
    - State tracking (pending  running  succeeded/failed)
    - Routing decision and execution results
    - Tool invocation history
    - Usage metrics
    """
    __tablename__ = "jobs"
    
    # Identifiers
    id = Column(String(36), primary_key=True)  # UUID
    session_id = Column(String(36), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Job specification (not encrypted - for routing/filtering)
    job_spec_version = Column(Integer, default=1, nullable=False)
    job_type = Column(String(50), nullable=False, index=True)
    resolved_job_type = Column(String(50), nullable=True)  # After alias resolution
    importance = Column(String(20), default="medium", nullable=False)
    data_sensitivity = Column(String(30), default="internal", nullable=False)
    
    # State tracking (not encrypted - for monitoring)
    state = Column(String(20), default="pending", nullable=False, index=True)
    triggered_by = Column(String(20), default="user", nullable=False)  # user | scheduler | planner
    parent_job_id = Column(String(36), nullable=True, index=True)  # For planner-spawned jobs
    
    # ENCRYPTED: Job envelope (contains messages, prompts, metadata)
    envelope_json = Column(EncryptedJSON, nullable=False)
    
    # ENCRYPTED: Primary output content
    output_content = Column(EncryptedText, nullable=True)
    output_contract = Column(String(50), default="text_response", nullable=False)
    artefact_id = Column(String(36), nullable=True, index=True)  # Reference to stored artefact
    
    # ENCRYPTED: Routing decision (JSON)
    routing_decision_json = Column(EncryptedJSON, nullable=True)
    
    # ENCRYPTED: Tool invocations (JSON list)
    tool_invocations_json = Column(EncryptedJSON, nullable=True)
    
    # ENCRYPTED: Critique issues (JSON list, if multi-model review)
    critique_issues_json = Column(EncryptedJSON, nullable=True)
    was_reviewed = Column(Boolean, default=False, nullable=False)
    unresolved_blockers = Column(Integer, default=0, nullable=False)
    
    # Usage metrics (not encrypted - for cost tracking)
    total_tokens = Column(Integer, default=0, nullable=False)
    total_cost_estimate = Column(Float, default=0.0, nullable=False)
    
    # ENCRYPTED: Usage details (JSON list of per-model usage)
    usage_metrics_json = Column(EncryptedJSON, nullable=True)
    
    # Error tracking (not encrypted - for monitoring)
    error_type = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # ENCRYPTED: Error details (JSON with stack traces, etc.)
    error_details_json = Column(EncryptedJSON, nullable=True)
    
    # Timing (not encrypted - for monitoring)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Relationships
    session = relationship("Session", back_populates="jobs")
    project = relationship("Project", backref="jobs")


class Artefact(Base):
    """
    Project artefact store for structured outputs.
    
    Stores:
    - Research dossiers
    - Architecture documents
    - Code patch proposals
    - Script proposals
    - Canonical prompts and configs
    
    Supports versioning and concurrency control via etag.
    """
    __tablename__ = "artefacts"
    
    # Identifiers
    id = Column(String(36), primary_key=True)  # UUID
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Artefact metadata (not encrypted - for search/versioning)
    artefact_type = Column(String(50), nullable=False, index=True)
    # Types: research_dossier, architecture_doc, code_patch_proposal, script_proposal, etc.
    
    name = Column(String(255), nullable=False)  # Human-readable name
    version = Column(Integer, default=1, nullable=False)
    etag = Column(String(64), nullable=False)  # For optimistic concurrency control
    
    status = Column(String(20), default="current", nullable=False, index=True)
    # Status: current, superseded, draft
    
    # ENCRYPTED: Artefact content
    content = Column(EncryptedText, nullable=False)
    
    # ENCRYPTED: Metadata (JSON)
    metadata_json = Column(EncryptedJSON, nullable=True)
    
    # Origin tracking (not encrypted)
    created_by_job_id = Column(String(36), nullable=True, index=True)
    supersedes_id = Column(String(36), nullable=True, index=True)  # Previous version
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", backref="artefacts")


class ScheduledJob(Base):
    """
    Periodic job scheduler definitions.
    
    Schedules jobs like:
    - Weekly model capability sync
    - Daily research dossier updates
    - Periodic maintenance tasks
    """
    __tablename__ = "scheduled_jobs"
    
    # Identifiers
    id = Column(String(36), primary_key=True)  # UUID
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Schedule configuration (not encrypted - for scheduler)
    enabled = Column(Boolean, default=True, nullable=False)
    schedule_type = Column(String(20), default="interval", nullable=False)
    # Types: interval, crontab
    
    schedule_spec = Column(String(100), nullable=False)
    # Examples: "every_7_days", "0 9 * * MON" (cron)
    
    # ENCRYPTED: Job template (JobEnvelope as JSON)
    job_template_json = Column(EncryptedJSON, nullable=False)
    
    # Execution tracking (not encrypted)
    last_run_at = Column(DateTime, nullable=True)
    last_run_job_id = Column(String(36), nullable=True)
    next_run_at = Column(DateTime, nullable=True)
    run_count = Column(Integer, default=0, nullable=False)
    failure_count = Column(Integer, default=0, nullable=False)
    
    # Metadata
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", backref="scheduled_jobs")


class ProjectConfigModel(Base):
    """
    Per-project configuration for Phase 4 job system.
    
    Stores overrides for:
    - Allowed/blocked providers
    - Default importance/sensitivity
    - Job type specific settings
    - Feature flags
    """
    __tablename__ = "project_configs"
    
    # Identifiers (one config per project)
    project_id = Column(Integer, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True)
    
    # ENCRYPTED: Configuration JSON
    config_json = Column(EncryptedJSON, nullable=False)
    # Contains: allowed_providers, blocked_providers, default_data_sensitivity,
    #           default_importance, job_type_overrides, local_only, allow_scheduled_jobs
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", backref="phase4_config", uselist=False)


class MigrationMetadata(Base):
    """
    Phase 4 migration tracking.
    
    Records:
    - Which migrations have been applied
    - Migration timestamps
    - Rollback information if needed
    """
    __tablename__ = "phase4_migrations"
    
    id = Column(Integer, primary_key=True)
    migration_name = Column(String(100), unique=True, nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Migration details (not encrypted - for diagnostics)
    description = Column(Text, nullable=True)
    version = Column(String(20), nullable=True)
    
    # Success tracking
    success = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)


# =============================================================================
# MODEL REGISTRY (for capabilities)
# =============================================================================

class ModelCapability(Base):
    """
    Model capability registry.
    
    Stores structured information about each model's:
    - Supported modalities
    - Context limits
    - Tool support
    - Strengths/weaknesses
    - Cost estimates
    
    Can be auto-populated via model_capability_sync jobs.
    """
    __tablename__ = "model_capabilities"
    
    id = Column(Integer, primary_key=True)
    
    # Model identification (not encrypted)
    model_id = Column(String(100), unique=True, nullable=False, index=True)
    provider = Column(String(50), nullable=False, index=True)
    tier = Column(String(10), default="B", nullable=False)  # S | A | B
    
    # Capabilities (not encrypted - for routing)
    modalities = Column(JSON, nullable=False)  # ["text", "image", ...]
    max_context_tokens = Column(Integer, nullable=False)
    max_output_tokens = Column(Integer, nullable=False)
    supports_tools = Column(Boolean, default=False, nullable=False)
    supports_streaming = Column(Boolean, default=True, nullable=False)
    
    # Data constraints (not encrypted)
    allowed_data_sensitivity_max = Column(String(30), default="internal", nullable=False)
    allowed_tools = Column(JSON, nullable=True)  # List of tool names
    forbidden_tools = Column(JSON, nullable=True)  # List of tool names
    
    # Cost (not encrypted - for budgeting)
    approx_cost_per_1k_tokens = Column(Float, default=0.01, nullable=False)
    
    # ENCRYPTED: Strengths and detailed metadata
    strengths_json = Column(EncryptedJSON, nullable=True)
    # Tags: ["agent_coding", "deep_reasoning", "multimodal", ...]
    
    metadata_json = Column(EncryptedJSON, nullable=True)
    # Additional metadata: latency_profile, release_date, deprecation_date, etc.
    
    # Source tracking
    source = Column(String(20), default="manual", nullable=False)  # manual | auto | mixed
    last_synced_at = Column(DateTime, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Session",
    "Job",
    "Artefact",
    "ScheduledJob",
    "ProjectConfigModel",
    "MigrationMetadata",
    "ModelCapability",
]