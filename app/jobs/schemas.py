# FILE: app/jobs/schemas.py
"""
Phase 4 Job System - Schemas and Envelopes (SPEC-COMPLIANT)

Job types, error taxonomy, and envelopes matching Phase 4 specification exactly.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS - PHASE 4 SPECIFICATION
# =============================================================================

class JobType(str, Enum):
    """
    Canonical job types from Phase 4 specification.
    
    DO NOT add aliases here. Use job_type_aliases in routing policy for friendly names.
    """
    # Chat and research
    CHAT_SIMPLE = "chat_simple"
    CHAT_RESEARCH = "chat_research"
    DEEP_RESEARCH_TASK = "deep_research_task"
    MODEL_CAPABILITY_SYNC = "model_capability_sync"
    
    # Code tasks
    CODE_SMALL = "code_small"
    CODE_REPO = "code_repo"
    
    # Architecture
    APP_ARCHITECTURE = "app_architecture"
    
    # Vision
    VISION_SIMPLE = "vision_simple"
    VISION_COMPLEX = "vision_complex"
    
    # Audio
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_MEETING = "audio_meeting"
    
    # Video
    VIDEO_SIMPLE = "video_simple"
    VIDEO_ADVANCED = "video_advanced"
    
    # Review and orchestration
    CRITIQUE_REVIEW = "critique_review"
    ORCHESTRATION_PLAN = "orchestration_plan"


class JobState(str, Enum):
    """Job lifecycle states (Phase 4 spec)."""
    PENDING = "pending"
    RUNNING = "running"
    NEEDS_SPEC_CLARIFICATION = "needs_spec_clarification"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class ErrorType(str, Enum):
    """
    Phase 4 error taxonomy - COMPLETE AND FIXED.
    
    All failures in the job engine, provider registry, and tools
    MUST map to exactly one of these error types.
    """
    VALIDATION_ERROR = "VALIDATION_ERROR"
    ROUTING_ERROR = "ROUTING_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    ADMISSION_CONTROL_REJECTED = "ADMISSION_CONTROL_REJECTED"
    CONFLICT_ERROR = "CONFLICT_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class Importance(str, Enum):
    """Job importance levels (affects routing and retry behavior)."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataSensitivity(str, Enum):
    """Data classification for routing constraints."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    HIGHLY_CONFIDENTIAL = "highly_confidential"


class Modality(str, Enum):
    """Input/output modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    CODE = "code"
    STRUCTURED = "structured"


class OutputContract(str, Enum):
    """Expected output shapes."""
    TEXT_RESPONSE = "text_response"
    RESEARCH_DOSSIER = "research_dossier"
    ARCHITECTURE_DOC = "architecture_doc"
    CODE_PATCH_PROPOSAL = "code_patch_proposal"
    PLAN_V1 = "plan_v1"
    MEETING_SUMMARY = "meeting_summary"
    CRITIQUE_REVIEW = "critique_review"
    SCRIPT_PROPOSAL = "script_proposal"


# =============================================================================
# JOB ENVELOPE - PHASE 4 SPECIFICATION
# =============================================================================

class JobBudget(BaseModel):
    """Resource constraints."""
    max_tokens: int = Field(default=8192, ge=1000)
    max_cost_estimate: float = Field(default=1.0, ge=0.0)
    max_wall_time_seconds: int = Field(default=300, ge=10)


class JobEnvelope(BaseModel):
    """
    Complete job specification matching Phase 4 spec.
    
    This is the authoritative structure. Every routing decision,
    tool usage, and safety constraint flows from this envelope.
    """
    # === REQUIRED FIELDS (Phase 4 spec) ===
    
    job_spec_version: int = Field(default=1, ge=1)
    
    # Identifiers
    job_id: Optional[str] = None  # Assigned by system
    session_id: str
    project_id: int
    
    # Job type (from canonical enum)
    job_type: JobType
    
    # Importance and sensitivity
    importance: Importance = Importance.MEDIUM
    data_sensitivity: DataSensitivity = DataSensitivity.INTERNAL
    
    # Modalities
    modalities_in: list[Modality] = Field(default_factory=lambda: [Modality.TEXT])
    
    # Network and tools
    needs_internet: bool = False
    needs_tools: list[str] = Field(default_factory=list)
    allowed_tools: Optional[list[str]] = None
    forbidden_tools: list[str] = Field(default_factory=list)
    
    # Multi-model review
    allow_multi_model_review: bool = False
    max_review_rounds: int = Field(default=1, ge=0, le=3)
    committee_size: int = Field(default=1, ge=1, le=3)
    
    # Budget
    budget: JobBudget = Field(default_factory=JobBudget)
    
    # Output contract
    output_contract: OutputContract = OutputContract.TEXT_RESPONSE
    
    # === PAYLOAD ===
    
    messages: list[dict] = Field(default_factory=list)
    system_prompt: Optional[str] = None
    attachments: Optional[list[dict]] = None
    
    # === METADATA ===
    
    metadata: dict[str, Any] = Field(default_factory=dict)
    triggered_by: str = "user"  # "user" | "scheduler" | "planner"
    parent_job_id: Optional[str] = None
    
    @field_validator("job_type", mode="before")
    @classmethod
    def validate_job_type(cls, v):
        """Ensure job_type is valid."""
        if isinstance(v, str):
            try:
                return JobType(v)
            except ValueError:
                raise ValueError(f"Invalid job_type: '{v}'. Must be one of: {[jt.value for jt in JobType]}")
        return v


# =============================================================================
# VALIDATION
# =============================================================================

class ValidationError(Exception):
    """Validation failed."""
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")


def validate_job_envelope(envelope: JobEnvelope) -> None:
    """
    Validate job envelope against Phase 4 rules.
    
    Raises:
        ValidationError: If validation fails (maps to VALIDATION_ERROR)
    
    Checks:
    - Job type is recognized
    - Modalities are consistent
    - Tool requirements are valid
    - Budget constraints are reasonable
    - Review configuration is valid
    """
    errors = []
    warnings = []
    
    # Check review configuration
    if envelope.allow_multi_model_review:
        if envelope.max_review_rounds > 3:
            errors.append("max_review_rounds cannot exceed 3")
        if envelope.committee_size > 3:
            errors.append("committee_size cannot exceed 3")
        if envelope.importance in (Importance.LOW, Importance.MEDIUM):
            warnings.append("Multi-model review on low/medium importance may be inefficient")
    
    # Check budget
    if envelope.budget.max_tokens < 1000:
        errors.append("budget.max_tokens must be at least 1000")
    if envelope.budget.max_wall_time_seconds < 10:
        errors.append("budget.max_wall_time_seconds must be at least 10")
    if envelope.budget.max_cost_estimate < 0:
        errors.append("budget.max_cost_estimate must be non-negative")
    
    # Check tool conflicts
    if envelope.allowed_tools and envelope.forbidden_tools:
        overlap = set(envelope.allowed_tools) & set(envelope.forbidden_tools)
        if overlap:
            errors.append(f"Tool(s) in both allowed and forbidden: {overlap}")
    
    # Check internet + sensitivity
    if envelope.needs_internet and envelope.data_sensitivity == DataSensitivity.HIGHLY_CONFIDENTIAL:
        warnings.append("Internet access with highly_confidential data - verify intended")
    
    # Check messages present
    if not envelope.messages:
        errors.append("messages list cannot be empty")
    
    # Check modalities consistency
    if Modality.IMAGE in envelope.modalities_in and not envelope.attachments:
        warnings.append("modalities_in includes IMAGE but no attachments provided")
    
    if errors:
        raise ValidationError(errors)
    
    # Log warnings but don't fail
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"[validation] {warning}")


# =============================================================================
# ROUTING DECISION
# =============================================================================

class ModelSelection(BaseModel):
    """Selected model with tier and role."""
    provider: str
    model_id: str
    tier: str  # "S" | "A" | "B"
    role: str  # "architect" | "reviewer" | "arbiter"


class RoutingDecision(BaseModel):
    """Complete routing decision for a job."""
    job_id: str
    job_type: str
    resolved_job_type: str
    
    # Model selections
    architect: ModelSelection
    reviewers: list[ModelSelection] = Field(default_factory=list)
    arbiter: Optional[ModelSelection] = None
    
    # Parameters
    temperature: float = 0.7
    max_tokens: int = 8192
    timeout_seconds: int = 120
    
    # Constraints
    data_sensitivity_constraint: str
    allowed_tools: list[str]
    forbidden_tools: list[str]
    
    # Fallback tracking
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None
    
    decided_at: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# JOB RESULT
# =============================================================================

class ToolInvocation(BaseModel):
    """Tool call record."""
    tool_name: str
    tool_version: str
    invoked_at: datetime
    side_effect_level: str
    ok: bool
    error_message: Optional[str] = None
    duration_ms: int = 0


class CritiqueIssue(BaseModel):
    """Issue raised by reviewer."""
    severity: str  # "blocker" | "major" | "minor" | "style"
    issue_type: str  # "security" | "correctness" | "scalability" | "ux"
    description: str
    fix_hint: Optional[str] = None
    resolved: bool = False
    architect_response: Optional[str] = None


class UsageMetrics(BaseModel):
    """
    Job-level aggregated token usage and cost.
    
    NOTE: This Pydantic model is for job-level aggregate metrics (stored in DB).
    For per-call low-level stats, see the dataclass UsageMetrics in
    app.providers.registry (not stored, used for internal tracking).
    """
    model_id: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_estimate: float = 0.0


class JobResult(BaseModel):
    """Complete job execution result."""
    job_id: str
    session_id: str
    project_id: int
    job_type: str
    
    state: JobState
    
    # Output
    content: str
    output_contract: OutputContract
    artefact_id: Optional[str] = None
    
    # Routing
    routing_decision: RoutingDecision
    
    # Tools
    tools_used: list[ToolInvocation] = Field(default_factory=list)
    
    # Critique
    was_reviewed: bool = False
    critique_issues: list[CritiqueIssue] = Field(default_factory=list)
    unresolved_blockers: int = 0
    
    # Usage
    usage_metrics: list[UsageMetrics] = Field(default_factory=list)
    total_cost_estimate: float = 0.0
    
    # Timing
    started_at: datetime
    completed_at: datetime
    duration_seconds: float = 0.0
    
    # Error
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    error_details: Optional[dict] = None


# =============================================================================
# API MODELS
# =============================================================================

class CreateJobRequest(BaseModel):
    """Request to create job."""
    session_id: Optional[str] = None
    project_id: int
    job_type: str  # String for API flexibility, validated on conversion
    messages: list[dict]
    
    importance: Optional[Importance] = None
    data_sensitivity: Optional[DataSensitivity] = None
    needs_internet: bool = False
    allow_multi_model_review: bool = False
    system_prompt: Optional[str] = None
    attachments: Optional[list[dict]] = None
    metadata: Optional[dict] = None
    
    @field_validator("job_type")
    @classmethod
    def validate_job_type_string(cls, v):
        """Validate that job_type is a recognized value."""
        try:
            # Check if it's a valid JobType enum value
            JobType(v)
            return v
        except ValueError:
            valid_types = [jt.value for jt in JobType]
            raise ValueError(
                f"Invalid job_type: '{v}'. Must be one of: {', '.join(valid_types)}"
            )


class CreateJobResponse(BaseModel):
    """Response after creating job."""
    job_id: str
    session_id: str
    state: JobState
    created_at: datetime


class GetJobResponse(BaseModel):
    """Response for job query."""
    job_id: str
    session_id: str
    project_id: int
    state: JobState
    job_type: str
    result: Optional[JobResult] = None
    created_at: datetime
    updated_at: datetime


class ListJobsRequest(BaseModel):
    """Request to list jobs."""
    project_id: Optional[int] = None
    session_id: Optional[str] = None
    state: Optional[JobState] = None
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class ListJobsResponse(BaseModel):
    """Response for list query."""
    jobs: list[GetJobResponse]
    total: int
    limit: int
    offset: int


__all__ = [
    # Enums (Phase 4 spec)
    "JobType",
    "JobState",
    "ErrorType",
    "Importance",
    "DataSensitivity",
    "Modality",
    "OutputContract",
    
    # Core
    "JobBudget",
    "JobEnvelope",
    
    # Validation
    "ValidationError",
    "validate_job_envelope",
    
    # Routing
    "ModelSelection",
    "RoutingDecision",
    
    # Results
    "ToolInvocation",
    "CritiqueIssue",
    "UsageMetrics",
    "JobResult",
    
    # API
    "CreateJobRequest",
    "CreateJobResponse",
    "GetJobResponse",
    "ListJobsRequest",
    "ListJobsResponse",
]