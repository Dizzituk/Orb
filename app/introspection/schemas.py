# FILE: app/introspection/schemas.py
"""
Schemas for log introspection feature.

All models are read-only representations of log data.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class LogRequestType(str, Enum):
    """Types of log introspection requests."""
    LAST_JOB = "last_job"
    TIME_WINDOW = "time_window"
    JOB_ID = "job_id"


class LogEvent(BaseModel):
    """Single log event from the ledger."""
    timestamp: datetime
    event_type: str
    job_id: Optional[str] = None
    stage_name: Optional[str] = None
    status: Optional[str] = None
    spec_id: Optional[str] = None
    expected_spec_hash: Optional[str] = None
    observed_spec_hash: Optional[str] = None
    verified: Optional[bool] = None
    error: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


class JobLogBundle(BaseModel):
    """Complete log bundle for a single job."""
    job_id: str
    job_type: Optional[str] = None
    state: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    events: list[LogEvent] = Field(default_factory=list)
    
    # Spec-gate summary fields
    spec_hash_computed: bool = False
    spec_hash_verified: bool = False
    spec_hash_mismatch: bool = False


class LogQueryRequest(BaseModel):
    """Request to query logs."""
    request_type: LogRequestType
    job_id: Optional[str] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None


class LogQueryResponse(BaseModel):
    """Response containing logs and summary."""
    request_type: LogRequestType
    summary: str
    job_bundles: list[JobLogBundle] = Field(default_factory=list)
    total_events: int = 0
    query_time_ms: int = 0
    error: Optional[str] = None


# =============================================================================
# Spec-Gate Hash Event Types
# =============================================================================

class SpecHashEventType(str, Enum):
    """Event types for spec-gate hash verification."""
    COMPUTED = "STAGE_SPEC_HASH_COMPUTED"
    VERIFIED = "STAGE_SPEC_HASH_VERIFIED"
    MISMATCH = "STAGE_SPEC_HASH_MISMATCH"


class SpecHashEvent(BaseModel):
    """Structured spec-hash verification event."""
    event: SpecHashEventType
    job_id: str
    stage_name: str
    spec_id: str
    expected_spec_hash: str
    observed_spec_hash: Optional[str] = None
    verified: bool
    timestamp: datetime
    severity: Optional[str] = None  # "INFO" | "WARNING" | "ERROR"
    message: Optional[str] = None


__all__ = [
    "LogRequestType",
    "LogEvent",
    "JobLogBundle",
    "LogQueryRequest",
    "LogQueryResponse",
    "SpecHashEventType",
    "SpecHashEvent",
]
