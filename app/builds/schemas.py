# FILE: app/builds/schemas.py
# Purpose: Pydantic schemas for build project endpoints.
# Called-by: app.builds.pipeline_bridge, app.builds.router, app.builds.service
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Pydantic schemas for build project endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime
from enum import Enum


class BuildStatusEnum(str, Enum):
    active = "active"
    completed = "completed"
    failed = "failed"
    archived = "archived"


class PipelineStageEnum(str, Enum):
    idle = "idle"
    weaver = "weaver"
    spec_gate = "spec_gate"
    critical_pipeline = "critical_pipeline"
    overwatcher = "overwatcher"
    implementer = "implementer"
    final_checkout = "final_checkout"
    complete = "complete"


class StageStatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    awaiting_input = "awaiting_input"
    passed = "passed"
    failed = "failed"
    skipped = "skipped"


# ── Requests ──

class CreateBuildRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    original_brief: Optional[str] = None


class UpdateBuildRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[BuildStatusEnum] = None


class UpdateStageRequest(BaseModel):
    """Advance or update a pipeline stage status."""
    stage: PipelineStageEnum
    stage_status: StageStatusEnum
    detail: Optional[str] = None


# ── Responses ──

class StageInfo(BaseModel):
    stage: PipelineStageEnum
    status: StageStatusEnum


class StageLogEntry(BaseModel):
    stage: str
    event: str
    timestamp: str
    detail: Optional[str] = None


class StageNarrativeSection(BaseModel):
    """A section within a stage narrative entry."""
    heading: str                          # e.g. "Inputs", "Decisions", "Output"
    body: str                             # Markdown-formatted content


class StageNarrative(BaseModel):
    """Rich narrative for a pipeline stage execution.

    Captures what happened, why, and what came out. Each stage
    can have multiple narrative entries (e.g. per-segment in
    Critical Pipeline, per-file in Implementer).
    """
    title: str                                        # e.g. "Weaver Synthesis"
    timestamp: str                                    # ISO timestamp
    duration_ms: Optional[int] = None                 # How long the stage took
    model_used: Optional[str] = None                  # Which LLM model ran
    input_summary: Optional[str] = None               # What went in (brief description)
    output_summary: Optional[str] = None              # What came out (brief description)
    sections: List[StageNarrativeSection] = []        # Detailed breakdown
    files_touched: List[str] = []                     # Files created/modified
    warnings: List[str] = []                          # Any warnings or issues
    tokens_used: Optional[int] = None                 # Token count if available


class AppendNarrativeRequest(BaseModel):
    """Request to append a narrative entry to a stage."""
    stage: str
    narrative: StageNarrative


class BuildProjectResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    status: BuildStatusEnum
    current_stage: PipelineStageEnum
    stages: List[StageInfo]
    spec_id: Optional[str] = None
    job_id: Optional[str] = None
    target_path: Optional[str] = None
    original_brief: Optional[str] = None
    weaver_extraction: Optional[dict] = None  # Trimmed Weaver output: requirements, preferences, constraints
    chat_project_id: Optional[int] = None
    total_segments: int = 0
    completed_segments: int = 0
    stage_log: List[StageLogEntry] = []
    stage_narratives: dict = {}           # {stage_name: [StageNarrative]}
    build_report: Optional[str] = None    # Compiled markdown report
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BuildProjectSummary(BaseModel):
    """Lightweight summary for the grid view."""
    id: str
    name: str
    status: BuildStatusEnum
    current_stage: PipelineStageEnum
    total_segments: int = 0
    completed_segments: int = 0
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
