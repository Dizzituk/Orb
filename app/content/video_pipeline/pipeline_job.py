# FILE: app/content/video_pipeline/pipeline_job.py
# Purpose: PipelineJob state container + on-disk job/tts dir constants for the video pipeline.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models
# Last-renovated: 2026-06-21
"""
PipelineJob — per-execution state container for the video pipeline.

Split out of orchestrator.py (BATCH 4) verbatim. Tracks a single
script-to-video run: scene/resolved/style plans, cost, error, the
SSE event buffer, and disk persistence of job state.
"""
import json
import uuid
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from app.content.video_pipeline.models import (
    PipelineJobRequest, PipelineStageUpdate,
    ScenePlan, ResolvedPlan, StyleProfile,
)

JOBS_DIR = Path("data/content/video_pipeline/jobs")
TTS_OUTPUT_DIR = Path("data/content/video_pipeline/tts")


class PipelineJob:
    """Tracks state for a single pipeline execution."""

    def __init__(self, request: PipelineJobRequest):
        self.job_id = str(uuid.uuid4())[:12]
        self.request = request
        self.created_at = datetime.now(timezone.utc)
        self.status = "pending"
        self.current_stage = ""
        self.scene_plan: Optional[ScenePlan] = None
        self.resolved_plan: Optional[ResolvedPlan] = None
        self.style_profile: Optional[StyleProfile] = None
        self.output_path: Optional[str] = None
        self.total_cost_usd: float = 0.0
        self.error: Optional[str] = None
        self._events: list = []

    def record_event(self, stage: str, status: str, message: str = "", **data):
        event = PipelineStageUpdate(
            job_id=self.job_id,
            stage=stage,
            status=status,
            message=message,
            data=data,
        )
        self._events.append(event)
        return event

    @property
    def job_dir(self) -> Path:
        path = JOBS_DIR / self.job_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_state(self):
        """Persist job state to disk."""
        state = {
            "job_id": self.job_id,
            "status": self.status,
            "current_stage": self.current_stage,
            "created_at": self.created_at.isoformat(),
            "total_cost_usd": self.total_cost_usd,
            "output_path": self.output_path,
            "error": self.error,
            "request": self.request.model_dump(),
        }
        if self.scene_plan:
            state["scene_plan"] = self.scene_plan.model_dump()
        if self.resolved_plan:
            state["resolved_plan"] = self.resolved_plan.model_dump()

        state_file = self.job_dir / "state.json"
        state_file.write_text(
            json.dumps(state, indent=2, default=str),
            encoding="utf-8",
        )
