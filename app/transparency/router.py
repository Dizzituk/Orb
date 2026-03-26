# FILE: app/transparency/router.py
"""
FastAPI router for Pipeline Transparency & User Feedback.

Endpoints:
- GET  /transparency/{build_project_id}/trace  — reasoning trace for a build project
- GET  /transparency/run/{job_id}/{run_id}      — trace for a specific pipeline run
- POST /transparency/corrections                — submit user correction
- GET  /transparency/corrections/project/{build_project_id} — corrections for a project
- GET  /transparency/corrections/event/{event_id}   — corrections for an event
- DELETE /transparency/corrections/{correction_id}   — delete a correction

v1.0 (2026-02): Initial implementation
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/transparency",
    tags=["Pipeline Transparency"],
    dependencies=[Depends(require_auth)],
)


# =============================================================================
# REQUEST / RESPONSE SCHEMAS
# =============================================================================

class AddCorrectionRequest(BaseModel):
    reasoning_event_id: str
    job_id: str = ""
    run_id: str = ""
    build_project_id: str = ""
    stage_name: str
    stage_index: int = 0
    decision_index: Optional[int] = None
    user_comment: str = Field(..., min_length=1, max_length=5000)
    severity: str = "note"              # "note" | "wrong_output" | "broke_things"
    correction_type: str = "wrong_decision"  # "wrong_evidence" | "wrong_decision" | "missing_context" | "wrong_output"


class CorrectionResponse(BaseModel):
    correction_id: str
    reasoning_event_id: str
    job_id: str = ""
    run_id: str = ""
    build_project_id: str = ""
    stage_name: str
    stage_index: int = 0
    decision_index: Optional[int] = None
    user_comment: str
    severity: str = "note"
    correction_type: str = "wrong_decision"
    context_keywords: List[str] = Field(default_factory=list)
    created_at: str = ""


class ReasoningEventResponse(BaseModel):
    event_id: str
    job_id: str = ""
    run_id: str = ""
    build_project_id: str = ""
    stage_name: str = ""
    stage_index: int = 0
    status: str = "running"
    confidence_score: float = 0.0
    reasoning_summary: str = ""
    reasoning_detail: str = ""
    evidence_sources: List = Field(default_factory=list)
    decisions: List = Field(default_factory=list)
    model_used: str = ""
    token_cost_usd: float = 0.0
    duration_ms: int = 0
    metadata: dict = Field(default_factory=dict)
    created_at: str = ""
    corrections: List[CorrectionResponse] = Field(default_factory=list)


# =============================================================================
# TRACE ENDPOINTS
# =============================================================================

@router.get("/{build_project_id}/trace", response_model=List[ReasoningEventResponse])
async def get_project_trace(build_project_id: str):
    """Get full reasoning trace for a build project."""
    from app.transparency.collector import ReasoningCollector
    from app.transparency.corrections import CorrectionStore

    events = await ReasoningCollector.get_project_trace(build_project_id)

    # Attach corrections to each event
    result = []
    for event_dict in events:
        corrections = CorrectionStore.get_corrections_for_event(
            event_dict.get("event_id", "")
        )
        event_dict["corrections"] = [c.to_dict() for c in corrections]
        result.append(event_dict)

    return result


@router.get("/run/{job_id}/{run_id}", response_model=List[ReasoningEventResponse])
async def get_run_trace(job_id: str, run_id: str):
    """Get reasoning trace for a specific pipeline run."""
    from app.transparency.collector import ReasoningCollector
    from app.transparency.corrections import CorrectionStore

    events = await ReasoningCollector.get_run_trace(job_id, run_id)

    result = []
    for event_dict in events:
        corrections = CorrectionStore.get_corrections_for_event(
            event_dict.get("event_id", "")
        )
        event_dict["corrections"] = [c.to_dict() for c in corrections]
        result.append(event_dict)

    return result


# =============================================================================
# CORRECTION ENDPOINTS
# =============================================================================

@router.post("/corrections", response_model=CorrectionResponse, status_code=201)
def add_correction(req: AddCorrectionRequest):
    """Submit a user correction for a reasoning event."""
    from app.transparency.corrections import CorrectionStore
    from app.transparency.schemas import UserCorrection

    correction = UserCorrection(
        reasoning_event_id=req.reasoning_event_id,
        job_id=req.job_id,
        run_id=req.run_id,
        build_project_id=req.build_project_id,
        stage_name=req.stage_name,
        stage_index=req.stage_index,
        decision_index=req.decision_index,
        user_comment=req.user_comment,
        severity=req.severity,
        correction_type=req.correction_type,
    )

    stored = CorrectionStore.add_correction(correction)

    # Notify Self-Model observer of the correction
    try:
        from app.self_model.hooks import on_user_correction
        on_user_correction(domain=req.stage_name or "general", user_comment=req.user_comment, stage_name=req.stage_name or "")
    except Exception:
        pass

    return stored.to_dict()


@router.get("/corrections/event/{event_id}", response_model=List[CorrectionResponse])
def get_event_corrections(event_id: str):
    """Get corrections for a specific reasoning event."""
    from app.transparency.corrections import CorrectionStore

    corrections = CorrectionStore.get_corrections_for_event(event_id)
    return [c.to_dict() for c in corrections]


@router.get("/corrections/project/{build_project_id}", response_model=List[CorrectionResponse])
def get_project_corrections(build_project_id: str):
    """Get all corrections for a build project."""
    from app.transparency.corrections import CorrectionStore

    corrections = CorrectionStore.get_corrections_for_project(build_project_id)
    return [c.to_dict() for c in corrections]


@router.delete("/corrections/{correction_id}", status_code=204)
def delete_correction(correction_id: str):
    """Delete a user correction."""
    from app.transparency.corrections import CorrectionStore

    if not CorrectionStore.delete_correction(correction_id):
        raise HTTPException(404, "Correction not found")


__all__ = ["router"]
