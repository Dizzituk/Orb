# FILE: app/debug/feedback.py
# Purpose: Debug session feedback — lets the user correct ASTRA's debug reasoning.
# Called-by: app.debug.debug_chat, main
# Depends-on: app.transparency.corrections, app.transparency.matcher, app.transparency.schemas
# Last-renovated: 2026-06-11
"""
Debug session feedback — lets the user correct ASTRA's debug reasoning.

Simple wrapper around the transparency/corrections system, adapted for
the debug chat context. Feedback is stored as corrections and injected
into future debug sessions via the CorrectionMatcher.

Usage from frontend:
    POST /api/debug/feedback
    { "message_id": 123, "feedback": "positive|negative|correction",
      "comment": "Your diagnosis was right but you edited the wrong files..." }
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["debug-feedback"])


class DebugFeedbackRequest(BaseModel):
    message_id: Optional[int] = None
    project_id: Optional[int] = None
    feedback: str = "correction"  # "positive" | "negative" | "correction"
    comment: str = Field(..., min_length=1, max_length=5000)
    severity: str = "note"  # "note" | "wrong_output" | "broke_things"


class DebugFeedbackResponse(BaseModel):
    stored: bool
    correction_id: Optional[str] = None
    message: str = ""


@router.post("/feedback", response_model=DebugFeedbackResponse)
def submit_debug_feedback(req: DebugFeedbackRequest):
    """Submit feedback on a debug session response.

    Stores the feedback as a correction in the transparency system,
    tagged with stage_name='debug_locked' so the CorrectionMatcher
    can surface it in future debug sessions.
    """
    try:
        from app.transparency.corrections import CorrectionStore
        from app.transparency.schemas import UserCorrection

        correction = UserCorrection(
            reasoning_event_id=f"debug-msg-{req.message_id or 'unknown'}",
            job_id="",
            run_id="",
            build_project_id=str(req.project_id or ""),
            stage_name="debug_locked",
            stage_index=0,
            user_comment=req.comment,
            severity=req.severity,
            correction_type=_feedback_to_correction_type(req.feedback),
        )

        stored = CorrectionStore.add_correction(correction)
        logger.info(
            "[debug_feedback] Stored feedback: %s (%s) — %s",
            req.feedback, req.severity, req.comment[:80],
        )
        return DebugFeedbackResponse(
            stored=True,
            correction_id=stored.correction_id if hasattr(stored, 'correction_id') else str(id(stored)),
            message="Feedback recorded — ASTRA will reference this in future debug sessions.",
        )
    except Exception as e:
        logger.error("[debug_feedback] Failed: %s", e)
        return DebugFeedbackResponse(stored=False, message=str(e))


def _feedback_to_correction_type(feedback: str) -> str:
    return {
        "positive": "note",
        "negative": "wrong_output",
        "correction": "wrong_decision",
    }.get(feedback, "note")


def get_relevant_corrections(project_id: Optional[int] = None, limit: int = 10) -> str:
    """Retrieve past debug corrections to inject into the system prompt.

    Returns a formatted string of relevant user feedback for the debug agent.
    """
    try:
        from app.transparency.matcher import CorrectionMatcher
        from app.transparency.corrections import CorrectionStore

        # Get corrections filtered by project — project-specific first, then universal
        corrections = CorrectionStore.get_corrections_by_stage_and_project(
            stage_name="debug_locked",
            build_project_id=str(project_id or ""),
            limit=limit,
        )

        if not corrections:
            return ""

        lines = ["## Past User Feedback (from previous debug sessions)"]
        for correction in corrections:
            severity_icon = {"broke_things": "🔴", "wrong_output": "🟡", "note": "🟢"}.get(
                correction.severity, "⚪"
            )
            lines.append(
                f"- {severity_icon} [{correction.severity}] {correction.user_comment}"
            )

        return "\n".join(lines) + "\n"

    except Exception as e:
        logger.debug("[debug_feedback] Failed to load corrections: %s", e)
        return ""