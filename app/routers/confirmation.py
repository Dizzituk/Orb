# FILE: app/routers/confirmation.py
# Purpose: REST endpoint for processing user confirmation gate responses.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.auth, app.db, app.llm.routing.confirmation_gate
# Last-renovated: 2026-06-11
"""
REST endpoint for processing user confirmation gate responses.

Receives yes/no from the chat UI when a confirmation gate fires,
logs the decision for training, and returns the action to take.

v2.1 (2026-02): Initial implementation.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["confirmation"])


class ConfirmationResponse(BaseModel):
    """User's response to a confirmation gate."""
    pattern_key: str
    gate_type: str          # "model_escalation" | "intent_routing"
    proposed_action: str
    approved: bool
    original_message: str
    confidence: float = 0.0
    correction: Optional[str] = None  # If rejected, what did the user actually want?


class ConfirmationResult(BaseModel):
    """Response back to the frontend."""
    logged: bool
    action: str             # "proceed" | "cancel" | "redirect"
    message: str


@router.post("/confirm_decision", response_model=ConfirmationResult)
async def confirm_decision(
    body: ConfirmationResponse,
    db: Session = Depends(get_db),
    auth: dict = Depends(require_auth),
):
    """
    Process a user's yes/no response to a confirmation gate.

    If approved: logs the decision and tells the frontend to re-send
    the original message (which will now auto-approve via history).

    If rejected: logs the decision with optional correction.
    """
    try:
        from app.llm.routing.confirmation_gate import process_confirmation_response

        process_confirmation_response(
            pattern_key=body.pattern_key,
            gate_type=body.gate_type,
            proposed_action=body.proposed_action,
            approved=body.approved,
            original_message=body.original_message,
            confidence=body.confidence,
            user_correction=body.correction,
        )

        if body.approved:
            return ConfirmationResult(
                logged=True,
                action="proceed",
                message="Approved. Processing your request.",
            )
        else:
            return ConfirmationResult(
                logged=True,
                action="cancel",
                message=body.correction or "Cancelled. Staying in chat mode.",
            )

    except ImportError:
        return ConfirmationResult(
            logged=False,
            action="proceed",
            message="Confirmation gate not available, proceeding.",
        )
    except Exception as e:
        logger.exception("[confirmation] Failed to process: %s", e)
        return ConfirmationResult(
            logged=False,
            action="proceed",
            message=f"Error logging decision: {e}",
        )
