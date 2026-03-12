# FILE: app/project_registry/confidence_tracker.py
"""
Confidence tracking for command intent classification.

Stores yes/no feedback from confirmation buttons and adjusts
confidence scores for phrasing patterns over time. This is the
learning loop that allows ASTRA to calibrate command detection
to the user's speech patterns.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.orm import Session

from app.db import Base

logger = logging.getLogger(__name__)


class CommandFeedback(Base):
    """
    Record of a command intent detection and user feedback.

    Each row represents one instance where ASTRA detected a potential
    command and surfaced a confirmation button. The user's yes/no
    response is stored alongside the phrasing for pattern learning.
    """
    __tablename__ = "command_feedback"

    id = Column(Integer, primary_key=True, index=True)

    # The raw message that triggered detection
    raw_message = Column(Text, nullable=False)

    # Normalised payload (after wake word stripped)
    payload = Column(Text, nullable=False)

    # The classified intent (e.g. "scan_project", "build_app")
    classified_intent = Column(String(100), nullable=False, index=True)

    # Confidence score at time of detection (0.0-1.0)
    confidence_at_detection = Column(Float, nullable=False)

    # User feedback: True=confirmed (yes), False=rejected (no), None=no response
    confirmed = Column(Boolean, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    feedback_at = Column(DateTime, nullable=True)


def record_detection(
    db: Session,
    raw_message: str,
    payload: str,
    classified_intent: str,
    confidence: float,
) -> int:
    """Record a command detection event. Returns the feedback ID."""
    entry = CommandFeedback(
        raw_message=raw_message,
        payload=payload,
        classified_intent=classified_intent,
        confidence_at_detection=confidence,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry.id


def record_feedback(
    db: Session,
    feedback_id: int,
    confirmed: bool,
) -> None:
    """Record the user's yes/no response to a confirmation button."""
    entry = db.query(CommandFeedback).get(feedback_id)
    if not entry:
        logger.warning("[confidence] Feedback ID %d not found", feedback_id)
        return
    entry.confirmed = confirmed
    entry.feedback_at = datetime.utcnow()
    db.commit()
    logger.info(
        "[confidence] Feedback recorded: id=%d, intent=%s, confirmed=%s",
        feedback_id, entry.classified_intent, confirmed,
    )


def get_confidence_adjustment(
    db: Session,
    classified_intent: str,
    payload_keywords: list[str],
) -> float:
    """
    Calculate a confidence adjustment based on historical feedback.

    Looks at past detections with the same intent and similar phrasing,
    and returns an adjustment value:
        - Positive: past confirmations suggest this IS a real command
        - Negative: past rejections suggest this is NOT a command
        - Zero: no relevant history

    Returns a float in [-0.3, +0.3] range.
    """
    # Get recent feedback for this intent
    recent = (
        db.query(CommandFeedback)
        .filter(
            CommandFeedback.classified_intent == classified_intent,
            CommandFeedback.confirmed.isnot(None),
        )
        .order_by(CommandFeedback.created_at.desc())
        .limit(50)
        .all()
    )

    if not recent:
        return 0.0

    confirmations = sum(1 for r in recent if r.confirmed)
    rejections = sum(1 for r in recent if not r.confirmed)
    total = confirmations + rejections

    if total == 0:
        return 0.0

    # Net ratio: 1.0 = all confirmed, 0.0 = all rejected
    ratio = confirmations / total

    # Map to [-0.3, +0.3] range
    # 0.5 ratio = 0.0 adjustment (balanced)
    # 1.0 ratio = +0.3 (strong confirmation pattern)
    # 0.0 ratio = -0.3 (strong rejection pattern)
    adjustment = (ratio - 0.5) * 0.6

    return round(adjustment, 3)