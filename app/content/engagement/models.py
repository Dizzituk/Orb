# FILE: app/content/engagement/models.py
# Purpose: Engagement Database Models.
# Called-by: app.content.engagement.dispatcher, app.content.engagement.responder, app.content.engagement.router, app.content.engagement.scanner (+1 more)
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
Engagement Database Models.

Tracks comments pulled from platforms, their classification,
auto-responses sent, and flags raised for human review.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Text, Integer, Float, Boolean,
    DateTime, ForeignKey, JSON,
)
from sqlalchemy.orm import relationship
from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─── Classification tiers ───
# positive    → auto-respond with template
# neutral     → ignore (monitor only)
# question    → flag for review (might want to answer)
# negative    → flag for review
# toxic       → flag urgently, do NOT auto-respond
# spam        → ignore and optionally hide

SENTIMENT_TIERS = (
    "positive", "neutral", "question",
    "negative", "toxic", "spam",
)


class EngagementComment(Base):
    """
    A comment pulled from any platform.
    Normalised into a single schema regardless of source.
    """
    __tablename__ = "engagement_comments"

    id = Column(String, primary_key=True, default=_uuid)

    # Which platform and post this came from
    platform = Column(String, nullable=False, index=True)
    platform_comment_id = Column(String, nullable=False, index=True)
    platform_post_id = Column(String, nullable=False, index=True)

    # Link back to content output (if we published it)
    output_id = Column(
        String, ForeignKey("content_outputs.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Comment content
    author_name = Column(String, nullable=True)
    author_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    posted_at = Column(DateTime, nullable=True)

    # Platform metrics on the comment itself
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)

    # Classification
    sentiment = Column(String, nullable=True, index=True)
    confidence = Column(Float, nullable=True)
    classification_method = Column(String, nullable=True)  # keyword | llm

    # Response tracking
    auto_responded = Column(Boolean, default=False)
    response_id = Column(String, nullable=True)
    responded_at = Column(DateTime, nullable=True)

    # Flag tracking
    flagged = Column(Boolean, default=False, index=True)
    flag_reason = Column(String, nullable=True)
    flag_resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)

    # Processing state
    processed = Column(Boolean, default=False)

    created_at = Column(DateTime, default=_now)

    # Relationships
    response = relationship(
        "EngagementResponse",
        back_populates="comment",
        uselist=False,
    )


class EngagementResponse(Base):
    """
    An auto-response sent to a comment.
    Template-based, not AI-generated.
    """
    __tablename__ = "engagement_responses"

    id = Column(String, primary_key=True, default=_uuid)
    comment_id = Column(
        String, ForeignKey("engagement_comments.id", ondelete="CASCADE"),
        nullable=False, unique=True,
    )

    # What we sent
    response_text = Column(Text, nullable=False)
    template_id = Column(String, nullable=True)

    # Platform response tracking
    platform_response_id = Column(String, nullable=True)
    sent_at = Column(DateTime, default=_now)
    send_status = Column(String, default="pending")  # pending | sent | failed

    created_at = Column(DateTime, default=_now)

    # Relationships
    comment = relationship("EngagementComment", back_populates="response")


class EngagementTemplate(Base):
    """
    Pre-approved response templates.
    Taz writes these, the system rotates through them.
    """
    __tablename__ = "engagement_templates"

    id = Column(String, primary_key=True, default=_uuid)

    # Which sentiment tier this template is for
    sentiment_tier = Column(String, nullable=False, index=True)

    # The response text (with optional {author} placeholder)
    text = Column(Text, nullable=False)

    # Which platforms this template is suitable for
    # JSON array: ["youtube", "tiktok", "instagram", "facebook"]
    platforms = Column(JSON, default=lambda: [
        "youtube", "tiktok", "instagram", "facebook",
    ])

    # Usage tracking for rotation
    use_count = Column(Integer, default=0)
    last_used_at = Column(DateTime, nullable=True)

    # Active flag
    active = Column(Boolean, default=True)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
