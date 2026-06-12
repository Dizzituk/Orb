# FILE: app/content/review.py
# Purpose: End-of-Day Review Service (Spec Section 6).
# Called-by: app.content.scout_router
# Depends-on: app.content.models
# Last-renovated: 2026-06-11
"""
End-of-Day Review Service (Spec Section 6).

Generates the daily content review summary that presents
all identified opportunities for user approval.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.content.models import (
    ContentConversation, ContentTag, ContentTopic,
    ContentPiece, ContentSeries,
)

logger = logging.getLogger(__name__)


def generate_daily_review(
    db: Session,
    date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """
    Generate the end-of-day content review for a given date.
    Returns a structured summary of conversations and content opportunities.
    
    Spec Section 6.1 — Review Presentation.
    """
    if date is None:
        date = datetime.now(timezone.utc)

    # Date boundaries
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)

    # Get conversations for the day
    conversations = (
        db.query(ContentConversation)
        .filter(
            ContentConversation.timestamp_start >= day_start,
            ContentConversation.timestamp_start < day_end,
        )
        .order_by(ContentConversation.timestamp_start.asc())
        .all()
    )

    total_duration = sum(
        c.duration_seconds or 0 for c in conversations
    )

    # Collect unique topics from the day's tags
    day_conv_ids = [c.id for c in conversations]
    topics_discussed = set()
    if day_conv_ids:
        tags = (
            db.query(ContentTag)
            .filter(ContentTag.conversation_id.in_(day_conv_ids))
            .all()
        )
        for tag in tags:
            if tag.topic and tag.topic.name:
                topics_discussed.add(tag.topic.name)

    # Get content pieces identified from today's conversations
    proposals = []
    if day_conv_ids:
        pieces = (
            db.query(ContentPiece)
            .filter(ContentPiece.status.in_(["identified", "proposed"]))
            .all()
        )
        # Filter to pieces sourced from today's conversations
        for piece in pieces:
            source_ids = piece.source_conversation_ids or []
            if any(cid in day_conv_ids for cid in source_ids):
                proposals.append(_format_proposal(db, piece))

    # Also include any pieces still in proposed/identified state from prior days
    # (in case the user hasn't reviewed them yet)
    backlog = (
        db.query(ContentPiece)
        .filter(ContentPiece.status.in_(["identified", "proposed"]))
        .order_by(ContentPiece.overall_score.desc().nullslast())
        .all()
    )
    backlog_ids = {p["id"] for p in proposals}
    for piece in backlog:
        if piece.id not in backlog_ids:
            proposals.append(_format_proposal(db, piece))

    # Sort proposals by overall score (highest first)
    proposals.sort(
        key=lambda p: p.get("overall_score") or 0,
        reverse=True,
    )

    # Count deferred pieces
    deferred_count = (
        db.query(ContentPiece)
        .filter(ContentPiece.status == "deferred")
        .count()
    )

    review = {
        "date": day_start.strftime("%Y-%m-%d"),
        "conversations_count": len(conversations),
        "total_duration_minutes": total_duration // 60,
        "topics_discussed": sorted(topics_discussed),
        "proposals": proposals,
        "proposal_count": len(proposals),
        "deferred_count": deferred_count,
        "conversations": [
            {
                "id": c.id,
                "started_at": c.timestamp_start.isoformat(),
                "duration_minutes": (c.duration_seconds or 0) // 60,
                "tag_count": len(c.content_tags) if c.content_tags else 0,
                "analysed": c.deep_analysis_done,
            }
            for c in conversations
        ],
    }

    logger.info(
        f"[review] Generated daily review for {review['date']}: "
        f"{review['conversations_count']} conversations, "
        f"{review['proposal_count']} proposals"
    )
    return review


def _format_proposal(db: Session, piece: ContentPiece) -> Dict[str, Any]:
    """Format a ContentPiece as a review proposal."""
    # Check if topic was previously published
    previously_covered = False
    last_published = None
    if piece.topic_id:
        published = (
            db.query(ContentPiece)
            .filter(
                ContentPiece.topic_id == piece.topic_id,
                ContentPiece.status == "published",
                ContentPiece.id != piece.id,
            )
            .order_by(ContentPiece.published_at.desc())
            .first()
        )
        if published:
            previously_covered = True
            last_published = (
                published.published_at.isoformat()
                if published.published_at else None
            )

    return {
        "id": piece.id,
        "title": piece.title,
        "description": piece.description,
        "content_category": piece.content_category,
        "topic_name": piece.topic.name if piece.topic else None,
        "series_name": piece.series.name if piece.series else None,
        "overall_score": piece.overall_score,
        "recommended_formats": piece.recommended_formats or [],
        "suggested_hooks": piece.suggested_hooks or [],
        "key_excerpts": piece.key_excerpts or [],
        "previously_covered": previously_covered,
        "last_published_on_topic": last_published,
        "status": piece.status,
        "created_at": piece.created_at.isoformat(),
    }
