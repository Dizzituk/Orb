# FILE: app/content/router.py
# Purpose: Content Pipeline API Router.
# Called-by: main
# Depends-on: app.auth, app.content, app.content.models, app.content.schemas (+2 more)
# Last-renovated: 2026-06-11
"""
Content Pipeline API Router.

FastAPI endpoints for the Content Creation Pipeline.
All endpoints require authentication.
"""
import logging
from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.content import schemas, service
from app.content.models import (
    ContentConversation, ContentTopic, ContentPiece,
    ContentSeries, ContentTag,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content",
    tags=["Content Pipeline"],
    dependencies=[Depends(require_auth)],
)


# ═══════════════════════════════════════════════════
# CONVERSATIONS
# ═══════════════════════════════════════════════════

@router.post("/conversations", response_model=dict)
def start_conversation(
    body: schemas.ConversationCreate,
    db: Session = Depends(get_db),
):
    """Start tracking a new content-eligible conversation."""
    conv = service.start_conversation(
        db, linked_video_path=body.linked_video_path
    )
    return {"conversation_id": conv.id, "started_at": conv.timestamp_start.isoformat()}


@router.post("/conversations/{conversation_id}/end", response_model=dict)
def end_conversation(
    conversation_id: str,
    body: schemas.ConversationEnd,
    db: Session = Depends(get_db),
):
    """End a conversation and store transcript."""
    try:
        conv = service.end_conversation(
            db, conversation_id, transcript_raw=body.transcript_raw
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {
        "conversation_id": conv.id,
        "duration_seconds": conv.duration_seconds,
        "ended_at": conv.timestamp_end.isoformat() if conv.timestamp_end else None,
    }


@router.get("/conversations", response_model=List[schemas.ConversationSummary])
def list_conversations(
    date: Optional[str] = Query(None, description="Filter by date (YYYY-MM-DD)"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """List tracked conversations, optionally filtered by date."""
    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Date must be YYYY-MM-DD")
        convs = service.get_conversations_for_date(db, dt)
    else:
        convs = (
            db.query(ContentConversation)
            .order_by(ContentConversation.timestamp_start.desc())
            .limit(limit)
            .all()
        )

    return [
        schemas.ConversationSummary(
            id=c.id,
            timestamp_start=c.timestamp_start,
            timestamp_end=c.timestamp_end,
            duration_seconds=c.duration_seconds,
            tag_count=len(c.content_tags) if c.content_tags else 0,
            scout_processed=c.scout_processed,
            deep_analysis_done=c.deep_analysis_done,
        )
        for c in convs
    ]


# ═══════════════════════════════════════════════════
# TOPICS
# ═══════════════════════════════════════════════════

@router.get("/topics", response_model=List[schemas.TopicOut])
def list_topics(
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List all tracked topics."""
    topics = (
        db.query(ContentTopic)
        .order_by(ContentTopic.last_discussed.desc())
        .limit(limit)
        .all()
    )
    results = []
    for t in topics:
        pub_count = (
            db.query(ContentPiece)
            .filter(
                ContentPiece.topic_id == t.id,
                ContentPiece.status == "published",
            )
            .count()
        )
        results.append(schemas.TopicOut(
            id=t.id,
            name=t.name,
            description=t.description,
            first_discussed=t.first_discussed,
            last_discussed=t.last_discussed,
            discussion_count=t.discussion_count,
            maturity_score=t.maturity_score,
            published_piece_count=pub_count,
        ))
    return results


@router.post("/topics", response_model=schemas.TopicOut)
def create_or_update_topic(
    body: schemas.TopicCreate,
    db: Session = Depends(get_db),
):
    """Create a new topic or update an existing one."""
    topic = service.get_or_create_topic(db, body.name, body.description)
    return schemas.TopicOut(
        id=topic.id,
        name=topic.name,
        description=topic.description,
        first_discussed=topic.first_discussed,
        last_discussed=topic.last_discussed,
        discussion_count=topic.discussion_count,
        maturity_score=topic.maturity_score,
    )


@router.get("/topics/{topic_id}/coverage", response_model=dict)
def check_topic_coverage(
    topic_id: str,
    window_days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Check if a topic has been recently covered (duplicate detection)."""
    return service.check_topic_coverage(db, topic_id, window_days)


# ═══════════════════════════════════════════════════
# CONTENT PIECES
# ═══════════════════════════════════════════════════

@router.get("/pieces", response_model=List[schemas.ContentPieceOut])
def list_pieces(
    status: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    """List content pieces with optional filters."""
    query = db.query(ContentPiece)
    if status:
        query = query.filter(ContentPiece.status == status)
    if category:
        query = query.filter(ContentPiece.content_category == category)

    pieces = query.order_by(ContentPiece.updated_at.desc()).limit(limit).all()

    return [
        schemas.ContentPieceOut(
            id=p.id,
            title=p.title,
            description=p.description,
            content_category=p.content_category,
            status=p.status,
            topic_name=p.topic.name if p.topic else None,
            series_name=p.series.name if p.series else None,
            overall_score=p.overall_score,
            recommended_formats=p.recommended_formats or [],
            output_count=len(p.outputs) if p.outputs else 0,
            created_at=p.created_at,
            updated_at=p.updated_at,
            published_at=p.published_at,
        )
        for p in pieces
    ]


@router.post("/pieces/{piece_id}/approve", response_model=dict)
def approve_piece(
    piece_id: str,
    body: schemas.ContentPieceApproval,
    db: Session = Depends(get_db),
):
    """Approve, reject, or defer a content piece."""
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise HTTPException(status_code=404, detail="Content piece not found")

    # Log the approval decision for preference learning
    service.log_approval(
        db,
        piece_id=piece_id,
        approval_type="content_piece",
        decision=body.decision,
        modifications=body.modifications,
        reason=body.rejection_reason,
    )

    # Transition status based on decision
    status_map = {
        "approved": "approved",
        "rejected": "rejected",
        "deferred": "deferred",
    }
    new_status = status_map.get(body.decision, piece.status)

    try:
        piece = service.transition_piece_status(
            db, piece_id, new_status,
            rejection_reason=body.rejection_reason,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "piece_id": piece.id,
        "status": piece.status,
        "decision": body.decision,
    }


# ═══════════════════════════════════════════════════
# SERIES
# ═══════════════════════════════════════════════════

@router.get("/series", response_model=List[schemas.SeriesOut])
def list_series(db: Session = Depends(get_db)):
    """List all content series."""
    series_list = service.get_all_series(db)
    return [
        schemas.SeriesOut(
            id=s.id,
            name=s.name,
            description=s.description,
            categories=s.categories or [],
            target_formats=s.target_formats or [],
            target_platforms=s.target_platforms or [],
            posting_cadence=s.posting_cadence,
            active=s.active,
            piece_count=len(s.pieces) if s.pieces else 0,
        )
        for s in series_list
    ]


@router.post("/series", response_model=schemas.SeriesOut)
def create_series(
    body: schemas.SeriesCreate,
    db: Session = Depends(get_db),
):
    """Create a new content series."""
    s = service.create_series(
        db,
        name=body.name,
        description=body.description,
        categories=body.categories,
        target_formats=body.target_formats,
        target_platforms=body.target_platforms,
        posting_cadence=body.posting_cadence,
    )
    return schemas.SeriesOut(
        id=s.id,
        name=s.name,
        description=s.description,
        categories=s.categories or [],
        target_formats=s.target_formats or [],
        target_platforms=s.target_platforms or [],
        posting_cadence=s.posting_cadence,
        active=s.active,
        piece_count=0,
    )


# ═══════════════════════════════════════════════════
# APPROVAL STATS (Preference Learning)
# ═══════════════════════════════════════════════════

@router.get("/approvals/stats", response_model=dict)
def get_approval_stats(
    approval_type: Optional[str] = Query(None),
    days: int = Query(90, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get approval rate statistics for preference learning."""
    return service.get_approval_stats(db, approval_type, days)


# ═══════════════════════════════════════════════════
# STYLE PROFILE
# ═══════════════════════════════════════════════════

@router.get("/style-profile", response_model=schemas.StyleProfileOut)
def get_style_profile(db: Session = Depends(get_db)):
    """Get the active style profile."""
    profile = service.ensure_default_style_profile(db)
    return schemas.StyleProfileOut(
        id=profile.id,
        name=profile.name,
        video_params=profile.video_params or {},
        voice_profile=profile.voice_profile or {},
        reference_video_count=len(profile.reference_videos or []),
        active=profile.active,
    )
