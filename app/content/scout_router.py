# FILE: app/content/scout_router.py
# Purpose: Content Scout & Review API endpoints.
# Called-by: main
# Depends-on: app.auth, app.content.models, app.content.review, app.content.scout (+1 more)
# Last-renovated: 2026-06-11
"""
Content Scout & Review API endpoints.

Async endpoints for content analysis and daily review.
Separated from the main router to keep files within size limits.
"""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.orm import Session

from app.db import get_db, get_db_session
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content",
    tags=["Content Scout"],
    dependencies=[Depends(require_auth)],
)


# ═══════════════════════════════════════════════════
# CONTENT SCOUT ENDPOINTS
# ═══════════════════════════════════════════════════

@router.post("/conversations/{conversation_id}/tag", response_model=dict)
async def tag_segment(
    conversation_id: str,
    segment: str = Query(..., description="Transcript segment to analyse"),
    offset_seconds: float = Query(0.0, description="Offset in conversation"),
    db: Session = Depends(get_db),
):
    """
    Tag a conversation segment for content potential (realtime).
    Uses Gemini Flash for speed. Non-blocking.
    """
    from app.content.scout import tag_transcript_segment

    tags = await tag_transcript_segment(
        db, conversation_id, segment, offset_seconds
    )
    return {
        "conversation_id": conversation_id,
        "tags_created": len(tags),
        "tags": [
            {
                "id": t.id,
                "tag_type": t.tag_type,
                "strength_score": t.strength_score,
                "excerpt": t.excerpt[:200],
            }
            for t in tags
        ],
    }


@router.post("/conversations/{conversation_id}/analyse", response_model=dict)
async def deep_analyse(
    conversation_id: str,
    db: Session = Depends(get_db),
):
    """
    Run deep analysis on a completed conversation.
    Uses Gemini Pro for comprehensive content opportunity identification.
    """
    from app.content.scout import deep_analyse_conversation

    try:
        pieces = await deep_analyse_conversation(db, conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "conversation_id": conversation_id,
        "opportunities_found": len(pieces),
        "pieces": [
            {
                "id": p.id,
                "title": p.title,
                "content_category": p.content_category,
                "overall_score": p.overall_score,
                "recommended_formats": p.recommended_formats or [],
                "suggested_hooks": p.suggested_hooks or [],
            }
            for p in pieces
        ],
    }


@router.post("/conversations/{conversation_id}/analyse-background")
async def deep_analyse_background(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Queue deep analysis as a background task.
    Returns immediately — results available via daily review.
    """
    from app.content.models import ContentConversation

    conv = db.query(ContentConversation).get(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conv.deep_analysis_done:
        return {
            "status": "already_analysed",
            "conversation_id": conversation_id,
        }

    async def _run_analysis():
        session = get_db_session()
        try:
            from app.content.scout import deep_analyse_conversation
            await deep_analyse_conversation(session, conversation_id)
        except Exception as e:
            logger.error(f"[scout] Background analysis failed: {e}")
        finally:
            session.close()

    background_tasks.add_task(_run_analysis)
    return {
        "status": "queued",
        "conversation_id": conversation_id,
    }


# ═══════════════════════════════════════════════════
# TOPIC INTELLIGENCE
# ═══════════════════════════════════════════════════

@router.post("/topics/classify", response_model=dict)
async def classify_excerpt_topic(
    excerpt: str = Query(..., description="Text to classify"),
    db: Session = Depends(get_db),
):
    """Classify a text excerpt into a topic."""
    from app.content.scout import classify_topic
    result = await classify_topic(db, excerpt)
    return result


@router.post("/topics/{topic_id}/evolution", response_model=dict)
async def detect_evolution(
    topic_id: str,
    excerpt: str = Query(..., description="Current discussion text"),
    db: Session = Depends(get_db),
):
    """
    Detect if the user's position on a topic has evolved.
    Compares current excerpt with stored position history.
    """
    from app.content.scout import detect_position_evolution
    result = await detect_position_evolution(db, topic_id, excerpt)
    return result


# ═══════════════════════════════════════════════════
# END-OF-DAY REVIEW
# ═══════════════════════════════════════════════════

@router.get("/review", response_model=dict)
def get_daily_review(
    date: Optional[str] = Query(
        None, description="Date for review (YYYY-MM-DD), defaults to today"
    ),
    db: Session = Depends(get_db),
):
    """
    Get the end-of-day content review.
    Presents all identified content opportunities for approval.
    """
    from app.content.review import generate_daily_review

    if date:
        try:
            dt = datetime.strptime(date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Date must be YYYY-MM-DD"
            )
    else:
        dt = datetime.now(timezone.utc)

    return generate_daily_review(db, dt)


@router.get("/review/backlog", response_model=dict)
def get_review_backlog(
    db: Session = Depends(get_db),
):
    """
    Get all unreviewed content opportunities across all dates.
    Useful for catching up on missed reviews.
    """
    from app.content.models import ContentPiece

    pieces = (
        db.query(ContentPiece)
        .filter(ContentPiece.status.in_(["identified", "proposed"]))
        .order_by(ContentPiece.overall_score.desc().nullslast())
        .all()
    )

    return {
        "backlog_count": len(pieces),
        "pieces": [
            {
                "id": p.id,
                "title": p.title,
                "content_category": p.content_category,
                "overall_score": p.overall_score,
                "status": p.status,
                "created_at": p.created_at.isoformat(),
                "topic_name": p.topic.name if p.topic else None,
                "series_name": p.series.name if p.series else None,
            }
            for p in pieces
        ],
    }
