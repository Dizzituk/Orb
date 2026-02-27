# FILE: app/content/engagement/router.py
"""
Engagement Management API endpoints.

Exposes comment scanning, classification review,
auto-response management, template CRUD, and
flagged comment review for the Social Media dashboard.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.db import get_db
from app.auth import require_auth
from app.content.engagement.models import (
    EngagementComment, EngagementResponse, EngagementTemplate,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/engagement",
    tags=["Engagement"],
    dependencies=[Depends(require_auth)],
)


# ─── REQUEST SCHEMAS ───

class TemplateCreate(BaseModel):
    sentiment_tier: str
    text: str
    platforms: list[str] = ["youtube", "tiktok", "instagram", "facebook"]


class TemplateUpdate(BaseModel):
    text: Optional[str] = None
    platforms: Optional[list[str]] = None
    active: Optional[bool] = None


class FlagResolve(BaseModel):
    action: str  # resolved | responded | ignored
    response_text: Optional[str] = None


# ═══════════════════════════════════════════════════
# COMMENT SCANNING
# ═══════════════════════════════════════════════════

@router.post("/scan", response_model=dict)
async def scan_comments(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Scan all recent posts for new comments."""
    from app.content.engagement.scanner import scan_all_recent
    return await scan_all_recent(db, days)


@router.post("/dispatch", response_model=dict)
async def dispatch_responses(
    db: Session = Depends(get_db),
):
    """Send all pending auto-responses."""
    from app.content.engagement.dispatcher import dispatch_pending
    return await dispatch_pending(db)


# ═══════════════════════════════════════════════════
# COMMENT LISTING & STATS
# ═══════════════════════════════════════════════════

@router.get("/comments", response_model=dict)
def list_comments(
    platform: Optional[str] = Query(None),
    sentiment: Optional[str] = Query(None),
    flagged_only: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """List comments with filtering."""
    query = db.query(EngagementComment)

    if platform:
        query = query.filter(EngagementComment.platform == platform)
    if sentiment:
        query = query.filter(EngagementComment.sentiment == sentiment)
    if flagged_only:
        query = query.filter(
            and_(
                EngagementComment.flagged.is_(True),
                EngagementComment.flag_resolved.is_(False),
            )
        )

    total = query.count()
    comments = (
        query.order_by(EngagementComment.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "total": total,
        "comments": [
            {
                "id": c.id,
                "platform": c.platform,
                "author_name": c.author_name,
                "text": c.text,
                "sentiment": c.sentiment,
                "confidence": c.confidence,
                "flagged": c.flagged,
                "flag_reason": c.flag_reason,
                "auto_responded": c.auto_responded,
                "posted_at": c.posted_at.isoformat() if c.posted_at else None,
                "created_at": c.created_at.isoformat(),
            }
            for c in comments
        ],
    }


@router.get("/stats", response_model=dict)
def engagement_stats(
    days: int = Query(7, ge=1, le=90),
    db: Session = Depends(get_db),
):
    """Get engagement statistics summary."""
    from datetime import datetime, timezone, timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Count by sentiment
    sentiment_counts = (
        db.query(
            EngagementComment.sentiment,
            func.count(EngagementComment.id),
        )
        .filter(EngagementComment.created_at >= cutoff)
        .group_by(EngagementComment.sentiment)
        .all()
    )

    # Count by platform
    platform_counts = (
        db.query(
            EngagementComment.platform,
            func.count(EngagementComment.id),
        )
        .filter(EngagementComment.created_at >= cutoff)
        .group_by(EngagementComment.platform)
        .all()
    )

    # Pending flags
    pending_flags = (
        db.query(EngagementComment)
        .filter(
            and_(
                EngagementComment.flagged.is_(True),
                EngagementComment.flag_resolved.is_(False),
            )
        )
        .count()
    )

    # Auto-response stats
    responses_sent = (
        db.query(EngagementResponse)
        .filter(
            and_(
                EngagementResponse.send_status == "sent",
                EngagementResponse.sent_at >= cutoff,
            )
        )
        .count()
    )

    return {
        "period_days": days,
        "total_comments": sum(c for _, c in sentiment_counts),
        "by_sentiment": {s: c for s, c in sentiment_counts},
        "by_platform": {p: c for p, c in platform_counts},
        "pending_flags": pending_flags,
        "auto_responses_sent": responses_sent,
    }


# ═══════════════════════════════════════════════════
# FLAGGED COMMENT REVIEW
# ═══════════════════════════════════════════════════

@router.get("/flags", response_model=dict)
def list_flags(
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Get unresolved flagged comments for review."""
    flags = (
        db.query(EngagementComment)
        .filter(
            and_(
                EngagementComment.flagged.is_(True),
                EngagementComment.flag_resolved.is_(False),
            )
        )
        .order_by(EngagementComment.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "count": len(flags),
        "flags": [
            {
                "id": c.id,
                "platform": c.platform,
                "author_name": c.author_name,
                "text": c.text,
                "sentiment": c.sentiment,
                "confidence": c.confidence,
                "flag_reason": c.flag_reason,
                "posted_at": c.posted_at.isoformat() if c.posted_at else None,
                "platform_post_id": c.platform_post_id,
            }
            for c in flags
        ],
    }


@router.post("/flags/{comment_id}/resolve", response_model=dict)
def resolve_flag(
    comment_id: str,
    body: FlagResolve,
    db: Session = Depends(get_db),
):
    """Resolve a flagged comment."""
    from datetime import datetime, timezone

    comment = db.query(EngagementComment).get(comment_id)
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    comment.flag_resolved = True
    comment.resolved_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "comment_id": comment_id,
        "action": body.action,
        "resolved": True,
    }


# ═══════════════════════════════════════════════════
# TEMPLATE MANAGEMENT
# ═══════════════════════════════════════════════════

@router.get("/templates", response_model=dict)
def list_templates(
    sentiment: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """List response templates."""
    query = db.query(EngagementTemplate)
    if sentiment:
        query = query.filter(EngagementTemplate.sentiment_tier == sentiment)

    templates = query.order_by(EngagementTemplate.sentiment_tier).all()

    return {
        "count": len(templates),
        "templates": [
            {
                "id": t.id,
                "sentiment_tier": t.sentiment_tier,
                "text": t.text,
                "platforms": t.platforms,
                "use_count": t.use_count,
                "active": t.active,
            }
            for t in templates
        ],
    }


@router.post("/templates", response_model=dict)
def create_template(
    body: TemplateCreate,
    db: Session = Depends(get_db),
):
    """Create a new response template."""
    valid_tiers = {"positive", "neutral", "question", "negative"}
    if body.sentiment_tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Choose from: {valid_tiers}",
        )

    template = EngagementTemplate(
        sentiment_tier=body.sentiment_tier,
        text=body.text,
        platforms=body.platforms,
    )
    db.add(template)
    db.commit()
    db.refresh(template)

    return {
        "id": template.id,
        "sentiment_tier": template.sentiment_tier,
        "text": template.text,
        "created": True,
    }


@router.put("/templates/{template_id}", response_model=dict)
def update_template(
    template_id: str,
    body: TemplateUpdate,
    db: Session = Depends(get_db),
):
    """Update a response template."""
    template = db.query(EngagementTemplate).get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    if body.text is not None:
        template.text = body.text
    if body.platforms is not None:
        template.platforms = body.platforms
    if body.active is not None:
        template.active = body.active

    db.commit()

    return {"id": template_id, "updated": True}


@router.delete("/templates/{template_id}", response_model=dict)
def delete_template(
    template_id: str,
    db: Session = Depends(get_db),
):
    """Delete a response template."""
    template = db.query(EngagementTemplate).get(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    db.delete(template)
    db.commit()

    return {"id": template_id, "deleted": True}
