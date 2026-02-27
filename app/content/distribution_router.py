# FILE: app/content/distribution_router.py
"""
Distribution & Analytics API endpoints.

Exposes scheduling, publishing, analytics, and
performance reporting for the content pipeline.
"""
import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db, get_db_session
from app.auth import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/content/distribution",
    tags=["Content Distribution"],
    dependencies=[Depends(require_auth)],
)


# ─── REQUEST SCHEMAS ───

class ScheduleRequest(BaseModel):
    scheduled_time: Optional[str] = None  # ISO format, auto if None


class StaggerRequest(BaseModel):
    output_ids: list[str]


# ═══════════════════════════════════════════════════
# SCHEDULING
# ═══════════════════════════════════════════════════

@router.post("/outputs/{output_id}/schedule", response_model=dict)
def schedule_output(
    output_id: str,
    body: ScheduleRequest,
    db: Session = Depends(get_db),
):
    """Schedule an output for publishing. Auto-selects time if not specified."""
    from app.content.distribution.scheduler import schedule_output as sched

    scheduled_time = None
    if body.scheduled_time:
        try:
            scheduled_time = datetime.fromisoformat(body.scheduled_time)
            if scheduled_time.tzinfo is None:
                scheduled_time = scheduled_time.replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")

    try:
        output = sched(db, output_id, scheduled_time)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {
        "output_id": output.id,
        "platform": output.platform,
        "scheduled_at": output.scheduled_at.isoformat(),
    }


@router.post("/schedule-staggered", response_model=dict)
def schedule_staggered(
    body: StaggerRequest,
    db: Session = Depends(get_db),
):
    """Schedule multiple outputs with staggered timing."""
    from app.content.distribution.scheduler import schedule_staggered as stagger

    outputs = stagger(db, body.output_ids)
    return {
        "scheduled_count": len(outputs),
        "schedule": [
            {
                "output_id": o.id,
                "platform": o.platform,
                "format": o.output_format,
                "scheduled_at": o.scheduled_at.isoformat(),
            }
            for o in outputs
        ],
    }


@router.get("/queue", response_model=dict)
def get_publishing_queue(
    platform: Optional[str] = Query(None),
    days_ahead: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Get the upcoming publishing queue."""
    from app.content.distribution.scheduler import get_scheduled_queue

    queue = get_scheduled_queue(db, platform, days_ahead)
    return {"queue_length": len(queue), "items": queue}


# ═══════════════════════════════════════════════════
# PUBLISHING
# ═══════════════════════════════════════════════════

@router.post("/outputs/{output_id}/publish", response_model=dict)
async def publish_output_endpoint(
    output_id: str,
    db: Session = Depends(get_db),
):
    """Publish a single output to its target platform."""
    from app.content.distribution.publisher import publish_output

    try:
        result = await publish_output(db, output_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return result


@router.post("/publish-due", response_model=dict)
async def publish_due_endpoint(
    db: Session = Depends(get_db),
):
    """Publish all outputs that are past their scheduled time."""
    from app.content.distribution.publisher import publish_due
    return await publish_due(db)


# ═══════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════

@router.post("/outputs/{output_id}/pull-analytics", response_model=dict)
async def pull_output_analytics(
    output_id: str,
    db: Session = Depends(get_db),
):
    """Pull fresh analytics for a published output."""
    from app.content.distribution.analytics import pull_analytics_for_output

    snapshot = await pull_analytics_for_output(db, output_id)
    if not snapshot:
        return {"status": "no_data", "output_id": output_id}

    return {
        "output_id": output_id,
        "snapshot_at": snapshot.snapshot_at.isoformat(),
        "views": snapshot.views,
        "likes": snapshot.likes,
        "comments": snapshot.comments,
        "shares": snapshot.shares,
        "saves": snapshot.saves,
        "engagement_rate": snapshot.engagement_rate,
    }


@router.post("/pull-all-analytics", response_model=dict)
async def pull_all_analytics(
    days: int = Query(7, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Pull analytics for all recently published outputs."""
    from app.content.distribution.analytics import pull_all_recent_analytics
    return await pull_all_recent_analytics(db, days)


@router.get("/analytics/summary", response_model=dict)
def analytics_summary(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get performance summary across platforms, formats, and categories."""
    from app.content.distribution.analytics import get_performance_summary
    return get_performance_summary(db, days)


@router.get("/analytics/top", response_model=dict)
def top_performing(
    metric: str = Query("views"),
    limit: int = Query(10, ge=1, le=50),
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get top performing content by a specific metric."""
    from app.content.distribution.analytics import get_top_performing

    valid_metrics = ["views", "likes", "comments", "shares", "saves", "engagement_rate"]
    if metric not in valid_metrics:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric. Choose from: {valid_metrics}",
        )

    results = get_top_performing(db, metric, limit, days)
    return {"metric": metric, "top": results}


# ═══════════════════════════════════════════════════
# PLATFORM STATUS
# ═══════════════════════════════════════════════════

@router.get("/platforms", response_model=dict)
def platform_status():
    """Check which platform integrations are configured."""
    from app.content.distribution.youtube import is_configured as yt_configured
    from app.content.distribution.instagram import is_configured as ig_configured
    from app.content.distribution.tiktok import is_configured as tt_configured
    from app.content.distribution.facebook import is_configured as fb_configured

    return {
        "youtube": {
            "configured": yt_configured(),
            "capabilities": ["upload", "schedule", "analytics"],
        },
        "instagram": {
            "configured": ig_configured(),
            "capabilities": ["reel", "carousel", "insights"],
        },
        "tiktok": {
            "configured": tt_configured(),
            "capabilities": ["upload", "insights", "comments"],
        },
        "facebook": {
            "configured": fb_configured(),
            "capabilities": ["video", "reel", "photo", "text", "insights", "comments"],
        },
    }
