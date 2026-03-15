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
async def schedule_output(
    output_id: str,
    body: ScheduleRequest,
    db: Session = Depends(get_db),
):
    """Schedule an output and immediately upload to the platform.

    The platform's own scheduler handles the actual publishing at the
    scheduled time. ASTRA doesn't need to be running when it goes live.

    YouTube: uploads as private with publishAt set — YouTube publishes it.
    Facebook: uses scheduled_publish_time in the Graph API.
    TikTok/Instagram: no native scheduling — publishes immediately if
        the scheduled time is in the past, otherwise stores locally.
    """
    from app.content.distribution.scheduler import schedule_output as sched
    from app.content.distribution.publisher import publish_output
    from app.content.models import ContentOutput

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

    # Immediately upload to the platform with the scheduled time.
    # The platform handles the actual publishing at the right moment.
    platform = output.platform
    upload_result = None
    upload_error = None

    try:
        upload_result = await publish_output(db, output_id)
        if upload_result.get("status") == "published":
            logger.info(
                "[distribution] Uploaded to %s with scheduled time %s",
                platform, output.scheduled_at,
            )
        elif upload_result.get("status") == "not_configured":
            upload_error = f"{platform} API not configured — stored schedule locally"
            logger.warning("[distribution] %s not configured, schedule stored locally", platform)
        else:
            upload_error = upload_result.get("error", "Upload failed")
            logger.warning("[distribution] Upload to %s failed: %s", platform, upload_error)
    except Exception as e:
        upload_error = str(e)
        logger.error("[distribution] Upload to %s error: %s", platform, e)

    result = {
        "output_id": output.id,
        "platform": platform,
        "scheduled_at": output.scheduled_at.isoformat(),
    }

    if upload_error:
        result["upload_status"] = "failed"
        result["upload_error"] = upload_error
        result["note"] = "Schedule saved locally. Upload to platform failed — you may need to retry."
    else:
        result["upload_status"] = "uploaded"
        result["platform_post_id"] = upload_result.get("post_id")
        result["note"] = f"Uploaded to {platform}. The platform will publish at the scheduled time."

    return result


@router.post("/outputs/{output_id}/unschedule", response_model=dict)
def unschedule_output(
    output_id: str,
    db: Session = Depends(get_db),
):
    """Remove scheduling from an output — unapprove it.

    Clears scheduled_at and published_at, allowing the output
    to be re-reviewed, re-scheduled, or deleted.
    """
    from app.content.models import ContentOutput

    output = db.query(ContentOutput).get(output_id)
    if not output:
        raise HTTPException(status_code=404, detail="Output not found")

    output.scheduled_at = None
    output.published_at = None
    # Reset platform metadata status to draft
    if output.platform_metadata:
        meta = dict(output.platform_metadata)
        meta["status"] = "draft"
        output.platform_metadata = meta
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(output, "platform_metadata")

    db.commit()
    db.refresh(output)

    logger.info("[distribution] Unscheduled output %s", output_id)

    return {
        "output_id": output.id,
        "status": "unscheduled",
        "message": f"Output unscheduled. You can now re-schedule or delete it.",
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

@router.post("/learn-posting-times", response_model=dict)
async def learn_posting_times(
    lookback_days: int = 90,
    db: Session = Depends(get_db),
):
    """Analyse posting performance and update optimal posting windows.

    Looks at all published content with analytics, groups by hour/day,
    and identifies the windows that drive the most views and engagement.
    Results are persisted and used by the scheduler automatically.

    Needs at least 10 posts with analytics to override defaults.
    """
    from app.content.distribution.posting_time_learner import update_learned_windows

    report = update_learned_windows(db, lookback_days)

    summary = {}
    for fmt, data in report.items():
        summary[fmt] = {
            "posts_analysed": data["total_posts_analysed"],
            "has_enough_data": data["has_enough_data"],
            "best_windows": data["best_windows"],
            "best_days": data["best_days"],
        }

    return {
        "status": "updated",
        "lookback_days": lookback_days,
        "formats": summary,
    }


@router.get("/posting-times", response_model=dict)
def get_posting_times():
    """Get current posting time configuration.

    Shows learned windows (from analytics) where available,
    and defaults where not enough data exists yet.
    """
    from app.content.distribution.posting_time_learner import (
        LEARNED_WINDOWS_PATH,
    )
    from app.content.distribution.scheduler import DEFAULT_WINDOWS
    import json

    learned = {}
    if LEARNED_WINDOWS_PATH.exists():
        try:
            learned = json.loads(LEARNED_WINDOWS_PATH.read_text())
        except Exception:
            pass

    result = {
        "defaults": DEFAULT_WINDOWS,
        "learned": learned.get("windows", {}),
        "last_updated": learned.get("updated_at"),
    }

    # Show what the scheduler will actually use per platform
    active = {}
    for platform, default_hours in DEFAULT_WINDOWS.items():
        learned_data = learned.get("windows", {}).get(platform, {})
        if learned_data.get("confidence") == "learned":
            active[platform] = {
                "source": "learned",
                "hours": learned_data["posting_hours"],
                "best_days": learned_data.get("best_days", []),
                "posts_analysed": learned_data.get("posts_analysed", 0),
            }
        else:
            active[platform] = {
                "source": "default",
                "hours": default_hours,
            }

    result["active"] = active
    return result


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
