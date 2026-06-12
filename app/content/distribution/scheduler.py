# FILE: app/content/distribution/scheduler.py
# Purpose: Publishing Scheduler (Spec Section 9.4).
# Called-by: app.content.distribution.publisher, app.content.distribution.youtube_router, app.content.distribution_router
# Depends-on: app.content.distribution.posting_time_learner, app.content.models
# Last-renovated: 2026-06-11
"""
Publishing Scheduler (Spec Section 9.4).

Manages the publishing calendar with platform-specific timing.
Handles:
- Optimal posting windows per platform
- Staggered cross-platform publishing
- Minimum intervals between posts
- Queue management
- Analytics-driven timing refinement

All scheduling is deterministic — no AI involved.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.content.models import ContentOutput, ContentAnalytics

logger = logging.getLogger(__name__)


# ─── DEFAULT POSTING WINDOWS (UK timezone) ───
# Format: list of (hour, minute) tuples representing optimal times

# Posting windows informed by algorithm_strategy.py
# YouTube times come from OPTIMAL_POSTING_TIMES in strategy module
DEFAULT_WINDOWS = {
    "instagram": [(7, 0), (12, 0), (18, 30)],
    "youtube": [(7, 0), (12, 0), (17, 0), (19, 0)],
    "tiktok": [(7, 0), (12, 30), (19, 0)],
    "facebook": [(9, 0), (13, 0), (18, 0)],
    "twitter": [(8, 0), (12, 0), (17, 0)],
    "blog": [(9, 0)],
}

# Minimum hours between posts on the same platform
# YouTube splits by format: shorts can be daily, longform weekly
MIN_INTERVAL_HOURS = {
    "instagram": 6,
    "youtube": 24,       # Default (shorts) — 1 per day
    "youtube_short": 24, # Shorts: 1 per day
    "youtube_longform": 168,  # Longform: 7 days (1 per week)
    "tiktok": 4,
    "facebook": 8,
    "twitter": 2,
    "blog": 48,
}

# Cross-platform stagger (hours between same content on different platforms)
CROSS_PLATFORM_STAGGER_HOURS = 3


def find_next_slot(
    db: Session,
    platform: str,
    after: Optional[datetime] = None,
    output_format: Optional[str] = None,
) -> datetime:
    """
    Find the next available posting slot for a platform.
    Considers: posting windows, minimum intervals, existing schedule.

    Uses learned posting windows (from analytics) when available,
    falling back to hardcoded defaults when not enough data exists.

    For YouTube, the interval depends on format:
    - youtube_short: 24h (1 per day)
    - youtube_longform: 168h (1 per week)
    """
    if after is None:
        after = datetime.now(timezone.utc)
    elif after.tzinfo is None:
        after = after.replace(tzinfo=timezone.utc)

    # Try learned windows first (analytics-driven), fall back to defaults
    windows = None
    try:
        from app.content.distribution.posting_time_learner import get_learned_windows
        # Check format-specific first (e.g. youtube_short), then platform
        if output_format:
            windows = get_learned_windows(output_format)
        if not windows:
            windows = get_learned_windows(platform)
    except Exception:
        pass

    if not windows:
        windows = DEFAULT_WINDOWS.get(platform, [(12, 0)])

    # Look up interval by format first (e.g. "youtube_longform"), fall back to platform
    interval_key = output_format or platform
    min_interval = timedelta(
        hours=MIN_INTERVAL_HOURS.get(interval_key, MIN_INTERVAL_HOURS.get(platform, 6))
    )

    # Find last scheduled/published post on this platform with the same format
    query = (
        db.query(ContentOutput)
        .filter(
            ContentOutput.platform == platform,
            ContentOutput.scheduled_at.isnot(None),
        )
    )
    # For YouTube, only check interval against the same format type
    if output_format and platform == "youtube":
        query = query.filter(ContentOutput.output_format == output_format)

    last_post = query.order_by(ContentOutput.scheduled_at.desc()).first()

    earliest = after
    if last_post and last_post.scheduled_at:
        sched = last_post.scheduled_at
        if sched.tzinfo is None:
            sched = sched.replace(tzinfo=timezone.utc)
        min_after = sched + min_interval
        if min_after > earliest:
            earliest = min_after

    # Find next window after earliest
    # All times are UTC — ensure timezone is always attached
    candidate = earliest
    for _ in range(14):  # Search up to 14 days ahead
        for hour, minute in windows:
            slot = candidate.replace(
                hour=hour, minute=minute, second=0, microsecond=0,
                tzinfo=timezone.utc,
            )
            if slot > earliest:
                return slot

        # Move to next day
        candidate = (candidate + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
            tzinfo=timezone.utc,
        )

    # Fallback: next day at noon
    return (earliest + timedelta(days=1)).replace(
        hour=12, minute=0, second=0, microsecond=0,
        tzinfo=timezone.utc,
    )


def schedule_output(
    db: Session,
    output_id: str,
    scheduled_time: Optional[datetime] = None,
) -> ContentOutput:
    """
    Schedule a content output for publishing.
    If no time specified, auto-selects the next optimal slot.
    """
    output = db.query(ContentOutput).get(output_id)
    if not output:
        raise ValueError(f"Output {output_id} not found")

    if scheduled_time is None:
        scheduled_time = find_next_slot(db, output.platform, output_format=output.output_format)

    output.scheduled_at = scheduled_time
    db.commit()
    db.refresh(output)

    logger.info(
        f"[scheduler] Scheduled {output.output_format} for "
        f"{scheduled_time.isoformat()} on {output.platform}"
    )
    return output


def schedule_staggered(
    db: Session,
    output_ids: List[str],
) -> List[ContentOutput]:
    """
    Schedule multiple outputs with staggered timing.
    Same content across platforms gets spaced out.
    """
    outputs = []
    last_scheduled = datetime.now(timezone.utc)

    for oid in output_ids:
        output = db.query(ContentOutput).get(oid)
        if not output:
            continue

        slot = find_next_slot(
            db, output.platform,
            after=last_scheduled,
        )
        output.scheduled_at = slot
        outputs.append(output)

        last_scheduled = slot + timedelta(hours=CROSS_PLATFORM_STAGGER_HOURS)

    db.commit()

    logger.info(
        f"[scheduler] Staggered {len(outputs)} outputs across platforms"
    )
    return outputs


def get_scheduled_queue(
    db: Session,
    platform: Optional[str] = None,
    days_ahead: int = 7,
) -> List[Dict[str, Any]]:
    """Get the upcoming publishing queue."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=days_ahead)

    query = db.query(ContentOutput).filter(
        and_(
            ContentOutput.scheduled_at.isnot(None),
            ContentOutput.published_at.is_(None),
            ContentOutput.scheduled_at >= now,
            ContentOutput.scheduled_at <= cutoff,
        )
    )
    if platform:
        query = query.filter(ContentOutput.platform == platform)

    outputs = query.order_by(ContentOutput.scheduled_at.asc()).all()

    return [
        {
            "output_id": o.id,
            "piece_id": o.piece_id,
            "format": o.output_format,
            "platform": o.platform,
            "device": o.publish_device,
            "scheduled_at": o.scheduled_at.isoformat(),
            "asset_path": o.primary_asset_path,
        }
        for o in outputs
    ]


def get_due_for_publishing(
    db: Session,
) -> List[ContentOutput]:
    """Get outputs that are past their scheduled time and not yet published."""
    now = datetime.now(timezone.utc)
    return (
        db.query(ContentOutput)
        .filter(
            and_(
                ContentOutput.scheduled_at.isnot(None),
                ContentOutput.scheduled_at <= now,
                ContentOutput.published_at.is_(None),
            )
        )
        .order_by(ContentOutput.scheduled_at.asc())
        .all()
    )


