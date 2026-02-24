# FILE: app/content/distribution/analytics.py
"""
Analytics Feedback Loop (Spec Section 9.3).

Pulls engagement metrics from platform APIs after publishing
and feeds them back into the content scoring and style profile
optimisation systems.

Metrics tracked per Spec Section 9.3:
- Watch time / retention
- Engagement rate
- Completion rate
- Follower growth per post
- Click-through rate
- Save/share rates
- Traffic sources
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from app.content.models import (
    ContentOutput, ContentAnalytics, ContentPiece,
)

logger = logging.getLogger(__name__)


async def pull_analytics_for_output(
    db: Session,
    output_id: str,
) -> Optional[ContentAnalytics]:
    """
    Pull fresh analytics from the platform API for a published output.
    Creates a new analytics snapshot.
    """
    output = db.query(ContentOutput).get(output_id)
    if not output or not output.published_at:
        return None

    platform = output.platform
    post_id = output.platform_post_id

    if not post_id:
        logger.warning(f"[analytics] No platform post ID for output {output_id}")
        return None

    metrics = None

    if platform == "youtube":
        from app.content.distribution.youtube import get_video_analytics
        raw = await get_video_analytics(post_id)
        if raw:
            metrics = {
                "views": raw.get("views", 0),
                "likes": raw.get("likes", 0),
                "comments": raw.get("comments", 0),
            }

    elif platform == "instagram":
        from app.content.distribution.instagram import get_media_insights
        raw = await get_media_insights(post_id)
        if raw:
            metrics = {
                "views": raw.get("impressions", 0),
                "likes": raw.get("likes", 0),
                "comments": raw.get("comments", 0),
                "saves": raw.get("saved", 0),
                "shares": raw.get("shares", 0),
            }

    if not metrics:
        return None

    # Calculate engagement rate
    views = metrics.get("views", 0)
    if views > 0:
        engagement_actions = (
            metrics.get("likes", 0) +
            metrics.get("comments", 0) +
            metrics.get("shares", 0) +
            metrics.get("saves", 0)
        )
        metrics["engagement_rate"] = engagement_actions / views
    else:
        metrics["engagement_rate"] = 0.0

    # Create snapshot
    snapshot = ContentAnalytics(
        output_id=output_id,
        views=metrics.get("views", 0),
        likes=metrics.get("likes", 0),
        comments=metrics.get("comments", 0),
        shares=metrics.get("shares", 0),
        saves=metrics.get("saves", 0),
        engagement_rate=metrics.get("engagement_rate"),
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)

    logger.info(
        f"[analytics] Snapshot for {platform}/{post_id}: "
        f"{metrics.get('views', 0)} views, "
        f"{metrics.get('engagement_rate', 0):.2%} engagement"
    )
    return snapshot


async def pull_all_recent_analytics(
    db: Session,
    max_age_days: int = 7,
) -> Dict[str, Any]:
    """
    Pull fresh analytics for all recently published outputs.
    Returns summary of updates.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    outputs = (
        db.query(ContentOutput)
        .filter(
            and_(
                ContentOutput.published_at.isnot(None),
                ContentOutput.published_at >= cutoff,
                ContentOutput.platform_post_id.isnot(None),
            )
        )
        .all()
    )

    updated = 0
    failed = 0
    for output in outputs:
        try:
            result = await pull_analytics_for_output(db, output.id)
            if result:
                updated += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"[analytics] Pull failed for {output.id}: {e}")
            failed += 1

    return {
        "outputs_checked": len(outputs),
        "updated": updated,
        "failed": failed,
    }


# ═══════════════════════════════════════════════════
# ANALYTICS QUERIES
# ═══════════════════════════════════════════════════

def get_performance_summary(
    db: Session,
    days: int = 30,
) -> Dict[str, Any]:
    """
    Get overall content performance summary.
    Used to inform content scoring and strategy.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Get latest snapshot per output
    outputs = (
        db.query(ContentOutput)
        .filter(ContentOutput.published_at >= cutoff)
        .all()
    )

    by_platform = {}
    by_format = {}
    by_category = {}

    for output in outputs:
        # Get latest analytics snapshot
        latest = (
            db.query(ContentAnalytics)
            .filter(ContentAnalytics.output_id == output.id)
            .order_by(ContentAnalytics.snapshot_at.desc())
            .first()
        )
        if not latest:
            continue

        # Aggregate by platform
        _aggregate(by_platform, output.platform, latest)

        # Aggregate by format
        _aggregate(by_format, output.output_format, latest)

        # Aggregate by content category
        piece = db.query(ContentPiece).get(output.piece_id)
        if piece:
            _aggregate(by_category, piece.content_category, latest)

    return {
        "period_days": days,
        "by_platform": _summarise(by_platform),
        "by_format": _summarise(by_format),
        "by_category": _summarise(by_category),
    }


def _aggregate(
    bucket: Dict, key: str, snapshot: ContentAnalytics,
) -> None:
    """Aggregate analytics into a bucket."""
    if key not in bucket:
        bucket[key] = {
            "count": 0, "total_views": 0, "total_likes": 0,
            "total_comments": 0, "total_shares": 0,
            "total_saves": 0, "engagement_rates": [],
        }

    b = bucket[key]
    b["count"] += 1
    b["total_views"] += snapshot.views or 0
    b["total_likes"] += snapshot.likes or 0
    b["total_comments"] += snapshot.comments or 0
    b["total_shares"] += snapshot.shares or 0
    b["total_saves"] += snapshot.saves or 0
    if snapshot.engagement_rate is not None:
        b["engagement_rates"].append(snapshot.engagement_rate)


def _summarise(bucket: Dict) -> Dict[str, Any]:
    """Summarise aggregated analytics."""
    result = {}
    for key, b in bucket.items():
        rates = b["engagement_rates"]
        avg_rate = sum(rates) / len(rates) if rates else 0.0

        result[key] = {
            "post_count": b["count"],
            "total_views": b["total_views"],
            "avg_views": b["total_views"] / b["count"] if b["count"] else 0,
            "total_engagement": (
                b["total_likes"] + b["total_comments"] +
                b["total_shares"] + b["total_saves"]
            ),
            "avg_engagement_rate": round(avg_rate, 4),
        }
    return result


def get_top_performing(
    db: Session,
    metric: str = "views",
    limit: int = 10,
    days: int = 30,
) -> List[Dict[str, Any]]:
    """Get top performing content by a specific metric."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Map metric names to model columns
    metric_col = {
        "views": ContentAnalytics.views,
        "likes": ContentAnalytics.likes,
        "comments": ContentAnalytics.comments,
        "shares": ContentAnalytics.shares,
        "saves": ContentAnalytics.saves,
        "engagement_rate": ContentAnalytics.engagement_rate,
    }.get(metric, ContentAnalytics.views)

    snapshots = (
        db.query(ContentAnalytics)
        .filter(ContentAnalytics.snapshot_at >= cutoff)
        .order_by(metric_col.desc())
        .limit(limit)
        .all()
    )

    results = []
    for snap in snapshots:
        output = db.query(ContentOutput).get(snap.output_id)
        piece = db.query(ContentPiece).get(output.piece_id) if output else None

        results.append({
            "piece_title": piece.title if piece else "Unknown",
            "format": output.output_format if output else "Unknown",
            "platform": output.platform if output else "Unknown",
            "views": snap.views,
            "likes": snap.likes,
            "engagement_rate": snap.engagement_rate,
            "published_at": (
                output.published_at.isoformat()
                if output and output.published_at else None
            ),
        })

    return results
