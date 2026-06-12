# FILE: app/content/distribution/posting_time_learner.py
# Purpose: Posting Time Learner — optimises scheduling based on real analytics.
# Called-by: app.content.distribution.algorithm_strategy, app.content.distribution.scheduler, app.content.distribution_router
# Depends-on: app.content.models
# Last-renovated: 2026-06-11
"""
Posting Time Learner — optimises scheduling based on real analytics.

Analyses historical performance data (views, engagement) by posting
hour and day-of-week, then adjusts the scheduler's posting windows
to favour times that drive the best reach.

The learner runs periodically (e.g. weekly) or on-demand, updates
a learned_windows.json file, and the scheduler reads from it instead
of the hardcoded defaults when enough data exists.

Minimum data threshold: 10 published posts with analytics before
the learner overrides defaults. Below that, defaults are used.

v1.0 (2026-03-13): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Where learned windows are persisted
LEARNED_WINDOWS_PATH = Path(
    os.getenv("ASTRA_DATA_DIR", "D:/Orb/data")
) / "content" / "learned_posting_windows.json"

# Minimum posts with analytics before we trust the learned data
MIN_DATA_POINTS = 10

# How far back to look for analytics (days)
DEFAULT_LOOKBACK_DAYS = 90


def analyse_posting_performance(
    db: Session,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, any]:
    """Analyse historical posting performance by hour and day.

    Groups all published content with analytics by:
    - platform + format (e.g. "youtube_short", "youtube_longform")
    - day of week
    - hour of day (UTC)

    For each bucket, calculates average views and engagement rate.

    Returns a structured report with the raw data, best windows,
    and whether we have enough data to override defaults.
    """
    from app.content.models import ContentOutput, ContentAnalytics

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    # Get all published outputs with analytics
    outputs = (
        db.query(ContentOutput)
        .filter(
            ContentOutput.published_at.isnot(None),
            ContentOutput.published_at >= cutoff,
        )
        .all()
    )

    # Build performance buckets: {format_key: {(day, hour): [metrics]}}
    buckets: Dict[str, Dict[Tuple[int, int], List[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for output in outputs:
        if not output.published_at:
            continue

        pub_time = output.published_at
        if pub_time.tzinfo is None:
            pub_time = pub_time.replace(tzinfo=timezone.utc)

        day_of_week = pub_time.weekday()  # 0=Mon, 6=Sun
        hour = pub_time.hour

        # Get the best analytics snapshot for this output
        latest_snap = (
            db.query(ContentAnalytics)
            .filter(ContentAnalytics.output_id == output.id)
            .order_by(ContentAnalytics.snapshot_at.desc())
            .first()
        )
        if not latest_snap:
            continue

        # Determine the format key
        format_key = output.output_format or output.platform
        platform_key = output.platform

        metrics = {
            "views": latest_snap.views or 0,
            "engagement_rate": latest_snap.engagement_rate or 0.0,
            "likes": latest_snap.likes or 0,
            "output_id": output.id,
        }

        # Store under both format-specific and platform-level keys
        buckets[format_key][(day_of_week, hour)].append(metrics)
        if format_key != platform_key:
            buckets[platform_key][(day_of_week, hour)].append(metrics)

    # Compute averages per bucket
    report = {}
    for format_key, time_buckets in buckets.items():
        total_posts = sum(len(v) for v in time_buckets.values())
        slots = {}

        for (day, hour), metrics_list in time_buckets.items():
            avg_views = sum(m["views"] for m in metrics_list) / len(metrics_list)
            avg_engagement = sum(m["engagement_rate"] for m in metrics_list) / len(metrics_list)

            slots[f"{day}_{hour}"] = {
                "day_of_week": day,
                "hour_utc": hour,
                "post_count": len(metrics_list),
                "avg_views": round(avg_views, 1),
                "avg_engagement_rate": round(avg_engagement, 4),
                # Combined score: views weighted 70%, engagement 30%
                # Normalised later
                "raw_score": avg_views,
            }

        # Rank slots by views (primary) then engagement (tiebreaker)
        ranked = sorted(
            slots.values(),
            key=lambda s: (s["avg_views"], s["avg_engagement_rate"]),
            reverse=True,
        )

        # Pick the best 4 windows
        best_windows = []
        seen_hours = set()
        for slot in ranked:
            h = slot["hour_utc"]
            if h not in seen_hours:
                best_windows.append((slot["hour_utc"], 0))
                seen_hours.add(h)
            if len(best_windows) >= 4:
                break

        # Sort by hour for readability
        best_windows.sort()

        has_enough_data = total_posts >= MIN_DATA_POINTS

        report[format_key] = {
            "total_posts_analysed": total_posts,
            "has_enough_data": has_enough_data,
            "slots": slots,
            "best_windows": best_windows,
            "best_days": _extract_best_days(ranked),
        }

    return report


def _extract_best_days(ranked_slots: list) -> List[str]:
    """Extract the best performing days from ranked slots."""
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]

    # Aggregate views by day
    day_views: Dict[int, float] = defaultdict(float)
    day_counts: Dict[int, int] = defaultdict(int)

    for slot in ranked_slots:
        day = slot["day_of_week"]
        day_views[day] += slot["avg_views"]
        day_counts[day] += 1

    # Average per day and rank
    day_avg = {
        day: day_views[day] / day_counts[day]
        for day in day_views
        if day_counts[day] > 0
    }

    ranked_days = sorted(day_avg.items(), key=lambda x: x[1], reverse=True)
    return [day_names[d] for d, _ in ranked_days[:3]]


def update_learned_windows(
    db: Session,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> Dict[str, any]:
    """Run the analysis and persist learned windows to disk.

    The scheduler reads this file at runtime to override defaults
    when enough data exists.

    Returns the full analysis report.
    """
    report = analyse_posting_performance(db, lookback_days)

    # Build the windows file
    learned = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "lookback_days": lookback_days,
        "windows": {},
    }

    for format_key, data in report.items():
        if data["has_enough_data"] and data["best_windows"]:
            learned["windows"][format_key] = {
                "posting_hours": data["best_windows"],
                "best_days": data["best_days"],
                "posts_analysed": data["total_posts_analysed"],
                "confidence": "learned",
            }
        else:
            learned["windows"][format_key] = {
                "posting_hours": data.get("best_windows", []),
                "best_days": data.get("best_days", []),
                "posts_analysed": data["total_posts_analysed"],
                "confidence": "insufficient_data",
            }

    # Persist
    LEARNED_WINDOWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_WINDOWS_PATH.write_text(json.dumps(learned, indent=2))
    logger.info(
        "[posting_time_learner] Updated learned windows: %d formats, %d total posts",
        len(learned["windows"]),
        sum(d["total_posts_analysed"] for d in report.values()),
    )

    return report


def get_learned_windows(
    format_key: str,
) -> Optional[List[Tuple[int, int]]]:
    """Read learned posting windows for a format/platform.

    Returns a list of (hour, minute) tuples, or None if no learned
    data exists (caller should fall back to defaults).
    """
    if not LEARNED_WINDOWS_PATH.exists():
        return None

    try:
        data = json.loads(LEARNED_WINDOWS_PATH.read_text())
        windows = data.get("windows", {}).get(format_key, {})

        if windows.get("confidence") != "learned":
            return None

        hours = windows.get("posting_hours", [])
        if not hours:
            return None

        # Convert to (hour, minute) tuples
        return [(h, m) if isinstance(h, int) else (h[0], h[1]) for h, m in hours]

    except Exception as e:
        logger.warning("[posting_time_learner] Failed to read learned windows: %s", e)
        return None


def get_learned_best_days(
    format_key: str,
) -> Optional[List[str]]:
    """Read learned best days for a format/platform.

    Returns a list of day names, or None if no learned data.
    """
    if not LEARNED_WINDOWS_PATH.exists():
        return None

    try:
        data = json.loads(LEARNED_WINDOWS_PATH.read_text())
        windows = data.get("windows", {}).get(format_key, {})

        if windows.get("confidence") != "learned":
            return None

        return windows.get("best_days") or None

    except Exception:
        return None
