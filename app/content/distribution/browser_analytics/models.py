# FILE: app/content/distribution/browser_analytics/models.py
# Purpose: Channel-level analytics — aggregated stats for a whole account/page,
# Called-by: app.content.distribution.browser_analytics.router, app.content.distribution.browser_analytics.scrape, app.content.distribution.browser_analytics.tests.test_scrape, main
# Depends-on: app.db
# Last-renovated: 2026-06-11
"""
Channel-level analytics — aggregated stats for a whole account/page,
separate from per-post metrics.

Distinction from ContentAnalytics:
    ContentAnalytics: one row per published post snapshot. Views,
        likes, engagement for ONE specific piece of content.
    ChannelAnalytics: one row per scrape of the platform's analytics
        dashboard. Total views across the channel, follower count,
        profile views — the channel as a whole.

A new row is written each time the scraper runs (append-only), which
lets us track trends over time. The dashboard reads the latest row
for each platform.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Float, DateTime, JSON
from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class ChannelAnalytics(Base):
    """
    One snapshot of channel-level metrics from a platform's analytics page.

    Most fields are nullable because platforms expose different metrics:
    TikTok shows profile_views but Meta doesn't; Meta shows follows_new
    but TikTok doesn't. Raw per-platform metrics get stashed in
    `metrics_json` for anything we couldn't flatten into columns.
    """
    __tablename__ = "channel_analytics"

    id = Column(String, primary_key=True, default=_uuid)
    platform = Column(String, nullable=False, index=True)
    # "facebook", "instagram", "tiktok", "youtube", "meta_business" (combined)
    captured_at = Column(DateTime(timezone=True), nullable=False, default=_now, index=True)

    # Time period the metrics cover as shown on the dashboard
    # (e.g. "7d", "28d", "90d"). String rather than int so we can
    # represent non-round periods if needed.
    period = Column(String, nullable=True)

    # Core metrics — nullable because not all platforms expose all fields
    views = Column(Integer, nullable=True)           # total views in period
    likes = Column(Integer, nullable=True)
    comments = Column(Integer, nullable=True)
    shares = Column(Integer, nullable=True)
    saves = Column(Integer, nullable=True)
    profile_views = Column(Integer, nullable=True)   # TikTok/IG specific
    reach = Column(Integer, nullable=True)           # Meta specific
    content_interactions = Column(Integer, nullable=True)  # Meta specific

    # Followers — absolute + delta in period
    followers_total = Column(Integer, nullable=True)
    followers_delta = Column(Integer, nullable=True)

    # Monetisation
    estimated_earnings = Column(Float, nullable=True)

    # Source tracking
    source = Column(String, nullable=False, default="browser_scrape")
    # "browser_scrape" | "api" | "manual"

    # URL we landed on when scraping (for debug)
    source_url = Column(String, nullable=True)

    # Everything we couldn't flatten into a column — raw parser output
    metrics_json = Column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ChannelAnalytics {self.platform} "
            f"period={self.period} views={self.views} "
            f"captured_at={self.captured_at}>"
        )
