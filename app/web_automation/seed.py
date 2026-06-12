# FILE: app/web_automation/seed.py
# Purpose: Seed the default set of web-automation sessions.
# Called-by: app.web_automation
# Depends-on: app.web_automation, app.web_automation.session_registry
# Last-renovated: 2026-06-11
"""
Seed the default set of web-automation sessions.

Idempotent — safe to call on every startup.

Landing URLs are the pages Electron navigates to when a session is
first opened. They're the authoritative "start here" for each platform.
User/ASTRA can edit these later via the PATCH /web_automation/sessions
endpoint without touching code.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

from sqlalchemy.orm import Session

from app.web_automation import session_registry

logger = logging.getLogger(__name__)


# (platform_key, label, partition, landing_url, purpose)
DEFAULT_SESSIONS: List[Tuple[str, str, str, str, str]] = [
    (
        "meta_business",
        "Meta Business Suite (FB + IG)",
        "persist:meta-business",
        "https://business.facebook.com/latest/home",
        "Unified control surface for the ASTRA UK Facebook Page and @astraukai Instagram. "
        "Publish posts and reels (cross-platform composer), read Insights for both FB and IG, "
        "and reply to comments/DMs via the unified Inbox. One login drives both platforms.",
    ),
    (
        "tiktok_astraukai",
        "TikTok — ASTRA UK",
        "persist:tt-astraukai",
        "https://www.tiktok.com/",
        "Upload TikToks, read analytics, manage comments on the ASTRA UK account.",
    ),
    (
        "youtube_studio",
        "YouTube Studio",
        "persist:yt-studio",
        "https://studio.youtube.com/",
        "Supplementary UI tasks on YouTube Studio. Primary YouTube work uses the Data API.",
    ),
    (
        "coursera",
        "Coursera",
        "persist:coursera",
        "https://www.coursera.org/",
        "Continue enrolled courses, extract lesson content, mark modules complete, pull transcripts.",
    ),
    (
        "wordpress_admin",
        "WordPress Admin",
        "persist:wp-admin",
        "about:blank",
        "Manage the ASTRA UK website — posts, pages, media. User sets the real landing URL via PATCH.",
    ),
]


def seed_sessions(db: Session) -> dict:
    """Create any missing default sessions. Returns a small summary."""
    created = 0
    for platform, label, partition, landing_url, purpose in DEFAULT_SESSIONS:
        before = session_registry.get_session_by_platform(db, platform)
        session_registry.create_session(
            db,
            platform=platform,
            label=label,
            partition=partition,
            landing_url=landing_url,
            purpose=purpose,
        )
        if before is None:
            created += 1
    logger.info(
        "[web_automation] seeded %s new session(s) (total definitions: %s)",
        created, len(DEFAULT_SESSIONS),
    )
    return {"created": created, "total": len(DEFAULT_SESSIONS)}
