# FILE: app/web_automation/seed.py
# Purpose: Seed the default set of web-automation sessions.
# Called-by: app.web_automation
# Depends-on: app.web_automation, app.web_automation.session_registry
# Last-renovated: 2026-07-01
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
        # My Learning dashboard, NOT the public home. Logged out, Coursera
        # 302s this to the public coursera.org home — that redirect is the
        # logged-out signal the health check keys on. (2026-07-01)
        "https://www.coursera.org/my-learning",
        "Continue enrolled courses, read progress, extract lesson content, pull "
        "transcripts. Landing is the logged-in My Learning dashboard; if it "
        "redirects to the public coursera.org home the session is LOGGED OUT — "
        "say Coursera needs a fresh login, never read the public page as course "
        "state. Module completion ticks are VISUAL: read progress via "
        "web_coursera_progress (vision), not extract_text/dom_snapshot.",
    ),
    (
        "wordpress_admin",
        "WordPress Admin",
        "persist:wp-admin",
        "about:blank",
        "Manage the ASTRA UK website — posts, pages, media. User sets the real landing URL via PATCH.",
    ),
    # ── Finance & tax (2026-06-14) ──────────────────────────────────────
    # Each gets its own persistent partition so the login survives restarts.
    # ASTRA EXPLAINS and ASSISTS on these — it never files a return, pays a
    # bill, or moves money; the user authorises every submission/payment.
    (
        "quickbooks",
        "QuickBooks",
        "persist:quickbooks",
        "https://app.qbo.intuit.com/",
        "Intuit QuickBooks Online — bookkeeping, invoices, expenses, VAT and the "
        "figures behind a Self Assessment return. Help read balances and reports, "
        "reconcile transactions, and explain accounting entries. ASTRA assists and "
        "explains; it never files or pays on the user's behalf.",
    ),
    (
        "hmrc",
        "HMRC — Tax & Self Assessment",
        "persist:hmrc",
        "https://www.gov.uk/log-in-file-self-assessment-tax-return",
        "UK Government / HMRC online services for Self Assessment tax returns. When "
        "the user is working through a return and gets stuck, explain what a section, "
        "question or field means in plain English and sense-check figures against "
        "QuickBooks. ASTRA explains and assists only — the user submits the return "
        "and authorises any payment themselves.",
    ),
    # ── Display manager + desktop media (2026-06-12) ────────────────────
    # All media windows share ONE persistent partition (persist:media) so a
    # single login per site survives restarts regardless of which screen a
    # window opens on.
    (
        "displays",
        "Display Manager (control)",
        "persist:media",
        "about:blank",
        "Control session for the desktop display manager. Carries the viewless "
        "displays_list/open/close/move/windows/save_alias actions; it never "
        "opens a page of its own.",
    ),
    (
        "soundcloud",
        "SoundCloud (media)",
        "persist:media",
        "https://soundcloud.com/feed",
        "Logged-in SoundCloud feed for desktop playback on named displays. "
        "Windows open via the display manager and register under this session "
        "for click/key control.",
    ),
    (
        "mixcloud",
        "Mixcloud (media)",
        "persist:media",
        "https://www.mixcloud.com/",
        "Logged-in Mixcloud for desktop playback on named displays.",
    ),
    (
        "youtube_watch",
        "YouTube (watch)",
        "persist:media",
        "https://www.youtube.com/",
        "YouTube VIEWING windows on named displays (distinct from "
        "youtube_studio). Search itself uses the Data API; playback windows "
        "register under this session for play/pause key control.",
    ),
    (
        "streaming",
        "Streaming services (movie night)",
        "persist:media",
        "about:blank",
        "Netflix / Prime / Disney+ / iPlayer windows for movie night. Landing "
        "URL is per-film; windows open via the display manager and register "
        "under this session for profile-click and play control.",
    ),
]


def seed_sessions(db: Session) -> dict:
    """Create any missing default sessions. Returns a small summary.

    Per-row guarded: one bad row must never strand the rows after it.
    (A legacy UNIQUE(partition) constraint did exactly that — the second
    persist:media insert aborted the whole loop at 8 of 12 defaults.)
    """
    created = 0
    failed = 0
    for platform, label, partition, landing_url, purpose in DEFAULT_SESSIONS:
        try:
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
        except Exception as exc:
            db.rollback()
            failed += 1
            logger.warning(
                "[web_automation] seed failed for %r: %s", platform, exc
            )
    logger.info(
        "[web_automation] seeded %s new session(s), %s failed (total definitions: %s)",
        created, failed, len(DEFAULT_SESSIONS),
    )
    return {"created": created, "failed": failed, "total": len(DEFAULT_SESSIONS)}
