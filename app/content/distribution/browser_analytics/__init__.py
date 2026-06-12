# FILE: app/content/distribution/browser_analytics/__init__.py
# Purpose: Browser-scraped analytics — reads insights pages from the logged-in
# Called-by: app.content.distribution.browser_analytics.tests.test_scrape, main
# Depends-on: app.content.distribution.browser_analytics.router
# Last-renovated: 2026-06-11
"""
Browser-scraped analytics — reads insights pages from the logged-in
WebContentsView sessions that ASTRA drives. Complements (and can
fully replace) API-based analytics when an API is walled/unavailable.

Scope in Phase 1 (recon):
    - One function per platform that navigates to the insights page,
      dismisses common popups, and dumps raw text + DOM snapshot to
      disk for review.
    - No parsing, no DB writes.

Scope in Phase 2 (parsers, next turn):
    - Per-platform parsers that extract numbers from the scraped text
    - Write channel-level rows to a new ChannelAnalytics table
    - Write per-post rows to the existing ContentAnalytics table

Scope in Phase 3 (automation):
    - LLM tool `scrape_platform_insights(platform)` for chat-driven pulls
    - Scheduler: periodic pulls on whichever sessions are live
    - Insights dashboard Refresh button triggers this path
"""
from app.content.distribution.browser_analytics.router import router

__all__ = ["router"]
