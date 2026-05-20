# FILE: app/content/distribution/browser_analytics/urls.py
"""
Best-guess insights URLs per platform. Editable without touching code
elsewhere — if a platform moves its analytics page, change it here.

Meta: https://business.facebook.com/latest/insights  — cross-account
  insights landing for the Business Suite. Auto-redirects to the
  currently-selected Page/IG account's insights.

TikTok Studio: https://www.tiktok.com/tiktokstudio/analytics/overview
  — web version of TikTok Studio, shows follower count and view stats.

YouTube Studio: https://studio.youtube.com/channel/UC/analytics/tab-overview
  — the 'UC' in the path is a wildcard Google resolves to the logged-in
  user's channel. Redirects to /channel/<actual-id>/analytics.
"""
from __future__ import annotations

# platform_key (matches WebSession.platform) -> URL to scrape
INSIGHTS_URLS: dict[str, str] = {
    # /insights/results renders aggregate numbers (reach, interactions,
    # follows, visits) directly in the DOM text, unlike /overview which
    # puts them in charts that don't surface in our AX snapshot.
    "meta_business":     "https://business.facebook.com/latest/insights/results",
    "tiktok_astraukai":  "https://www.tiktok.com/tiktokstudio/analytics/overview",
    "youtube_studio":    "https://studio.youtube.com/channel/UC/analytics/tab-overview",
}


def get_insights_url(platform: str) -> str | None:
    return INSIGHTS_URLS.get(platform)
