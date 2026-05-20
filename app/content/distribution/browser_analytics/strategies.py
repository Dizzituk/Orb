# FILE: app/content/distribution/browser_analytics/strategies.py
"""
Per-platform acquisition strategies.

A strategy is an ordered list of *attempts*. Each attempt describes a
URL to navigate to and optional behaviours (scroll, extra wait). The
scrape runner tries them in order until one produces metrics that pass
the meaningfulness check (at least one real number extracted).

Why a list, not a single URL? Modern SaaS analytics dashboards render
their aggregate numbers inside <canvas> or lazy-loaded React components
that don't always surface in an accessibility-tree snapshot. Meta is
the worst offender — the big hero numbers on /insights/results are
charts, invisible to our DOM grab. So we try a content-table page
first, then force lazy-loads via scrolling, then fall back to summing
per-post data from the overview page. Whichever gives us real numbers
wins; the others just leave debug dumps.
"""
from __future__ import annotations

from typing import TypedDict


class Attempt(TypedDict, total=False):
    url: str
    label: str           # short tag used in debug dumps & logs
    scroll: bool         # trigger lazy-loads by scrolling the viewport
    wait_ms: int         # initial post-navigate wait (overrides default)


# Per-platform, ordered best-first.
STRATEGIES: dict[str, list[Attempt]] = {
    # Meta: three attempts with progressively more aggressive tactics.
    "meta_business": [
        # (B) content_summary — Meta renders a content-performance table
        #     here. Tables tend to land in the DOM tree reliably.
        {
            "url":   "https://business.facebook.com/latest/insights/content_summary",
            "label": "content_summary",
            "scroll": False,
        },
        # (C) results with scroll — force the charts to initialise by
        #     scrolling past them, then snapshot. Sometimes the numeric
        #     text overlays get rendered once the chart is in-view.
        {
            "url":   "https://business.facebook.com/latest/insights/results",
            "label": "results_scrolled",
            "scroll": True,
        },
        # (A) overview — the page with the "Recent content" grid.
        #     Post cards carry their own views/likes/comments/shares
        #     counts in plain text, which the parser sums as a fallback.
        {
            "url":   "https://business.facebook.com/latest/insights/overview",
            "label": "overview_post_cards",
            "scroll": False,
        },
    ],

    # TikTok: single URL, no scroll needed.
    "tiktok_astraukai": [
        {
            "url":   "https://www.tiktok.com/tiktokstudio/analytics/overview",
            "label": "tiktok_overview",
            "scroll": False,
        },
    ],

    # YouTube: single URL, no parser registered but we still capture a
    # dump for future reference. API is the source of truth here.
    "youtube_studio": [
        {
            "url":   "https://studio.youtube.com/channel/UC/analytics/tab-overview",
            "label": "youtube_overview",
            "scroll": False,
        },
    ],
}


def get_strategy(platform: str) -> list[Attempt]:
    """
    Return the ordered list of attempts for a platform. Empty list
    means no strategy defined — caller should treat as 'unknown'.
    """
    return STRATEGIES.get(platform, [])
