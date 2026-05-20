# FILE: app/content/distribution/browser_analytics/parsers/__init__.py
"""
Per-platform parsers that extract channel-level metrics from a
DOM accessibility tree snapshot.

Each parser takes the raw `dom_snapshot` result (dict with 'elements'
list) and returns a dict with standardised keys mapping to
ChannelAnalytics columns, plus a 'raw' key for anything platform-
specific that couldn't be flattened.
"""
from app.content.distribution.browser_analytics.parsers.tiktok import (
    parse_tiktok_overview,
)
from app.content.distribution.browser_analytics.parsers.meta import (
    parse_meta_overview,
)

# Registry: platform (matching WebSession.platform) -> parser function
PARSERS = {
    "tiktok_astraukai": parse_tiktok_overview,
    "meta_business":   parse_meta_overview,
    # "youtube_studio": skipped — YouTube Data API covers channel stats better
}


def get_parser(platform: str):
    return PARSERS.get(platform)


__all__ = ["PARSERS", "get_parser", "parse_tiktok_overview", "parse_meta_overview"]
