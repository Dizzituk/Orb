# FILE: app/content/distribution/browser_analytics/parsers/tiktok.py
# Purpose: TikTok Studio analytics parser.
# Called-by: app.content.distribution.browser_analytics.parsers, app.content.distribution.browser_analytics.tests.test_scrape
# Depends-on: app.content.distribution.browser_analytics.parsers.common
# Last-renovated: 2026-06-11
"""
TikTok Studio analytics parser.

Source page: https://www.tiktok.com/tiktokstudio/analytics/overview

The overview page shows six metric cards at the top of the main area,
each rendered as a button element with text in the form:

    "<metric_name>\\n<current_value>\\n<delta>\\n<delta_pct>"

Examples from recon (all zero because nothing posted yet):
    "Video views\\n0\\n0\\n(--)"
    "Profile views\\n0\\n-1\\n(-100.0%)"
    "Likes\\n0\\n0\\n(--)"
    "Comments\\n0\\n0\\n(--)"
    "Shares\\n0\\n0\\n(--)"
    "Est. rewards\\n$0\\n0\\n(0.0%)"

The date range is in a separate button near the top-right:
    "Last 7 days" (default), "Last 28 days", "Last 60 days", etc.
"""
from __future__ import annotations

from typing import Any

from app.content.distribution.browser_analytics.parsers.common import (
    Element,
    find_elements,
    first,
    parse_int,
    parse_number,
    parse_percent,
)


# Metric card names as they appear in TikTok Studio's UI.
# Mapping: UI label -> (our_column_name, parser_fn)
_METRIC_CARDS = {
    "Video views":    ("views",            parse_int),
    "Profile views":  ("profile_views",    parse_int),
    "Likes":          ("likes",            parse_int),
    "Comments":       ("comments",         parse_int),
    "Shares":         ("shares",           parse_int),
    "Est. rewards":   ("estimated_earnings", parse_number),
}


def parse_tiktok_overview(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a TikTok Studio overview DOM snapshot.

    Returns a dict with keys matching ChannelAnalytics columns:
        views, likes, comments, shares, profile_views, estimated_earnings,
        period, metrics_json

    metrics_json carries the per-metric delta info (current, delta,
    delta_pct) so we can chart week-over-week movement later without
    needing to re-scrape.
    """
    elements = snapshot.get("elements", []) or []
    result: dict[str, Any] = {"metrics_json": {"raw_cards": {}}}

    # ─── Date range (period) ──────────────────────────────────
    # The range button sits around x=868 y=18 — but the safer approach
    # is text-based: "Last 7 days", "Last 28 days", etc.
    period_el = first(elements, text_starts_with="Last ")
    if period_el:
        period_text = period_el["text"].strip()
        # Normalise: "Last 7 days" -> "7d", "Last 28 days" -> "28d"
        result["period"] = _normalise_period(period_text)
        result["metrics_json"]["period_label"] = period_text

    # ─── Metric cards ─────────────────────────────────────────
    for label, (column, parser_fn) in _METRIC_CARDS.items():
        card = _find_metric_card(elements, label)
        if card is None:
            continue

        parsed = _parse_card_text(card["text"])
        if parsed is None:
            continue

        current_value, delta_raw, delta_pct_raw = parsed
        value = parser_fn(current_value)
        if value is not None:
            result[column] = value

        # Stash the delta info for trend analysis
        result["metrics_json"]["raw_cards"][label] = {
            "value": current_value,
            "delta": delta_raw,
            "delta_pct": delta_pct_raw,
            "delta_numeric": parse_number(delta_raw),
            "delta_pct_numeric": parse_percent(delta_pct_raw),
        }

    return result


# ─── Internal helpers ────────────────────────────────────────────────


def _find_metric_card(elements: list[Element], label: str) -> Element | None:
    """
    Find the metric card whose text starts with the given label.
    We match on text_starts_with rather than contains because some
    labels are substrings of others (e.g. "Likes" vs "Like rate").
    """
    matches = find_elements(elements, text_starts_with=f"{label}\n")
    return matches[0] if matches else None


def _parse_card_text(text: str) -> tuple[str, str, str] | None:
    """
    A card's text is "label\ncurrent\ndelta\n(delta_pct)".
    Returns (current, delta, delta_pct) as raw strings, or None if
    the shape isn't what we expect.
    """
    lines = text.split("\n")
    if len(lines) < 4:
        return None
    # lines[0] = label (already matched)
    current = lines[1].strip()
    delta = lines[2].strip()
    delta_pct = lines[3].strip()
    return current, delta, delta_pct


def _normalise_period(label: str) -> str:
    """'Last 7 days' -> '7d'.  'Last 28 days' -> '28d'.  Falls back to raw."""
    import re
    m = re.match(r"Last\s+(\d+)\s+days?", label, re.IGNORECASE)
    if m:
        return f"{m.group(1)}d"
    return label
