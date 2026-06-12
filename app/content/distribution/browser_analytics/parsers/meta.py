# FILE: app/content/distribution/browser_analytics/parsers/meta.py
# Purpose: Meta Business Suite (Facebook + Instagram combined) insights parser.
# Called-by: app.content.distribution.browser_analytics.parsers, app.content.distribution.browser_analytics.tests.test_scrape
# Depends-on: app.content.distribution.browser_analytics.parsers.common
# Last-renovated: 2026-06-11
"""
Meta Business Suite (Facebook + Instagram combined) insights parser.

Primary source: https://business.facebook.com/latest/insights/results
This page renders aggregate metric cards as plain text in the DOM,
unlike /overview which puts the numbers inside <canvas> charts.

Fallback strategy: if no aggregate cards match, sum the per-post
numbers visible on /overview's "Recent content" grid. Each post card
has text in the shape:
    "<title>\\n<title repeat>\\n<timestamp>\\n<views>\\n<reactions>\\n<comments>\\n<shares>"

The fallback gives us a *lower bound* on activity — only counts the
posts Meta chose to surface (typically the last ~5). It's enough for
"post to populate" to flip the dashboard when content exists.
"""
from __future__ import annotations

import re
from typing import Any

from app.content.distribution.browser_analytics.parsers.common import (
    Element,
    find_elements,
    first,
    parse_int,
)


# Aggregate card label -> (column, parser). Meta's labels on /results:
#   "Reach", "Content interactions", "Visits", "Follows", "Views",
#   "Views from followers", "Views from non-followers", "Viewers"
# We flatten the most useful into our schema; the rest land in metrics_json.
_AGGREGATE_LABELS: dict[str, tuple[str, Any]] = {
    "Views":                ("views",                 parse_int),
    "Viewers":              ("reach",                 parse_int),
    "Reach":                ("reach",                 parse_int),
    "Content interactions": ("content_interactions",  parse_int),
    "Follows":              ("followers_delta",       parse_int),
    "Follows net":          ("followers_delta",       parse_int),
    "Visits":               ("profile_views",         parse_int),
    "Facebook visits":      ("profile_views",         parse_int),
}


def parse_meta_overview(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Parse a Meta Business Suite insights DOM snapshot.

    Tries aggregates first (works on /insights/results); falls back
    to per-post summation if aggregates aren't present.
    """
    elements = snapshot.get("elements", []) or []
    result: dict[str, Any] = {"metrics_json": {}}

    # ─── Period (from the date-range pill) ────────────────────
    period_el = first(
        elements,
        predicate=lambda e: "days:" in (e.get("text") or "").lower()
                            or (e.get("text", "") or "").startswith("Last "),
    )
    if period_el:
        result["period"] = _normalise_period(period_el["text"])
        result["metrics_json"]["period_label"] = period_el["text"].strip()

    # ─── Strategy A: aggregate cards ──────────────────────────
    aggregates = _extract_aggregate_cards(elements)
    if aggregates:
        result["metrics_json"]["source_strategy"] = "aggregates"
        result["metrics_json"]["raw_aggregates"] = aggregates
        for label, value_str in aggregates.items():
            mapping = _AGGREGATE_LABELS.get(label)
            if mapping is None:
                continue
            column, parser_fn = mapping
            value = parser_fn(value_str)
            if value is not None:
                # Don't overwrite a higher-priority column already set
                # (e.g. Views beats Viewers for the `views` column).
                result.setdefault(column, value)

    # ─── Strategy B: per-post fallback ────────────────────────
    if "views" not in result:
        post_totals = _extract_post_totals(elements)
        if post_totals:
            result["metrics_json"]["source_strategy"] = "per_post_fallback"
            result["metrics_json"]["posts_summed"] = post_totals["post_count"]
            result["views"] = post_totals["total_views"]
            result["likes"] = post_totals["total_likes"]
            result["comments"] = post_totals["total_comments"]
            result["shares"] = post_totals["total_shares"]

    return result


# ─── Aggregate-card extraction ───────────────────────────────────────


# A metric card on /insights/results has a distinctive text shape:
# either "Label\n<number>" or "Label\n<number>\n<trend info>".
# We match labels against our known list.
def _extract_aggregate_cards(elements: list[Element]) -> dict[str, str]:
    """
    Find text nodes that look like metric cards: known label as first
    line, number as second line. Returns label -> raw value string.
    """
    hits: dict[str, str] = {}
    for el in elements:
        text = (el.get("text") or "").strip()
        if "\n" not in text:
            continue
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) < 2:
            continue
        label = lines[0]
        # Match known label exactly or with trailing "from followers" etc.
        # suffix (Meta sometimes stacks "Views\n755\nViews from followers\n33%")
        for known in _AGGREGATE_LABELS:
            if label == known and _looks_numeric(lines[1]):
                hits[known] = lines[1]
                break
    return hits


def _looks_numeric(s: str) -> bool:
    """Cheap check: does this string contain a digit?"""
    return any(ch.isdigit() for ch in s)


# ─── Per-post fallback ───────────────────────────────────────────────


# A post card link has text like:
#   "What happens is I'm your hands\nWhat happens is I'm your hands\n9 April 14:58\n81\n2\n0\n1"
# Structure: title, title (duplicated), timestamp, views, reactions, comments, shares.
# The duplication comes from Meta rendering alt-text and display-text.
_POST_CARD_MIN_LINES = 6  # title + timestamp + 4 numbers minimum
_TIMESTAMP_HINT = re.compile(
    r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", re.IGNORECASE,
)


def _extract_post_totals(elements: list[Element]) -> dict[str, int] | None:
    """
    Walk elements with href="/latest/insights/object_insights/..." and
    sum their last four numeric lines (views, reactions, comments,
    shares).
    """
    totals = {"total_views": 0, "total_likes": 0,
              "total_comments": 0, "total_shares": 0, "post_count": 0}

    for el in elements:
        href = el.get("href", "") or ""
        if "insights/object_insights" not in href:
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) < _POST_CARD_MIN_LINES:
            continue

        # The last 4 lines are the numeric metrics. Guard: they should
        # ALL parse as numbers. If not, skip this card — its shape isn't
        # what we expect.
        numeric_tail = lines[-4:]
        parsed = [parse_int(x) for x in numeric_tail]
        if any(n is None for n in parsed):
            continue

        views, likes, comments, shares = parsed
        totals["total_views"] += views or 0
        totals["total_likes"] += likes or 0
        totals["total_comments"] += comments or 0
        totals["total_shares"] += shares or 0
        totals["post_count"] += 1

    if totals["post_count"] == 0:
        return None
    return totals


# ─── Period normalisation ────────────────────────────────────────────


def _normalise_period(label: str) -> str:
    """
    'Last 28 days: 25 Mar 2026 - 21 Apr 2026' -> '28d'.
    'Last 7 days' -> '7d'. Falls back to raw label.
    """
    m = re.search(r"Last\s+(\d+)\s+days?", label, re.IGNORECASE)
    if m:
        return f"{m.group(1)}d"
    return label.strip()
