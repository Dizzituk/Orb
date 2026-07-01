# FILE: app/watchers/snippet_source.py
# Purpose: Shared observation strategy — Brave/DDG search snippets -> median price Reading.
# Called-by: app.watchers.portugal_land, app.watchers.hardware
# Depends-on: app.tools.registry (web_search_handler), app.watchers.price_extract
# Last-renovated: 2026-07-01
"""
Deterministic scrape-and-store via the existing web_search tool (Brave
primary, DuckDuckGo fallback — both whitelisted non-LLM calls). Prices are
regex-parsed from result titles+descriptions, bounds-filtered, and the
median recorded with an honest note about the method. When Taz supplies a
stable source URL per key, a flow-memory site walk can replace this per
instance without touching the framework.
"""

from __future__ import annotations

import logging

from app.watchers.framework import Reading
from app.watchers.price_extract import extract_prices, median

logger = logging.getLogger(__name__)


async def observe_via_search(
    key: str,
    query: str,
    currency: str,
    lo: float,
    hi: float,
    max_results: int = 8,
) -> Reading:
    """One key, one search burst, one median. value=None when nothing parses."""
    from app.tools.registry import web_search_handler

    resp = await web_search_handler({"query": query, "max_results": max_results}, None)
    results = resp.get("results") or []
    provider = resp.get("provider") or "none"

    prices: list[float] = []
    domains: list[str] = []
    for r in results:
        text = f"{r.get('title') or ''} {r.get('description') or ''}"
        found = extract_prices(text, currency=currency, lo=lo, hi=hi)
        if found:
            prices.extend(found)
            url = str(r.get("url") or "")
            domain = url.split("/")[2] if url.startswith("http") and url.count("/") >= 2 else url
            if domain and domain not in domains:
                domains.append(domain)

    value = median(prices)
    if value is None:
        return Reading(
            key=key,
            value=None,
            source=f"{provider}",
            note=f"no parseable {currency} prices in {len(results)} results for: {query}",
        )
    return Reading(
        key=key,
        value=value,
        source=f"{provider}:{','.join(domains[:5])}",
        note=f"median of {len(prices)} prices from {len(domains)} source(s)",
    )
