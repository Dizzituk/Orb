# FILE: app/investments/news_service.py
# Purpose: Per-holding news via ASTRA's existing Brave Search integration.
# Called-by: app.investments.chat_service, app.investments.router
# Depends-on: app.tools.registry
# Last-renovated: 2026-06-11
"""
Per-holding news via ASTRA's existing Brave Search integration.

Queries:
- For stocks: "{company_name} stock news"
- For crypto: "{coin_name} crypto news"

Returns structured news items with headline, source, snippet, URL.
"""
import logging
from typing import List, Dict, Any, Optional

from app.tools.registry import execute_tool_async

logger = logging.getLogger(__name__)


class NewsItem:
    """Structured news result."""

    def __init__(
        self,
        headline: str,
        source: str,
        url: str,
        snippet: str = "",
        published: str = "",
        ticker: str = "",
    ) -> None:
        self.headline = headline
        self.source = source
        self.url = url
        self.snippet = snippet
        self.published = published
        self.ticker = ticker

    def to_dict(self) -> Dict[str, str]:
        return {
            "headline": self.headline,
            "source": self.source,
            "url": self.url,
            "snippet": self.snippet,
            "published": self.published,
            "ticker": self.ticker,
        }


async def search_news(
    query: str,
    ticker: str = "",
    max_results: int = 5,
) -> List[Dict[str, str]]:
    """
    Search for news articles using ASTRA's web search tool.
    Returns list of NewsItem dicts.
    """
    try:
        result = await execute_tool_async(
            "web_search",
            "v1",
            {"query": query, "max_results": max_results},
        )

        raw_results = []
        if isinstance(result, dict):
            raw_results = result.get("results", [])
        elif hasattr(result, "result") and isinstance(result.result, dict):
            raw_results = result.result.get("results", [])

        items: List[Dict[str, str]] = []
        for r in raw_results:
            item = NewsItem(
                headline=r.get("title", ""),
                source=_extract_domain(r.get("url", "")),
                url=r.get("url", ""),
                snippet=r.get("description", r.get("snippet", "")),
                published=r.get("age", ""),
                ticker=ticker,
            )
            items.append(item.to_dict())

        return items

    except Exception as exc:
        logger.error("[news_service] Search failed for '%s': %s", query, exc)
        return []


async def fetch_news_for_ticker(
    ticker: str,
    name: str = "",
    asset_type: str = "stock",
    max_results: int = 3,
) -> List[Dict[str, str]]:
    """
    Fetch news for a specific holding.
    Builds appropriate query based on asset type.
    """
    display = name if name else ticker
    if asset_type == "crypto":
        query = f"{display} crypto news"
    else:
        query = f"{display} stock news"

    return await search_news(query, ticker=ticker, max_results=max_results)


async def fetch_news_aggregated(
    holdings: List[Dict[str, str]],
    max_per_holding: int = 3,
) -> List[Dict[str, str]]:
    """
    Fetch news across all holdings.
    Returns combined list sorted by recency (best effort).
    """
    all_news: List[Dict[str, str]] = []

    for h in holdings:
        items = await fetch_news_for_ticker(
            ticker=h.get("ticker", ""),
            name=h.get("name", ""),
            asset_type=h.get("asset_type", "stock"),
            max_results=max_per_holding,
        )
        all_news.extend(items)

    return all_news


def _extract_domain(url: str) -> str:
    """Extract clean domain from URL."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""
