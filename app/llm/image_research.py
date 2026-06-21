# FILE: app/llm/image_research.py
# Purpose: Research-before-chart helper for data-driven image requests (leaf).
# Called-by: app.llm.image_router (shim), app.llm.image_stream, app.llm.image_core
# Depends-on: app.llm.chart_research / app.llm.web_search (lazy)
# Last-renovated: 2026-06-21
"""Targeted multi-search research used before chart rendering.

Pure leaf split out of image_router.py (batch 3, 2026-06-21). _needs_research is
retained verbatim for completeness (its live counterpart lives in
image_type_classifier.py). Imports nothing back from the image modules.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Keywords that suggest the user wants data gathered before image creation
_RESEARCH_KEYWORDS = [
    "latest", "recent", "current", "data", "benchmark", "benchmarks",
    "statistics", "stats", "compare", "comparison", "research",
    "findings", "results", "performance", "ranking", "rankings",
    "trends", "trend", "chart", "graph", "infographic",
    "numbers", "figures", "metrics", "scores",
]


def _needs_research(message: str) -> bool:
    """Detect if the image request implies data gathering first."""
    lower = message.lower()
    return any(kw in lower for kw in _RESEARCH_KEYWORDS)


async def _run_research(message: str) -> Optional[str]:
    """Run targeted multi-search research for chart data.

    Uses LLM to plan focused queries, runs multiple Brave searches,
    and combines results for richer data extraction.
    """
    try:
        from app.llm.chart_research import run_multi_search
        return await run_multi_search(message)
    except ImportError:
        logger.warning("[image_stream] chart_research not available, falling back to single search")
        # Fallback: single search with cleaned query
        try:
            from app.llm.web_search import WebSearchRequest, search_and_answer
            from app.llm.chart_research import _clean_query
            query = _clean_query(message)
            req = WebSearchRequest(query=query, max_results=5)
            result = await search_and_answer(req)
            if result.ok and result.answer:
                return result.answer
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning("[image_stream] Research step failed: %s", e)
        return None
