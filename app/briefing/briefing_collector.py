# FILE: app/briefing/briefing_collector.py
"""
Briefing Collector — Gathers news stories via multi-topic web searches.

For each enabled topic category, runs the configured search queries,
deduplicates results, tags source credibility, and returns a structured
collection of stories grouped by topic.

Uses the existing web search infrastructure (Brave primary, DDG fallback)
and source classifier.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Story model
# =========================================================================

@dataclass
class BriefingStory:
    """A single news story for the briefing."""
    title: str
    url: str
    snippet: str = ""
    source_name: str = ""
    credibility_label: str = "unknown"
    source_type: str = "unknown"
    bias_note: str = ""
    topic_key: str = ""               # Which category this belongs to
    astra_relevant: bool = False      # Flagged as relevant to ASTRA's domain
    page_text: str = ""               # Full page text (if fetched)


@dataclass
class TopicCollection:
    """Stories collected for a single topic category."""
    topic_key: str
    topic_name: str
    description: str = ""
    stories: List[BriefingStory] = field(default_factory=list)
    query_count: int = 0
    error: str = ""


@dataclass
class BriefingCollection:
    """Complete collection across all topics."""
    topics: List[TopicCollection] = field(default_factory=list)
    total_stories: int = 0
    errors: List[str] = field(default_factory=list)


# =========================================================================
# Collection logic
# =========================================================================

async def _search_topic(
    topic_key: str,
    topic_name: str,
    description: str,
    queries: list[str],
    max_stories: int,
    astra_relevant: bool,
    context: Optional[dict] = None,
) -> TopicCollection:
    """Run searches for a single topic and return collected stories."""
    try:
        from app.llm.web_search import WebSearchRequest, search_and_answer
    except ImportError:
        return TopicCollection(
            topic_key=topic_key,
            topic_name=topic_name,
            description=description,
            error="web_search module not available",
        )

    seen_urls: set[str] = set()
    stories: list[BriefingStory] = []

    for query in queries:
        if len(stories) >= max_stories:
            break
        try:
            req = WebSearchRequest(query=query, max_results=5)
            resp = await search_and_answer(req, context=context)

            if not resp.ok or not resp.sources:
                continue

            for src in resp.sources:
                if src.url in seen_urls:
                    continue
                if len(stories) >= max_stories:
                    break
                seen_urls.add(src.url)
                stories.append(BriefingStory(
                    title=src.title,
                    url=src.url,
                    snippet=src.snippet,
                    credibility_label=src.credibility_label,
                    source_type=src.source_type,
                    bias_note=src.bias_note,
                    topic_key=topic_key,
                    astra_relevant=astra_relevant,
                ))
        except Exception as e:
            logger.warning("[briefing_collector] Search failed: query='%s', error=%s", query, e)
            continue

    return TopicCollection(
        topic_key=topic_key,
        topic_name=topic_name,
        description=description,
        stories=stories,
        query_count=len(queries),
    )


async def collect_all_topics(
    context: Optional[dict] = None,
) -> BriefingCollection:
    """
    Collect stories for all enabled topic categories.

    Runs searches concurrently across topics for speed.
    Returns a BriefingCollection with all gathered stories.
    """
    from app.briefing.briefing_config import get_topics

    topics = get_topics()
    if not topics:
        return BriefingCollection(errors=["No topics configured"])

    # Run topic searches concurrently
    tasks = [
        _search_topic(
            topic_key=t.key,
            topic_name=t.name,
            description=t.description,
            queries=t.search_queries,
            max_stories=t.max_stories,
            astra_relevant=t.astra_relevant,
            context=context,
        )
        for t in topics
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    collection = BriefingCollection()
    for result in results:
        if isinstance(result, Exception):
            collection.errors.append(str(result))
            continue
        collection.topics.append(result)
        collection.total_stories += len(result.stories)

    logger.info(
        "[briefing_collector] Collection complete: %d topics, %d stories, %d errors",
        len(collection.topics), collection.total_stories, len(collection.errors),
    )
    return collection


__all__ = [
    "BriefingStory",
    "TopicCollection",
    "BriefingCollection",
    "collect_all_topics",
]
