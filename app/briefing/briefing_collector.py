from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BriefingStory:
    title: str
    url: str
    snippet: str = ""
    source_name: str = ""
    credibility_label: str = "unknown"
    source_type: str = "unknown"
    bias_note: str = ""
    topic_key: str = ""
    astra_relevant: bool = False
    page_text: str = ""


@dataclass
class TopicCollection:
    topic_key: str
    topic_name: str
    description: str = ""
    stories: list[BriefingStory] = field(default_factory=list)
    query_count: int = 0
    error: str = ""


@dataclass
class BriefingCollection:
    topics: list[TopicCollection] = field(default_factory=list)
    total_stories: int = 0
    errors: list[str] = field(default_factory=list)
    profile: str = "default"


async def _search_topic(
    topic_key: str,
    topic_name: str,
    description: str,
    queries: list[str],
    max_stories: int,
    astra_relevant: bool,
    context: Optional[dict] = None,
) -> TopicCollection:
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
        except Exception as exc:
            logger.warning("[briefing_collector] Search failed: query='%s', error=%s", query, exc)
            continue

    return TopicCollection(
        topic_key=topic_key,
        topic_name=topic_name,
        description=description,
        stories=stories,
        query_count=len(queries),
    )


async def collect_all_topics(context: Optional[dict] = None, profile: str = "default") -> BriefingCollection:
    from app.briefing.briefing_config import get_topics

    topics = get_topics(profile=profile)
    if not topics:
        return BriefingCollection(errors=["No topics configured"], profile=profile)

    tasks = [
        _search_topic(
            topic_key=topic.key,
            topic_name=topic.name,
            description=topic.description,
            queries=topic.search_queries,
            max_stories=topic.max_stories,
            astra_relevant=topic.astra_relevant,
            context=context,
        )
        for topic in topics
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    collection = BriefingCollection(profile=profile)
    for result in results:
        if isinstance(result, Exception):
            collection.errors.append(str(result))
            continue
        collection.topics.append(result)
        collection.total_stories += len(result.stories)

    logger.info(
        "[briefing_collector] Collection complete: profile=%s, topics=%d, stories=%d, errors=%d",
        profile, len(collection.topics), collection.total_stories, len(collection.errors),
    )
    return collection


__all__ = [
    "BriefingStory",
    "TopicCollection",
    "BriefingCollection",
    "collect_all_topics",
]
