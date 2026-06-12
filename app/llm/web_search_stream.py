# FILE: app/llm/web_search_stream.py
# Purpose: SSE stream handler for web search intent.
# Called-by: app.llm.routing.handler_registry
# Depends-on: app.llm.web_search, app.memory, app.memory.schemas, app.memory.service
# Last-renovated: 2026-06-11
"""
SSE stream handler for web search intent.

Receives a natural language query, runs it through the web search
orchestrator (Brave primary, DuckDuckGo fallback), and streams
the answer back as SSE events.

v2.7 (2026-03): Multi-topic query splitting + better logging + resilient streaming.
v2.6 (2026-03): Replaced search status token with structured web_search_status/web_search_sources events.
v2.5 (2026-02): Saves user + assistant messages to history (persistence fix).
v2.4 (2026-02): Sends sources as structured 'sources' event for rich rendering.
v2.3 (2026-02): Shows actual synthesis model in metadata badge.
v2.2 (2026-02): Fixed SSE event types to match frontend expectations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import AsyncIterator, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _sse(event_type: str, **kwargs) -> str:
    """Build an SSE data line with the given type and fields."""
    payload = {"type": event_type, **kwargs}
    return f"data: {json.dumps(payload)}\n\n"


def _save_to_history(
    db: Session,
    project_id: int,
    user_message: str,
    assistant_content: str,
    provider: str,
    model: str,
) -> None:
    """Persist user + assistant messages so they survive reload."""
    try:
        from app.memory import service as mem_svc, schemas as mem_schemas
        mem_svc.create_message(db, mem_schemas.MessageCreate(
            project_id=project_id,
            role="user",
            content=user_message,
            provider="local",
        ))
        mem_svc.create_message(db, mem_schemas.MessageCreate(
            project_id=project_id,
            role="assistant",
            content=assistant_content,
            provider=provider,
            model=model,
        ))
    except Exception as e:
        logger.warning("[web_search_stream] Failed to save history: %s", e)


def _extract_search_queries(raw_query: str) -> list[str]:
    """Extract focused keyword queries from a natural language request.

    For multi-topic requests like "update on frontier AI, robotics, and quantum computing",
    this splits into separate Brave queries per topic for better coverage.

    Returns a list of 1-5 short keyword queries.
    """
    q = raw_query.strip()
    if not q:
        return []

    # Strip conversational preamble
    preamble_patterns = [
        r"^(?:astra[,.]?\s+)?(?:i\s+need\s+you\s+to\s+)?(?:go\s+on\s+to\s+the\s+internet\s+and\s+)?",
        r"(?:get\s+me\s+)?(?:a\s+)?(?:full\s+)?(?:update\s+(?:of|on)\s+)?",
        r"(?:where\s+)?(?:(?:things\s+)?(?:stand|are)\s+(?:at\s+)?(?:with\s+)?)?",
        r"(?:(?:at\s+)?(?:this\s+)?(?:present\s+)?(?:moment\s+)?(?:in\s+time)?[,.]?\s*)?",
        r"(?:which\s+is\s+\w+\s+(?:the\s+)?\d+(?:st|nd|rd|th)?[,.]?\s*\d*[,.]?\s*)?",
        r"(?:(?:in|as\s+of)\s+\w+\s+\d{4}[,.]?\s*)?",
    ]
    stripped = q
    for pat in preamble_patterns:
        stripped = re.sub(pat, "", stripped, flags=re.IGNORECASE).strip()

    # If stripping removed everything, fall back to original
    if len(stripped) < 5:
        stripped = q

    # Try to detect comma/and-separated topics
    topic_split = re.split(
        r'\s*(?:,\s*(?:and\s+)?|(?:\s+and\s+)|(?:\s*&\s*))',
        stripped,
        flags=re.IGNORECASE,
    )
    topics = [t.strip().rstrip('.') for t in topic_split if t.strip() and len(t.strip()) > 2]

    if len(topics) >= 2:
        queries = []
        date_suffix = " 2026"
        for topic in topics[:5]:
            q_clean = topic.strip()[:50]
            queries.append(f"{q_clean} latest{date_suffix}")
        return queries

    # Single topic: use the cleaned query
    single = stripped[:120].strip()
    if not single:
        single = q[:120].strip()
    return [single]


def _build_sources_data(sources) -> tuple[list[dict], list[str]]:
    """Build sources data dicts and markdown links from source objects."""
    sources_data = []
    source_lines = []
    for src in sources:
        sources_data.append({
            "title": src.title,
            "url": src.url,
            "snippet": src.snippet,
            "credibility_label": src.credibility_label,
            "source_type": src.source_type,
        })
        source_lines.append(f"[{src.title}]({src.url})")
    return sources_data, source_lines


async def generate_web_search_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[object] = None,
    extracted_query: Optional[str] = None,
) -> AsyncIterator[str]:
    """Stream web search results as SSE events."""
    synth_provider = "brave"
    synth_model = "web_search"
    full_response = ""

    try:
        from app.llm.web_search import (
            WebSearchRequest,
            search_and_answer,
        )
    except ImportError:
        yield _sse("error", error="Web search module not available.")
        yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
        return

    raw_query = (extracted_query or message).strip()
    logger.info(
        "[web_search_stream] extracted_query=%r, message=%r, raw_query=%r",
        extracted_query, message[:60], raw_query[:80],
    )
    if not raw_query:
        yield _sse("error", error="No search query provided.")
        yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
        return

    # v2.7: Split multi-topic queries into focused Brave searches
    search_queries = _extract_search_queries(raw_query)
    logger.info(
        "[web_search_stream] Split into %d queries: %s",
        len(search_queries), [q[:60] for q in search_queries],
    )

    yield _sse("metadata", provider="brave", model="web_search")
    yield _sse("web_search_status", status="searching")

    try:
        # Run all queries (single or multi-topic) in parallel
        per_query = max(2, 8 // len(search_queries)) if len(search_queries) > 1 else 5
        search_tasks = [
            search_and_answer(WebSearchRequest(query=sq, max_results=per_query))
            for sq in search_queries
        ]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results from all queries
        all_sources = []
        all_answers = []
        any_ok = False

        for i, res in enumerate(results):
            if isinstance(res, Exception):
                logger.warning("[web_search_stream] Sub-query %d failed: %s", i, res)
                continue
            if not res.ok:
                logger.warning("[web_search_stream] Sub-query %d not ok: %s", i, res.error)
                continue
            any_ok = True
            all_sources.extend(res.sources)
            if res.answer:
                all_answers.append(res.answer)
            if res.answer_provider:
                synth_provider = res.answer_provider
            if res.answer_model:
                synth_model = res.answer_model

        if not any_ok:
            yield _sse("web_search_status", status="error")
            yield _sse("error", error="All search queries failed")
            yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
            return

        # Deduplicate sources by URL
        seen_urls: set[str] = set()
        deduped_sources = []
        for src in all_sources:
            if src.url not in seen_urls:
                seen_urls.add(src.url)
                deduped_sources.append(src)

        # Build sources data
        sources_data, source_lines = _build_sources_data(deduped_sources)
        if sources_data:
            yield _sse("web_search_sources", sources=sources_data)

        # Update badge to synthesis model and signal synthesising
        yield _sse("metadata", provider=synth_provider, model=synth_model)
        yield _sse("web_search_status", status="synthesising")

        # Build combined answer
        combined_answer = "\n\n".join(all_answers)

        # Stream the answer as token events
        if combined_answer:
            full_response = combined_answer
            logger.info("[web_search_stream] Streaming %d chars of answer", len(combined_answer))
            chunk_size = 80
            for i in range(0, len(combined_answer), chunk_size):
                yield _sse("token", content=combined_answer[i:i + chunk_size])
        else:
            logger.warning("[web_search_stream] Synthesis returned empty answer")

        # Signal search complete
        yield _sse("web_search_status", status="complete")

        # Also send sources event for rich cards on the final message
        if sources_data:
            # Collect missing_perspectives from the first successful result
            missing = []
            for res in results:
                if not isinstance(res, Exception) and res.ok:
                    missing = res.missing_perspectives or []
                    break
            yield _sse("sources", sources=sources_data, missing_perspectives=missing)
            if source_lines:
                full_response += "\n\nSources: " + " | ".join(source_lines)

    except Exception as e:
        logger.exception("[web_search_stream] Failed: %s", e)
        yield _sse("error", error=f"Search error: {str(e)}")

    # Save to history so messages persist across reloads
    if full_response.strip():
        logger.info("[web_search_stream] Saving %d chars to history", len(full_response))
        _save_to_history(db, project_id, message, full_response, synth_provider, synth_model)
    else:
        logger.warning("[web_search_stream] No response to save — full_response is empty")

    yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
    logger.info("[web_search_stream] Stream complete (done event emitted)")
