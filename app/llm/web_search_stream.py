# FILE: app/llm/web_search_stream.py
"""
SSE stream handler for web search intent.

Receives a natural language query, runs it through the web search
orchestrator (Brave primary, DuckDuckGo fallback), and streams
the answer back as SSE events.

v2.1 (2026-02): Initial implementation.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def generate_web_search_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[object] = None,
    extracted_query: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream web search results as SSE events.

    Args:
        project_id: Current project ID.
        message: The user's original message.
        db: Database session.
        trace: Optional stage trace.
        extracted_query: Pre-extracted search query from Tier 0 rules.
                         Falls back to the full message if not provided.
    """
    try:
        from app.llm.web_search import (
            WebSearchRequest,
            search_and_answer,
        )
    except ImportError:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Web search module not available.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Use the extracted query if available, otherwise use the full message
    query = (extracted_query or message).strip()
    if not query:
        yield f"data: {json.dumps({'type': 'error', 'content': 'No search query provided.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    # Signal that we're searching
    yield f"data: {json.dumps({'type': 'status', 'content': f'Searching the web for: {query}'})}\n\n"

    try:
        req = WebSearchRequest(query=query, max_results=5)
        result = await search_and_answer(req)

        if not result.ok:
            error_msg = result.error or "Search failed"
            yield f"data: {json.dumps({'type': 'error', 'content': f'Search failed: {error_msg}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream the answer
        if result.answer:
            # Stream answer in chunks for a more responsive feel
            answer = result.answer
            chunk_size = 80
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        # Append sources with credibility tags
        if result.sources:
            source_text = "\n\n---\n**Sources:**\n"
            for i, src in enumerate(result.sources, 1):
                cred = ""
                if src.source_type and src.source_type != "unknown":
                    cred = f" `{src.credibility_label}` · {src.source_type}"
                source_text += f"\n[{i}] [{src.title}]({src.url}){cred}"
            if result.missing_perspectives:
                missing = ", ".join(result.missing_perspectives)
                source_text += f"\n\n*Note: Results lack {missing} perspectives.*"
            yield f"data: {json.dumps({'type': 'content', 'content': source_text})}\n\n"

        # Provider + diversity info
        provider = result.provider or "unknown"
        meta = {
            'provider': provider,
            'source_count': len(result.sources),
            'diversity_score': result.diversity_score,
        }
        yield f"data: {json.dumps({'type': 'metadata', 'content': json.dumps(meta)})}\n\n"

    except Exception as e:
        logger.exception("[web_search_stream] Failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'content': f'Search error: {str(e)}'})}\n\n"

    yield "data: [DONE]\n\n"

