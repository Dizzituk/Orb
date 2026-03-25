# FILE: app/llm/web_search_stream.py
"""
SSE stream handler for web search intent.

Receives a natural language query, runs it through the web search
orchestrator (Brave primary, DuckDuckGo fallback), and streams
the answer back as SSE events.

v2.6 (2026-03): Replaced search status token with structured web_search_status/web_search_sources events.
v2.5 (2026-02): Saves user + assistant messages to history (persistence fix).
v2.4 (2026-02): Sends sources as structured 'sources' event for rich rendering.
v2.3 (2026-02): Shows actual synthesis model in metadata badge.
v2.2 (2026-02): Fixed SSE event types to match frontend expectations.
"""
from __future__ import annotations

import json
import logging
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
    full_response = ""  # Accumulate for history

    try:
        from app.llm.web_search import (
            WebSearchRequest,
            search_and_answer,
        )
    except ImportError:
        yield _sse("error", error="Web search module not available.")
        yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
        return

    # Use the extracted query if available, otherwise use the full message
    query = (extracted_query or message).strip()
    logger.info("[web_search_stream] extracted_query=%r, message=%r, using query=%r",
                extracted_query, message[:60], query[:60])
    if not query:
        yield _sse("error", error="No search query provided.")
        yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
        return

    # v2.6: Send structured search status (no text token — frontend shows animated dots)
    yield _sse("metadata", provider="brave", model="web_search")
    yield _sse("web_search_status", status="searching")

    try:
        req = WebSearchRequest(query=query, max_results=5)
        result = await search_and_answer(req)

        if not result.ok:
            yield _sse("web_search_status", status="error")
            yield _sse("error", error=f"Search failed: {result.error or 'Unknown error'}")
            yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
            return

        # v2.6: Send sources as structured data so frontend can show collapsible dropdown
        sources_data = []
        source_lines = []
        if result.sources:
            for src in result.sources:
                sources_data.append({
                    "title": src.title,
                    "url": src.url,
                    "snippet": src.snippet,
                    "credibility_label": src.credibility_label,
                    "source_type": src.source_type,
                })
                source_lines.append(f"[{src.title}]({src.url})")
            yield _sse("web_search_sources", sources=sources_data)

        # Update badge to synthesis model and signal synthesising
        synth_provider = result.answer_provider or "openai"
        synth_model = result.answer_model or "web_search"
        yield _sse("metadata", provider=synth_provider, model=synth_model)
        yield _sse("web_search_status", status="synthesising")

        # Stream the LLM-synthesised answer
        if result.answer:
            full_response = result.answer
            chunk_size = 80
            for i in range(0, len(result.answer), chunk_size):
                yield _sse("token", content=result.answer[i:i + chunk_size])

        # Signal search complete
        yield _sse("web_search_status", status="complete")

        # Also send sources event for any existing frontend handling
        if sources_data:
            yield _sse("sources", sources=sources_data,
                        missing_perspectives=result.missing_perspectives or [])
            if source_lines:
                full_response += "\n\nSources: " + " | ".join(source_lines)

    except Exception as e:
        logger.exception("[web_search_stream] Failed: %s", e)
        yield _sse("error", error=f"Search error: {str(e)}")

    # Save to history so messages persist across reloads
    if full_response.strip():
        _save_to_history(db, project_id, message, full_response, synth_provider, synth_model)

    yield _sse("done", provider=synth_provider, model=synth_model, total_length=0)
