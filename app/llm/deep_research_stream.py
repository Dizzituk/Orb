# FILE: app/llm/deep_research_stream.py
# Purpose: SSE stream handler for deep research.
# Called-by: app.llm.routing.handler_registry
# Depends-on: app.llm.deep_research_engine
# Last-renovated: 2026-06-11
"""
SSE stream handler for deep research.

Streams progress updates during the iterative research loop,
then streams the final synthesis.

v2.1 (2026-02): Initial implementation — Capability #4.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def generate_deep_research_stream(
    project_id: int,
    message: str,
    db: Session,
    trace: Optional[object] = None,
    extracted_query: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream deep research results as SSE events.

    Emits progress events during each research round,
    then streams the final synthesis.
    """
    try:
        from app.llm.deep_research_engine import (
            DeepResearchRequest,
            run_deep_research,
        )
    except ImportError:
        yield f"data: {json.dumps({'type': 'error', 'content': 'Deep research engine not available.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    query = (extracted_query or message).strip()
    if not query:
        yield f"data: {json.dumps({'type': 'error', 'content': 'No research query provided.'})}\n\n"
        yield "data: [DONE]\n\n"
        return

    yield f"data: {json.dumps({'type': 'status', 'content': f'Starting deep research: {query}'})}\n\n"

    try:
        req = DeepResearchRequest(query=query)
        result = await run_deep_research(req)

        if result.error:
            yield f"data: {json.dumps({'type': 'error', 'content': f'Research failed: {result.error}'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Stream round summaries
        for rnd in result.rounds:
            summary = (
                f"**Round {rnd.round_number}:** "
                f"{len(rnd.queries)} queries, "
                f"{len(rnd.sources)} sources found, "
                f"{rnd.pages_fetched} pages fetched"
            )
            yield f"data: {json.dumps({'type': 'status', 'content': summary})}\n\n"
            if rnd.gap_analysis and "NO_GAPS" not in rnd.gap_analysis.upper():
                yield f"data: {json.dumps({'type': 'status', 'content': f'Gaps identified: {rnd.gap_analysis[:200]}'})}\n\n"

        # Stream synthesis
        if result.synthesis:
            yield f"data: {json.dumps({'type': 'content', 'content': '\\n\\n---\\n**Research Synthesis:**\\n\\n'})}\n\n"
            chunk_size = 100
            for i in range(0, len(result.synthesis), chunk_size):
                chunk = result.synthesis[i:i + chunk_size]
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"

        # Sources
        if result.all_sources:
            seen_urls = set()
            source_text = "\n\n---\n**Sources:**\n"
            idx = 0
            for src in result.all_sources:
                url = src.get("url", "")
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                idx += 1
                cred = ""
                if src.get("source_type", "unknown") != "unknown":
                    cred = f" `{src.get('credibility_label', '')}` · {src.get('source_type', '')}"
                source_text += f"\n[{idx}] [{src.get('title', 'Untitled')}]({url}){cred}"
                if idx >= 15:
                    source_text += f"\n... and {len(result.all_sources) - idx} more"
                    break
            if result.missing_perspectives:
                missing = ", ".join(result.missing_perspectives)
                source_text += f"\n\n*Note: Results lack {missing} perspectives.*"
            yield f"data: {json.dumps({'type': 'content', 'content': source_text})}\n\n"

        # Metadata
        meta = {
            "rounds": result.total_rounds,
            "total_queries": result.total_queries,
            "total_pages": result.total_pages_fetched,
            "unique_sources": len(set(s.get("url", "") for s in result.all_sources)),
            "diversity_score": result.diversity_score,
            "elapsed_seconds": result.elapsed_seconds,
        }
        yield f"data: {json.dumps({'type': 'metadata', 'content': json.dumps(meta)})}\n\n"

    except Exception as e:
        logger.exception("[deep_research_stream] Failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'content': f'Research error: {str(e)}'})}\n\n"

    yield "data: [DONE]\n\n"
