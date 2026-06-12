# FILE: app/llm/chart_research.py
# Purpose: Research pipeline for data-driven chart generation.
# Called-by: app.llm.image_router
# Depends-on: app.llm.web_search
# Last-renovated: 2026-06-11
"""
Research pipeline for data-driven chart generation.

Takes a user's chart request, plans targeted search queries via LLM,
runs multiple Brave searches, and combines the results into a rich
evidence block for the data extractor.

v1.0 (2026-03-20): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_QUERY_PLANNER_PROMPT = """You are a research query planner. The user wants to create a data chart.
Your job is to generate 2-3 specific, data-focused web search queries that will find the
NUMERICAL DATA needed for the chart.

USER'S REQUEST:
{user_message}

Rules:
- Output ONLY a JSON array of search query strings. No explanation.
- Each query should target SPECIFIC numerical data, rankings, or statistics.
- Include the current year (2026) or recent date range in at least one query.
- Remove any "create/make/generate" verbs — focus on finding the DATA.
- Target authoritative sources: leaderboards, benchmark sites, official docs.
- If the request is about model comparisons, search for specific benchmark names (Elo, MMLU, etc.)

Examples:
  User: "Create a chart comparing the top AI image generation models and their benchmark scores"
  Output: ["AI image generation model Elo scores LM Arena 2026", "best AI image generators benchmark ranking comparison 2026", "GPT Image Flux Midjourney Imagen benchmark scores"]

  User: "Make a graph of UK house prices over the past 5 years"
  Output: ["UK average house prices 2021 2022 2023 2024 2025 2026", "UK house price index annual data ONS", "UK property market statistics yearly trend"]

  User: "Chart the growth of electric vehicle sales worldwide"
  Output: ["global electric vehicle sales numbers 2020 2021 2022 2023 2024 2025", "worldwide EV market share percentage by year", "electric car annual sales statistics IEA"]

Respond with ONLY the JSON array."""


async def plan_search_queries(user_message: str) -> list[str]:
    """Use an LLM to generate focused search queries for chart data.

    Returns:
        List of 2-3 search query strings, or fallback to cleaned user message.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("[chart_research] No API key, using raw message as query")
            return [_clean_query(user_message)]

        model = os.getenv("IMAGE_PROMPT_SYNTH_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)

        prompt = _QUERY_PLANNER_PROMPT.format(user_message=user_message)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=300,
            ),
        )

        raw = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    raw += part.text

        raw = raw.strip()

        # Strip markdown fences
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 3:
                raw = parts[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            else:
                raw = raw.replace("```json", "").replace("```", "")
        raw = raw.strip()

        queries = json.loads(raw)

        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            logger.info("[chart_research] Planned %d search queries: %s",
                         len(queries), queries)
            return queries[:3]  # Cap at 3
        else:
            logger.warning("[chart_research] Invalid query plan format, using fallback")
            return [_clean_query(user_message)]

    except Exception as e:
        logger.warning("[chart_research] Query planning failed: %s, using fallback", e)
        return [_clean_query(user_message)]


def _clean_query(message: str) -> str:
    """Strip command verbs from the user message to make a better search query."""
    import re
    # Remove common command prefixes
    cleaned = re.sub(
        r'^(?:create|make|generate|build|draw|design|compile|produce|put together)\s+'
        r'(?:me\s+)?(?:a\s+|an\s+|the\s+)?'
        r'(?:chart|graph|image|picture|infographic|visual)\s+'
        r'(?:of\s+|on\s+|about\s+|for\s+|showing\s+|comparing\s+)?',
        '',
        message,
        flags=re.IGNORECASE,
    )
    return cleaned.strip() or message


async def run_multi_search(user_message: str) -> Optional[str]:
    """Plan queries, run multiple searches, combine results.

    Returns:
        Combined research text from all searches, or None if all fail.
    """
    from app.llm.web_search import WebSearchRequest, search_and_answer

    # Stage 1: Plan targeted queries
    queries = await plan_search_queries(user_message)

    # Stage 2: Run all searches
    all_answers = []
    all_sources = set()

    for i, query in enumerate(queries):
        logger.info("[chart_research] Search %d/%d: %s", i + 1, len(queries), query)
        try:
            req = WebSearchRequest(query=query, max_results=5)
            result = await search_and_answer(req)
            if result.ok and result.answer:
                all_answers.append(f"[Search {i+1}: {query}]\n{result.answer}")
                for src in result.sources:
                    all_sources.add(src.title)
                logger.info("[chart_research] Search %d returned %d chars",
                             i + 1, len(result.answer))
            else:
                logger.warning("[chart_research] Search %d returned no results", i + 1)
        except Exception as e:
            logger.warning("[chart_research] Search %d failed: %s", i + 1, e)

    if not all_answers:
        return None

    # Stage 3: Combine into a single evidence block
    combined = "\n\n".join(all_answers)

    if all_sources:
        combined += f"\n\n[Sources consulted: {', '.join(list(all_sources)[:10])}]"

    logger.info("[chart_research] Combined research: %d chars from %d searches",
                 len(combined), len(all_answers))

    return combined


__all__ = ["plan_search_queries", "run_multi_search"]
