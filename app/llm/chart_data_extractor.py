# FILE: app/llm/chart_data_extractor.py
# Purpose: LLM-powered data extraction for chart rendering.
# Called-by: app.llm.image_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
LLM-powered data extraction for chart rendering.

Takes raw research text and the user's request, asks an LLM to extract
structured chart data (labels, values, chart type) that can be fed
directly to chart_renderer.py.

Uses the IMAGE_PROMPT_SYNTH_MODEL (Gemini Flash by default) — cheap, fast.

v1.0 (2026-03-20): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are a data extraction specialist. Your job is to read research text
and extract structured data for a chart.

USER'S REQUEST:
{user_message}

RESEARCH DATA:
{research_text}

Extract the data and respond with ONLY a JSON object (no markdown, no backticks, no explanation).
The JSON must have these fields:

{{
  "chart_type": "bar" | "horizontal_bar" | "line" | "pie" | "scatter" | "grouped_bar",
  "title": "Chart title",
  "subtitle": "Optional subtitle with date range or context",
  "labels": ["Label 1", "Label 2", ...],
  "values": [number1, number2, ...],
  "series": [
    {{"name": "Series A", "values": [n1, n2, ...]}},
    {{"name": "Series B", "values": [n1, n2, ...]}}
  ],
  "x_label": "X axis label",
  "y_label": "Y axis label",
  "source_note": "Data source attribution"
}}

CRITICAL RULES:
- Extract as MANY data points as possible. A chart with 2 bars is useless. Aim for 5-15 items.
- ONLY include items that are RELEVANT to the user's request. If they asked about image generation
  models, do NOT include text-only LLMs like Claude or GPT-5 — only include models that GENERATE IMAGES.
- Use "series" ONLY for grouped_bar or multi-line charts. Otherwise leave it as [].
- For single-series charts, put data in "values" not "series".
- Choose the chart_type that best represents the data:
  - Comparisons between items → horizontal_bar (preferred for many items, labels readable on left)
  - Trends over time → line
  - Parts of a whole → pie
  - Correlations → scatter
  - Multiple metrics per item → grouped_bar
  - Use horizontal_bar over bar when there are 5+ items (labels read better horizontally).
- Extract REAL numbers from the research. Do NOT invent data.
- If a number has a unit (Elo, %, $), include the unit in the y_label not the value.
- If the research mentions scores/rankings for many items, include ALL of them, not just the top 2.
- If the research doesn't contain enough numerical data for a chart (fewer than 3 data points),
  set "values": [] and "title": "INSUFFICIENT_DATA".
- Keep labels short and readable (model names, years, categories).
- Round numbers to reasonable precision.
- Sort by value descending so the chart reads naturally (highest first)."""


async def extract_chart_data(
    user_message: str,
    research_text: str,
) -> Optional[dict]:
    """Extract structured chart data from research text using an LLM.

    Returns:
        Dict with chart_type, title, labels, values, etc. or None on failure.
        Returns None if the LLM determines there's insufficient data.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("[chart_data_extractor] GOOGLE_API_KEY not set")
            return None

        model = os.getenv("IMAGE_PROMPT_SYNTH_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)

        prompt = _EXTRACTION_PROMPT.format(
            user_message=user_message,
            research_text=research_text[:6000],  # Cap to avoid token overflow
        )

        logger.info("[chart_data_extractor] Extracting chart data with %s", model)

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temp for accurate extraction
                max_output_tokens=1500,
            ),
        )

        # Extract text from response
        raw = ""
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    raw += part.text

        raw = raw.strip()
        logger.info("[chart_data_extractor] Raw LLM response (%d chars): %s",
                     len(raw), raw[:500])

        # Strip markdown fences if present (various formats)
        if "```" in raw:
            # Extract content between first ``` and last ```
            parts = raw.split("```")
            if len(parts) >= 3:
                # Take the content between first and last fence
                raw = parts[1]
                # Remove language hint on first line (e.g. 'json')
                if raw.startswith("json"):
                    raw = raw[4:]
                elif raw.startswith("JSON"):
                    raw = raw[4:]
            else:
                # Single fence — strip it
                raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "")
        raw = raw.strip()

        # Parse JSON
        chart_data = json.loads(raw)
        logger.info("[chart_data_extractor] Parsed keys: %s", list(chart_data.keys()))

        # Validate minimum fields
        title = chart_data.get("title", "")
        labels = chart_data.get("labels", [])

        if not title:
            logger.warning("[chart_data_extractor] Missing title field")
            return None

        if not labels:
            logger.warning("[chart_data_extractor] Missing labels field (got: %s)",
                         type(labels).__name__)
            return None

        if chart_data.get("title") == "INSUFFICIENT_DATA":
            logger.info("[chart_data_extractor] LLM reports insufficient data for chart")
            return None

        if not chart_data.get("values") and not chart_data.get("series"):
            logger.warning("[chart_data_extractor] No values or series in extracted data")
            return None

        logger.info(
            "[chart_data_extractor] Extracted: type=%s, title=%s, %d labels",
            chart_data.get("chart_type"),
            chart_data.get("title", "")[:60],
            len(chart_data.get("labels", [])),
        )

        return chart_data

    except json.JSONDecodeError as e:
        logger.error("[chart_data_extractor] JSON parse failed: %s\nRaw: %s", e, raw[:500])
        return None
    except Exception as e:
        logger.error("[chart_data_extractor] Failed: %s", e)
        return None


__all__ = ["extract_chart_data"]
