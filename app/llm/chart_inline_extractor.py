# FILE: app/llm/chart_inline_extractor.py
# Purpose: Extract chart data directly from user-provided inline data.
# Called-by: app.llm.image_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Extract chart data directly from user-provided inline data.

When the user provides percentages, labels, and values in their message,
this module extracts them into Plotly-compatible structured data WITHOUT
any web search or LLM hallucination.

Uses a lightweight LLM call (Gemini Flash) to parse the user's natural
language data description into structured JSON — but the LLM is ONLY
reading data the user already provided, not inventing new data.

v1.0 (2026-04-02): Initial implementation.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_INLINE_EXTRACTION_PROMPT = """You are a data parsing specialist. The user has provided EXACT data
in their message. Your ONLY job is to extract their data into structured JSON.

USER'S MESSAGE:
{user_message}

CRITICAL RULES:
- Extract ONLY the data the user provided. Do NOT invent, modify, or supplement any numbers.
- Every percentage, count, and label must come directly from the user's text.
- If the user specified colours, include them as HEX codes (e.g. "#E07A5F") in a "colours" field.
  If they described colours in words (e.g. "coral", "grey-blue"), convert them to appropriate hex codes.
- If the user specified a title, use it exactly.
- If the user specified a footer/source, include it in "source_note".
- If the user asked for annotation text / explanatory note / caption in the chart, include it in "annotation".

Respond with ONLY a JSON object (no markdown, no backticks, no explanation):

{{
  "chart_type": "horizontal_bar" | "bar" | "grouped_bar" | "pie" | "line",
  "title": "Exact title from user's message",
  "subtitle": "Optional subtitle if specified",
  "labels": ["Label 1", "Label 2", ...],
  "values": [number1, number2, ...],
  "series": [
    {{"name": "Series A", "values": [n1, n2, ...], "colour": "#hexcode"}},
    {{"name": "Series B", "values": [n1, n2, ...], "colour": "#hexcode"}}
  ],
  "x_label": "",
  "y_label": "",
  "source_note": "Source attribution if the user provided one",
  "annotation": "Short explanatory text if the user asked for one",
  "colours": ["#hex1", "#hex2"],
  "size": "1080x1080"
}}

RULES FOR COLOURS:
- ALWAYS output colours as hex codes starting with "#".
- If the user writes "coral" -> "#E07A5F", "grey-blue" -> "#7B9AA6", "red" -> "#E05050", "green" -> "#50C878".
- Series-level "colour" fields MUST also be hex codes.

RULES FOR SERIES:
- If the user has TWO separate datasets (e.g. prisoners vs population), use "series" with 2 entries.
- If only one dataset, use "values" and leave "series" as [].
- For grouped/stacked charts, ALWAYS use "series".

RULES FOR CHART TYPE:
- horizontal_bar is best when comparing categories with bars going left to right (labels on left axis).
- grouped_bar is VERTICAL bars side by side — only use if user explicitly says "vertical".
- If the user says "horizontal bar" or shows data that compares categories, default to "horizontal_bar".
- Use the chart type the user specified if they mentioned one."""


async def extract_inline_chart_data(user_message: str) -> Optional[dict]:
    """Extract structured chart data from user-provided inline data.

    This is the SAFE path — the user already gave us the numbers.
    The LLM only organises what's already there.

    Returns:
        Dict with chart_type, title, labels, values/series, etc. or None on failure.
    """
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("[chart_inline_extractor] GOOGLE_API_KEY not set")
            return None

        model = os.getenv("IMAGE_PROMPT_SYNTH_MODEL", "gemini-2.5-flash")
        client = genai.Client(api_key=api_key)

        prompt = _INLINE_EXTRACTION_PROMPT.format(user_message=user_message)

        logger.info("[chart_inline_extractor] Extracting inline data with %s", model)

        # v1.1: Retry up to 2 times on truncated/unparseable responses.
        # Gemini Flash sometimes returns incomplete JSON on the first attempt.
        raw = ""
        for attempt in range(3):
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.05,  # Near-zero — just parse, don't create
                    max_output_tokens=4000,  # v1.1: increased from 1500
                    response_mime_type="application/json",  # v1.1: force JSON output
                ),
            )

            raw = ""
            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        raw += part.text

            raw = raw.strip()
            logger.info("[chart_inline_extractor] Attempt %d: raw response (%d chars): %s",
                         attempt + 1, len(raw), raw[:500])

            # Strip markdown fences if present (shouldn't be with response_mime_type but just in case)
            if "```" in raw:
                parts = raw.split("```")
                if len(parts) >= 3:
                    raw = parts[1]
                    if raw.lower().startswith("json"):
                        raw = raw[4:]
                else:
                    raw = raw.replace("```json", "").replace("```JSON", "").replace("```", "")
            raw = raw.strip()

            # Try to parse JSON — if it fails, retry
            try:
                chart_data = json.loads(raw)
                logger.info("[chart_inline_extractor] JSON parsed successfully on attempt %d", attempt + 1)
                break  # Success — exit retry loop
            except json.JSONDecodeError as e:
                logger.warning(
                    "[chart_inline_extractor] JSON parse failed on attempt %d: %s (raw=%d chars)",
                    attempt + 1, e, len(raw),
                )
                if attempt < 2:
                    logger.info("[chart_inline_extractor] Retrying...")
                    continue
                else:
                    logger.error("[chart_inline_extractor] All 3 attempts failed. Last raw: %s", raw[:500])
                    return None
        else:
            # Loop completed without break — all retries exhausted
            return None

        # Validate
        if not chart_data.get("title"):
            logger.warning("[chart_inline_extractor] No title extracted")
            return None

        has_values = bool(chart_data.get("values"))
        has_series = bool(chart_data.get("series"))

        if not has_values and not has_series:
            logger.warning("[chart_inline_extractor] No values or series extracted")
            return None

        logger.info(
            "[chart_inline_extractor] Extracted: type=%s, title=%s, labels=%d, series=%d",
            chart_data.get("chart_type"),
            chart_data.get("title", "")[:60],
            len(chart_data.get("labels", [])),
            len(chart_data.get("series", [])),
        )

        return chart_data

    except Exception as e:
        logger.error("[chart_inline_extractor] Failed: %s", e)
        return None


__all__ = ["extract_inline_chart_data"]
