# FILE: app/finance/services/ocr_vision_adapters.py
"""
Vision-model adapters for the OCR pipeline.

Each adapter takes raw image bytes + mime type and returns a
ScreenshotOCRResult. Adapters are called by ocr_pipeline.py — they
should NEVER be called directly from routers or sync workers.

Both adapters use the same prompt (via _PROMPT) and the same JSON
parser (_parse_json_response) so the only difference is the HTTP call.

SIZE: ~6KB — one adapter per provider, no business logic, no fallback
orchestration (that's the pipeline's job).
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from datetime import datetime

from app.finance.schemas import ScreenshotOCRResult

logger = logging.getLogger(__name__)


_PROMPT = """You are extracting data from a Yodel delivery driver's "Finish Tour" screenshot.

The screenshot shows a mobile app screen with these fields:
- Tour Date (e.g., "21st November")
- User ID (e.g., "DA9735")
- Tour ID (e.g., "TO09")
- Successfully Completed Parcels (total, with Deliveries + Collections breakdown)
- Stops (total, with To Do / Attempted / Done breakdown)
- Sometimes: Enter End Mileage, earnings info

Extract ALL visible fields. Return ONLY valid JSON, no other text:

{
    "tour_id": "string or null",
    "user_id": "string or null",
    "work_date": "YYYY-MM-DD or null",
    "delivery_count": "integer (from Deliveries line)",
    "collections": "integer (from Collections line)",
    "stops": "integer (total stops number)",
    "attempted": "integer (from Attempted line)",
    "done": "integer (from Done line)",
    "failed_deliveries": "integer (from To Do line, these are undelivered)",
    "gross_earnings": "float or 0.0 (if visible)",
    "route_area": "string or null"
}

IMPORTANT: "To Do" count means parcels NOT delivered (failed/remaining).
If you cannot read a field clearly, use null or 0."""


# ── Gemini Flash (preferred \u2014 matches ASTRA's ambient vision stack) ──

async def extract_via_gemini(
    image_bytes: bytes, mime_type: str = "image/png",
) -> ScreenshotOCRResult:
    """Send screenshot to Gemini Flash for structured extraction.

    Uses the model configured via GEMINI_VISION_MODEL_FAST (default:
    gemini-flash-latest). Returns a failure result with confidence=0
    if the API key isn't set so the pipeline can fall through.
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return ScreenshotOCRResult(
            success=False,
            message="Gemini API key not configured (GOOGLE_API_KEY / GEMINI_API_KEY).",
        )

    model = os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-flash-latest")

    try:
        from google import genai
        from google.genai import types as gt

        client = genai.Client(api_key=api_key)
        response = await client.aio.models.generate_content(
            model=model,
            contents=[
                gt.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                _PROMPT,
            ],
            config=gt.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        raw_text = (response.text or "").strip()
        logger.debug("[ocr/gemini] Response: %s", raw_text[:200])
        return _parse_json_response(raw_text, source="gemini")
    except Exception as e:
        logger.error("[ocr/gemini] Extraction failed: %s", e)
        return ScreenshotOCRResult(
            success=False,
            message=f"Gemini extraction failed: {e}",
            raw_text=str(e),
        )


# ── OpenAI vision (final fallback) ──

async def extract_via_openai(
    image_bytes: bytes, mime_type: str = "image/png",
) -> ScreenshotOCRResult:
    """Send screenshot to OpenAI vision for structured extraction.

    Handles GPT-5.x family param rules correctly:
      - No `temperature` (reasoning models reject non-default values)
      - `max_completion_tokens` NOT `max_tokens`
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ScreenshotOCRResult(
            success=False,
            message="OpenAI API key not configured.",
        )

    model = (
        os.getenv("OPENAI_VISION_MODEL")
        or os.getenv("OPENAI_DEFAULT_MODEL")
        or "gpt-5.4-mini"
    )

    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Build kwargs conditionally \u2014 GPT-5.x rejects temperature + max_tokens,
        # older models still want max_tokens. Detect by name prefix.
        kwargs = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64}",
                            "detail": "high",
                        },
                    },
                ],
            }],
        }
        if _is_reasoning_model(model):
            kwargs["max_completion_tokens"] = 500
        else:
            kwargs["max_tokens"] = 500
            kwargs["temperature"] = 0.0

        response = await client.chat.completions.create(**kwargs)
        raw_text = (response.choices[0].message.content or "").strip()
        logger.debug("[ocr/openai] Response: %s", raw_text[:200])
        return _parse_json_response(raw_text, source="openai")

    except Exception as e:
        logger.error("[ocr/openai] Extraction failed: %s", e)
        return ScreenshotOCRResult(
            success=False,
            message=f"OpenAI extraction failed: {e}",
            raw_text=str(e),
        )


# ── Shared JSON parser ──

def _parse_json_response(raw_text: str, source: str) -> ScreenshotOCRResult:
    """Parse a JSON response from any vision model into ScreenshotOCRResult."""
    result = ScreenshotOCRResult(raw_text=raw_text)

    try:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)

        result.tour_id = data.get("tour_id")
        result.user_id = data.get("user_id")
        result.route_area = data.get("route_area")

        # Parse date \u2014 accept several LLM output formats
        date_str = data.get("work_date")
        if date_str:
            for fmt in ("%Y-%m-%d", "%d %B %Y", "%d/%m/%Y", "%B %d, %Y"):
                try:
                    result.work_date = datetime.strptime(date_str, fmt).date()
                    break
                except ValueError:
                    continue

        # Parse integers safely
        for field in ("delivery_count", "collections", "stops",
                      "attempted", "done", "failed_deliveries"):
            val = data.get(field, 0)
            try:
                setattr(result, field, int(val) if val else 0)
            except (ValueError, TypeError):
                setattr(result, field, 0)

        # Parse earnings
        earnings = data.get("gross_earnings", 0.0)
        if isinstance(earnings, str):
            earnings = earnings.replace("\u00a3", "").replace(",", "").strip() or "0"
            earnings = float(earnings)
        result.gross_earnings = float(earnings or 0.0)

        result.success = True
        result.confidence = 85.0
        result.message = f"Extracted via {source}. Please verify before saving."
    except (json.JSONDecodeError, ValueError) as e:
        result.success = False
        result.confidence = 0.0
        result.message = f"Could not parse {source} response: {e}"

    return result


def _is_reasoning_model(model: str) -> bool:
    """Identify GPT-5.x / o-series reasoning models that reject temperature."""
    m = (model or "").lower()
    return (
        m.startswith("gpt-5")
        or m.startswith("o1")
        or m.startswith("o3")
        or m.startswith("o4")
    )
