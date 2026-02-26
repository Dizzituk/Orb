"""
Screenshot OCR service for Yodel 'Finish Tour' screenshots.

Uses OpenAI GPT-4o vision directly for structured data extraction.
Falls back gracefully if API key unavailable.

Yodel Finish Tour screenshot typically contains:
- Tour ID (e.g., "TO09")
- User ID (e.g., "DA9735")
- Tour Date
- Deliveries / Collections counts
- Stops / To Do / Attempted / Done counts
- Enter End Mileage prompt
- FINISH button
"""

import os
import json
import base64
import logging
import re
from pathlib import Path
from datetime import datetime, date
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Storage config ────────────────────────────────────────

SCREENSHOT_DIR = Path("data/finance/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class OCRExtraction:
    success: bool = False
    work_date: Optional[date] = None
    tour_id: Optional[str] = None
    user_id: Optional[str] = None
    delivery_count: int = 0
    collections: int = 0
    stops: int = 0
    attempted: int = 0
    done: int = 0
    failed_deliveries: int = 0
    gross_earnings: float = 0.0
    route_area: Optional[str] = None
    raw_text: Optional[str] = None
    confidence: float = 0.0
    message: str = ""


# ── Save screenshot ───────────────────────────────────────

def save_screenshot(file_bytes: bytes, filename: str) -> Path:
    """Save uploaded screenshot to data directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r'[^\w\-_.]', '_', filename)
    save_path = SCREENSHOT_DIR / f"{timestamp}_{safe_name}"
    save_path.write_bytes(file_bytes)
    return save_path


# ── LLM-based extraction ─────────────────────────────────

async def extract_via_llm(image_bytes: bytes, mime_type: str = "image/png") -> OCRExtraction:
    """
    Send screenshot to GPT-4o vision to extract structured data.
    Uses OpenAI client directly (vision calls bypass Phase 4 envelope).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return OCRExtraction(
            success=False,
            message="OpenAI API key not configured. Please enter data manually.",
        )

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key)
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt = """You are extracting data from a Yodel delivery driver's "Finish Tour" screenshot.

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

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
            temperature=0.0,
        )

        raw_text = response.choices[0].message.content or ""
        logger.info("[ocr] GPT-4o response: %s", raw_text[:200])
        return _parse_llm_response(raw_text)

    except Exception as e:
        logger.error("[ocr] LLM extraction failed: %s", e)
        return OCRExtraction(
            success=False,
            message=f"OCR extraction failed: {str(e)}. Please enter data manually.",
            raw_text=str(e),
        )


# ── Parse LLM response ───────────────────────────────────

def _parse_llm_response(raw_text: str) -> OCRExtraction:
    """Parse JSON response from LLM vision model."""
    result = OCRExtraction(raw_text=raw_text)

    try:
        # Strip markdown code fences if present
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
            cleaned = re.sub(r'\s*```$', '', cleaned)

        data = json.loads(cleaned)

        result.tour_id = data.get("tour_id")
        result.user_id = data.get("user_id")
        result.route_area = data.get("route_area")

        # Parse date — handle various formats
        date_str = data.get("work_date")
        if date_str:
            try:
                result.work_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                # Try other formats the LLM might return
                for fmt in ("%d %B %Y", "%d/%m/%Y", "%B %d, %Y"):
                    try:
                        result.work_date = datetime.strptime(date_str, fmt).date()
                        break
                    except ValueError:
                        continue

        # Parse integers safely
        for field in ["delivery_count", "collections", "stops", "attempted", "done", "failed_deliveries"]:
            val = data.get(field, 0)
            try:
                setattr(result, field, int(val) if val else 0)
            except (ValueError, TypeError):
                setattr(result, field, 0)

        # Parse earnings
        earnings = data.get("gross_earnings", 0.0)
        if isinstance(earnings, str):
            earnings = float(earnings.replace("\u00a3", "").replace(",", "").strip() or "0")
        result.gross_earnings = float(earnings or 0.0)

        result.success = True
        result.confidence = 0.85
        result.message = "Data extracted from screenshot. Please verify before saving."

    except (json.JSONDecodeError, ValueError) as e:
        result.success = False
        result.confidence = 0.0
        result.message = f"Could not parse screenshot data: {e}. Please enter manually."

    return result
