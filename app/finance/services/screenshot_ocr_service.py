# FILE: app/finance/services/screenshot_ocr_service.py
"""
Backwards-compatible facade over the unified OCR pipeline.

The original implementation hard-coded a single OpenAI call with
parameters that break on GPT-5.x models (temperature + max_tokens).
All new code should use `app.finance.services.ocr_pipeline` directly.

This file now only provides:
  - `save_screenshot` (still used by some callers to write to disk)
  - `extract_via_llm` as a thin wrapper around the pipeline, returning
    the same `ScreenshotOCRResult` pydantic schema it always has.

The legacy `OCRExtraction` dataclass remains as an alias so callers
that imported it don't break.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from app.finance.schemas import ScreenshotOCRResult

logger = logging.getLogger(__name__)

SCREENSHOT_DIR = Path("data/finance/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Back-compat alias \u2014 some callers import OCRExtraction from here
OCRExtraction = ScreenshotOCRResult


def save_screenshot(file_bytes: bytes, filename: str) -> Path:
    """Save uploaded screenshot to the screenshots directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-_.]", "_", filename)
    save_path = SCREENSHOT_DIR / f"{timestamp}_{safe_name}"
    save_path.write_bytes(file_bytes)
    return save_path


async def extract_via_llm(
    image_bytes: bytes, mime_type: str = "image/png",
) -> ScreenshotOCRResult:
    """Back-compat entry point. Delegates to the unified pipeline.

    Prefer calling `ocr_pipeline.extract_from_bytes` directly in new code.
    """
    from app.finance.services.ocr_pipeline import extract_from_bytes
    return await extract_from_bytes(image_bytes, mime_type=mime_type)
