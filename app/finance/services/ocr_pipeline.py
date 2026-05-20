# FILE: app/finance/services/ocr_pipeline.py
"""
Unified OCR pipeline for Yodel Finish Tour screenshots.

ONE pipeline used by THREE surfaces:
  1. Upload endpoint (POST /finance/upload/screenshot)
  2. Drive folder sync (auto-import from Drive)
  3. Future: agentic chat tool (drop image in chat → log created)

Routing order (cheapest/fastest first):
  Tier 1: Tesseract deterministic OCR (local, free, millisecond latency)
  Tier 2: Gemini Flash vision (fast, cheap, matches ASTRA's ambient vision model)
  Tier 3: OpenAI vision (only if both above fail)

Returns a ScreenshotOCRResult (pydantic schema from app.finance.schemas)
so every caller gets the same shape regardless of which tier answered.

SIZE: Pipeline orchestration only. Tesseract logic lives in
screenshot_ocr_deterministic.py; LLM-specific adapters live in
ocr_vision_adapters.py. This file stays under 6KB.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.finance.schemas import ScreenshotOCRResult

logger = logging.getLogger(__name__)

# Confidence threshold below which we fall back to a higher tier
DETERMINISTIC_MIN_CONFIDENCE = 70.0

# Required fields for an OCR result to be considered "good enough"
REQUIRED_FIELDS = ("work_date", "delivery_count")


@dataclass
class OCRPipelineConfig:
    """Per-call overrides for pipeline behaviour."""
    prefer_tier: str = "tesseract"      # tesseract | gemini | openai
    allow_fallback: bool = True
    min_confidence: float = DETERMINISTIC_MIN_CONFIDENCE


async def extract_from_bytes(
    image_bytes: bytes,
    mime_type: str = "image/png",
    filename: Optional[str] = None,
    config: Optional[OCRPipelineConfig] = None,
) -> ScreenshotOCRResult:
    """Extract Yodel tour data from raw image bytes.

    This is the single entry point. Saves the image locally for audit
    trail, then walks the tier ladder until one succeeds.
    """
    from app.finance.services.screenshot_ocr_deterministic import extract_from_image

    cfg = config or OCRPipelineConfig()
    save_path = _save_screenshot(image_bytes, filename or "screenshot.png")

    # Tier 1: Tesseract
    if cfg.prefer_tier == "tesseract":
        try:
            det = extract_from_image(save_path)
            if det.is_valid and det.confidence >= cfg.min_confidence:
                logger.info(
                    "[ocr] Tesseract succeeded (%s%%) for %s",
                    det.confidence, save_path.name,
                )
                return _deterministic_to_schema(det)
            logger.info(
                "[ocr] Tesseract insufficient (%s%%, valid=%s) for %s, falling back",
                det.confidence, det.is_valid, save_path.name,
            )
        except Exception as e:
            logger.warning("[ocr] Tesseract threw: %s", e)

    if not cfg.allow_fallback:
        return ScreenshotOCRResult(
            success=False,
            message=f"Tesseract did not meet {cfg.min_confidence}% threshold",
        )

    # Tier 2: Gemini Flash
    try:
        from app.finance.services.ocr_vision_adapters import extract_via_gemini
        result = await extract_via_gemini(image_bytes, mime_type)
        if result.success:
            logger.info("[ocr] Gemini Flash succeeded for %s", save_path.name)
            return result
        logger.info("[ocr] Gemini returned no result for %s: %s", save_path.name, result.message)
    except Exception as e:
        logger.warning("[ocr] Gemini adapter threw: %s", e)

    # Tier 3: OpenAI vision
    try:
        from app.finance.services.ocr_vision_adapters import extract_via_openai
        result = await extract_via_openai(image_bytes, mime_type)
        if result.success:
            logger.info("[ocr] OpenAI vision succeeded for %s", save_path.name)
        return result
    except Exception as e:
        logger.error("[ocr] OpenAI adapter threw: %s", e)
        return ScreenshotOCRResult(
            success=False,
            message=f"All OCR tiers failed. Last error: {e}",
        )


def _save_screenshot(image_bytes: bytes, filename: str) -> Path:
    """Save uploaded image to the screenshots directory for audit trail."""
    import re
    from datetime import datetime

    screenshot_dir = Path("data/finance/screenshots")
    screenshot_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-_.]", "_", filename)
    save_path = screenshot_dir / f"{timestamp}_{safe_name}"
    save_path.write_bytes(image_bytes)
    return save_path


def _deterministic_to_schema(det) -> ScreenshotOCRResult:
    """Map YodelOCRResult (dataclass) to ScreenshotOCRResult (pydantic)."""
    return ScreenshotOCRResult(
        success=True,
        work_date=det.work_date,
        tour_id=det.tour_id,
        user_id=det.user_id,
        delivery_count=det.delivery_count,
        collections=det.collections,
        stops=det.attempted_stops,
        attempted=det.attempted_stops,
        done=det.deliveries or det.delivery_count,
        failed_deliveries=det.not_attempted,
        gross_earnings=det.gross_earnings,
        route_area=None,
        raw_text=det.raw_text,
        confidence=det.confidence,
        message=f"Tesseract ({det.confidence}% confidence, {len(det.fields_extracted)} fields)",
    )
