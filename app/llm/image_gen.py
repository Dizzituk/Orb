# FILE: app/llm/image_gen.py
"""
OpenAI image generation backend (GPT Image 1.5).

Called by image_service.py when IMAGE_GEN_PROVIDER=openai.
All config read from .env at runtime — no hardcoded models.

v2.0 (2026-03-20): Env-driven config, consistent return schema with nano_banana.
v1.0 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.llm.image_output_dir import get_image_output_dir

logger = logging.getLogger(__name__)


def _get_model() -> str:
    return os.getenv("IMAGE_GEN_MODEL", "gpt-image-1.5")


def _get_quality() -> str:
    return os.getenv("IMAGE_GEN_QUALITY", "standard")


def _get_size() -> str:
    return os.getenv("IMAGE_GEN_SIZE", "1024x1024")


async def generate_image(
    prompt: str,
    model: Optional[str] = None,
    quality: Optional[str] = None,
    size: Optional[str] = None,
    output_filename: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> Optional[dict]:
    """Generate an image using OpenAI GPT Image API.

    All defaults read from .env (IMAGE_GEN_MODEL, IMAGE_GEN_QUALITY, IMAGE_GEN_SIZE).
    aspect_ratio is mapped to size if provided and size is not explicit.

    Returns:
        Dict with path, filename, size_bytes, base64_data, mime_type, prompt, text
        (consistent with nano_banana return schema) or None on failure.
    """
    model = model or _get_model()
    quality = quality or _get_quality()
    size = size or _map_aspect_to_size(aspect_ratio) or _get_size()

    try:
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[image_gen] OPENAI_API_KEY not set")
            return None

        client = AsyncOpenAI(api_key=api_key)

        logger.info(
            "[image_gen] Generating with %s (size=%s, quality=%s): %s",
            model, size, quality, prompt[:100],
        )

        # GPT Image models (gpt-image-1, gpt-image-1.5) return b64_json by default.
        # They use output_format (png/jpeg/webp) not response_format.
        # Quality values: 'low', 'medium', 'high' (not 'standard'/'hd').
        gen_kwargs = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": size,
        }

        # Map legacy quality values to GPT Image format
        quality_map = {"standard": "medium", "hd": "high"}
        gen_kwargs["quality"] = quality_map.get(quality, quality)

        # Only add response_format for DALL-E models (not GPT Image)
        if "dall-e" in model:
            gen_kwargs["response_format"] = "b64_json"

        response = await client.images.generate(**gen_kwargs)

        if not response.data:
            logger.warning("[image_gen] No image data returned")
            return None

        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        # Save to output directory (resolved via shared helper — honours
        # IMAGE_OUTPUT_DIR if set, else falls back to legacy ASTRA_OUTPUT_DIR/images)
        output_dir = get_image_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_filename:
            h = hashlib.md5(prompt.encode()).hexdigest()[:8]
            ts = datetime.now(timezone.utc).strftime("%H%M%S")
            output_filename = f"gpt-{h}-{ts}.png"

        filepath = output_dir / output_filename
        filepath.write_bytes(image_bytes)

        logger.info("[image_gen] Saved %s (%d bytes)", filepath, len(image_bytes))

        return {
            "path": str(filepath),
            "filename": output_filename,
            "size_bytes": len(image_bytes),
            "base64_data": b64_data,
            "mime_type": "image/png",
            "prompt": prompt,
            "text": None,
        }

    except Exception as e:
        logger.error("[image_gen] Failed: %s", e)
        return None


def _map_aspect_to_size(aspect_ratio: Optional[str]) -> Optional[str]:
    """Map aspect ratio string to OpenAI size parameter."""
    if not aspect_ratio:
        return None
    mapping = {
        "16:9": "1536x1024",
        "9:16": "1024x1536",
        "4:3": "1536x1024",
        "21:9": "1536x1024",
        "1:1": "1024x1024",
    }
    return mapping.get(aspect_ratio)


def image_to_data_uri(b64_data: str, mime_type: str = "image/png") -> str:
    """Convert base64 image data to a data URI for inline HTML embedding."""
    return f"data:{mime_type};base64,{b64_data}"


__all__ = ["generate_image", "image_to_data_uri"]
