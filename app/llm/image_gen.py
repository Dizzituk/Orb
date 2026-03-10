# FILE: app/llm/image_gen.py
"""
GPT Image 1.5 integration for generating images via OpenAI API.

Used by the content/blog pipeline to generate concept images,
hero graphics, and visual elements for HTML pages.

v1.0 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("ASTRA_OUTPUT_DIR", r"D:\Orb\output")


async def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "standard",
    model: str = "gpt-image-1.5",
    output_filename: Optional[str] = None,
) -> Optional[dict]:
    """Generate an image using OpenAI GPT Image 1.5.

    Args:
        prompt: Description of the image to generate
        size: Image dimensions (1024x1024, 1536x1024, 1024x1536)
        quality: 'standard' or 'hd'
        model: Model to use (gpt-image-1.5)
        output_filename: Filename to save as (auto-generated if None)

    Returns:
        Dict with path, filename, size_bytes, base64_data or None on failure
    """
    try:
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("[image_gen] OPENAI_API_KEY not set")
            return None

        client = AsyncOpenAI(api_key=api_key)

        logger.info("[image_gen] Generating: %s (size=%s, quality=%s)", prompt[:80], size, quality)

        response = await client.images.generate(
            model=model,
            prompt=prompt,
            n=1,
            size=size,
            quality=quality,
            response_format="b64_json",
        )

        if not response.data:
            logger.warning("[image_gen] No image data returned")
            return None

        b64_data = response.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        # Save to output directory
        output_dir = Path(OUTPUT_DIR) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_filename:
            import hashlib
            h = hashlib.md5(prompt.encode()).hexdigest()[:8]
            output_filename = f"gen-{h}.png"

        filepath = output_dir / output_filename
        filepath.write_bytes(image_bytes)

        logger.info("[image_gen] Saved %s (%d bytes)", filepath, len(image_bytes))

        return {
            "path": str(filepath),
            "filename": output_filename,
            "size_bytes": len(image_bytes),
            "base64_data": b64_data,
            "prompt": prompt,
        }

    except Exception as e:
        logger.error("[image_gen] Failed: %s", e)
        return None


def image_to_data_uri(b64_data: str, mime_type: str = "image/png") -> str:
    """Convert base64 image data to a data URI for inline HTML embedding."""
    return f"data:{mime_type};base64,{b64_data}"


__all__ = ["generate_image", "image_to_data_uri"]
