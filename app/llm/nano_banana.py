# FILE: app/llm/nano_banana.py
"""
Google Gemini image generation backend (Nano Banana).

Called by image_service.py when IMAGE_GEN_PROVIDER=google,
or as fallback when OpenAI primary fails.
All config read from .env at runtime — no hardcoded models.

v2.0 (2026-03-20): Env-driven config via IMAGE_GEN_FALLBACK_MODEL.
v1.1 (2026-03-09): Fixed to use google-genai SDK with response_modalities.
v1.0 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.llm.image_output_dir import get_image_output_dir

logger = logging.getLogger(__name__)


def _get_model() -> str:
    """Read model from env. Checks fallback vars then primary."""
    return (
        os.getenv("IMAGE_GEN_FALLBACK_MODEL")
        or os.getenv("IMAGE_GEN_MODEL")
        or "gemini-2.5-flash-image"
    )


async def generate_image(
    prompt: str,
    model: Optional[str] = None,
    output_filename: Optional[str] = None,
    aspect_ratio: Optional[str] = None,
) -> Optional[dict]:
    """Generate an image using Google Gemini (Nano Banana).

    Model read from .env (IMAGE_GEN_FALLBACK_MODEL) at runtime.

    Returns:
        Dict with path, filename, size_bytes, base64_data, mime_type or None
    """
    model = model or _get_model()
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.error("[nano_banana] GOOGLE_API_KEY not set")
            return None

        client = genai.Client(api_key=api_key)

        logger.info("[nano_banana] Generating image with %s (ar=%s): %s",
                     model, aspect_ratio or "default", prompt[:100])

        config_kwargs = {"response_modalities": ["TEXT", "IMAGE"]}
        if aspect_ratio:
            config_kwargs["image_config"] = types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        # Extract image data from response parts
        image_data = None
        mime_type = "image/png"
        text_parts = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    mime_type = part.inline_data.mime_type or "image/png"
                elif hasattr(part, 'text') and part.text:
                    text_parts.append(part.text)

        if not image_data:
            logger.warning("[nano_banana] No image data in response. Text: %s",
                         " ".join(text_parts)[:200] if text_parts else "none")
            return None

        # Ensure bytes
        if isinstance(image_data, str):
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = bytes(image_data) if not isinstance(image_data, bytes) else image_data

        b64 = base64.b64encode(image_bytes).decode('ascii')

        # Save to output directory (shared helper — see image_output_dir.py)
        output_dir = get_image_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        if not output_filename:
            import hashlib
            h = hashlib.md5(prompt.encode()).hexdigest()[:8]
            ts = datetime.now(timezone.utc).strftime('%H%M%S')
            ext = "png" if "png" in mime_type else "jpeg"
            output_filename = f"nano-{h}-{ts}.{ext}"

        filepath = output_dir / output_filename
        filepath.write_bytes(image_bytes)
        logger.info("[nano_banana] Saved %s (%d bytes)", filepath, len(image_bytes))

        return {
            "path": str(filepath),
            "filename": output_filename,
            "size_bytes": len(image_bytes),
            "base64_data": b64,
            "mime_type": mime_type,
            "prompt": prompt,
            "text": " ".join(text_parts) if text_parts else None,
        }

    except Exception as e:
        logger.error("[nano_banana] Failed: %s", e)
        return None


def image_to_data_uri(b64_data: str, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{b64_data}"


__all__ = ["generate_image", "image_to_data_uri"]
