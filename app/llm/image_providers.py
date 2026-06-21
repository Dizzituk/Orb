# FILE: app/llm/image_providers.py
# Purpose: Image-gen provider selection + backend dispatch + request body (leaf).
# Called-by: app.llm.image_router (shim), app.llm.image_stream, app.llm.image_core
# Depends-on: app.llm.image_gen / app.llm.nano_banana (lazy)
# Last-renovated: 2026-06-21
"""Provider selection and backend dispatch for image generation.

Pure leaf split out of image_router.py (batch 3, 2026-06-21); imported by the
router shim, image_stream and image_core. Imports nothing back from them.
"""
import os
import logging
from typing import Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _get_provider() -> str:
    return os.getenv("IMAGE_GEN_PROVIDER", "openai").lower()


def _get_fallback_provider() -> str:
    return os.getenv("IMAGE_GEN_FALLBACK_PROVIDER", "google").lower()


class ImageGenRequest(BaseModel):
    project_id: int
    prompt: str


async def _generate_with_provider(
    provider: str,
    prompt: str,
    aspect_ratio: Optional[str] = None,
) -> Optional[dict]:
    """Call the appropriate backend based on provider string."""
    if provider == "openai":
        from app.llm.image_gen import generate_image
        return await generate_image(prompt=prompt, aspect_ratio=aspect_ratio)
    elif provider in ("google", "gemini"):
        from app.llm.nano_banana import generate_image
        return await generate_image(prompt=prompt, aspect_ratio=aspect_ratio)
    else:
        logger.error("[image_router] Unknown provider: %s", provider)
        return None
