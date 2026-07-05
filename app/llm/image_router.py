# FILE: app/llm/image_router.py
# Purpose: Image generation HTTP endpoint + re-export shim for stream/core (batch-3 split).
# Called-by: main (router), app.bridge.capability_layer (generate_image_core), app.llm.routing.handler_registry (generate_image_stream)
# Depends-on: app.llm.image_providers, app.llm.image_stream, app.llm.image_core, app.llm.image_research
# Last-renovated: 2026-06-21
"""Image generation endpoint + re-export shim.

Split 2026-06-21 (batch 3) into single-responsibility modules; this file keeps
the /generate-image HTTP route and re-exports the streaming + bridge entry
points so their lazy importers resolve unchanged:
  - image_providers.py -- provider selection / dispatch / request body (leaf)
  - image_research.py  -- research-before-chart helper (leaf)
  - image_stream.py    -- generate_image_stream (SSE chat path)
  - image_core.py      -- generate_image_core (bridge non-streaming path)

Provider routing read from .env (IMAGE_GEN_PROVIDER / IMAGE_GEN_MODEL +
IMAGE_GEN_FALLBACK_PROVIDER / IMAGE_GEN_FALLBACK_MODEL).
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth.middleware import require_auth, AuthResult
from app.memory import service as memory_service, schemas as memory_schemas

from app.llm.image_providers import (
    _get_provider,
    _get_fallback_provider,
    _generate_with_provider,
    ImageGenRequest,
)
# Re-export the streaming + bridge entry points for their lazy importers
# (handler_registry.generate_image_stream / capability_layer.generate_image_core).
from app.llm.image_stream import generate_image_stream
from app.llm.image_core import generate_image_core
# Surface preservation: keep the moved helpers importable from this module path.
from app.llm.image_stream import _sse, _sse_token, _sse_status  # noqa: F401
from app.llm.image_research import _RESEARCH_KEYWORDS, _needs_research, _run_research  # noqa: F401

logger = logging.getLogger(__name__)

router = APIRouter(tags=["image_generation"])


@router.post("/generate-image")
async def generate_image_endpoint(
    req: ImageGenRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """Generate an image using configured provider, with automatic fallback."""
    primary = _get_provider()
    fallback = _get_fallback_provider()
    used_provider = primary

    logger.info("[image_router] Generating image: primary=%s, fallback=%s", primary, fallback)

    # Try primary
    result = await _generate_with_provider(primary, prompt=req.prompt)

    # Fallback if primary failed and fallback is different
    if not result and fallback and fallback != primary:
        logger.warning("[image_router] Primary (%s) failed, trying fallback (%s)", primary, fallback)
        result = await _generate_with_provider(fallback, prompt=req.prompt)
        used_provider = fallback

    if not result:
        raise HTTPException(500, "Image generation failed (primary and fallback)")

    # Resolve model name for memory log
    if used_provider == "openai":
        model_name = os.getenv("IMAGE_GEN_MODEL", "")
    else:
        model_name = os.getenv("IMAGE_GEN_FALLBACK_MODEL", "")

    # Save to memory
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="user",
        content=f"[Image request] {req.prompt}",
        provider="local",
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="assistant",
        content=f"Generated image: {result['filename']}",
        provider=used_provider,
        model=model_name,
    ))

    return {
        "success": True,
        "provider": used_provider,
        "model": model_name,
        "file": {
            "path": result["path"],
            "filename": result["filename"],
            "type": "image",
            "size": result["size_bytes"],
            "description": f"Generated: {req.prompt[:80]}",
        },
        "data_uri": f"data:{result['mime_type']};base64,{result['base64_data']}",
    }
