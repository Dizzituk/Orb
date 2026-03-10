# FILE: app/llm/image_router.py
"""
Image generation endpoint - routes "create an image" requests to Nano Banana.

v1.0 (2026-03-09): Initial implementation.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth.middleware import require_auth, AuthResult
from app.memory import service as memory_service, schemas as memory_schemas

logger = logging.getLogger(__name__)

router = APIRouter(tags=["image_generation"])


class ImageGenRequest(BaseModel):
    project_id: int
    prompt: str


@router.post("/generate-image")
async def generate_image_endpoint(
    req: ImageGenRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """Generate an image using Nano Banana and return file info."""
    from app.llm.nano_banana import generate_image
    from app.llm.file_output import sse_file_outputs

    result = await generate_image(prompt=req.prompt)

    if not result:
        raise HTTPException(500, "Image generation failed")

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
        provider="google",
        model="nano-banana-2",
    ))

    return {
        "success": True,
        "file": {
            "path": result["path"],
            "filename": result["filename"],
            "type": "image",
            "size": result["size_bytes"],
            "description": f"Generated: {req.prompt[:80]}",
        },
        "data_uri": f"data:{result['mime_type']};base64,{result['base64_data']}",
    }
