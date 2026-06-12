# FILE: app/debug/recordings_router.py
# Purpose: Recordings Router — upload screen recordings and prepare for Gemini analysis.
# Called-by: main
# Depends-on: app.auth, app.auth.middleware, app.debug.screen_capture
# Last-renovated: 2026-06-11
"""
Recordings Router — upload screen recordings and prepare for Gemini analysis.

Endpoint: POST /api/recordings/upload
    Accepts multipart video + metadata JSON.
    Saves to disk, uploads to Gemini Files API, returns file_uri.

Standalone: the router path is /api/recordings (not /api/debug/recordings)
so it can be used from any feature. The 'context' field in metadata
tracks which feature initiated the recording (e.g. "debug", "content").
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.debug.screen_capture import upload_video_to_gemini, GeminiFileInfo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/recordings", tags=["Recordings"])

# Base directory for saved recordings
RECORDINGS_DIR = Path(os.getenv("ASTRA_DATA_DIR", "D:/Orb/data")) / "recordings"


class UploadResponse(BaseModel):
    """Response from a successful recording upload."""
    recording_id: str
    file_uri: str
    file_name: str
    mime_type: str
    size_bytes: int
    duration_seconds: Optional[float] = None
    local_path: str


@router.post("/upload", response_model=UploadResponse)
async def upload_recording(
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    auth: AuthResult = Depends(require_auth),
):
    """Upload a screen recording and prepare it for Gemini multimodal analysis.

    Accepts:
        file: The video file (WebM expected)
        metadata: JSON string with optional fields:
            - context: str ("debug", "content", etc.) — used for subdirectory
            - session_id: str — links recording to a chat session
            - duration_seconds: float — recording duration

    Returns:
        UploadResponse with the Gemini file_uri for use in multimodal messages.
    """
    # 1. Parse metadata
    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        meta = {}

    context = meta.get("context", "general")
    session_id = meta.get("session_id", "")
    duration_seconds = meta.get("duration_seconds", None)

    # 2. Determine MIME type
    mime_type = file.content_type or "video/webm"
    if not mime_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"Expected video file, got {mime_type}")

    # 3. Save to disk
    save_dir = RECORDINGS_DIR / context
    save_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    recording_id = f"rec-{timestamp}-{uuid.uuid4().hex[:8]}"
    ext = ".webm" if "webm" in mime_type else ".mp4"
    filename = f"{recording_id}{ext}"
    file_path = save_dir / filename

    try:
        contents = await file.read()
        file_path.write_bytes(contents)
        size_bytes = len(contents)
        logger.info(
            "[recordings] Saved %s (%d bytes) to %s",
            recording_id, size_bytes, file_path,
        )
    except Exception as e:
        logger.error("[recordings] Failed to save file: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save recording")

    # 4. Upload to Gemini Files API
    try:
        gemini_info: GeminiFileInfo = await upload_video_to_gemini(
            file_path=str(file_path),
            mime_type=mime_type,
        )
    except TimeoutError as e:
        logger.error("[recordings] Gemini upload timed out: %s", e)
        raise HTTPException(status_code=504, detail="Video processing timed out")
    except Exception as e:
        logger.error("[recordings] Gemini upload failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Gemini upload failed: {e}")

    return UploadResponse(
        recording_id=recording_id,
        file_uri=gemini_info.uri,
        file_name=gemini_info.name,
        mime_type=mime_type,
        size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        local_path=str(file_path),
    )