# FILE: app/debug/screen_capture.py
# Purpose: Gemini multimodal video utilities.
# Called-by: app.debug.file_upload_router, app.debug.gemini_vision, app.debug.recordings_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Gemini multimodal video utilities.

Handles uploading video files to the Google Generative AI Files API,
polling for processing completion, and building content parts for
multimodal messages.

Standalone utility — not tied to any specific feature.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# How long to wait for Gemini to process the uploaded video
MAX_POLL_SECONDS = 120
POLL_INTERVAL_SECONDS = 2


@dataclass
class GeminiFileInfo:
    """Result of uploading a file to Gemini Files API."""
    name: str        # e.g. "files/abc123"
    uri: str         # e.g. "https://generativelanguage.googleapis.com/..."
    mime_type: str
    state: str       # "ACTIVE", "PROCESSING", etc.
    size_bytes: int


async def upload_video_to_gemini(
    file_path: str,
    mime_type: str = "video/webm",
) -> GeminiFileInfo:
    """Upload a video file to the Gemini Files API and wait until ACTIVE.

    Args:
        file_path: Absolute path to the video file on disk.
        mime_type: MIME type of the file.

    Returns:
        GeminiFileInfo with the file URI needed for content parts.

    Raises:
        TimeoutError: If the file doesn't reach ACTIVE within MAX_POLL_SECONDS.
        RuntimeError: If the upload or processing fails.
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)

    logger.info("[screen_capture] Uploading %s (%s) to Gemini Files API", file_path, mime_type)

    # Upload (synchronous SDK call — run in thread to avoid blocking)
    loop = asyncio.get_event_loop()
    uploaded = await loop.run_in_executor(
        None,
        lambda: genai.upload_file(file_path, mime_type=mime_type),
    )

    logger.info("[screen_capture] Uploaded as %s, state=%s", uploaded.name, uploaded.state.name)

    # Poll until ACTIVE
    start = time.time()
    while uploaded.state.name == "PROCESSING":
        if time.time() - start > MAX_POLL_SECONDS:
            raise TimeoutError(
                f"Gemini file {uploaded.name} still PROCESSING after {MAX_POLL_SECONDS}s"
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)
        uploaded = await loop.run_in_executor(
            None,
            lambda: genai.get_file(uploaded.name),
        )
        logger.debug("[screen_capture] Poll: %s state=%s", uploaded.name, uploaded.state.name)

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini file {uploaded.name} in unexpected state: {uploaded.state.name}")

    logger.info("[screen_capture] File %s is ACTIVE", uploaded.name)

    return GeminiFileInfo(
        name=uploaded.name,
        uri=uploaded.uri,
        mime_type=mime_type,
        state="ACTIVE",
        size_bytes=getattr(uploaded, "size_bytes", 0) or os.path.getsize(file_path),
    )


def build_video_content_part(file_uri: str, mime_type: str = "video/webm"):
    """Build a genai.protos.Part for a Gemini multimodal message.

    Args:
        file_uri: The URI from GeminiFileInfo.uri
        mime_type: The MIME type of the file.

    Returns:
        A genai.protos.Part that can be included in message content.
    """
    import google.generativeai as genai

    return genai.protos.Part(
        file_data=genai.protos.FileData(
            file_uri=file_uri,
            mime_type=mime_type,
        )
    )


async def cleanup_gemini_file(file_name: str) -> bool:
    """Delete a file from Gemini storage after analysis.

    Args:
        file_name: The file name (e.g. "files/abc123") from GeminiFileInfo.

    Returns:
        True if deleted, False on error.
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return False

    genai.configure(api_key=api_key)

    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: genai.delete_file(file_name))
        logger.info("[screen_capture] Deleted Gemini file %s", file_name)
        return True
    except Exception as e:
        logger.warning("[screen_capture] Failed to delete %s: %s", file_name, e)
        return False