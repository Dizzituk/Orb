# FILE: app/debug/gemini_vision.py
"""
Gemini Vision Analyser — extracts structured observations from video/image uploads.

This module handles ONLY the visual analysis phase. It sends video or image
content to Gemini 3.1 and gets back a structured text description of what
was observed. No tool calls, no code changes — pure observation.

The output text is then injected into Claude Opus's context for the actual
investigation and fix phase.

v1.0 (2026-03-21): Initial — split from monolithic Gemini debug flow.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_VISION_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash"]


async def analyse_visual_content(
    user_message: str,
    video_file_uri: Optional[str] = None,
    video_mime_type: Optional[str] = None,
    file_upload_uri: Optional[str] = None,
    file_upload_mime: Optional[str] = None,
    file_upload_name: Optional[str] = None,
) -> Optional[str]:
    """Send video/image to Gemini 3.1 for visual analysis only.

    Returns a structured text description of what was observed,
    or None if no visual content was provided or analysis failed.

    This is a single-shot call — no tool use, no conversation history.
    Gemini's job is purely observational: describe what screens are shown,
    what the user clicked, what errors appeared, what the UI state looks like.
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        logger.warning("[gemini_vision] GOOGLE_API_KEY not set")
        return None

    # Only proceed if there's actually visual content
    has_video = bool(video_file_uri)
    has_image = bool(file_upload_uri)
    if not has_video and not has_image:
        return None

    genai.configure(api_key=api_key)

    # Build the analysis prompt — instruct Gemini to be a detailed observer
    analysis_prompt = (
        "You are a visual analysis assistant. Your ONLY job is to describe "
        "what you observe in the attached media as accurately and thoroughly "
        "as possible. You are NOT fixing anything or suggesting solutions.\n\n"
        "Provide a structured analysis covering:\n"
        "1. **Screens shown**: What app screens or UI elements are visible\n"
        "2. **User actions**: What buttons were tapped, what text was entered, "
        "what gestures were made (in chronological order)\n"
        "3. **UI state changes**: What happened after each action — did the UI "
        "update, stay the same, show an error, crash, etc.\n"
        "4. **Error messages**: Any error text, crash dialogs, status messages, "
        "HTTP error codes, or toast notifications visible on screen\n"
        "5. **User narration**: If the user is speaking in the video, summarise "
        "what they said about the problem\n"
        "6. **Timing**: Note any significant delays, freezes, or loading states\n\n"
        "Be specific and factual. Quote exact text from the screen where possible. "
        "Do not speculate about causes — just describe what you see.\n\n"
        f"The user's message about this issue: {user_message}"
    )

    # Build multimodal content parts
    content_parts = [analysis_prompt]

    if has_video:
        from app.debug.screen_capture import build_video_content_part
        video_part = build_video_content_part(
            file_uri=video_file_uri,
            mime_type=video_mime_type or "video/webm",
        )
        content_parts.append(video_part)
        logger.info("[gemini_vision] Attached video: %s (%s)", video_file_uri, video_mime_type)

    if has_image:
        from app.debug.screen_capture import build_video_content_part
        image_part = build_video_content_part(
            file_uri=file_upload_uri,
            mime_type=file_upload_mime or "image/png",
        )
        content_parts.append(image_part)
        logger.info("[gemini_vision] Attached image: %s (%s)", file_upload_name, file_upload_mime)

    last_error = None
    for model_name in _VISION_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(content_parts)
            analysis_text = response.text
            if analysis_text and len(analysis_text) > 50:
                logger.info("[gemini_vision] Analysis complete via %s: %d chars", model_name, len(analysis_text))
                return analysis_text
        except Exception as e:
            last_error = e
            error_str = str(e)
            logger.warning("[gemini_vision] %s failed: %s", model_name, error_str[:200])
            if "API_KEY" in error_str or "expired" in error_str.lower():
                break
            continue
    logger.error("[gemini_vision] All vision models failed. Last error: %s", last_error)
    return None


async def cleanup_uploads(
    video_file_uri: Optional[str] = None,
    video_local_path: Optional[str] = None,
    file_upload_gemini_name: Optional[str] = None,
    file_upload_local_path: Optional[str] = None,
):
    """Clean up uploaded files from Gemini Files API and local disk."""
    import os

    if video_file_uri:
        try:
            gemini_file_name = "/".join(video_file_uri.rstrip("/").split("/")[-2:])
            from app.debug.screen_capture import cleanup_gemini_file
            await cleanup_gemini_file(gemini_file_name)
        except Exception as e:
            logger.warning("[gemini_vision] Gemini video cleanup failed: %s", e)

    if video_local_path:
        try:
            if os.path.exists(video_local_path):
                os.remove(video_local_path)
                logger.info("[gemini_vision] Deleted local recording: %s", video_local_path)
        except Exception as e:
            logger.warning("[gemini_vision] Local video cleanup failed: %s", e)

    if file_upload_gemini_name:
        try:
            from app.debug.screen_capture import cleanup_gemini_file
            await cleanup_gemini_file(file_upload_gemini_name)
        except Exception as e:
            logger.warning("[gemini_vision] Gemini file cleanup failed: %s", e)

    if file_upload_local_path:
        try:
            if os.path.exists(file_upload_local_path):
                os.remove(file_upload_local_path)
                logger.info("[gemini_vision] Deleted local upload: %s", file_upload_local_path)
        except Exception as e:
            logger.warning("[gemini_vision] Local upload cleanup failed: %s", e)