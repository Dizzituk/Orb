# FILE: app/llm/gemini_vision.py
# Purpose: Vision client for image and video analysis.
# Called-by: app.bridge.attachment_describe, app.content.style_analyser, app.debug.executors.user_files, app.endpoints._chat_media_processors (+11 more)
# Depends-on: app.llm._gemini_vision_utils_2, app.llm._gemini_vision_utils_3
# Last-renovated: 2026-06-11
"""
Vision client for image and video analysis.
Supports Google Gemini and OpenAI GPT-4 Vision with automatic fallback.

v0.14.2 - Flash Removal from Auto-Routing:
- REMOVED: Gemini Flash from all auto-selection paths
- ALL images now route to Gemini 2.5 Pro (was Flash for single images)
- ALL videos now route to Gemini 3 Pro (was Flash for videos <10MB)
- Default tier changed from Flash to 2.5 Pro
- "fast" tier retained only for explicit manual override
- select_vision_tier() simplified: always returns "complex" or "video_heavy"

v0.14.1 - Video Transcription for Pipelines:
- Added transcribe_video_for_context() for Video+Code and high-stakes video pre-steps
- Uses Gemini 3 Pro to generate structured text descriptions for code debugging context
- Supports both Video+Code debug pipeline and high-stakes critical pipeline with video

v0.13.10 - Video Tier Parameter:
- Added optional tier parameter to analyze_video() to support explicit tier override
- When tier is provided, skips internal select_vision_tier() logic
- Enables job_classifier-driven routing for multi-video and mixed-media scenarios

v0.13.0 - Phase 4 Routing Fix:
- Added VIDEO_HEAVY tier using GEMINI_VIDEO_HEAVY_MODEL (gemini-2.5-pro)
- Added OPUS_CRITIC tier using GEMINI_OPUS_CRITIC_MODEL (gemini-2.5-pro)
- Updated tier selection for 8-route system

v0.12.7:
- Added analyze_video() for video analysis with Gemini
- Video files uploaded via Gemini File API for processing

v0.12.6:
- Added OpenAI GPT-4 Vision as fallback provider
- Automatic fallback when Gemini fails (rate limit, quota, errors)
"""
import os
import json
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv
from app.llm._gemini_vision_utils_2 import VIDEO_SIZE_THRESHOLD, VIDEO_TRANSCRIPTION_PROMPT, _get_openai_client, analyze_with_gemini, get_vision_model_for_complexity, is_image_mime_type, select_vision_tier, transcribe_video_for_context_sync
from app.llm._gemini_vision_utils_3 import _detect_mime_type, _get_openai_api_key, _openai_vision_analyze, _read_image_bytes, analyze_video, check_vision_available, transcribe_video_for_context

load_dotenv()

_vision_models = {}
_openai_client = None

# Size threshold for video tier escalation (10MB)


# =============================================================================
# API KEY HELPERS
# =============================================================================

def _get_google_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'")
    return key if key else None


# =============================================================================
# MODEL NAME HELPERS (8-ROUTE SYSTEM)
# =============================================================================

def _get_model_name(tier: str = "default") -> str:
    """
    Get model name for tier.
    
    v0.14.2: Default changed from Flash to 2.5 Pro
    
    Tiers:
    - fast: gemini-2.0-flash (LEGACY - only for explicit manual override)
    - complex: gemini-2.5-pro (ALL images - now the default)
    - video_heavy: gemini-3.0-pro-preview (ALL videos)
    - opus_critic: gemini-3.0-pro-preview (Opus output review)
    - default: same as complex (was fast pre-v0.14.2)
    """
    if tier == "fast":
        return os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash")
    elif tier == "complex":
        return os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro")
    elif tier == "video_heavy":
        return os.getenv("GEMINI_VIDEO_HEAVY_MODEL", "gemini-3.0-pro-preview")
    elif tier == "opus_critic":
        return os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-3.0-pro-preview")
    else:
        # v0.14.2: Default to 2.5 Pro (was Flash)
        return os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro")


def _get_vision_model(tier: str = "default"):
    """Get or create vision model for tier."""
    global _vision_models
    
    if tier in _vision_models:
        return _vision_models[tier]
    
    api_key = _get_google_api_key()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not set")
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    
    model_name = _get_model_name(tier)
    model = genai.GenerativeModel(model_name)
    _vision_models[tier] = model
    print(f"[vision] Initialized Gemini model: {model_name} (tier={tier})")
    
    return model


# =============================================================================
# OPENAI VISION (Fallback)
# =============================================================================


# =============================================================================
# IMAGE HELPERS
# =============================================================================


# =============================================================================
# ANALYZE IMAGE
# =============================================================================

def analyze_image(
    image_source: Union[bytes, str, Path],
    mime_type: Optional[str] = None,
    user_prompt: Optional[str] = None,
    tier: str = "default",
) -> dict:
    """
    Analyze an image and return structured metadata.
    
    Returns:
        dict with: summary, tags, type, provider, error (optional)
    """
    try:
        image_bytes = _read_image_bytes(image_source)
    except Exception as e:
        return {
            "summary": f"Could not read image: {str(e)}",
            "tags": ["image", "error"],
            "type": "image",
            "error": str(e),
        }
    
    if not mime_type and not isinstance(image_source, bytes):
        mime_type = _detect_mime_type(image_source)
    elif not mime_type:
        mime_type = "image/jpeg"
    
    analysis_prompt = user_prompt or """Analyze this image and provide:
1. A brief description (1-2 sentences)
2. Key elements/objects visible
3. Image type (photo, screenshot, diagram, etc.)

Return as JSON: {"summary": "...", "tags": [...], "type": "..."}"""

    # Try Gemini first
    gemini_error = None
    try:
        model = _get_vision_model(tier)
        
        import PIL.Image
        import io
        image = PIL.Image.open(io.BytesIO(image_bytes))
        
        response = model.generate_content([analysis_prompt, image])
        
        response_text = response.text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        try:
            result = json.loads(response_text)
            return {
                "summary": result.get("summary", "Image analyzed"),
                "tags": result.get("tags", ["image"]),
                "type": result.get("type", "image"),
                "provider": "google",
                "model": _get_model_name(tier),
            }
        except json.JSONDecodeError:
            return {
                "summary": response_text[:300],
                "tags": ["image"],
                "type": "image",
                "provider": "google",
                "model": _get_model_name(tier),
            }
            
    except Exception as e:
        gemini_error = str(e)
        print(f"[vision] Gemini failed, trying OpenAI: {e}")
    
    # Fallback to OpenAI
    try:
        openai_result = _openai_vision_analyze(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=analysis_prompt,
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"),
        )
        
        if "error" in openai_result:
            return {
                "summary": f"Analysis failed: {openai_result['error']}",
                "tags": ["image", "error"],
                "type": "image",
                "error": openai_result["error"],
            }
        
        response_text = openai_result["answer"]
        try:
            result = json.loads(response_text)
            return {
                "summary": result.get("summary", "Image analyzed"),
                "tags": result.get("tags", ["image"]),
                "type": result.get("type", "image"),
                "provider": "openai",
                "model": os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"),
            }
        except json.JSONDecodeError:
            return {
                "summary": response_text[:300],
                "tags": ["image"],
                "type": "image",
                "provider": "openai",
                "model": os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"),
            }
            
    except Exception as e:
        return {
            "summary": f"Analysis failed: Gemini ({gemini_error}), OpenAI ({str(e)})",
            "tags": ["image", "error"],
            "type": "image",
            "error": str(e),
        }


# =============================================================================
# ASK ABOUT IMAGE
# =============================================================================

def ask_about_image(
    image_source: Union[bytes, str, Path],
    user_question: str,
    mime_type: Optional[str] = None,
    context: Optional[str] = None,
    tier: str = "default",
) -> dict:
    """
    Ask a question about an image using Gemini Vision (with OpenAI fallback).
    
    Returns:
        dict with: answer, provider, model, error (optional)
    """
    try:
        image_bytes = _read_image_bytes(image_source)
    except Exception as e:
        return {
            "answer": f"Couldn't read image: {str(e)}",
            "provider": "none",
            "model": "none",
            "error": str(e),
        }
    
    if not mime_type and not isinstance(image_source, bytes):
        mime_type = _detect_mime_type(image_source)
    elif not mime_type:
        mime_type = "image/jpeg"
    
    prompt_parts = []
    if context:
        prompt_parts.append(f"Context: {context}\n\n")
    prompt_parts.append(f"User's question: {user_question}")
    full_prompt = "".join(prompt_parts)
    
    # Try Gemini
    gemini_error = None
    try:
        model = _get_vision_model(tier)
        
        import PIL.Image
        import io
        image = PIL.Image.open(io.BytesIO(image_bytes))
        
        response = model.generate_content([full_prompt, image])
        
        return {
            "answer": response.text,
            "provider": "google",
            "model": _get_model_name(tier),
        }
        
    except Exception as e:
        gemini_error = str(e)
        print(f"[vision] Gemini Q&A failed, trying OpenAI: {e}")
    
    # Fallback to OpenAI
    try:
        result = _openai_vision_analyze(
            image_bytes=image_bytes,
            mime_type=mime_type,
            prompt=full_prompt,
            model=os.getenv("OPENAI_VISION_MODEL", "gpt-5.4-mini"),
        )
        return result
        
    except Exception as e:
        return {
            "answer": f"Vision failed: Gemini ({gemini_error}), OpenAI ({str(e)})",
            "provider": "none",
            "model": "none",
            "error": str(e),
        }


# =============================================================================
# ANALYZE VIDEO (with VIDEO_HEAVY tier support)
# =============================================================================


# =============================================================================
# ANALYZE WITH GEMINI (Generic entry point)
# =============================================================================


# =============================================================================
# MODEL SELECTION HELPERS
# =============================================================================


# =============================================================================
# VIDEO TRANSCRIPTION FOR PIPELINES (v0.14.1)
# =============================================================================

# Default prompt for video transcription (optimized for debugging context)


# =============================================================================
# CHECK AVAILABILITY
# =============================================================================