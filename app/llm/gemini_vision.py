# FILE: app/llm/gemini_vision.py
"""
Vision client for image and video analysis.
Supports Google Gemini and OpenAI GPT-4 Vision with automatic fallback.

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
import base64
import json
import time
from pathlib import Path
from typing import Optional, Union
from dotenv import load_dotenv

load_dotenv()

_vision_models = {}
_openai_client = None

# Size threshold for video tier escalation (10MB)
VIDEO_SIZE_THRESHOLD = 10 * 1024 * 1024


# =============================================================================
# API KEY HELPERS
# =============================================================================

def _get_google_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'")
    return key if key else None


def _get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'")
    return key if key else None


# =============================================================================
# MODEL NAME HELPERS (8-ROUTE SYSTEM)
# =============================================================================

def _get_model_name(tier: str = "default") -> str:
    """
    Get model name for tier.
    
    Tiers:
    - fast: gemini-2.0-flash (simple images, small videos)
    - complex: gemini-2.5-pro (complex PDFs with images, multi-image)
    - video_heavy: gemini-2.5-pro (videos >10MB)
    - opus_critic: gemini-2.5-pro (Opus output review)
    - default: same as fast
    """
    if tier == "fast":
        return os.getenv("GEMINI_VISION_MODEL_FAST", "gemini-2.0-flash")
    elif tier == "complex":
        return os.getenv("GEMINI_VISION_MODEL_COMPLEX", "gemini-2.5-pro")
    elif tier == "video_heavy":
        return os.getenv("GEMINI_VIDEO_HEAVY_MODEL", "gemini-2.5-pro")
    elif tier == "opus_critic":
        return os.getenv("GEMINI_OPUS_CRITIC_MODEL", "gemini-2.5-pro")
    else:
        return os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")


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


def select_vision_tier(
    attachment_size_bytes: int = 0,
    is_video: bool = False,
    image_count: int = 1,
    job_type: Optional[str] = None,
) -> str:
    """
    Select the appropriate Gemini tier based on content.
    
    Rules:
    - opus.critic job_type → opus_critic tier
    - Video >10MB → video_heavy tier
    - Video ≤10MB → fast tier
    - Multiple images (>2) or complex keywords → complex tier
    - Simple image → fast tier
    
    Returns tier name: "fast", "complex", "video_heavy", or "opus_critic"
    """
    # Explicit opus.critic
    if job_type == "opus.critic":
        return "opus_critic"
    
    # Video handling
    if is_video:
        if attachment_size_bytes > VIDEO_SIZE_THRESHOLD:
            return "video_heavy"
        return "fast"
    
    # Multiple images → complex
    if image_count > 2:
        return "complex"
    
    return "fast"


# =============================================================================
# OPENAI VISION (Fallback)
# =============================================================================

def _get_openai_client():
    """Lazy init for OpenAI client."""
    global _openai_client
    
    if _openai_client is None:
        api_key = _get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
        print("[vision] OpenAI client initialized")
    
    return _openai_client


def _openai_vision_analyze(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
    model: str = "gpt-4o-mini",
) -> dict:
    """Analyze image using OpenAI Vision."""
    try:
        client = _get_openai_client()
        
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_image}"
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=2000,
        )
        
        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "provider": "openai",
            "model": model,
        }
        
    except Exception as e:
        print(f"[vision] OpenAI vision error: {e}")
        return {
            "answer": f"OpenAI vision failed: {str(e)}",
            "error": str(e),
            "provider": "openai",
            "model": model,
        }


# =============================================================================
# IMAGE HELPERS
# =============================================================================

def _read_image_bytes(source: Union[bytes, str, Path]) -> bytes:
    """Read image bytes from various sources."""
    if isinstance(source, bytes):
        return source
    
    path = Path(source) if isinstance(source, str) else source
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    return path.read_bytes()


def _detect_mime_type(path: Union[str, Path]) -> str:
    """Detect MIME type from file extension."""
    suffix = Path(path).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    return mime_map.get(suffix, "application/octet-stream")


def is_image_mime_type(mime_type: str) -> bool:
    """Check if MIME type is an image."""
    return mime_type.startswith("image/") if mime_type else False


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
            model="gpt-4o-mini",
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
                "model": "gpt-4o-mini",
            }
        except json.JSONDecodeError:
            return {
                "summary": response_text[:300],
                "tags": ["image"],
                "type": "image",
                "provider": "openai",
                "model": "gpt-4o-mini",
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
            model="gpt-4o",
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

def analyze_video(
    video_path: Union[str, Path],
    user_question: Optional[str] = None,
    context: Optional[str] = None,
    job_type: Optional[str] = None,
    tier: Optional[str] = None,
) -> dict:
    """
    Analyze a video file using Gemini Vision.
    
    Gemini supports native video analysis via the File API.
    
    v0.13.10: Added optional tier parameter for explicit tier override.
    When tier is provided, it skips internal select_vision_tier() logic.
    This enables job_classifier-driven routing for multi-video scenarios.
    
    Tier selection (when tier not provided):
    - Videos >10MB → video_heavy (gemini-2.5-pro)
    - Videos ≤10MB → fast (gemini-2.0-flash)
    - opus.critic job_type → opus_critic (gemini-2.5-pro)
    
    Args:
        video_path: Path to the video file
        user_question: Optional question about the video
        context: Optional additional context
        job_type: Optional job type for tier override
        tier: Optional explicit tier override (v0.13.10)
    
    Returns:
        dict with: answer, provider, model, tier, error (optional)
    """
    path = Path(video_path)
    if not path.exists():
        return {
            "answer": f"Video file not found: {video_path}",
            "provider": "none",
            "model": "none",
            "error": "File not found",
        }
    
    file_size = path.stat().st_size
    print(f"[vision] Analyzing video: {path.name} ({file_size // 1024}KB)")
    
    # Build prompt
    prompt_parts = []
    if context:
        prompt_parts.append(f"Context: {context}\n\n")
    
    if user_question:
        prompt_parts.append(f"User's question about this video: {user_question}")
    else:
        prompt_parts.append("""Analyze this video and describe:
1. What is happening in the video
2. Key events or actions
3. Duration and pacing
4. Any text, speech, or important audio elements
5. Overall summary""")
    
    full_prompt = "".join(prompt_parts)
    
    # v0.13.10: Use explicit tier if provided, otherwise select based on file size/job_type
    if tier is None:
        tier = select_vision_tier(
            attachment_size_bytes=file_size,
            is_video=True,
            job_type=job_type,
        )
        print(f"[vision] Video tier auto-selected: {file_size / 1024 / 1024:.1f}MB → {tier}")
    else:
        print(f"[vision] Video tier explicitly provided: {tier}")
    
    try:
        import google.generativeai as genai
        
        api_key = _get_google_api_key()
        if not api_key:
            return {
                "answer": "GOOGLE_API_KEY not set - video analysis requires Gemini",
                "provider": "none",
                "model": "none",
                "error": "No API key",
            }
        
        genai.configure(api_key=api_key)
        
        # Upload video file to Gemini
        print(f"[vision] Uploading video to Gemini File API...")
        video_file = genai.upload_file(path=str(path))
        
        # Wait for processing
        print(f"[vision] Waiting for video processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            return {
                "answer": "Video processing failed",
                "provider": "google",
                "model": _get_model_name(tier),
                "tier": tier,
                "error": "Processing failed",
            }
        
        print(f"[vision] Video ready, generating response with {tier} tier...")
        
        # Generate response
        model = _get_vision_model(tier)
        response = model.generate_content([video_file, full_prompt])
        
        # Clean up uploaded file
        try:
            genai.delete_file(video_file.name)
            print(f"[vision] Cleaned up video file from Gemini")
        except Exception as cleanup_error:
            print(f"[vision] Warning: Could not delete uploaded file: {cleanup_error}")
        
        return {
            "answer": response.text,
            "provider": "google",
            "model": _get_model_name(tier),
            "tier": tier,
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"[vision] Video analysis failed: {e}")
        
        # Check for quota/rate limit errors
        if "429" in error_msg or "quota" in error_msg.lower():
            return {
                "answer": "Gemini API quota exceeded. Video analysis is temporarily unavailable.",
                "provider": "google",
                "model": _get_model_name(tier),
                "tier": tier,
                "error": "Quota exceeded",
            }
        
        return {
            "answer": f"Video analysis failed: {error_msg}",
            "provider": "google",
            "model": _get_model_name(tier),
            "tier": tier,
            "error": error_msg,
        }


# =============================================================================
# ANALYZE WITH GEMINI (Generic entry point)
# =============================================================================

async def analyze_with_gemini(
    prompt: str,
    attachments: Optional[list] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> dict:
    """
    Generic entry point for Gemini analysis.
    
    Used by router.py for vision jobs.
    """
    import google.generativeai as genai
    
    api_key = _get_google_api_key()
    if not api_key:
        return {
            "content": "GOOGLE_API_KEY not set",
            "error": "No API key",
        }
    
    genai.configure(api_key=api_key)
    
    # Use provided model or default
    model_name = model or _get_model_name("default")
    
    try:
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
        )
        
        # Build content parts
        parts = [prompt]
        
        # Add attachments if any
        if attachments:
            import PIL.Image
            import io
            
            for att in attachments:
                if att.get("data"):
                    # Base64 image data
                    image_bytes = base64.b64decode(att["data"])
                    image = PIL.Image.open(io.BytesIO(image_bytes))
                    parts.append(image)
                elif att.get("path"):
                    # File path
                    image = PIL.Image.open(att["path"])
                    parts.append(image)
        
        response = gemini_model.generate_content(parts)
        
        return {
            "content": response.text,
            "provider": "google",
            "model": model_name,
        }
        
    except Exception as e:
        return {
            "content": f"Gemini analysis failed: {str(e)}",
            "error": str(e),
        }


# =============================================================================
# MODEL SELECTION HELPERS
# =============================================================================

def get_vision_model_for_complexity(question: str) -> str:
    """
    Select vision model tier based on question complexity.
    """
    q_lower = question.lower()
    
    # Complex analysis keywords -> use complex model
    complex_keywords = [
        "analyze", "detailed", "compare", "critique", "review",
        "explain in detail", "step by step", "comprehensive",
        "technical", "professional", "expert",
    ]
    
    if any(kw in q_lower for kw in complex_keywords):
        return _get_model_name("complex")
    
    # Simple questions -> use fast model
    simple_keywords = [
        "what is", "what's this", "describe", "read",
        "ocr", "text", "simple",
    ]
    
    if any(kw in q_lower for kw in simple_keywords):
        return _get_model_name("fast")
    
    return _get_model_name("default")


# =============================================================================
# CHECK AVAILABILITY
# =============================================================================

def check_vision_available() -> dict:
    """Check if vision capability is available."""
    google_available = False
    openai_available = False
    
    google_key = _get_google_api_key()
    if google_key:
        try:
            model = _get_vision_model("default")
            google_available = True
        except Exception as e:
            print(f"[vision] Google check failed: {e}")
    
    openai_key = _get_openai_api_key()
    if openai_key:
        openai_available = True
    
    if google_available:
        return {
            "available": True,
            "provider": "google",
            "model": _get_model_name("default"),
            "fallback": openai_available,
            "video_support": True,
            "tiers": {
                "fast": _get_model_name("fast"),
                "complex": _get_model_name("complex"),
                "video_heavy": _get_model_name("video_heavy"),
                "opus_critic": _get_model_name("opus_critic"),
            },
        }
    elif openai_available:
        return {
            "available": True,
            "provider": "openai",
            "model": "gpt-4o-mini",
            "fallback": False,
            "video_support": False,
        }
    else:
        return {
            "available": False,
            "error": "No vision provider configured. Set GOOGLE_API_KEY or OPENAI_API_KEY.",
            "video_support": False,
        }