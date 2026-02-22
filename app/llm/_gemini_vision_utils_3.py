from __future__ import annotations
import base64
import os
import time
from app.llm._gemini_vision_utils_2 import VIDEO_TRANSCRIPTION_PROMPT, _get_openai_client, select_vision_tier
from pathlib import Path
from typing import Optional, Union


def _get_openai_api_key() -> Optional[str]:
    """Get OpenAI API key from environment."""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'")
    return key if key else None

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
    from .gemini_vision import _get_google_api_key, _get_model_name, _get_vision_model
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

async def transcribe_video_for_context(
    video_path: str,
    custom_prompt: str | None = None,
) -> str:
    """
    Transcribe/describe a video for use as context in code pipelines.
    
    v0.14.1: Used by Video+Code debug pipeline and high-stakes video pre-step.
    
    Uses Gemini 3 Pro to generate a structured textual description of the video
    content, optimized for debugging and code analysis context.
    
    Args:
        video_path: Path to the video file
        custom_prompt: Optional custom prompt (defaults to VIDEO_TRANSCRIPTION_PROMPT)
    
    Returns:
        Structured text description of the video content.
        On error, returns an error message string.
    """
    from .gemini_vision import _get_google_api_key, _get_vision_model
    from pathlib import Path
    
    path = Path(video_path)
    if not path.exists():
        error_msg = f"[video-transcribe] Video file not found: {video_path}"
        print(error_msg)
        return f"ERROR: {error_msg}"
    
    file_size = path.stat().st_size
    prompt = custom_prompt or VIDEO_TRANSCRIPTION_PROMPT
    
    print(f"[video-transcribe] Transcribing video: {path.name} ({file_size / 1024 / 1024:.1f}MB)")
    
    try:
        import google.generativeai as genai
        
        api_key = _get_google_api_key()
        if not api_key:
            error_msg = "[video-transcribe] GOOGLE_API_KEY not set - video transcription requires Gemini"
            print(error_msg)
            return f"ERROR: {error_msg}"
        
        genai.configure(api_key=api_key)
        
        # Upload video file to Gemini
        print(f"[video-transcribe] Uploading video to Gemini File API...")
        video_file = genai.upload_file(path=str(path))
        
        # Wait for processing
        print(f"[video-transcribe] Waiting for video processing...")
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            error_msg = f"[video-transcribe] Video processing failed for {path.name}"
            print(error_msg)
            return f"ERROR: {error_msg}"
        
        print(f"[video-transcribe] Video ready, generating transcript with Gemini 3 Pro...")
        
        # Use video_heavy tier (Gemini 3 Pro) for transcription
        model = _get_vision_model("video_heavy")
        response = model.generate_content([video_file, prompt])
        
        # Clean up uploaded file
        try:
            genai.delete_file(video_file.name)
            print(f"[video-transcribe] Cleaned up video file from Gemini")
        except Exception as cleanup_error:
            print(f"[video-transcribe] Warning: Could not delete uploaded file: {cleanup_error}")
        
        transcript = response.text
        print(f"[video-transcribe] Transcript complete: {len(transcript)} chars")
        
        return transcript
        
    except Exception as e:
        error_msg = f"[video-transcribe] Error transcribing video {path.name}: {e}"
        print(error_msg)
        return f"ERROR: {error_msg}"

def check_vision_available() -> dict:
    """Check if vision capability is available."""
    from .gemini_vision import _get_google_api_key, _get_model_name, _get_vision_model
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
