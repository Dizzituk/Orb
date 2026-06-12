# Purpose: gemini vision utils 2
# Called-by: app.llm._gemini_vision_utils_3, app.llm.gemini_vision
# Depends-on: app.llm.gemini_vision
# Last-renovated: 2026-06-11
from __future__ import annotations
import base64
from typing import Optional
_openai_client = None


VIDEO_SIZE_THRESHOLD = 10 * 1024 * 1024

def select_vision_tier(
    attachment_size_bytes: int = 0,
    is_video: bool = False,
    image_count: int = 1,
    job_type: Optional[str] = None,
) -> str:
    """
    Select the appropriate Gemini tier based on content.
    
    v0.14.2: Flash removed from auto-selection
    - ALL videos → video_heavy (Gemini 3 Pro)
    - ALL images → complex (Gemini 2.5 Pro)
    - Flash ("fast") only for explicit manual override
    
    Rules:
    - opus.critic job_type → opus_critic tier
    - Any video → video_heavy tier (was size-dependent pre-v0.14.2)
    - Any image(s) → complex tier (was count-dependent pre-v0.14.2)
    
    Returns tier name: "complex", "video_heavy", or "opus_critic"
    """
    # Explicit opus.critic
    if job_type == "opus.critic":
        return "opus_critic"
    
    # v0.14.2: ALL videos → video_heavy (Gemini 3 Pro)
    if is_video:
        return "video_heavy"
    
    # v0.14.2: ALL images → complex (Gemini 2.5 Pro)
    # No more Flash for any images
    return "complex"

def _get_openai_client():
    """Lazy init for OpenAI client."""
    from .gemini_vision import _get_openai_api_key
    global _openai_client
    
    if _openai_client is None:
        api_key = _get_openai_api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
        print("[vision] OpenAI client initialized")
    
    return _openai_client

def is_image_mime_type(mime_type: str) -> bool:
    """Check if MIME type is an image."""
    return mime_type.startswith("image/") if mime_type else False

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
    from .gemini_vision import _get_google_api_key, _get_model_name
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

def get_vision_model_for_complexity(question: str) -> str:
    """
    Select vision model tier based on question complexity.
    """
    from .gemini_vision import _get_model_name
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

VIDEO_TRANSCRIPTION_PROMPT = """Analyze this video and provide a detailed structured description for debugging/code analysis purposes.

Focus on:
1. **Timeline of Events**: Describe what happens chronologically with approximate timestamps
2. **UI/Screen Activity**: Any visible user interface elements, screens, dialogs, forms
3. **Error Messages**: Any visible error messages, warnings, stack traces, or exceptions
4. **Log Output**: Any visible log output, console messages, or terminal output
5. **User Actions**: Mouse clicks, keyboard input, navigation, file operations
6. **Code/Text Visible**: Any code snippets, configuration, or text shown on screen
7. **Audio/Speech**: Any relevant spoken instructions or narration (if present)

Format your response as structured text that can be used as context for code debugging.
Be specific about what you observe - timestamps, exact error text, UI element names, etc."""

def transcribe_video_for_context_sync(
    video_path: str,
    custom_prompt: str | None = None,
) -> str:
    """
    Synchronous wrapper for transcribe_video_for_context.
    
    For use in contexts where async is not available.
    """
    from .gemini_vision import transcribe_video_for_context
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    transcribe_video_for_context(video_path, custom_prompt)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                transcribe_video_for_context(video_path, custom_prompt)
            )
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(transcribe_video_for_context(video_path, custom_prompt))
