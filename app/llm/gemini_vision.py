# FILE: app/llm/gemini_vision.py
"""
Gemini Vision client for image analysis.
Uses Google's Gemini multimodal API.
"""
import os
import base64
import json
from typing import Optional
from dotenv import load_dotenv

# Load .env in this module
load_dotenv()

# Lazy-loaded model
_vision_model = None


def _get_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        key = key.strip().strip('"').strip("'")
    return key if key else None


def _get_model_name() -> str:
    """Get Gemini vision model name from environment."""
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
    return model.strip().strip('"').strip("'")


def _get_vision_model():
    """Lazy init for Gemini vision model."""
    global _vision_model
    if _vision_model is None:
        import google.generativeai as genai
        
        api_key = _get_api_key()
        if not api_key:
            # Debug: show what's happening
            print("[gemini_vision] ERROR: GOOGLE_API_KEY not found in environment")
            print(f"[gemini_vision] Current working directory: {os.getcwd()}")
            
            # Check if .env file exists
            env_paths = [
                os.path.join(os.getcwd(), ".env"),
                "D:\\Orb\\.env",
            ]
            for env_path in env_paths:
                if os.path.exists(env_path):
                    print(f"[gemini_vision] Found .env at: {env_path}")
                    # Try loading it explicitly
                    load_dotenv(env_path)
                    api_key = os.getenv("GOOGLE_API_KEY")
                    if api_key:
                        api_key = api_key.strip().strip('"').strip("'")
                        print(f"[gemini_vision] Loaded API key from {env_path}")
                        break
            
            if not api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set. "
                    "Please add GOOGLE_API_KEY=your_key to D:\\Orb\\.env"
                )
        
        model_name = _get_model_name()
        print(f"[gemini_vision] Initializing model: {model_name}")
        
        genai.configure(api_key=api_key)
        _vision_model = genai.GenerativeModel(model_name)
        print(f"[gemini_vision] Model initialized successfully")
    
    return _vision_model


def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    user_prompt: Optional[str] = None,
) -> dict:
    """
    Analyze an image using Gemini Vision.
    
    Args:
        image_bytes: Raw image bytes
        mime_type: MIME type (e.g., "image/jpeg", "image/png")
        user_prompt: Optional user context about the image
    
    Returns:
        dict with keys: summary, tags, type, and optionally error
    """
    print(f"[gemini_vision] Analyzing image: mime_type={mime_type}, size={len(image_bytes)} bytes")
    
    try:
        model = _get_vision_model()
    except RuntimeError as e:
        print(f"[gemini_vision] Model initialization failed: {e}")
        return {
            "summary": f"Image analysis unavailable: {str(e)}",
            "tags": ["image", "error"],
            "type": "image",
            "error": str(e),
        }
    
    # Build the prompt
    analysis_prompt = """Analyze this image and provide:
1. A brief summary (1-3 sentences) describing what's in the image and any relevant details.
2. A list of descriptive tags (3-8 tags).
3. The image type category (e.g., "photo", "screenshot", "diagram", "document", "artwork", "chart", "meme", "logo", etc.)

Respond in this exact JSON format:
{
    "summary": "Your summary here",
    "tags": ["tag1", "tag2", "tag3"],
    "type": "image_type"
}

Only respond with the JSON, no other text."""

    if user_prompt:
        analysis_prompt += f"\n\nUser's question/context: {user_prompt}"
    
    # Create image part for Gemini
    image_part = {
        "mime_type": mime_type,
        "data": base64.b64encode(image_bytes).decode("utf-8"),
    }
    
    try:
        print(f"[gemini_vision] Calling Gemini API...")
        response = model.generate_content([analysis_prompt, image_part])
        response_text = response.text.strip()
        print(f"[gemini_vision] Raw response: {response_text[:300]}...")
        
        # Parse JSON response - handle markdown code blocks
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            # Remove first line if it's a code fence
            if lines[0].startswith("```"):
                lines = lines[1:]
            # Remove last line if it's a code fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        result = json.loads(response_text)
        
        parsed_result = {
            "summary": result.get("summary", "Image analyzed successfully"),
            "tags": result.get("tags", ["image"]),
            "type": result.get("type", "image"),
        }
        print(f"[gemini_vision] Analysis complete: {parsed_result['summary'][:100]}...")
        return parsed_result
        
    except json.JSONDecodeError as e:
        print(f"[gemini_vision] JSON parse error: {e}")
        print(f"[gemini_vision] Response was: {response_text[:500]}")
        # Return the raw text as summary if JSON parsing fails
        return {
            "summary": response_text[:300] if response_text else "Image analyzed",
            "tags": ["image"],
            "type": "image",
        }
    except Exception as e:
        print(f"[gemini_vision] API error: {type(e).__name__}: {e}")
        return {
            "summary": f"Image analysis failed: {str(e)}",
            "tags": ["image", "error"],
            "type": "image",
            "error": str(e),
        }


def is_image_mime_type(mime_type: str) -> bool:
    """Check if a MIME type is an image."""
    if not mime_type:
        return False
    return mime_type.lower().startswith("image/")