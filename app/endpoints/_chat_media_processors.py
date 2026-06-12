# FILE: app/endpoints/_chat_media_processors.py
# Purpose: Media processing helpers for chat_attachments endpoint.
# Called-by: app.endpoints.chat_attachments
# Depends-on: app.endpoints.chat_attachments, app.helpers.context, app.llm, app.llm.file_analyzer (+5 more)
# Last-renovated: 2026-06-11
"""
Media processing helpers for chat_attachments endpoint.

Extracted from chat_attachments.py to reduce file size.
Handles image, video, and document attachment processing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from app.llm import (
    analyze_image, is_image_mime_type,
    extract_text_content, detect_document_type,
    generate_document_summary,
)
from app.llm.schemas import AttachmentInfo
from app.llm.file_analyzer import is_video_mime_type
from app.llm.gemini_vision import ask_about_image, check_vision_available, analyze_video
from sqlalchemy.orm import Session

from app.helpers.context import build_document_context
from app.memory import service as memory_service, schemas as memory_schemas

logger = logging.getLogger(__name__)

def _process_image_attachment(
    content: bytes,
    mime_type: str,
    original_name: str,
    suffix: str,
) -> tuple[dict, Optional[str], str]:
    """Process image attachment, return (analysis, raw_text, doc_type)."""
    print(f"[chat_attachments] Detected IMAGE: {original_name}")
    
    try:
        analysis_result = analyze_image(
            image_source=content,
            mime_type=mime_type,
            user_prompt=None,
        )
        analysis = {
            "summary": analysis_result.get("summary", f"Image: {original_name}"),
            "tags": analysis_result.get("tags", ["image", suffix.lstrip(".")]),
            "type": "image",
        }
        raw_text = analysis["summary"]
    except Exception as e:
        print(f"[chat_attachments] Image analysis failed: {e}")
        analysis = {
            "summary": f"Image: {original_name}",
            "tags": ["image", suffix.lstrip(".")],
            "type": "image",
        }
        raw_text = None
    
    return analysis, raw_text, "image"


def _process_video_attachment(
    content: bytes,
    original_name: str,
    suffix: str,
) -> tuple[dict, None, str]:
    """Process video attachment, return (analysis, None, doc_type)."""
    print(f"[chat_attachments] Detected VIDEO: {original_name} ({len(content)} bytes)")
    
    analysis = {
        "summary": f"Video: {original_name} ({len(content) // 1024}KB)",
        "tags": ["video", suffix.lstrip(".")],
        "type": "video",
    }
    return analysis, None, "video"


def _process_document_attachment(
    file_path: str,
    mime_type: str,
    original_name: str,
    suffix: str,
    simple_llm_call=None,
) -> tuple[dict, Optional[str], str, Optional[str]]:
    """Process document attachment, return (analysis, raw_text, doc_type, structured_data)."""
    print(f"[chat_attachments] Extracting text: {original_name}")
    raw_text = extract_text_content(file_path, mime_type)
    structured_data = None
    
    if raw_text:
        doc_type = detect_document_type(raw_text, original_name)
        print(f"[chat_attachments] Detected doc_type: {doc_type}")
        
        if doc_type == "cv":
            print(f"[chat_attachments] Parsing CV...")
            parsed_data = parse_cv_with_llm(raw_text, original_name, simple_llm_call)
            structured_data = json.dumps(parsed_data)
            
            role_count = len(parsed_data.get("roles", []))
            name = parsed_data.get("name", "Unknown")
            summary_text = f"CV for {name} with {role_count} work experiences"
            if parsed_data.get("skills"):
                summary_text += f", skills: {', '.join(parsed_data['skills'][:5])}"
            
            analysis = {
                "summary": summary_text,
                "tags": ["cv", "resume", suffix.lstrip(".")] if suffix else ["cv", "resume"],
                "type": "cv",
            }
        else:
            print(f"[chat_attachments] Generating summary...")
            summary_text = generate_document_summary(raw_text, original_name, doc_type, simple_llm_call)
            analysis = {
                "summary": summary_text,
                "tags": [doc_type, suffix.lstrip(".")] if suffix else [doc_type],
                "type": doc_type,
            }
    else:
        print(f"[chat_attachments] No text extracted: {original_name}")
        analysis = {
            "summary": f"File uploaded: {original_name}",
            "tags": [suffix.lstrip(".") if suffix else "file"],
            "type": "document",
        }
        doc_type = "document"
    
    return analysis, raw_text, doc_type, structured_data


def _build_document_content_section(
    raw_text: str,
    original_name: str,
    doc_type: str,
    summary: str,
) -> str:
    """Build document content section for LLM context."""
    file_size_kb = len(raw_text) // 1024
    
    if len(raw_text) <= 50 * 1024:
        # Small file - include full text
        return f"""
=== FILE: {original_name} ===
Type: {doc_type}
Size: {file_size_kb}KB
Summary: {summary}

--- FULL CONTENT ---
{raw_text}
"""
    else:
        # Large file - truncate intelligently
        first_chunk = raw_text[:40 * 1024]
        last_chunk = raw_text[-10 * 1024:]
        return f"""
=== FILE: {original_name} ===
Type: {doc_type}
Size: {file_size_kb}KB (TRUNCATED - showing first 40KB + last 10KB)
Summary: {summary}

--- BEGINNING OF CONTENT ---
{first_chunk}

... [CONTENT TRUNCATED] ...

--- END OF CONTENT ---
{last_chunk}
"""


# ============================================================================
# VISION ROUTING
# ============================================================================

def _route_to_vision(
    image_attachments: List[dict],
    video_attachments: List[dict],
    vision_prompt: str,
    vision_context: str,
    tier: str,
    override_model_name: Optional[str],
    attachments_summary: list,
    project_id: int,
    user_message: str,
    db: Session,
):
    """Route media to Gemini Vision and return response."""
    
    image_count = len(image_attachments)
    video_count = len(video_attachments)
    
    print(f"[chat_attachments] Routing to Gemini Vision: {image_count} image(s), {video_count} video(s)")
    
    # Multiple videos
    if len(video_attachments) > 1:
        vision_result = _process_multiple_videos(
            video_attachments, vision_prompt, vision_context, tier, override_model_name
        )
        media_type = "videos"
    
    # Single video
    elif video_attachments:
        primary_media = video_attachments[0]
        media_type = "video"
        
        try:
            vision_result = analyze_video(
                video_path=primary_media["path"],
                user_question=vision_prompt,
                context=vision_context,
                tier=tier,
            )
        except TypeError:
            print("[chat_attachments] analyze_video doesn't support tier parameter, using default")
            vision_result = analyze_video(
                video_path=primary_media["path"],
                user_question=vision_prompt,
                context=vision_context,
            )
    
    # Multiple images
    elif len(image_attachments) > 1:
        vision_result = _process_multiple_images(
            image_attachments, vision_prompt, vision_context, tier, override_model_name
        )
        media_type = "images"
    
    # Single image
    else:
        primary_media = image_attachments[0]
        media_type = "image"
        
        vision_result = ask_about_image(
            image_source=primary_media["bytes"],
            user_question=vision_prompt,
            mime_type=primary_media["mime_type"],
            context=vision_context,
            tier=tier,
        )
    
    provider_str = vision_result.get("provider", "google")
    model_str = vision_result.get("model", "gemini-2.0-flash")
    reply_text = vision_result.get("answer", f"I couldn't analyze the {media_type}.")
    
    if vision_result.get("error"):
        print(f"[chat_attachments] Vision error: {vision_result['error']}")
    
    print(f"[chat_attachments] Vision response from: {provider_str} / {model_str}")
    
    # Deferred import to avoid circular dependency
    from app.endpoints.chat_attachments import ChatResponse
    
    # Log messages
    user_content = user_message if user_message else "[Attachment only]"
    if attachments_summary:
        user_content += f" [Uploaded: {', '.join(a.client_filename for a in attachments_summary)}]"
    
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id,
        role="user",
        content=user_content,
        provider="local",
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id,
        role="assistant",
        content=reply_text,
        provider=provider_str,
        model=model_str,
    ))
    
    return ChatResponse(
        project_id=project_id,
        provider=provider_str,
        model=model_str,
        reply=reply_text,
        was_reviewed=False,
        critic_review=None,
        attachments_summary=attachments_summary,
    )


def _process_multiple_videos(
    video_attachments: List[dict],
    vision_prompt: str,
    vision_context: str,
    tier: str,
    override_model_name: Optional[str],
) -> dict:
    """Process multiple videos with Gemini."""
    try:
        import google.generativeai as genai
        from app.llm.gemini_vision import _get_google_api_key, _get_model_name
        
        api_key = _get_google_api_key()
        if not api_key:
            return {
                "answer": "GOOGLE_API_KEY not set",
                "provider": "google",
                "model": override_model_name or (_get_model_name(tier) if tier else "gemini-2.5-pro"),
                "error": "No API key"
            }
        
        genai.configure(api_key=api_key)
        model_name = override_model_name or _get_model_name(tier)
        model = genai.GenerativeModel(model_name)
        
        # Upload all videos
        video_files = []
        for i, video_att in enumerate(video_attachments):
            print(f"[chat_attachments] Uploading video {i+1}/{len(video_attachments)}: {video_att['filename']}...")
            video_file = genai.upload_file(path=str(video_att["path"]))
            
            import time
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                print(f"[chat_attachments] Video {i+1} processing failed")
                continue
            
            video_files.append(video_file)
            print(f"[chat_attachments] Video {i+1} ready")
        
        if not video_files:
            return {
                "answer": "All video processing failed",
                "provider": "google",
                "model": model_name,
                "error": "Processing failed"
            }
        
        # Build prompt
        prompt_parts = []
        if vision_context:
            prompt_parts.append(f"Context: {vision_context}\n\n")
        prompt_parts.append(f"User's question about these {len(video_files)} videos: {vision_prompt}")
        full_prompt = "".join(prompt_parts)
        
        # Generate response
        content_parts = video_files + [full_prompt]
        response = model.generate_content(content_parts)
        
        result = {
            "answer": response.text,
            "provider": "google",
            "model": model_name,
        }
        
        # Clean up
        for video_file in video_files:
            try:
                genai.delete_file(video_file.name)
            except Exception as cleanup_error:
                print(f"[chat_attachments] Warning: Could not delete video: {cleanup_error}")
        
        print(f"[chat_attachments] Cleaned up {len(video_files)} video file(s)")
        return result
        
    except Exception as e:
        print(f"[chat_attachments] Multi-video analysis failed: {e}")
        return {
            "answer": f"Multi-video analysis failed: {str(e)}",
            "provider": "google",
            "model": override_model_name or "gemini-2.5-pro",
            "error": str(e)
        }


def _process_multiple_images(
    image_attachments: List[dict],
    vision_prompt: str,
    vision_context: str,
    tier: str,
    override_model_name: Optional[str],
) -> dict:
    """Process multiple images with Gemini."""
    try:
        import google.generativeai as genai
        from app.llm.gemini_vision import _get_google_api_key, _get_model_name
        
        api_key = _get_google_api_key()
        if not api_key:
            return {
                "answer": "GOOGLE_API_KEY not set",
                "provider": "google",
                "model": override_model_name or "gemini-2.5-pro",
                "error": "No API key"
            }
        
        genai.configure(api_key=api_key)
        model_name = override_model_name or _get_model_name(tier)
        model = genai.GenerativeModel(model_name)
        
        # Build prompt
        prompt_parts = []
        if vision_context:
            prompt_parts.append(f"Context: {vision_context}\n\n")
        prompt_parts.append(f"User's question: {vision_prompt}")
        full_prompt = "".join(prompt_parts)
        
        # Load images
        import PIL.Image
        import io
        images = []
        for img_att in image_attachments:
            image = PIL.Image.open(io.BytesIO(img_att["bytes"]))
            images.append(image)
        
        # Generate response
        content_parts = [full_prompt] + images
        response = model.generate_content(content_parts)
        
        return {
            "answer": response.text,
            "provider": "google",
            "model": model_name,
        }
        
    except Exception as e:
        print(f"[chat_attachments] Multi-image analysis failed: {e}")
        from app.llm.gemini_vision import _get_model_name
        return {
            "answer": f"Multi-image analysis failed: {str(e)}",
            "provider": "google",
            "model": _get_model_name(tier) if tier else "gemini-2.5-pro",
            "error": str(e)
        }

