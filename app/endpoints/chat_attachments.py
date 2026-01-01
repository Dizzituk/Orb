# FILE: app/endpoints/chat_attachments.py
"""
Chat with attachments endpoint - handles file uploads with chat.

Refactored from main.py with BUG FIX:
- Current document content now appears FIRST in context (not after old docs)
- This fixes the "file lag" bug where model responded about previous file

v0.16.1: BUG FIX - Current attachment context ordering
v0.15.1: OVERRIDE command support
v0.14.2: Document content logging
v0.13.10: Job classifier for media routing
v0.13.5: Document content injection
"""

import os
import json
import time
from pathlib import Path
from uuid import uuid4
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File as FastAPIFile, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas
from app.llm import (
    call_llm, LLMTask, LLMResult, JobType,
    analyze_image, is_image_mime_type,
    extract_text_content, detect_document_type,
    parse_cv_with_llm, generate_document_summary,
)
from app.llm.schemas import Provider, AttachmentInfo
from app.llm.file_analyzer import is_video_mime_type
from app.llm.gemini_vision import ask_about_image, check_vision_available, analyze_video
from app.llm.job_classifier import classify_job, detect_frontier_override

from app.helpers.context import build_context_block, build_document_context
from app.helpers.llm_utils import (
    simple_llm_call,
    sync_await,
    extract_provider_value,
    extract_model_value,
    map_model_to_vision_tier,
)

router = APIRouter(tags=["Chat"])


# ============================================================================
# MODELS
# ============================================================================

class AttachmentSummary(BaseModel):
    client_filename: str
    stored_id: str
    type: str
    summary: str
    tags: List[str]


class ChatResponse(BaseModel):
    project_id: int
    provider: str
    model: Optional[str] = None
    reply: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None
    attachments_summary: Optional[List[AttachmentSummary]] = None


# ============================================================================
# ATTACHMENT PROCESSORS
# ============================================================================

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
    attachments_summary: List[AttachmentSummary],
    project_id: int,
    user_message: str,
    db: Session,
) -> ChatResponse:
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


# ============================================================================
# MAIN ENDPOINT
# ============================================================================

@router.post("/chat_with_attachments", response_model=ChatResponse)
def chat_with_attachments(
    project_id: int = Form(...),
    message: Optional[str] = Form(None),
    job_type: str = Form("casual_chat"),
    files: List[UploadFile] = FastAPIFile(...),
    auth: AuthResult = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Chat with file attachments.
    
    v0.16.1: BUG FIX - Current attachment content now FIRST in context
    v0.15.1: OVERRIDE command support
    v0.13.10: Multi-image/video routing uses job_classifier
    v0.13.5: Document content injected into LLM context
    """
    from app.embeddings import service as embeddings_service
    
    project = memory_service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    attachments_summary: List[AttachmentSummary] = []
    attachment_context_parts: List[str] = []
    document_content_parts: List[str] = []  # CURRENT upload content
    attachment_metadata: List[AttachmentInfo] = []
    indexed_file_ids: List[int] = []
    
    # Track media for Gemini Vision routing
    image_attachments: List[dict] = []
    video_attachments: List[dict] = []
    
    project_dir = Path(f"data/files/{project_id}")
    project_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # PROCESS EACH FILE
    # =========================================================================
    
    for upload_file in files:
        original_name = upload_file.filename or "unknown"
        suffix = Path(original_name).suffix.lower()
        unique_name = f"{uuid4().hex}{suffix}"
        file_path = project_dir / unique_name
        relative_path = f"{project_id}/{unique_name}"
        
        content = upload_file.file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"[chat_attachments] Saved: {original_name} -> {file_path}")
        
        mime_type = upload_file.content_type or ""
        
        raw_text: Optional[str] = None
        structured_data: Optional[str] = None
        
        # Process based on type
        if is_image_mime_type(mime_type):
            analysis, raw_text, doc_type = _process_image_attachment(
                content, mime_type, original_name, suffix
            )
            image_attachments.append({
                "bytes": content,
                "mime_type": mime_type,
                "filename": original_name,
                "path": str(file_path),
                "size": len(content),
            })
        
        elif is_video_mime_type(mime_type):
            analysis, raw_text, doc_type = _process_video_attachment(
                content, original_name, suffix
            )
            video_attachments.append({
                "bytes": content,
                "mime_type": mime_type,
                "filename": original_name,
                "path": str(file_path),
                "size": len(content),
            })
        
        else:
            analysis, raw_text, doc_type, structured_data = _process_document_attachment(
                str(file_path), mime_type, original_name, suffix
            )
            
            # Build document content section for current upload
            if raw_text:
                doc_section = _build_document_content_section(
                    raw_text, original_name, doc_type, analysis.get("summary", "N/A")
                )
                document_content_parts.append(doc_section)
                print(f"[chat_attachments] Added document content: {original_name} ({len(raw_text) // 1024}KB)")
        
        # Create file record
        file_record = memory_service.create_file_for_project(
            db, project_id,
            memory_schemas.FileCreate(
                project_id=project_id,
                path=relative_path,
                original_name=original_name,
                file_type=mime_type or suffix.lstrip("."),
                description=analysis.get("summary", ""),
            )
        )
        
        # Store document content
        if raw_text:
            memory_service.create_document_content(
                db,
                memory_schemas.DocumentContentCreate(
                    project_id=project_id,
                    file_id=file_record.id,
                    filename=original_name,
                    doc_type=doc_type,
                    raw_text=raw_text,
                    summary=analysis.get("summary", ""),
                    structured_data=structured_data,
                )
            )
            indexed_file_ids.append(file_record.id)
        
        stored_id = f"file_{file_record.id}"
        summary = AttachmentSummary(
            client_filename=original_name,
            stored_id=stored_id,
            type=analysis.get("type", "document"),
            summary=analysis.get("summary", ""),
            tags=analysis.get("tags", []),
        )
        attachments_summary.append(summary)
        attachment_context_parts.append(f"[Uploaded: {original_name}] {analysis.get('summary', '')}")
        
        attachment_metadata.append(AttachmentInfo(
            filename=original_name,
            mime_type=mime_type,
            size_bytes=len(content),
            file_id=file_record.id,
        ))

    # Index for semantic search
    for file_id in indexed_file_ids:
        try:
            from app.memory.models import DocumentContent
            doc = db.query(DocumentContent).filter(DocumentContent.file_id == file_id).first()
            if doc:
                embeddings_service.index_document(db, doc, force=True)
                print(f"[chat_attachments] Indexed file_id={file_id}")
        except Exception as e:
            print(f"[chat_attachments] Index failed for file_id={file_id}: {e}")

    # =========================================================================
    # CHECK FOR OVERRIDE COMMAND
    # =========================================================================
    
    user_message = message.strip() if message else ""
    
    override_result = detect_frontier_override(user_message)
    frontier_override_active = override_result is not None
    
    override_image_data = None
    
    if frontier_override_active:
        override_provider, override_model_id, _ = override_result
        print(f"[chat_attachments] OVERRIDE detected → {override_provider} / {override_model_id}")
        print(f"[chat_attachments] HARD OVERRIDE: Bypassing vision routing, sending to router.py")
        
        if image_attachments:
            import base64
            override_image_data = []
            for img in image_attachments:
                override_image_data.append({
                    "bytes_b64": base64.b64encode(img["bytes"]).decode("utf-8"),
                    "mime_type": img["mime_type"],
                    "filename": img["filename"],
                })
            print(f"[chat_attachments] Captured {len(override_image_data)} image(s) for OVERRIDE path")

    # =========================================================================
    # VISION ROUTING (skip if OVERRIDE)
    # =========================================================================
    
    has_media = image_attachments or video_attachments
    
    if has_media and not frontier_override_active:
        vision_prompt = user_message if user_message else "Describe this image/video in detail."
        
        # Use job_classifier for tier
        try:
            classification = classify_job(
                message=vision_prompt,
                attachments=attachment_metadata,
            )
            selected_model = classification.model
            tier = map_model_to_vision_tier(selected_model)
            
            if os.getenv("ORB_ROUTER_DEBUG") == "1":
                print(f"[chat_attachments] Classifier selected: {classification.provider.value} / {selected_model}")
                print(f"[chat_attachments] Mapped to vision tier: {tier}")
        except Exception as e:
            print(f"[chat_attachments] Classifier failed, using default tier: {e}")
            tier = "fast"
        
        override_model_name = override_model_id if frontier_override_active and tier == "override" else None
        
        vision_status = check_vision_available()
        if not vision_status.get("available"):
            error_msg = vision_status.get("error", "Vision not available")
            print(f"[chat_attachments] Vision unavailable: {error_msg}")
            attachment_context_parts.insert(0, f"[Warning: {error_msg}]")
        else:
            # Build vision context
            vision_context = f"Project: {project.name}."
            if project.description:
                vision_context += f" {project.description}"
            
            # Include CURRENT document content in vision context
            if document_content_parts:
                vision_context += "\n\n=== DOCUMENT CONTENT ===\n"
                vision_context += "\n".join(document_content_parts)
                vision_context += "\n=== END DOCUMENTS ==="
                
                doc_count = len(document_content_parts)
                context_chars = len(vision_context)
                print(f"[chat_attachments] ✓ DOCUMENT CONTEXT INJECTED: {doc_count} document(s), {context_chars} total chars")
                
                if os.getenv("ORB_ROUTER_DEBUG") == "1":
                    print(f"[chat_attachments] Document content preview (first 500 chars):")
                    print(f"  {vision_context[:500]}...")
            
            return _route_to_vision(
                image_attachments=image_attachments,
                video_attachments=video_attachments,
                vision_prompt=vision_prompt,
                vision_context=vision_context,
                tier=tier,
                override_model_name=override_model_name,
                attachments_summary=attachments_summary,
                project_id=project_id,
                user_message=user_message,
                db=db,
            )

    # =========================================================================
    # TEXT-ONLY PATH
    # =========================================================================

    full_message = user_message if user_message else ""
    
    if attachments_summary and not full_message:
        full_message = "[User uploaded files without a message]"

    if not full_message and not attachments_summary:
        return ChatResponse(
            project_id=project_id,
            provider="none",
            model=None,
            reply="No message or attachments received.",
            attachments_summary=[],
        )

    # =========================================================================
    # BUG FIX: Build context with CURRENT document FIRST
    # =========================================================================
    
    full_context = ""
    
    # 1. CURRENT document content FIRST (this is the fix!)
    if document_content_parts:
        document_content_block = "\n".join(document_content_parts)
        full_context += f"=== CURRENT UPLOAD (JUST UPLOADED) ===\n{document_content_block}\n\n"
        
        total_docs = len(document_content_parts)
        print(f"[chat_attachments] ✓ CURRENT DOCUMENT CONTEXT FIRST: {total_docs} document(s)")
    
    # 2. Then attachment metadata
    if attachment_context_parts:
        attachment_info = "\n".join(attachment_context_parts)
        full_context += f"=== UPLOADED FILES (METADATA) ===\n{attachment_info}\n\n"
    
    # 3. Then notes/tasks context
    context_block = build_context_block(db, project_id)
    if context_block:
        full_context += context_block + "\n\n"
    
    # 4. Then OLD document summaries (from DB) - these come LAST now
    doc_context = build_document_context(db, project_id, full_message)
    if doc_context:
        full_context += "=== PREVIOUS UPLOADS (HISTORY) ===\n" + doc_context

    # Build messages
    history = memory_service.list_messages(db, project_id, limit=50)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    
    # Handle multimodal OVERRIDE
    if frontier_override_active and override_image_data:
        content_parts = []
        
        for img_data in override_image_data:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img_data['mime_type']};base64,{img_data['bytes_b64']}"
                }
            })
        
        text_content = user_message if user_message else "Describe this image in detail."
        content_parts.append({"type": "text", "text": text_content})
        
        messages = history_dicts + [{"role": "user", "content": content_parts}]
        print(f"[chat_attachments] Built multimodal message: {len(override_image_data)} image(s) + text")
    else:
        messages = history_dicts + [{"role": "user", "content": full_message}]

    # Job type
    if attachment_metadata:
        jt = JobType.UNKNOWN
        print(f"[chat_attachments] {len(attachment_metadata)} attachment(s) - router will classify")
    else:
        jt = JobType.UNKNOWN
        print(f"[chat_attachments] No attachments - router will use default")

    system_prompt = f"Project: {project.name}. {project.description or ''}"
    
    if attachments_summary:
        filenames = ', '.join(a.client_filename for a in attachments_summary)
        system_prompt += f"""

IMPORTANT - CURRENT UPLOAD:
The user JUST uploaded these files in THIS message: {filenames}

When the user asks about "this", "the file", "the document", or "what is this", they are referring ONLY to the file(s) listed above that were JUST uploaded.

Do NOT discuss or reference any files from previous messages in the chat history. Focus ONLY on the newly uploaded file(s): {filenames}

The content of the current upload appears first in the context under "=== CURRENT UPLOAD (JUST UPLOADED) ==="."""

    task = LLMTask(
        job_type=jt,
        messages=messages,
        attachments=attachment_metadata if attachment_metadata else None,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
    )

    try:
        result: LLMResult = sync_await(call_llm(task))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = extract_provider_value(result)
    model_str = extract_model_value(result)
    
    print(f"[chat_attachments] Response from: {provider_str} / {model_str}")

    user_content = message if message else "[Attachment only]"
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
        content=result.content,
        provider=provider_str,
        model=model_str,
    ))

    return ChatResponse(
        project_id=project_id,
        provider=provider_str,
        model=model_str,
        reply=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
        attachments_summary=attachments_summary,
    )