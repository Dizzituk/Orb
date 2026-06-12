# FILE: app/endpoints/chat_attachments.py
# Purpose: Chat with attachments endpoint - handles file uploads with chat.
# Called-by: app.endpoints, app.endpoints._chat_media_processors, app.endpoints._video_code_tools
# Depends-on: app.auth, app.auth.middleware, app.db, app.embeddings (+18 more)
# Last-renovated: 2026-06-11
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
# ATTACHMENT PROCESSORS (extracted to _chat_media_processors.py)
# ============================================================================
from app.endpoints._chat_media_processors import (
    _process_image_attachment,
    _process_video_attachment,
    _process_document_attachment,
    _build_document_content_section,
    _route_to_vision,
    _process_multiple_videos,
    _process_multiple_images,
)

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
                str(file_path), mime_type, original_name, suffix, simple_llm_call
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
    # v0.14.0: KNOWLEDGE PROMOTION — extract durable facts to ASTRA memory
    # =========================================================================
    for file_id in indexed_file_ids:
        try:
            from app.memory.models import DocumentContent
            from app.memory.document_knowledge_promoter import promote_document_knowledge_sync
            doc = db.query(DocumentContent).filter(DocumentContent.file_id == file_id).first()
            if doc and doc.raw_text:
                promo_result = promote_document_knowledge_sync(
                    db=db,
                    filename=doc.filename,
                    raw_text=doc.raw_text,
                    structured_data=doc.structured_data,
                )
                _cls = promo_result.get("classification", "?")
                _ext = promo_result.get("extractions", 0)
                _wrt = promo_result.get("written", 0)
                _skip = promo_result.get("skipped_reason")
                if _wrt > 0:
                    print(f"[chat_attachments] Knowledge promoted: {doc.filename} ({_cls}) — {_wrt} facts written")
                elif _skip:
                    print(f"[chat_attachments] Knowledge skipped: {doc.filename} ({_cls}) — {_skip}")
        except Exception as e:
            print(f"[chat_attachments] Knowledge promotion failed for file_id={file_id}: {e}")

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
    
    # v10.1: File creation intent overrides vision-only routing.
    # When the user wants an HTML file created AND attaches reference files,
    # we still run vision to describe images, but inject the description into
    # the text LLM path rather than returning the vision response directly.
    _file_creation_override = False
    try:
        from app.llm.routing.chat_routing import _detect_file_creation_intent, _get_sticky_model
        _is_file_intent = user_message and _detect_file_creation_intent(user_message)
        _sticky = _get_sticky_model(project_id)
        # Only override vision path if:
        # 1. Current message explicitly asks for file creation, OR
        # 2. Sticky model is GPT-5.4 (set by a previous file creation intent in this session)
        _sticky_is_file_creation = _sticky and _sticky[1] and 'gpt-5' in _sticky[1]
        if _is_file_intent or _sticky_is_file_creation:
            _file_creation_override = True
            print(f"[chat_attachments] File creation/sticky override — will use text LLM path, not vision-only")
    except Exception:
        pass
    
    if has_media and not frontier_override_active and not _file_creation_override:
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
            classification = None
            tier = "fast"

        # v9.0: VIDEO_CODE_TOOLS — single model with video + tool access
        if classification and classification.job_type == JobType.VIDEO_CODE_TOOLS:
            print(f"[chat_attachments] VIDEO_CODE_TOOLS detected — routing to video+tools pipeline")
            from app.endpoints._video_code_tools import route_video_code_tools
            return route_video_code_tools(
                video_attachments=video_attachments,
                user_message=user_message,
                project_id=project_id,
                project=project,
                attachments_summary=attachments_summary,
                db=db,
                model=selected_model,
            )
        
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
    # v10.1: If file creation override, get image descriptions via vision first
    # =========================================================================
    if _file_creation_override and image_attachments:
        try:
            for img in image_attachments:
                vision_result = ask_about_image(
                    image_source=img["path"],
                    user_question="Describe this image in detail — it's a logo or design reference the user wants incorporated into an HTML page.",
                    tier="fast",
                )
                if vision_result:
                    document_content_parts.append(f"=== IMAGE DESCRIPTION: {img.get('filename', 'image')} ===\n{vision_result}")
                    print(f"[chat_attachments] Vision description for file creation: {len(vision_result)} chars")
        except Exception as ve:
            print(f"[chat_attachments] Vision for file creation failed (non-fatal): {ve}")

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

    # Job type — documents need a capable model, not chat.light
    # v0.17: Force OPENAI_DEFAULT_MODEL for document uploads (avoids gpt-4.1-mini)
    # v10.0: Check session model stickiness FIRST — if the conversation already
    # upgraded to a specific model (e.g. GPT-5.4 for file creation), hold it.
    doc_force_provider = None
    doc_force_model = None

    try:
        from app.llm.routing.chat_routing import _get_sticky_model, _detect_file_creation_intent, _set_sticky_model
        sticky = _get_sticky_model(project_id)
        # Only use sticky model if it's a file-creation model (GPT-5.x)
        # Other sticky models (Gemini deep, Opus) should NOT override attachment routing
        if sticky and sticky[1] and 'gpt-5' in sticky[1]:
            sticky_prov_str, sticky_mod = sticky
            _prov_map = {"openai": Provider.OPENAI, "google": Provider.GOOGLE, "anthropic": Provider.ANTHROPIC}
            doc_force_provider = _prov_map.get(sticky_prov_str, Provider.OPENAI)
            doc_force_model = sticky_mod
            jt = JobType.TEXT_HEAVY
            print(f"[chat_attachments] File creation sticky override: {sticky_prov_str}/{sticky_mod}")
        elif user_message and _detect_file_creation_intent(user_message):
            doc_force_provider = Provider.OPENAI
            doc_force_model = os.getenv("FILE_CREATION_MODEL", "gpt-5.4")
            jt = JobType.TEXT_HEAVY
            _set_sticky_model(project_id, "openai", doc_force_model)
            print(f"[chat_attachments] File creation intent detected -> openai/{doc_force_model}")
    except Exception as _sticky_err:
        print(f"[chat_attachments] Sticky/intent check failed: {_sticky_err}")

    if doc_force_provider is None and document_content_parts:
        jt = JobType.TEXT_HEAVY
        doc_force_provider = Provider.OPENAI
        doc_force_model = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5.2")
        print(f"[chat_attachments] {len(attachment_metadata)} attachment(s) with document content - forcing {doc_force_model}")
    elif doc_force_provider is None and attachment_metadata:
        jt = JobType.UNKNOWN
        print(f"[chat_attachments] {len(attachment_metadata)} attachment(s) - router will classify")
    elif doc_force_provider is None:
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

    # v10.1: File creation uses streaming to avoid timeout on large HTML generation
    if _file_creation_override and doc_force_provider is not None:
        from fastapi.responses import StreamingResponse
        from app.llm.stream_handlers import generate_sse_stream
        from app.llm.routing.prompt_builders import _CONVERSATIONAL_GUIDELINES

        # Inject the file generation instructions into the system prompt
        system_prompt += _CONVERSATIONAL_GUIDELINES

        print(f"[chat_attachments] File creation: routing to streaming path -> {doc_force_provider}/{doc_force_model}")

        # Convert Provider enum to string for the stream handler
        prov_str = doc_force_provider.value if hasattr(doc_force_provider, 'value') else str(doc_force_provider)

        # Save user message to memory before streaming
        user_content = message if message else "[Attachment only]"
        if attachments_summary:
            user_content += f" [Uploaded: {', '.join(a.client_filename for a in attachments_summary)}]"
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=user_content, provider="local",
        ))

        return StreamingResponse(
            generate_sse_stream(
                project_id=project_id,
                message=user_content,
                provider=prov_str,
                model=doc_force_model,
                system_prompt=system_prompt,
                messages=messages,
                db=db,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    task = LLMTask(
        job_type=jt,
        messages=messages,
        attachments=attachment_metadata if attachment_metadata else None,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
        provider=doc_force_provider,
        model=doc_force_model,
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