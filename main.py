# FILE: main.py
"""
Orb Backend - FastAPI Application
Version: 0.12.1

Security Features:
- Password authentication with bcrypt
- Session token management
- Database encryption at rest (master key via ORB_MASTER_KEY)

Features:
- File upload with text extraction and storage
- CV parsing and structured data extraction
- Document content retrieval for Q&A
- Image analysis via Gemini Vision
- Streaming LLM responses
- Web search grounding via Gemini
- Semantic search (RAG) with embeddings
- Phase 4: Job system with unified provider routing (optional)

v0.12.1 Changes:
- Fixed /chat to return model in response
- Fixed /chat to save provider/model to message database
- Added model field to ChatResponse schema
"""
import os
import json
import re
from pathlib import Path
from uuid import uuid4
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File as FastAPIFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

# Load .env FIRST before any other imports that might need env vars
load_dotenv()

from app.db import init_db, get_db
from app.memory.router import router as memory_router
from app.memory import service as memory_service, schemas as memory_schemas
from app.auth import require_auth, is_auth_configured
from app.auth.router import router as auth_router
from app.auth.middleware import AuthResult
from app.llm import (
    call_llm, LLMTask, LLMResult, JobType,
    analyze_image, is_image_mime_type,
    extract_text_content, detect_document_type,
    parse_cv_with_llm, generate_document_summary,
)
from app.llm.schemas import Provider
from app.llm.clients import check_provider_availability, call_openai
from app.llm.stream_router import router as stream_router
from app.llm.web_search_router import router as web_search_router
from app.embeddings.router import router as embeddings_router, search_router as embeddings_search_router
from app.embeddings import service as embeddings_service

app = FastAPI(
    title="Orb Assistant",
    version="0.12.1",
    description="Personal AI assistant with multi-LLM orchestration and semantic search",
)

# ====== CORS ======

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "file://",  # For Electron file:// protocol
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== STARTUP ======

@app.on_event("startup")
def on_startup():
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/files", exist_ok=True)
    
    # Security Level 4: Initialize master key encryption FIRST
    print("[startup] Initializing encryption...")
    from app.crypto import require_master_key_or_exit, is_master_key_initialized
    require_master_key_or_exit()  # Exits if ORB_MASTER_KEY not set or invalid
    
    if is_master_key_initialized():
        print("[startup] Database encryption: [OK] master key active")
    
    init_db()
    
    # Check auth status
    print("[startup] Checking authentication...")
    if is_auth_configured():
        print("[startup] Password authentication: [OK] configured")
    else:
        print("[startup] Password authentication: [X] NOT CONFIGURED")
        print("[startup] Call POST /auth/setup to set a password")
    
    # Verify critical env vars
    print("[startup] Checking environment variables...")
    if os.getenv("GOOGLE_API_KEY"):
        print("[startup] GOOGLE_API_KEY: [OK] set (enables image analysis + web search)")
    else:
        print("[startup] GOOGLE_API_KEY: [X] NOT SET - image analysis and web search will fail")
    
    if os.getenv("OPENAI_API_KEY"):
        print("[startup] OPENAI_API_KEY: [OK] set (enables chat + embeddings)")
    else:
        print("[startup] OPENAI_API_KEY: [X] NOT SET - chat and semantic search will fail")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        print("[startup] ANTHROPIC_API_KEY: [OK] set")
    else:
        print("[startup] ANTHROPIC_API_KEY: [X] NOT SET")
    
    # Check Phase 4 feature flag
    print("[startup] Checking Phase 4 status...")
    phase4_enabled = os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true"
    if phase4_enabled:
        print("[startup] Phase 4 Job System: [OK] ENABLED")
        print("[startup]   - POST /jobs/create")
        print("[startup]   - GET /jobs/{job_id}")
        print("[startup]   - GET /jobs/list")
        print("[startup]   - GET /artefacts/{artefact_id}")
        print("[startup]   - GET /artefacts/list")
    else:
        print("[startup] Phase 4 Job System: [X] DISABLED")
        print("[startup]   Set ORB_ENABLE_PHASE4=true to enable")


# ====== ROUTERS ======

# Auth router - public endpoints for setup/validation
app.include_router(auth_router)

# Memory router - protected
app.include_router(memory_router)

# Streaming router - protected
app.include_router(stream_router)

# Web search router - protected
app.include_router(web_search_router)

# Embeddings router - protected
app.include_router(embeddings_router)
app.include_router(embeddings_search_router)

# Phase 4 routers - protected (optional based on feature flag)
if os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true":
    try:
        from app.jobs.router import router as jobs_router
        from app.artefacts.router import router as artefacts_router
        
        app.include_router(
            jobs_router,
            prefix="/jobs",
            tags=["Phase 4 Jobs"],
            dependencies=[Depends(require_auth)]
        )
        
        app.include_router(
            artefacts_router,
            prefix="/artefacts",
            tags=["Phase 4 Artefacts"],
            dependencies=[Depends(require_auth)]
        )
        
        print("[startup] Phase 4 routers registered successfully")
    except ImportError as e:
        print(f"[startup] WARNING: Phase 4 enabled but failed to import routers: {e}")


# ====== MODELS ======

class ChatRequest(BaseModel):
    project_id: int
    message: str
    job_type: str = "casual_chat"
    force_provider: Optional[str] = None
    use_semantic_search: bool = True  # NEW: Enable/disable semantic search


class AttachmentSummary(BaseModel):
    client_filename: str
    stored_id: str
    type: str
    summary: str
    tags: List[str]


class ChatResponse(BaseModel):
    project_id: int
    provider: str
    model: Optional[str] = None  # ADDED: Model that generated the response
    reply: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None
    attachments_summary: Optional[List[AttachmentSummary]] = None


# ====== STATIC FILES ======

app.mount("/static", StaticFiles(directory="static"), name="static")


# ====== PUBLIC ENDPOINTS ======

@app.get("/")
def read_index():
    """Serve static UI (public)."""
    return FileResponse("static/index.html")


@app.get("/ping")
def ping():
    """Health check (public)."""
    return {"status": "ok", "auth_configured": is_auth_configured()}


# ====== PROTECTED ENDPOINTS ======

@app.get("/providers")
def list_providers(auth: AuthResult = Depends(require_auth)):
    """List available LLM providers (protected)."""
    return check_provider_availability()


@app.get("/job-types")
def list_job_types(auth: AuthResult = Depends(require_auth)):
    """List available job types (protected)."""
    from app.llm.schemas import RoutingConfig
    return {
        "gpt_only": [jt.value for jt in RoutingConfig.GPT_ONLY_JOBS],
        "medium_dev": [jt.value for jt in RoutingConfig.MEDIUM_DEV_JOBS],
        "claude_primary": [jt.value for jt in RoutingConfig.CLAUDE_PRIMARY_JOBS],
        "high_stakes_reviewed": [jt.value for jt in RoutingConfig.HIGH_STAKES_JOBS],
        "gemini": [jt.value for jt in RoutingConfig.GEMINI_JOBS],
    }


# ====== HELPERS ======

def simple_llm_call(prompt: str) -> str:
    """Quick LLM call for analysis tasks."""
    try:
        content, _ = call_openai(
            system_prompt="You are a helpful assistant. Respond with only what is asked, no extra text.",
            messages=[{"role": "user", "content": prompt}],
        )
        return content
    except Exception as e:
        print(f"[simple_llm_call] Error: {e}")
        return ""


def build_context_block(db: Session, project_id: int) -> str:
    """Build context from notes + tasks."""
    sections = []

    notes = memory_service.list_notes(db, project_id)[:10]
    if notes:
        notes_text = "\n".join(
            f"- [{n.id}] {n.title}: {n.content[:200]}..."
            for n in notes
        )
        sections.append(f"PROJECT NOTES:\n{notes_text}")

    tasks = memory_service.list_tasks(db, project_id, status_filter="pending")[:10]
    if tasks:
        tasks_text = "\n".join(f"- {t.title}" for t in tasks)
        sections.append(f"PENDING TASKS:\n{tasks_text}")

    return "\n\n".join(sections) if sections else ""


def build_document_context(db: Session, project_id: int, user_message: str) -> str:
    """Build context from uploaded documents with semantic search."""
    try:
        # Search documents that might be relevant
        from app.memory.models import DocumentContent
        
        # Get recent documents
        recent_docs = (db.query(DocumentContent)
                      .filter(DocumentContent.project_id == project_id)
                      .order_by(DocumentContent.created_at.desc())
                      .limit(5)
                      .all())
        
        if not recent_docs:
            return ""
        
        # Build context from recent documents
        context_parts = []
        for doc in recent_docs:
            summary = doc.summary[:300] if doc.summary else "(no summary)"
            context_parts.append(f"[{doc.filename}]: {summary}")
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"[build_document_context] Error: {e}")
        return ""


def _extract_provider_value(result: LLMResult) -> str:
    """
    Extract provider string from LLMResult.
    Handles both enum (result.provider.value) and string (result.provider) cases.
    """
    if result.provider is None:
        return "unknown"
    if hasattr(result.provider, 'value'):
        return result.provider.value
    return str(result.provider)


def _extract_model_value(result: LLMResult) -> Optional[str]:
    """
    Extract model string from LLMResult.
    """
    return result.model if hasattr(result, 'model') else None


# ====== CHAT ENDPOINTS (PROTECTED) ======

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    auth: AuthResult = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Send chat message with context (protected).
    
    Features:
    - Project context injection (notes, tasks, docs)
    - Semantic search for document retrieval (if enabled)
    - Message history management
    - Multi-provider routing
    - Returns provider and model info for UI display
    """
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {req.project_id}")

    # Build context
    context_block = build_context_block(db, req.project_id)
    
    # Semantic search context (if enabled)
    semantic_context = ""
    if req.use_semantic_search:
        try:
            semantic_results = embeddings_service.search(
                db=db,
                project_id=req.project_id,
                query=req.message,
                top_k=5,
            )
            if semantic_results:
                semantic_context = "=== RELEVANT DOCUMENTS (semantic search) ===\n"
                for result in semantic_results:
                    semantic_context += f"\n[Score: {result.similarity_score:.3f}] {result.content_preview}\n"
        except Exception as e:
            print(f"[chat] Semantic search failed: {e}")
    
    # Document context (recent docs)
    doc_context = build_document_context(db, req.project_id, req.message)
    
    # Combine all context
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== RECENT UPLOADS ===\n" + doc_context

    # Get conversation history
    history = memory_service.list_messages(db, req.project_id, limit=50)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    messages = history_dicts + [{"role": "user", "content": req.message}]

    # Parse job type
    try:
        jt = JobType(req.job_type)
    except ValueError:
        jt = JobType.CASUAL_CHAT

    # Build task
    system_prompt = f"Project: {project.name}. {project.description or ''}"
    
    task = LLMTask(
        job_type=jt,
        messages=messages,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
        force_provider=Provider(req.force_provider) if req.force_provider else None,
    )

    # Call LLM
    try:
        result: LLMResult = call_llm(task)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Extract provider and model for storage and response
    provider_str = _extract_provider_value(result)
    model_str = _extract_model_value(result)

    # Store messages with provider/model info
    # User message: mark provider as "local" (typed by user)
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="user",
        content=req.message,
        provider="local",
    ))
    # Assistant message: record which provider/model generated the response
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="assistant",
        content=result.content,
        provider=provider_str,
        model=model_str,
    ))

    return ChatResponse(
        project_id=req.project_id,
        provider=provider_str,
        model=model_str,
        reply=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
    )


@app.post("/chat_with_attachments", response_model=ChatResponse)
async def chat_with_attachments(
    project_id: int = Form(...),
    message: Optional[str] = Form(None),
    job_type: str = Form("casual_chat"),
    files: List[UploadFile] = FastAPIFile(...),
    auth: AuthResult = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """
    Chat with file uploads (protected).
    
    Supports:
    - PDF, DOCX, TXT file extraction
    - Image analysis via Gemini Vision
    - CV parsing with structured extraction
    - Semantic search indexing
    """
    project = memory_service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    
    # Ensure project files directory exists
    project_files_dir = Path("data/files") / str(project_id)
    project_files_dir.mkdir(parents=True, exist_ok=True)
    
    attachments_summary: List[AttachmentSummary] = []
    attachment_context_parts: List[str] = []
    indexed_file_ids: List[int] = []
    
    for upload_file in files:
        original_name = upload_file.filename or "unnamed_file"
        suffix = Path(original_name).suffix.lower()
        mime_type = upload_file.content_type
        
        # Generate unique filename
        unique_id = str(uuid4())[:8]
        safe_name = re.sub(r'[^\w\-_.]', '_', original_name)
        stored_name = f"{unique_id}_{safe_name}"
        file_path = project_files_dir / stored_name
        relative_path = f"{project_id}/{stored_name}"
        
        # Save file
        contents = await upload_file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        print(f"[chat_with_attachments] Saved: {file_path}")
        
        # Analyze file
        raw_text = ""
        structured_data = None
        analysis = {}
        doc_type = "document"
        
        # Check if it's an image
        if is_image_mime_type(mime_type) or suffix in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"]:
            print(f"[chat_with_attachments] Analyzing image: {original_name}")
            try:
                # Use Gemini Vision
                image_analysis = analyze_image(
                    str(file_path),
                    "Describe this image in detail. What do you see?"
                )
                raw_text = image_analysis.get("description", "")
                analysis = {
                    "summary": raw_text[:200] + "..." if len(raw_text) > 200 else raw_text,
                    "tags": image_analysis.get("tags", ["image"]),
                    "type": "image",
                }
                doc_type = "image"
            except Exception as e:
                print(f"[chat_with_attachments] Image analysis failed: {e}")
                analysis = {
                    "summary": f"Image uploaded: {original_name}",
                    "tags": ["image", suffix.lstrip(".")] if suffix else ["image"],
                    "type": "image",
                }
                doc_type = "image"
                raw_text = f"[Image: {original_name}] (not analyzed)"
        else:
            # Extract text from document
            print(f"[chat_with_attachments] Extracting text from: {original_name}")
            raw_text = extract_text_content(str(file_path))
            
            if raw_text:
                print(f"[chat_with_attachments] Extracted {len(raw_text)} characters")
                
                # Detect document type
                doc_type = detect_document_type(original_name, raw_text)
                print(f"[chat_with_attachments] Document type: {doc_type}")
                
                # Parse CV if detected
                if doc_type == "cv":
                    print(f"[chat_with_attachments] Parsing CV with LLM...")
                    parsed_data = parse_cv_with_llm(raw_text, original_name, simple_llm_call)
                    structured_data = json.dumps(parsed_data)
                    
                    # Generate summary
                    role_count = len(parsed_data.get("roles", []))
                    name = parsed_data.get("name", "Unknown")
                    summary_text = f"CV for {name} with {role_count} work experiences"
                    if parsed_data.get("skills"):
                        summary_text += f", skills include: {', '.join(parsed_data['skills'][:5])}"
                    
                    analysis = {
                        "summary": summary_text,
                        "tags": ["cv", "resume", suffix.lstrip(".")] if suffix else ["cv", "resume"],
                        "type": "cv",
                    }
                else:
                    # Generate general summary
                    print(f"[chat_with_attachments] Generating summary...")
                    summary_text = generate_document_summary(raw_text, original_name, doc_type, simple_llm_call)
                    
                    analysis = {
                        "summary": summary_text,
                        "tags": [doc_type, suffix.lstrip(".")] if suffix else [doc_type],
                        "type": doc_type,
                    }
            else:
                print(f"[chat_with_attachments] Could not extract text from: {original_name}")
                analysis = {
                    "summary": f"File uploaded: {original_name}",
                    "tags": [suffix.lstrip(".") if suffix else "file"],
                    "type": "document",
                }
                doc_type = "document"
        
        # Create file record
        file_record = memory_service.create_file_for_project(
            db, project_id,
            memory_schemas.FileCreate(
                project_id=project_id,
                path=relative_path,
                original_name=original_name,
                file_type=mime_type or suffix.lstrip("."),
                description=analysis["summary"],
            )
        )
        
        # Store document content for retrieval (for both text docs and image descriptions)
        if raw_text:
            print(f"[chat_with_attachments] Storing document content for file_id={file_record.id}")
            memory_service.create_document_content(
                db,
                memory_schemas.DocumentContentCreate(
                    project_id=project_id,
                    file_id=file_record.id,
                    filename=original_name,
                    doc_type=doc_type,
                    raw_text=raw_text,
                    summary=analysis["summary"],
                    structured_data=structured_data,
                )
            )
            # Track for embedding indexing
            indexed_file_ids.append(file_record.id)
        
        # Build summary
        stored_id = f"file_{file_record.id}"
        summary = AttachmentSummary(
            client_filename=original_name,
            stored_id=stored_id,
            type=analysis["type"],
            summary=analysis["summary"],
            tags=analysis["tags"],
        )
        attachments_summary.append(summary)
        
        # Context for LLM
        attachment_context_parts.append(
            f"[Uploaded: {original_name}] {analysis['summary']}"
        )

    # Index newly uploaded files for semantic search
    for file_id in indexed_file_ids:
        try:
            from app.memory.models import DocumentContent
            doc = db.query(DocumentContent).filter(DocumentContent.file_id == file_id).first()
            if doc:
                embeddings_service.index_document(db, doc, force=True)
                print(f"[chat_with_attachments] Indexed embeddings for file_id={file_id}")
        except Exception as e:
            print(f"[chat_with_attachments] Failed to index file_id={file_id}: {e}")

    # Build message with attachment context
    full_message = message.strip() if message else ""
    
    if attachment_context_parts:
        attachment_info = "\n".join(attachment_context_parts)
        if full_message:
            full_message = f"{full_message}\n\n[User uploaded:]\n{attachment_info}"
        else:
            full_message = f"[User uploaded:]\n{attachment_info}\n\nPlease acknowledge these files."

    if not full_message and not attachments_summary:
        return ChatResponse(
            project_id=project_id,
            provider="none",
            model=None,
            reply="No message or attachments received.",
            attachments_summary=[],
        )

    # Build context
    context_block = build_context_block(db, project_id)
    doc_context = build_document_context(db, project_id, full_message)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===\n" + doc_context

    history = memory_service.list_messages(db, project_id, limit=50)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    messages = history_dicts + [{"role": "user", "content": full_message}]

    try:
        jt = JobType(job_type)
    except ValueError:
        jt = JobType.CASUAL_CHAT

    system_prompt = f"Project: {project.name}. {project.description or ''}"
    system_prompt += """

The user just uploaded files. Acknowledge receipt and briefly describe what you found in them.
If it's a CV, mention key details like name, number of roles, and notable positions.
If it's an image, describe what you see based on the image analysis provided."""

    task = LLMTask(
        job_type=jt,
        messages=messages,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
    )

    try:
        result: LLMResult = call_llm(task)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Extract provider and model
    provider_str = _extract_provider_value(result)
    model_str = _extract_model_value(result)

    # Log messages with provider/model info
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

    print(f"[chat_with_attachments] Done. Returning {len(attachments_summary)} summaries")

    return ChatResponse(
        project_id=project_id,
        provider=provider_str,
        model=model_str,
        reply=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
        attachments_summary=attachments_summary,
    )


# ====== DIRECT LLM (PROTECTED) ======

class DirectLLMRequest(BaseModel):
    job_type: str
    message: str
    force_provider: Optional[str] = None


class DirectLLMResponse(BaseModel):
    provider: str
    model: Optional[str] = None  # ADDED: Model that generated the response
    content: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None


@app.post("/llm", response_model=DirectLLMResponse)
def direct_llm(
    req: DirectLLMRequest,
    auth: AuthResult = Depends(require_auth),
) -> DirectLLMResponse:
    """Direct LLM call without project context (protected)."""
    try:
        job_type = JobType(req.job_type)
    except ValueError:
        job_type = JobType.UNKNOWN

    task = LLMTask(
        job_type=job_type,
        messages=[{"role": "user", "content": req.message}],
    )

    try:
        result = call_llm(task)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = _extract_provider_value(result)
    model_str = _extract_model_value(result)

    return DirectLLMResponse(
        provider=provider_str,
        model=model_str,
        content=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
    )