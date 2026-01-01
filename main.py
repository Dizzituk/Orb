# FILE: main.py
"""
Orb Backend - FastAPI Application
Version: 0.16.0

v0.16.0 Changes (Log Introspection):
- Added read-only log introspection feature
- GET /introspection/logs/last - Last completed job logs
- GET /introspection/logs - Time-based log query
- GET /introspection/logs/{job_id} - Specific job logs
- LLM-powered log summarization
- Spec-gate hash verification events (COMPUTED/VERIFIED/MISMATCH)
- Chat layer integration for natural language log queries

v0.15.1 Changes (OVERRIDE command support):
- Added OVERRIDE detection before vision routing in chat_with_attachments
- OVERRIDE keyword bypasses vision path, routes through router.py
- Image data included as multimodal content when OVERRIDE + images
- Supports OVERRIDE Gemini/Claude/GPT for explicit frontier model selection

v0.14.2 Changes (Flash removal + context logging):
- Added debug logging to verify document content reaches vision models
- When images/videos + text files uploaded, confirms text injection to vision context
- ORB_ROUTER_DEBUG=1 shows first 500 chars of injected context
- All vision routing now uses Gemini 2.5 Pro (images) or 3 Pro (videos)

v0.13.10 Changes (chat_with_attachments routing fix):
- CRITICAL FIX: Multi-image/video routing now uses job_classifier
- 2+ images → Gemini 2.5 Pro (was Flash)
- 2+ videos → Gemini 3.0 Pro (was Flash)  
- Mixed images + video → Gemini 3.0 Pro (was Flash)
- Added _map_model_to_vision_tier() helper for model→tier mapping
- Vision routing now consistent with /chat and /stream/chat

v0.13.5 Changes:
- CRITICAL FIX: Document content now injected into LLM context for chat_with_attachments
- Extracted text from uploaded docs now available to the LLM (was being extracted but not used)
- Added document_content_parts to build actual file content sections
- Files <50KB: Full text included; Files >50KB: First 40KB + last 10KB (truncated)
- Added debug logging showing total context size sent to LLM

v0.12.17 Changes:
- CRITICAL FIX: Don't inject filenames into message text for classifier
- Classifier receives clean user message, attachment metadata passed separately
- File info goes to context (for LLM) not message (for classifier)
- Fixed missing attachment_metadata list initialization

v0.12.8 Changes:
- Images/videos now route to Gemini Vision even without user message
- Default prompt "Describe this image/video in detail." used when no message

v0.12.7 Changes:
- Video files now route to Gemini Vision (same as images)
- Integrated new job_classifier for automatic routing
- Fixed file_analyzer to skip binary files from text extraction
- Added video_attachments tracking alongside image_attachments
"""
import os
import json
import re
from pathlib import Path
from uuid import uuid4
import time
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File as FastAPIFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session

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
from app.llm.schemas import Provider, AttachmentInfo
from app.llm.clients import check_provider_availability, call_openai
from app.llm.stream_router import router as stream_router
from app.llm.telemetry_router import router as telemetry_router
from app.llm.audit_logger import get_audit_logger, RoutingTrace
from app.llm.web_search_router import router as web_search_router
from app.embeddings.router import router as embeddings_router, search_router as embeddings_search_router
from app.embeddings import service as embeddings_service

# v0.16.0: Log introspection feature
from app.introspection.router import router as introspection_router
from app.astra_memory.router import router as astra_memory_router

# v0.12.7: Import video detection and vision functions
from app.llm.file_analyzer import is_video_mime_type
from app.llm.gemini_vision import ask_about_image, check_vision_available, analyze_video

# v0.13.10: Import job_classifier for proper media routing
# v0.15.1: Import detect_frontier_override for OVERRIDE command support
from app.llm.job_classifier import classify_job, detect_frontier_override

app = FastAPI(
    title="Orb Assistant",
    version="0.16.0",
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
        "file://",
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
    
    print("[startup] Initializing encryption...")
    from app.crypto import require_master_key_or_exit, is_master_key_initialized
    require_master_key_or_exit()
    
    if is_master_key_initialized():
        print("[startup] Database encryption: [OK] master key active")
    
    init_db()
    
    print("[startup] Checking authentication...")
    if is_auth_configured():
        print("[startup] Password authentication: [OK] configured")
    else:
        print("[startup] Password authentication: [X] NOT CONFIGURED")
        print("[startup] Call POST /auth/setup to set a password")
    
    print("[startup] Checking environment variables...")
    if os.getenv("GOOGLE_API_KEY"):
        print("[startup] GOOGLE_API_KEY: [OK] set (enables vision + web search)")
    else:
        print("[startup] GOOGLE_API_KEY: [X] NOT SET - vision and web search will fail")
    
    if os.getenv("OPENAI_API_KEY"):
        print("[startup] OPENAI_API_KEY: [OK] set (enables chat + embeddings)")
    else:
        print("[startup] OPENAI_API_KEY: [X] NOT SET - chat and semantic search will fail")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        print("[startup] ANTHROPIC_API_KEY: [OK] set")
    else:
        print("[startup] ANTHROPIC_API_KEY: [X] NOT SET")
    
    print("[startup] Checking Phase 4 status...")
    phase4_enabled = os.getenv("ORB_ENABLE_PHASE4", "false").lower() == "true"
    if phase4_enabled:
        print("[startup] Phase 4 Job System: [OK] ENABLED")
    else:
        print("[startup] Phase 4 Job System: [X] DISABLED")


    # ASTRA Memory: Auto-index on startup
    try:
        from app.astra_memory.indexer import run_full_index
        from app.db import SessionLocal
        _db = SessionLocal()
        _results = run_full_index(_db)
        print(f"[startup] ASTRA memory indexed: {sum(_results.values())} records")
        _db.close()
    except Exception as e:
        print(f"[startup] ASTRA memory indexing skipped: {e}")

# ====== ROUTERS ======

app.include_router(auth_router)
app.include_router(memory_router)
app.include_router(stream_router)
app.include_router(telemetry_router)
app.include_router(web_search_router)
app.include_router(embeddings_router)
app.include_router(embeddings_search_router)
app.include_router(astra_memory_router)

# v0.16.0: Log introspection (read-only, requires auth)
app.include_router(
    introspection_router,
    tags=["Introspection"],
    dependencies=[Depends(require_auth)]
)

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
        print(f"[startup] WARNING: Phase 4 import failed: {e}")


# ====== MODELS ======

class ChatRequest(BaseModel):
    project_id: int
    message: str
    job_type: str = "casual_chat"
    force_provider: Optional[str] = None
    use_semantic_search: bool = True


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


# ====== STATIC FILES ======

app.mount("/static", StaticFiles(directory="static"), name="static")


# ====== PUBLIC ENDPOINTS ======

@app.get("/")
def read_index():
    return FileResponse("static/index.html")


@app.get("/ping")
def ping():
    return {"status": "ok", "auth_configured": is_auth_configured()}


# ====== PROTECTED ENDPOINTS ======

@app.get("/providers")
def list_providers(auth: AuthResult = Depends(require_auth)):
    return check_provider_availability()


@app.get("/job-types")
def list_job_types(auth: AuthResult = Depends(require_auth)):
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

    tasks = memory_service.list_tasks(db, project_id, status="pending")[:10]
    if tasks:
        tasks_text = "\n".join(f"- {t.title}" for t in tasks)
        sections.append(f"PENDING TASKS:\n{tasks_text}")

    return "\n\n".join(sections) if sections else ""


def build_document_context(db: Session, project_id: int, user_message: str) -> str:
    """Build context from uploaded documents."""
    try:
        from app.memory.models import DocumentContent
        
        recent_docs = (db.query(DocumentContent)
                      .filter(DocumentContent.project_id == project_id)
                      .order_by(DocumentContent.created_at.desc())
                      .limit(5)
                      .all())
        
        if not recent_docs:
            return ""
        
        context_parts = []
        for doc in recent_docs:
            summary = doc.summary[:300] if doc.summary else "(no summary)"
            context_parts.append(f"[{doc.filename}]: {summary}")
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"[build_document_context] Error: {e}")
        return ""




def _make_session_id(auth: AuthResult) -> str:
    """Best-effort stable session id for audit correlation."""
    sid = getattr(auth, "session_id", None)
    if sid:
        return str(sid)
    uid = getattr(auth, "user_id", None)
    if uid:
        return str(uid)
    # Fallback: avoid crashing if auth object changes
    return "unknown"


def _extract_provider_value(result: LLMResult) -> str:
    if result.provider is None:
        return "unknown"
    if hasattr(result.provider, 'value'):
        return result.provider.value
    return str(result.provider)


def _extract_model_value(result: LLMResult) -> Optional[str]:
    return result.model if hasattr(result, 'model') else None


def _classify_job_type(message: str, requested_type: str) -> JobType:
    """
    Simple job type passthrough - lets router.py handle all classification.
    
    The router's classify_and_route() does the real classification using job_classifier.
    This just validates the requested type or defaults to CHAT_LIGHT.
    """
    # If user explicitly requested a type, validate and use it
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            print(f"[classify] Invalid job_type '{requested_type}', defaulting to CHAT_LIGHT")
            return JobType.CHAT_LIGHT
    
    # Default to chat_light (matches job_classifier.py)
    print(f"[classify] Defaulting to CHAT_LIGHT")
    return JobType.CHAT_LIGHT


def _map_model_to_vision_tier(model: str) -> str:
    """
    Map job_classifier's model selection to gemini_vision tier.
    
    v0.13.10: Maps classifier's chosen model to vision tier parameter.
    
    Tier mapping:
    - gemini-2.0-flash → "fast"
    - gemini-2.5-pro → "complex" (for IMAGE_COMPLEX)
    - gemini-2.5-pro → "video_heavy" (for VIDEO_HEAVY - when .env has wrong model)
    - gemini-3.0-pro-preview or gemini-3-pro → "video_heavy"
    - default → "fast"
    """
    if not model:
        return "fast"
    
    model_lower = model.lower()
    
    # Check for specific model patterns
    if "flash" in model_lower or "2.0" in model_lower:
        return "fast"
    elif "3.0" in model_lower or "3-pro" in model_lower or "3.0-pro" in model_lower:
        # gemini-3.0-pro-preview, gemini-3-pro, etc.
        return "video_heavy"
    elif "2.5-pro" in model_lower:
        # gemini-2.5-pro can be used for both complex images and video_heavy
        # Default to complex, but video_heavy tier also supports 2.5-pro
        return "complex"
    else:
        return "fast"  # Default to fast tier


# ====== CHAT ENDPOINTS ======

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    auth: AuthResult = Depends(require_auth),
    db: Session = Depends(get_db),
) -> ChatResponse:
    """Send chat message with context."""
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {req.project_id}")

    context_block = build_context_block(db, req.project_id)
    
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
                semantic_context = "=== RELEVANT DOCUMENTS ===\n"
                for result in semantic_results:
                    semantic_context += f"\n[Score: {result.similarity_score:.3f}] {result.content_preview}\n"
        except Exception as e:
            print(f"[chat] Semantic search failed: {e}")
    
    doc_context = build_document_context(db, req.project_id, req.message)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== RECENT UPLOADS ===\n" + doc_context

    history = memory_service.list_messages(db, req.project_id, limit=50)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    messages = history_dicts + [{"role": "user", "content": req.message}]

    jt = _classify_job_type(req.message, req.job_type)
    print(f"[chat] Job type: {jt.value}")

    system_prompt = f"Project: {project.name}. {project.description or ''}"
    
    task = LLMTask(
        job_type=jt,
        messages=messages,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
        force_provider=Provider(req.force_provider) if req.force_provider else None,
    )

    try:
        result: LLMResult = call_llm(task)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = _extract_provider_value(result)
    model_str = _extract_model_value(result)
    
    print(f"[chat] Response from: {provider_str} / {model_str}")

    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="user",
        content=req.message,
        provider="local",
    ))
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
    
    v0.13.10: Multi-image/video routing now uses job_classifier (2+ images → 2.5 Pro, mixed → 3.0 Pro)
    v0.13.5: Document content now injected into LLM context (was extracted but not used)
    v0.12.17: Fixed classifier receiving filenames in message text
    v0.12.7: Video files now route to Gemini Vision alongside images.
    """
    project = memory_service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")

    attachments_summary: List[AttachmentSummary] = []
    attachment_context_parts: List[str] = []
    document_content_parts: List[str] = []  # v0.13.5: Store actual document text
    attachment_metadata: List[dict] = []  # v0.12.17: For router's job_classifier
    indexed_file_ids: List[int] = []
    
    # Track media for Gemini Vision routing
    image_attachments: List[dict] = []
    video_attachments: List[dict] = []
    
    project_dir = Path(f"data/files/{project_id}")
    project_dir.mkdir(parents=True, exist_ok=True)

    for upload_file in files:
        original_name = upload_file.filename or "unknown"
        suffix = Path(original_name).suffix.lower()
        unique_name = f"{uuid4().hex}{suffix}"
        file_path = project_dir / unique_name
        relative_path = f"{project_id}/{unique_name}"
        
        content = upload_file.file.read()
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"[chat_with_attachments] Saved: {original_name} -> {file_path}")
        
        mime_type = upload_file.content_type or ""
        
        analysis: dict = {}
        raw_text: Optional[str] = None
        structured_data: Optional[str] = None
        doc_type: str = "document"
        
        # ==========================================================================
        # v0.12.7: DETECT IMAGES AND VIDEO FOR GEMINI VISION
        # ==========================================================================
        
        if is_image_mime_type(mime_type):
            print(f"[chat_with_attachments] Detected IMAGE: {original_name}")
            
            image_attachments.append({
                "bytes": content,
                "mime_type": mime_type,
                "filename": original_name,
                "path": str(file_path),
                "size": len(content),
            })
            
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
                doc_type = "image"
            except Exception as e:
                print(f"[chat_with_attachments] Image analysis failed: {e}")
                analysis = {
                    "summary": f"Image: {original_name}",
                    "tags": ["image", suffix.lstrip(".")],
                    "type": "image",
                }
                doc_type = "image"
        
        elif is_video_mime_type(mime_type):
            print(f"[chat_with_attachments] Detected VIDEO: {original_name} ({len(content)} bytes)")
            
            video_attachments.append({
                "bytes": content,
                "mime_type": mime_type,
                "filename": original_name,
                "path": str(file_path),
                "size": len(content),
            })
            
            # For videos, we don't extract text - they go to Gemini Vision
            analysis = {
                "summary": f"Video: {original_name} ({len(content) // 1024}KB)",
                "tags": ["video", suffix.lstrip(".")],
                "type": "video",
            }
            doc_type = "video"
            raw_text = None  # No text extraction for video
        
        else:
            # Text-based document extraction
            print(f"[chat_with_attachments] Extracting text: {original_name}")
            raw_text = extract_text_content(str(file_path), mime_type)
            
            if raw_text:
                doc_type = detect_document_type(raw_text, original_name)
                print(f"[chat_with_attachments] Detected doc_type: {doc_type}")
                
                if doc_type == "cv":
                    print(f"[chat_with_attachments] Parsing CV...")
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
                    print(f"[chat_with_attachments] Generating summary...")
                    summary_text = generate_document_summary(raw_text, original_name, doc_type, simple_llm_call)
                    analysis = {
                        "summary": summary_text,
                        "tags": [doc_type, suffix.lstrip(".")] if suffix else [doc_type],
                        "type": doc_type,
                    }
                
                # ==========================================================================
                # v0.13.5: BUILD DOCUMENT CONTENT FOR LLM CONTEXT
                # ==========================================================================
                # Now we capture the actual document text to pass to the LLM
                
                file_size_kb = len(raw_text) // 1024
                
                # For small files (<50KB), include full text
                # For large files, truncate to first 40KB + last 10KB
                if len(raw_text) <= 50 * 1024:
                    # Small file - include full text
                    document_section = f"""
=== FILE: {original_name} ===
Type: {doc_type}
Size: {file_size_kb}KB
Summary: {analysis.get('summary', 'N/A')}

--- FULL CONTENT ---
{raw_text}
"""
                else:
                    # Large file - truncate intelligently
                    first_chunk = raw_text[:40 * 1024]
                    last_chunk = raw_text[-10 * 1024:]
                    document_section = f"""
=== FILE: {original_name} ===
Type: {doc_type}
Size: {file_size_kb}KB (TRUNCATED - showing first 40KB + last 10KB)
Summary: {analysis.get('summary', 'N/A')}

--- BEGINNING OF CONTENT ---
{first_chunk}

... [CONTENT TRUNCATED] ...

--- END OF CONTENT ---
{last_chunk}
"""
                
                document_content_parts.append(document_section)
                print(f"[chat_with_attachments] Added document content: {original_name} ({file_size_kb}KB)")
            
            else:
                print(f"[chat_with_attachments] No text extracted: {original_name}")
                analysis = {
                    "summary": f"File uploaded: {original_name}",
                    "tags": [suffix.lstrip(".") if suffix else "file"],
                    "type": "document",
                }
        
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
        
        # Store document content (for text docs and image descriptions, NOT video)
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
        
        # v0.12.17: Collect metadata for router's job_classifier
        # v0.13.10: Use AttachmentInfo objects instead of dicts
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
                print(f"[chat_with_attachments] Indexed file_id={file_id}")
        except Exception as e:
            print(f"[chat_with_attachments] Index failed for file_id={file_id}: {e}")

    # ==========================================================================
    # v0.15.1: CHECK FOR OVERRIDE COMMAND BEFORE VISION ROUTING
    # ==========================================================================
    # OVERRIDE is a HARD RULE - bypasses ALL routing logic including vision routing
    # If user says "OVERRIDE Claude" with an image, it goes to Claude, period.
    # We detect OVERRIDE here to skip vision routing, but DON'T strip the line -
    # router.py will handle full OVERRIDE logic including model selection and line stripping.
    
    user_message = message.strip() if message else ""
    
    override_result = detect_frontier_override(user_message)
    frontier_override_active = override_result is not None
    
    # Store image data for OVERRIDE path (will be included in multimodal message)
    override_image_data = None
    
    if frontier_override_active:
        override_provider, override_model_id, _ = override_result  # Don't use cleaned_message
        print(f"[chat_with_attachments] OVERRIDE detected → {override_provider} / {override_model_id}")
        print(f"[chat_with_attachments] HARD OVERRIDE: Bypassing vision routing, sending to router.py")
        # NOTE: We do NOT strip the OVERRIDE line here - router.py will handle it
        # This ensures router.py knows exactly which model to use
        
        # Capture image data for inclusion in LLMTask (multimodal message)
        if image_attachments:
            import base64
            override_image_data = []
            for img in image_attachments:
                override_image_data.append({
                    "bytes_b64": base64.b64encode(img["bytes"]).decode("utf-8"),
                    "mime_type": img["mime_type"],
                    "filename": img["filename"],
                })
            print(f"[chat_with_attachments] Captured {len(override_image_data)} image(s) for OVERRIDE path")
    
    # ==========================================================================
    # v0.13.10: USE JOB_CLASSIFIER FOR MEDIA ROUTING
    # ==========================================================================
    
    # Check if we have media - route to Gemini Vision with proper tier
    # v0.15.1: SKIP vision routing entirely if OVERRIDE is active
    has_media = image_attachments or video_attachments
    
    if has_media and not frontier_override_active:
        # Provide default prompt if no user message
        vision_prompt = user_message if user_message else "Describe this image/video in detail."
        
        image_count = len(image_attachments)
        video_count = len(video_attachments)
        
        print(f"[chat_with_attachments] Routing to Gemini Vision: {image_count} image(s), {video_count} video(s)")
        
        # v0.13.10: Use job_classifier to determine correct tier
        # This makes multi-image and mixed-media routing consistent with job_classifier.py v0.12.17
        try:
            classification = classify_job(
                message=vision_prompt,
                attachments=attachment_metadata,
            )
            
            # Map classifier's model choice to vision tier
            selected_model = classification.model
            tier = _map_model_to_vision_tier(selected_model)
            
            if os.getenv("ORB_ROUTER_DEBUG") == "1":
                print(f"[chat_with_attachments] Classifier selected: {classification.provider.value} / {selected_model}")
                print(f"[chat_with_attachments] Mapped to vision tier: {tier}")
        
        except Exception as e:
            print(f"[chat_with_attachments] Classifier failed, using default tier: {e}")
            tier = "fast"  # Fallback to fast tier
        
        # v0.15.1: Track explicit override model name
        # When tier=="override", use override_model_id directly instead of _get_model_name(tier)
        override_model_name = override_model_id if tier == "override" else None
        
        vision_status = check_vision_available()
        if not vision_status.get("available"):
            error_msg = vision_status.get("error", "Vision not available")
            print(f"[chat_with_attachments] Vision unavailable: {error_msg}")
            attachment_context_parts.insert(0, f"[Warning: {error_msg}]")
        else:
            vision_context = f"Project: {project.name}."
            if project.description:
                vision_context += f" {project.description}"
            
            # v0.14.2: Include document text content in vision context
            if document_content_parts:
                vision_context += "\n\n=== DOCUMENT CONTENT ===\n"
                vision_context += "\n".join(document_content_parts)
                vision_context += "\n=== END DOCUMENTS ==="
                
                # v0.14.2: Debug logging to verify text content reaches vision model
                doc_count = len(document_content_parts)
                context_chars = len(vision_context)
                print(f"[chat_with_attachments] ✓ DOCUMENT CONTEXT INJECTED: {doc_count} document(s), {context_chars} total chars in vision_context")
                if os.getenv("ORB_ROUTER_DEBUG") == "1":
                    print(f"[chat_with_attachments] Document content preview (first 500 chars):")
                    print(f"  {vision_context[:500]}...")
            else:
                print(f"[chat_with_attachments] No document content to inject into vision context")
            
            # Prefer video if present (more complex), otherwise use image(s)
            if len(video_attachments) > 1:
                # v0.13.10: Multiple videos - upload all to Gemini
                media_type = "videos"
                print(f"[chat_with_attachments] Analyzing {len(video_attachments)} videos with Gemini...")
                
                try:
                    import google.generativeai as genai
                    from app.llm.gemini_vision import _get_google_api_key, _get_model_name
                    import time
                    
                    api_key = _get_google_api_key()
                    if not api_key:
                        vision_result = {
                            "answer": "GOOGLE_API_KEY not set",
                            "provider": "google",
                            "model": override_model_name or (_get_model_name(tier) if tier else "gemini-2.5-pro"),
                            "error": "No API key"
                        }
                    else:
                        genai.configure(api_key=api_key)
                        
                        # Get the model for this tier (or use override)
                        model_name = override_model_name or _get_model_name(tier)
                        model = genai.GenerativeModel(model_name)
                        
                        # Upload all videos
                        video_files = []
                        for i, video_att in enumerate(video_attachments):
                            print(f"[chat_with_attachments] Uploading video {i+1}/{len(video_attachments)}: {video_att['filename']}...")
                            video_file = genai.upload_file(path=str(video_att["path"]))
                            
                            # Wait for processing
                            while video_file.state.name == "PROCESSING":
                                time.sleep(2)
                                video_file = genai.get_file(video_file.name)
                            
                            if video_file.state.name == "FAILED":
                                print(f"[chat_with_attachments] Video {i+1} processing failed")
                                continue
                            
                            video_files.append(video_file)
                            print(f"[chat_with_attachments] Video {i+1} ready")
                        
                        if not video_files:
                            vision_result = {
                                "answer": "All video processing failed",
                                "provider": "google",
                                "model": model_name,
                                "error": "Processing failed"
                            }
                        else:
                            # Build prompt with context
                            prompt_parts = []
                            if vision_context:
                                prompt_parts.append(f"Context: {vision_context}\n\n")
                            prompt_parts.append(f"User's question about these {len(video_files)} videos: {vision_prompt}")
                            full_prompt = "".join(prompt_parts)
                            
                            # Generate response with all videos
                            content_parts = video_files + [full_prompt]
                            response = model.generate_content(content_parts)
                            
                            vision_result = {
                                "answer": response.text,
                                "provider": "google",
                                "model": model_name,
                            }
                        
                        # Clean up uploaded videos
                        for video_file in video_files:
                            try:
                                genai.delete_file(video_file.name)
                            except Exception as cleanup_error:
                                print(f"[chat_with_attachments] Warning: Could not delete video: {cleanup_error}")
                        
                        print(f"[chat_with_attachments] Cleaned up {len(video_files)} video file(s)")
                
                except Exception as e:
                    print(f"[chat_with_attachments] Multi-video analysis failed: {e}")
                    vision_result = {
                        "answer": f"Multi-video analysis failed: {str(e)}",
                        "provider": "google",
                        "model": override_model_name or (_get_model_name(tier) if tier else "gemini-2.5-pro"),
                        "error": str(e)
                    }
            
            elif video_attachments:
                # Single video
                primary_media = video_attachments[0]
                media_type = "video"
                
                # v0.13.10: Pass tier to analyze_video
                try:
                    vision_result = analyze_video(
                        video_path=primary_media["path"],
                        user_question=vision_prompt,
                        context=vision_context,
                        tier=tier,  # NEW: Pass classifier's tier decision
                    )
                except TypeError:
                    # Fallback if analyze_video doesn't support tier yet
                    print("[chat_with_attachments] analyze_video doesn't support tier parameter, using default")
                    vision_result = analyze_video(
                        video_path=primary_media["path"],
                        user_question=vision_prompt,
                        context=vision_context,
                    )
            elif len(image_attachments) > 1:
                # v0.13.10: Multiple images - pass all to Gemini at once
                media_type = "images"
                print(f"[chat_with_attachments] Analyzing {len(image_attachments)} images with Gemini...")
                
                try:
                    import google.generativeai as genai
                    from app.llm.gemini_vision import _get_google_api_key, _get_model_name
                    
                    api_key = _get_google_api_key()
                    if not api_key:
                        vision_result = {
                            "answer": "GOOGLE_API_KEY not set",
                            "provider": "google",
                            "model": override_model_name or "gemini-2.5-pro",
                            "error": "No API key"
                        }
                    else:
                        genai.configure(api_key=api_key)
                        
                        # Get the model for this tier (or use override)
                        model_name = override_model_name or _get_model_name(tier)
                        model = genai.GenerativeModel(model_name)
                        
                        # Build prompt with context
                        prompt_parts = []
                        if vision_context:
                            prompt_parts.append(f"Context: {vision_context}\n\n")
                        prompt_parts.append(f"User's question: {vision_prompt}")
                        full_prompt = "".join(prompt_parts)
                        
                        # Load all images
                        import PIL.Image
                        import io
                        images = []
                        for img_att in image_attachments:
                            image = PIL.Image.open(io.BytesIO(img_att["bytes"]))
                            images.append(image)
                        
                        # Build content with prompt and all images
                        content_parts = [full_prompt] + images
                        
                        # Call Gemini with all images
                        response = model.generate_content(content_parts)
                        
                        vision_result = {
                            "answer": response.text,
                            "provider": "google",
                            "model": model_name,
                        }
                        
                except Exception as e:
                    print(f"[chat_with_attachments] Multi-image analysis failed: {e}")
                    vision_result = {
                        "answer": f"Multi-image analysis failed: {str(e)}",
                        "provider": "google",
                        "model": _get_model_name(tier) if tier else "gemini-2.5-pro",
                        "error": str(e)
                    }
            else:
                # Single image
                primary_media = image_attachments[0]
                media_type = "image"
                
                # v0.13.10: Pass tier to ask_about_image
                vision_result = ask_about_image(
                    image_source=primary_media["bytes"],
                    user_question=vision_prompt,
                    mime_type=primary_media["mime_type"],
                    context=vision_context,
                    tier=tier,  # NEW: Pass classifier's tier decision
                )
            
            provider_str = vision_result.get("provider", "google")
            model_str = vision_result.get("model", "gemini-2.0-flash")
            reply_text = vision_result.get("answer", f"I couldn't analyze the {media_type}.")
            
            if vision_result.get("error"):
                print(f"[chat_with_attachments] Vision error: {vision_result['error']}")
            
            print(f"[chat_with_attachments] Vision response from: {provider_str} / {model_str}")
            
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
    
    # ==========================================================================
    # FALLBACK: Text-only path
    # ==========================================================================

    # v0.12.17 FIX: Don't inject file upload metadata into user message
    # The classifier already receives attachment metadata separately via task.attachments
    # Injecting filenames into the message causes false positives (e.g., "architect" in filename)
    
    full_message = user_message if user_message else ""
    
    # For logging/storage, note that files were uploaded, but don't include filenames
    # (filenames are already in attachment_metadata and will be passed to classifier)
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

    context_block = build_context_block(db, project_id)
    doc_context = build_document_context(db, project_id, full_message)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===\n" + doc_context
    
    # v0.12.17: Add attachment summaries to context, not to user message
    if attachment_context_parts:
        attachment_info = "\n".join(attachment_context_parts)
        full_context += f"\n\n=== USER UPLOADED FILES (METADATA) ===\n{attachment_info}"
    
    # ==========================================================================
    # v0.13.5: ADD ACTUAL DOCUMENT CONTENT TO CONTEXT
    # ==========================================================================
    # This is the critical fix: now the LLM can actually see the document text
    
    if document_content_parts:
        document_content_block = "\n".join(document_content_parts)
        full_context += f"\n\n=== DOCUMENT CONTENT ===\n{document_content_block}"
        
        # Debug logging to show context was built
        total_docs = len(document_content_parts)
        doc_types = {}
        for att in attachment_metadata:
            # v0.13.10: Handle both dict and Pydantic AttachmentInfo objects
            if isinstance(att, dict):
                dt = att.get("doc_type", "unknown")
            else:
                dt = getattr(att, "doc_type", "unknown")
            doc_types[dt] = doc_types.get(dt, 0) + 1
        
        type_summary = ", ".join([f"{dt}={count}" for dt, count in doc_types.items()])
        print(f"[chat_with_attachments] Built context for LLM: {len(full_context)} chars "
              f"({total_docs} documents with content: {type_summary})")

    history = memory_service.list_messages(db, project_id, limit=50)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    
    # v0.15.1: Build multimodal message when OVERRIDE with images
    # NOTE: We include the OVERRIDE line in the message - router.py will strip it
    if frontier_override_active and override_image_data:
        # Build multimodal content with images + text
        content_parts = []
        
        # Add images first
        for img_data in override_image_data:
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{img_data['mime_type']};base64,{img_data['bytes_b64']}"
                }
            })
        
        # Add text prompt (including OVERRIDE line for router.py to detect)
        text_content = user_message if user_message else "Describe this image in detail."
        content_parts.append({"type": "text", "text": text_content})
        
        messages = history_dicts + [{"role": "user", "content": content_parts}]
        print(f"[chat_with_attachments] Built multimodal message: {len(override_image_data)} image(s) + text")
    else:
        messages = history_dicts + [{"role": "user", "content": full_message}]

    # Classification logic:
    # - If attachments present: Use UNKNOWN, let router's job_classifier decide
    # - If no attachments: Use UNKNOWN, router will default to CHAT_LIGHT
    if attachment_metadata:
        jt = JobType.UNKNOWN  # Let router's job_classifier handle it with attachment data
        print(f"[chat_with_attachments] {len(attachment_metadata)} attachment(s) - router will classify")
    else:
        jt = JobType.UNKNOWN  # No attachments, router will use default
        print(f"[chat_with_attachments] No attachments - router will use default")

    system_prompt = f"Project: {project.name}. {project.description or ''}"
    
    # v0.12.17: Only add upload instruction if files were actually uploaded
    if attachments_summary:
        system_prompt += "\n\nThe user uploaded files. Review the file information in the context and respond appropriately."

    # v0.15.1: Let router.py handle OVERRIDE detection and routing
    # The OVERRIDE line is preserved in the message for router.py to process
    task = LLMTask(
        job_type=jt,
        messages=messages,
        attachments=attachment_metadata if attachment_metadata else None,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
    )

    try:
        result: LLMResult = call_llm(task)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = _extract_provider_value(result)
    model_str = _extract_model_value(result)
    
    print(f"[chat_with_attachments] Response from: {provider_str} / {model_str}")

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


# ====== DIRECT LLM ======

class DirectLLMRequest(BaseModel):
    job_type: str
    message: str
    force_provider: Optional[str] = None


class DirectLLMResponse(BaseModel):
    provider: str
    model: Optional[str] = None
    content: str
    was_reviewed: bool = False
    critic_review: Optional[str] = None


@app.post("/llm", response_model=DirectLLMResponse)
def direct_llm(
    req: DirectLLMRequest,
    auth: AuthResult = Depends(require_auth),
) -> DirectLLMResponse:
    """Direct LLM call without project context."""
    jt = _classify_job_type(req.message, req.job_type)
    print(f"[llm] Job type: {jt.value}")

    task = LLMTask(
        job_type=jt,
        messages=[{"role": "user", "content": req.message}],
        force_provider=Provider(req.force_provider) if req.force_provider else None,
    )

    try:
        audit = get_audit_logger()
        trace: RoutingTrace | None = None
        request_id = str(uuid4())
        if getattr(audit, "enabled", False):
            trace = audit.start_trace(
                session_id=_make_session_id(auth),
                project_id=project_id,
                user_text=req.message,
                request_id=request_id,
                attachments=None,
            )
            trace.log_request_start(
                job_type=jt.value,
                resolved_job_type=jt.value,
                provider=provider_value,
                model=model_value,
                reason="main.py /chat",
                frontier_override=None,
                file_map_injected=False,
                attachments=None,
            )

        t0 = time.perf_counter()
        try:
            result = call_llm(task)
        except Exception as e:
            if trace:
                trace.log_error(
                    where="main.chat",
                    error_type=type(e).__name__,
                    message=str(e),
                )
                trace.finalize(success=False, error_type=type(e).__name__, message=str(e))
            raise

        dt_ms = int((time.perf_counter() - t0) * 1000)
        if trace:
            trace.log_routing_decision(
                job_type=jt.value,
                provider=getattr(result, "provider", provider_value),
                model=getattr(result, "model", model_value),
                reason="call_llm result",
                frontier_override=None,
            )
            trace.log_model_call(
                stage="primary",
                provider=getattr(result, "provider", provider_value),
                model=getattr(result, "model", model_value),
                role="primary",
                prompt_tokens=0,
                completion_tokens=0,
                duration_ms=dt_ms,
                success=True,
            )
            trace.finalize(success=True)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = _extract_provider_value(result)
    model_str = _extract_model_value(result)
    
    print(f"[llm] Response from: {provider_str} / {model_str}")

    return DirectLLMResponse(
        provider=provider_str,
        model=model_str,
        content=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
    )







