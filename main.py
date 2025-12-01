# FILE: main.py
"""
Orb Backend - FastAPI Application
Version: 0.11.0

Security Features:
- Password authentication with bcrypt
- Session token management
- Database encryption at rest

Features:
- File upload with text extraction and storage
- CV parsing and structured data extraction
- Document content retrieval for Q&A
- Image analysis via Gemini Vision
- Streaming LLM responses
- Web search grounding via Gemini
- Semantic search (RAG) with embeddings
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
    version="0.11.0",
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
    init_db()
    
    # Check auth status
    print("[startup] Checking authentication...")
    if is_auth_configured():
        print("[startup] Password authentication: ✓ configured")
    else:
        print("[startup] Password authentication: ✗ NOT CONFIGURED")
        print("[startup] Call POST /auth/setup to set a password")
    
    # Verify critical env vars
    print("[startup] Checking environment variables...")
    if os.getenv("GOOGLE_API_KEY"):
        print("[startup] GOOGLE_API_KEY: ✓ set (enables image analysis + web search)")
    else:
        print("[startup] GOOGLE_API_KEY: ✗ NOT SET - image analysis and web search will fail")
    
    if os.getenv("OPENAI_API_KEY"):
        print("[startup] OPENAI_API_KEY: ✓ set (enables chat + embeddings)")
    else:
        print("[startup] OPENAI_API_KEY: ✗ NOT SET - chat and semantic search will fail")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        print("[startup] ANTHROPIC_API_KEY: ✓ set")
    else:
        print("[startup] ANTHROPIC_API_KEY: ✗ NOT SET")


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

    tasks = memory_service.list_tasks(db, project_id)
    open_tasks = [t for t in tasks if t.status in ("todo", "in_progress")]
    if open_tasks:
        tasks_text = "\n".join(
            f"- [{t.id}] [{t.status.upper()}] {t.title}"
            for t in open_tasks
        )
        sections.append(f"OPEN TASKS:\n{tasks_text}")

    return "\n\n".join(sections) if sections else ""


def build_semantic_context(db: Session, project_id: int, user_message: str, top_k: int = 5) -> str:
    """
    Build context using semantic search.
    Retrieves the most relevant content based on the user's message.
    """
    results, total = embeddings_service.search_embeddings(
        db,
        project_id,
        user_message,
        top_k=top_k,
    )
    
    if not results:
        return ""
    
    context_parts = []
    
    for result in results:
        source_label = {
            "note": "NOTE",
            "message": "PREVIOUS CONVERSATION",
            "file": "DOCUMENT",
        }.get(result.source_type, "CONTENT")
        
        # Truncate content if too long
        content = result.content
        if len(content) > 1500:
            content = content[:1500] + "..."
        
        context_parts.append(
            f"[{source_label} - similarity: {result.similarity:.2f}]\n{content}"
        )
    
    return "\n\n---\n\n".join(context_parts)


def build_document_context(db: Session, project_id: int, user_message: str) -> str:
    """
    Build document context for RAG (fallback keyword-based method).
    Used when semantic search is disabled or has no embeddings.
    """
    message_lower = user_message.lower()
    
    # Check if user is asking about a specific file or CV
    cv_keywords = ["cv", "resume", "curriculum", "my cv", "your cv"]
    is_cv_query = any(kw in message_lower for kw in cv_keywords)
    
    # Check for specific filename mentions
    doc_contents = memory_service.list_document_contents(db, project_id)
    
    if not doc_contents:
        return ""
    
    relevant_docs = []
    
    for doc in doc_contents:
        # Check if this document is mentioned by name
        if doc.filename.lower() in message_lower:
            relevant_docs.append(doc)
            continue
        
        # Check if it's a CV query and this is a CV
        if is_cv_query and doc.doc_type == "cv":
            relevant_docs.append(doc)
            continue
        
        # Check if the content might be relevant (simple keyword match)
        if doc.raw_text:
            # Look for key terms from user message in document
            words = re.findall(r'\b\w{4,}\b', message_lower)
            for word in words:
                if word in doc.raw_text.lower():
                    relevant_docs.append(doc)
                    break
    
    if not relevant_docs:
        # If no specific match, include the most recent document
        relevant_docs = [doc_contents[0]]
    
    # Build context string
    context_parts = []
    for doc in relevant_docs[:3]:  # Limit to 3 documents
        context = f"\n=== DOCUMENT: {doc.filename} (type: {doc.doc_type}) ===\n"
        
        if doc.structured_data:
            try:
                data = json.loads(doc.structured_data)
                context += f"STRUCTURED DATA:\n{json.dumps(data, indent=2)}\n"
            except json.JSONDecodeError:
                pass
        
        if doc.summary:
            context += f"SUMMARY: {doc.summary}\n"
        
        # Include relevant portion of raw text
        if doc.raw_text:
            context += f"CONTENT:\n{doc.raw_text[:4000]}\n"
        
        context_parts.append(context)
    
    return "\n".join(context_parts)


# ====== CHAT ENDPOINT (PROTECTED) ======

@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
) -> ChatResponse:
    """Main chat endpoint with semantic search RAG (protected)."""
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")

    # Build context from notes/tasks
    context_block = build_context_block(db, req.project_id)
    
    # Build document context
    doc_context = ""
    if req.use_semantic_search:
        # Try semantic search first
        doc_context = build_semantic_context(db, req.project_id, req.message, top_k=5)
        
    if not doc_context:
        # Fallback to keyword-based if semantic search returns nothing
        doc_context = build_document_context(db, req.project_id, req.message)
    
    # Combine contexts
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if doc_context:
        full_context += "=== RELEVANT CONTEXT ===\n" + doc_context

    # Load message history
    history = memory_service.list_messages(db, req.project_id, limit=50)
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in history]
    messages = history_dicts + [{"role": "user", "content": req.message}]

    # Parse job type
    try:
        job_type = JobType(req.job_type)
    except ValueError:
        job_type = JobType.CASUAL_CHAT

    # Build system prompt with document awareness
    system_prompt = f"Project: {project.name}. {project.description or ''}"
    if doc_context:
        system_prompt += """

IMPORTANT: Relevant context from notes, documents, and previous conversations has been provided above.
Use this context to answer accurately. Quote specific details when relevant.
If you cannot find the information in the context, say so."""

    task = LLMTask(
        job_type=job_type,
        messages=messages,
        project_context=full_context if full_context else None,
        system_prompt=system_prompt,
    )

    try:
        result: LLMResult = call_llm(task)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Log messages
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id, role="user", content=req.message,
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id, role="assistant", content=result.content,
    ))

    return ChatResponse(
        project_id=req.project_id,
        provider=result.provider.value,
        reply=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
    )


# ====== CHAT WITH ATTACHMENTS (PROTECTED) ======

@app.post("/chat_with_attachments", response_model=ChatResponse)
async def chat_with_attachments(
    project_id: int = Form(...),
    message: str = Form(""),
    job_type: str = Form("casual_chat"),
    files: List[UploadFile] = FastAPIFile(default=[]),
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
) -> ChatResponse:
    """Chat with file attachments - extracts, parses, and stores document content (protected)."""
    print(f"[chat_with_attachments] project_id={project_id}, message='{message}', files={len(files)}")
    
    project = memory_service.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    attachments_summary: List[AttachmentSummary] = []
    attachment_context_parts: List[str] = []
    indexed_file_ids: List[int] = []  # Track files to index
    
    data_root = Path("data")
    project_dir = data_root / "files" / str(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)

    for upload_file in files:
        print(f"[chat_with_attachments] Processing: {upload_file.filename}, type: {upload_file.content_type}")
        
        file_bytes = await upload_file.read()
        original_name = upload_file.filename or "uploaded_file"
        suffix = Path(original_name).suffix
        unique_name = f"{uuid4().hex}{suffix}"
        file_path = project_dir / unique_name
        
        # Save file
        file_path.write_bytes(file_bytes)
        print(f"[chat_with_attachments] Saved: {file_path}")
        
        relative_path = str(file_path.relative_to(data_root)).replace("\\", "/")
        mime_type = upload_file.content_type or ""
        
        # Initialize variables
        raw_text = None
        doc_type = "document"
        structured_data = None
        analysis = None
        
        # Analyze based on type
        if is_image_mime_type(mime_type):
            print(f"[chat_with_attachments] Analyzing image with Gemini...")
            try:
                analysis = analyze_image(file_bytes, mime_type, message if message else None)
                
                # Check if there was an error in analysis
                if "error" in analysis:
                    print(f"[chat_with_attachments] Image analysis error: {analysis['error']}")
                else:
                    print(f"[chat_with_attachments] Image analysis success: {analysis['summary'][:100]}...")
                
                doc_type = analysis.get("type", "image")
                raw_text = f"[Image: {original_name}] {analysis['summary']}"
                
            except Exception as e:
                print(f"[chat_with_attachments] Image analysis exception: {type(e).__name__}: {e}")
                analysis = {
                    "summary": f"Image uploaded: {original_name} (analysis unavailable: {str(e)})",
                    "tags": ["image"],
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

    # Log messages
    user_content = message if message else "[Attachment only]"
    if attachments_summary:
        user_content += f" [Uploaded: {', '.join(a.client_filename for a in attachments_summary)}]"
    
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id, role="user", content=user_content,
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id, role="assistant", content=result.content,
    ))

    print(f"[chat_with_attachments] Done. Returning {len(attachments_summary)} summaries")

    return ChatResponse(
        project_id=project_id,
        provider=result.provider.value,
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

    return DirectLLMResponse(
        provider=result.provider.value,
        content=result.content,
        was_reviewed=result.was_reviewed,
        critic_review=result.critic_review,
    )