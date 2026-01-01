# FILE: app/endpoints/chat.py
"""
Chat endpoint - text-only chat without attachments.

Refactored from main.py.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas
from app.llm import call_llm, LLMTask, LLMResult, JobType
from app.llm.schemas import Provider
from app.embeddings import service as embeddings_service

from app.helpers.context import build_context_block, build_document_context
from app.helpers.llm_utils import (
    sync_await,
    extract_provider_value,
    extract_model_value,
    classify_job_type,
)

router = APIRouter(tags=["Chat"])


# ============================================================================
# MODELS
# ============================================================================

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


# ============================================================================
# ENDPOINT
# ============================================================================

@router.post("/chat", response_model=ChatResponse)
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

    jt = classify_job_type(req.message, req.job_type)
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
        result: LLMResult = sync_await(call_llm(task))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    provider_str = extract_provider_value(result)
    model_str = extract_model_value(result)
    
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
