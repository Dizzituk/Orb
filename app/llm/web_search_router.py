# app/llm/web_search_router.py
"""
Web search endpoints for real-time grounded responses.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas
from .web_search import search_and_answer, is_web_search_available, format_sources_markdown

router = APIRouter(prefix="/search", tags=["web_search"])


class WebSearchRequest(BaseModel):
    project_id: int
    query: str
    include_history: bool = True
    history_limit: int = 10


class WebSearchResponse(BaseModel):
    project_id: int
    query: str
    answer: str
    sources: list
    search_queries: list
    provider: str
    error: Optional[str] = None


@router.get("/status")
async def search_status(auth: AuthResult = Depends(require_auth)):
    """Check if web search is available."""
    return {
        "available": is_web_search_available(),
        "provider": "gemini" if is_web_search_available() else None,
    }


@router.post("/query", response_model=WebSearchResponse)
async def web_search_query(
    req: WebSearchRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """
    Perform a web-grounded search query.
    Returns answer with sources.
    """
    if not is_web_search_available():
        raise HTTPException(
            status_code=503,
            detail="Web search not available. Check GOOGLE_API_KEY."
        )
    
    # Verify project
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")
    
    # Build context
    context = f"Project: {project.name}"
    if project.description:
        context += f". {project.description}"
    
    # Get conversation history
    history = None
    if req.include_history:
        messages = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        history = [{"role": msg.role, "content": msg.content} for msg in messages]
    
    # Perform search
    result = search_and_answer(
        query=req.query,
        context=context,
        history=history,
    )
    
    # Format answer with sources
    answer_with_sources = result["answer"]
    if result["sources"]:
        answer_with_sources += format_sources_markdown(result["sources"])
    
    # Save to message history
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="user",
        content=f"🔍 {req.query}",  # Mark as search query
    ))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=req.project_id,
        role="assistant",
        content=answer_with_sources,
    ))
    
    return WebSearchResponse(
        project_id=req.project_id,
        query=req.query,
        answer=answer_with_sources,
        sources=result["sources"],
        search_queries=result["search_queries"],
        provider=result["provider"],
        error=result.get("error"),
    )
