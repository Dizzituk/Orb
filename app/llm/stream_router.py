# app/llm/stream_router.py
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).
"""

import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas
from .streaming import stream_llm, get_available_streaming_provider

router = APIRouter(prefix="/stream", tags=["streaming"])


class StreamChatRequest(BaseModel):
    project_id: int
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    # Optional routing hints (currently unused, but accepted for future policy routing)
    job_type: Optional[str] = None
    use_policy: bool = False
    include_history: bool = True
    history_limit: int = 20


async def generate_sse_stream(
    project_id: int,
    message: str,
    provider: Optional[str],
    model: Optional[str],
    system_prompt: str,
    messages: List[dict],
    db: Session,
):
    """Generate SSE stream with proper formatting."""
    full_response = ""
    metadata_sent = False
    # Track which provider/model actually handled the response
    current_provider = provider or "unknown"
    current_model = model

    async for chunk in stream_llm(
        messages=messages,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
    ):
        # First chunk is metadata JSON
        if not metadata_sent and chunk.startswith("{"):
            try:
                metadata = json.loads(chunk.strip())
                if metadata.get("type") == "metadata":
                    current_provider = metadata.get("provider", current_provider or "unknown")
                    current_model = metadata.get("model", current_model)
                    yield f"data: {json.dumps(metadata)}\n\n"
                    metadata_sent = True
                    continue
                elif metadata.get("error"):
                    yield f"data: {json.dumps({'type': 'error', 'error': metadata['error']})}\n\n"
                    return
            except json.JSONDecodeError:
                pass

        # Stream token
        full_response += chunk
        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"

    # Save messages to database
    # User message: mark provider as "local" (typed by the user, not an LLM)
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id,
        role="user",
        content=message,
        provider="local",
    ))
    # Assistant message: record which provider/model actually generated the text
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id,
        role="assistant",
        content=full_response,
        provider=current_provider,
        model=current_model,
    ))

    # Send completion event
    yield f"data: {json.dumps({'type': 'done', 'provider': current_provider, 'model': current_model, 'total_length': len(full_response)})}\n\n"


@router.post("/chat")
async def stream_chat(
    req: StreamChatRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """
    Stream chat response using Server-Sent Events.

    Returns an SSE stream with:
    - metadata: {type: "metadata", provider: "...", model: "..."}
    - tokens: {type: "token", content: "..."}
    - completion: {type: "done", provider: "...", model: "...", total_length: N}
    - errors: {type: "error", error: "..."}
    """

    # Verify project exists
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")

    # Check provider availability (fallback to whatever the streaming layer can find)
    if not req.provider and not get_available_streaming_provider():
        raise HTTPException(status_code=503, detail="No LLM provider available")

    # Build message history
    messages: List[dict] = []
    if req.include_history:
        history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        messages = [{"role": msg.role, "content": msg.content} for msg in history]

    # Add current message
    messages.append({"role": "user", "content": req.message})

    # Build system prompt
    system_prompt = f"Project: {project.name}."
    if project.description:
        system_prompt += f" {project.description}"

    # NOTE: For now we simply pass through provider/model from the request.
    # Future work: use req.job_type / req.use_policy to invoke the routing policy.
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=req.provider,
            model=req.model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # For Nginx / reverse proxies
        },
    )


@router.get("/providers")
async def list_streaming_providers(
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """List available providers for streaming."""
    from .streaming import HAS_OPENAI, HAS_ANTHROPIC, HAS_GEMINI, DEFAULT_MODELS
    import os

    providers = {}

    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        providers["openai"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["openai"],
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        }

    if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
        providers["anthropic"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["anthropic"],
            "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        }

    if HAS_GEMINI and os.getenv("GOOGLE_API_KEY"):
        providers["gemini"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["gemini"],
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        }

    return {
        "providers": providers,
        "default": get_available_streaming_provider(),
    }
