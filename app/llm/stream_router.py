# FILE: app/llm/stream_router.py
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

v0.13.9: CRITICAL FIX - Streaming event format normalization
- Fixed: AttributeError 'str' object has no attribute 'get'
- Added: Defensive normalization for legacy string tokens
- Updated: Handle dict events with "text" field (not "content")
- Updated: Error events use "message" field (not "error")
- Added: Comprehensive event type logging

v0.13.7: CRITICAL FIX - Security classification over-triggering prevented
- Removed generic keywords ("high stakes", "critical") from security_keywords
- Security keywords now concrete: "security review", "threat model", "encryption", "vulnerability"
- Added classification logging (first 200 chars) for debugging
- Architecture prompts default to architecture_design unless security is explicit main topic

v0.13.4: CRITIQUE PIPELINE INTEGRATION COMPLETE
v0.13.3: CRITICAL FIX - Security reviews correctly classified
v0.13.2: CRITICAL FIX - High-stakes jobs route to Opus
v0.12.5: Added document context and semantic search
v0.12.4: Fixed THINKING/ANSWER tags
v0.12.2: Added Phase-4 job-type routing
"""

import os
import re
import json
import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import get_db
from app.auth import require_auth
from app.auth.middleware import AuthResult
from app.memory import service as memory_service, schemas as memory_schemas
from app.llm.schemas import JobType, RoutingConfig, Provider, LLMTask, RoutingOptions
from .streaming import stream_llm, get_available_streaming_provider

# Critique pipeline imports
from app.llm.router import (
    run_high_stakes_with_critique,
    synthesize_envelope_from_task,
    is_high_stakes_job,
    is_opus_model,
    HIGH_STAKES_JOB_TYPES,
)

router = APIRouter(prefix="/stream", tags=["streaming"])
logger = logging.getLogger(__name__)

# Default models
DEFAULT_MODELS = {
    "openai": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini"),
    "anthropic": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    "gemini": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash"),
}


def _parse_reasoning_tags(raw: str) -> tuple:
    """Parse THINKING/ANSWER tags from streamed content."""
    thinking_match = re.search(r'<THINKING>([\s\S]*?)</THINKING>', raw, re.IGNORECASE)
    answer_match = re.search(r'<ANSWER>([\s\S]*?)</ANSWER>', raw, re.IGNORECASE)
    
    if thinking_match and answer_match:
        reasoning = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return answer, reasoning
    
    cleaned = re.sub(r'</?THINKING[^>]*>', '', raw, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?ANSWER[^>]*>', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()
    
    return cleaned if cleaned else raw, ""


def _build_context_block(db: Session, project_id: int) -> str:
    """Build context from notes + tasks."""
    sections = []
    notes = memory_service.list_notes(db, project_id)[:10]
    if notes:
        notes_text = "\n".join(f"- [{n.id}] {n.title}: {n.content[:200]}..." for n in notes)
        sections.append(f"PROJECT NOTES:\n{notes_text}")
    
    tasks = memory_service.list_tasks(db, project_id, status="pending")[:10]
    if tasks:
        tasks_text = "\n".join(f"- {t.title}" for t in tasks)
        sections.append(f"PENDING TASKS:\n{tasks_text}")
    
    return "\n\n".join(sections) if sections else ""


def _build_document_context(db: Session, project_id: int) -> str:
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
            summary = doc.summary[:500] if doc.summary else ""
            raw_preview = doc.raw_text[:1000] if doc.raw_text else ""
            if summary or raw_preview:
                context_parts.append(f"[{doc.filename}]:\nSummary: {summary}\nContent: {raw_preview}...")
        
        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[stream_router] Error building document context: {e}")
        return ""


def _get_semantic_context(db: Session, project_id: int, query: str) -> str:
    """Get relevant context via semantic search."""
    try:
        from app.embeddings import service as embeddings_service
        results = embeddings_service.search(db=db, project_id=project_id, query=query, top_k=5)
        
        if not results:
            return ""
        
        context_parts = ["=== RELEVANT CONTEXT (semantic search) ==="]
        for result in results:
            content_preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            context_parts.append(f"\n[Score: {result.similarity:.3f}] {content_preview}")
        
        return "\n".join(context_parts)
    except Exception as e:
        print(f"[stream_router] Semantic search failed: {e}")
        return ""


def _classify_job_type(message: str, requested_type: str) -> JobType:
    """
    Classify message to determine appropriate job type.
    
    v0.13.7: Fixed security classification over-triggering
    """
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            pass
    
    msg_lower = message.lower()
    
    # v0.13.7: Log first 200 chars for debugging
    print(f"[stream_router] Classifying message (first 200 chars): {repr(message[:200])}")
    
    # v0.13.7: CONCRETE security keywords only
    security_keywords = [
        "security review", "security audit", "security assessment",
        "penetration test", "pentest", "threat model", "threat modeling",
        "vulnerability", "vulnerabilities", "vulnerability assessment",
        "exploit", "attack vector", "attack surface",
        "sql injection", "xss", "csrf", "authentication bypass",
        "privilege escalation", "session fixation", "session hijacking",
        "security analysis", "security check",
        "encryption review", "key management", "secrets management",
        "authentication security", "authorization security",
        "security hardening", "security posture",
    ]
    
    arch_keywords = [
        "architect", "architecture", "design a system", "system design",
        "microservice", "micro-service", "infrastructure", "infra",
        "scalab", "database schema", "db schema", "api design",
        "high-level design", "hld", "distributed system", "design pattern", "tech stack",
    ]
    
    review_keywords = [
        "review this", "review my", "code review", "check this code",
        "find bugs", "audit this", "critique", "what's wrong with",
    ]
    
    code_keywords = [
        "write a function", "write code", "implement", "debug",
        "fix this code", "refactor", "def ", "function ", "```",
    ]
    
    language_keywords = [
        "python", "javascript", "typescript", "java", "c++", "rust",
        "react", "vue", "fastapi", "django",
    ]
    
    # PRIORITY 1: Security (explicit terms only)
    if any(kw in msg_lower for kw in security_keywords):
        print(f"[stream_router] Classified: SECURITY_REVIEW (explicit security keyword)")
        return JobType.SECURITY_REVIEW
    
    # PRIORITY 2: Architecture
    if any(kw in msg_lower for kw in arch_keywords):
        print(f"[stream_router] Classified: ARCHITECTURE_DESIGN")
        return JobType.ARCHITECTURE_DESIGN
    
    # PRIORITY 3: Code review
    if any(kw in msg_lower for kw in review_keywords):
        print(f"[stream_router] Classified: CODE_REVIEW")
        return JobType.CODE_REVIEW
    
    # PRIORITY 4: Code tasks
    is_code_related = (
        any(kw in msg_lower for kw in code_keywords) or
        any(kw in msg_lower for kw in language_keywords)
    )
    
    if is_code_related:
        complex_indicators = ["complex", "full file", "entire file", "production"]
        if any(x in msg_lower for x in complex_indicators):
            print(f"[stream_router] Classified: COMPLEX_CODE_CHANGE")
            return JobType.COMPLEX_CODE_CHANGE
        print(f"[stream_router] Classified: SIMPLE_CODE_CHANGE")
        return JobType.SIMPLE_CODE_CHANGE
    
    print(f"[stream_router] Classified: CASUAL_CHAT (default)")
    return JobType.CASUAL_CHAT


def _select_provider_for_job_type(job_type: JobType) -> tuple:
    """Select provider and model based on job_type."""
    if job_type in RoutingConfig.GPT_ONLY_JOBS:
        return ("openai", DEFAULT_MODELS["openai"])
    
    if job_type in RoutingConfig.HIGH_STAKES_JOBS:
        print(f"[stream_router] High-stakes job '{job_type.value}' → Opus")
        return ("anthropic", DEFAULT_MODELS["anthropic_opus"])
    
    if job_type in RoutingConfig.CLAUDE_PRIMARY_JOBS:
        return ("anthropic", DEFAULT_MODELS["anthropic"])
    
    if job_type in RoutingConfig.GEMINI_JOBS:
        return ("gemini", DEFAULT_MODELS["gemini"])
    
    if job_type in RoutingConfig.MEDIUM_DEV_JOBS:
        smart_provider = RoutingConfig.SMART_PROVIDER
        provider_key = smart_provider.value
        return (provider_key, DEFAULT_MODELS.get(provider_key, DEFAULT_MODELS["anthropic"]))
    
    print(f"[stream_router] No routing rule, defaulting to OpenAI")
    return ("openai", DEFAULT_MODELS["openai"])


class StreamChatRequest(BaseModel):
    project_id: int
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    job_type: Optional[str] = None
    use_policy: bool = False
    include_history: bool = True
    history_limit: int = 20
    enable_reasoning: bool = False
    use_semantic_search: bool = True


def chunk_text(text: str, chunk_size: int = 80):
    """Split text into chunks for fake streaming."""
    if not text:
        return
    
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_len = len(word) + 1
        if current_length + word_len > chunk_size and current_chunk:
            yield " ".join(current_chunk) + " "
            current_chunk = [word]
            current_length = word_len
        else:
            current_chunk.append(word)
            current_length += word_len
    
    if current_chunk:
        yield " ".join(current_chunk)


async def generate_high_stakes_critique_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    full_context: str,
    job_type_str: str,
    db: Session,
    enable_reasoning: bool = False,
):
    """Generate SSE stream for high-stakes jobs using critique pipeline."""
    import asyncio
    
    yield f"data: {json.dumps({'type': 'metadata', 'provider': provider, 'model': model})}\n\n"
    
    print(f"[stream] High-stakes: routing through critique pipeline (job_type={job_type_str})")
    
    task = LLMTask(
        job_type=JobType.ORCHESTRATOR,
        messages=messages,
        system_prompt=system_prompt,
        project_context=full_context,
        project_id=project_id,
    )
    
    try:
        envelope = synthesize_envelope_from_task(task, project_id=project_id)
        
        result = await run_high_stakes_with_critique(
            task=task,
            provider_id=provider,
            model_id=model,
            envelope=envelope,
            job_type_str=job_type_str,
        )
        
        if result.error_message:
            yield f"data: {json.dumps({'type': 'error', 'error': result.error_message})}\n\n"
            return
        
        final_text = result.content
        answer_content, reasoning_content = _parse_reasoning_tags(final_text)
        
        if enable_reasoning and reasoning_content:
            yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning_content})}\n\n"
        
        for chunk in chunk_text(answer_content, chunk_size=80):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.05)
        
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="user", content=message, provider="local"))
        memory_service.create_message(db, memory_schemas.MessageCreate(
            project_id=project_id, role="assistant", content=answer_content,
            provider=provider, model=model, reasoning=reasoning_content or None))
        
        yield f"data: {json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(answer_content)})}\n\n"
        
    except Exception as e:
        logger.exception("[stream] Critique pipeline failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"


async def generate_sse_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    db: Session,
    enable_reasoning: bool = False,
):
    """
    Generate SSE stream for normal jobs.
    
    v0.13.9: Fixed event handling to support canonical dict events.
    Defensively normalizes legacy string tokens for backward compatibility.
    """
    yield f"data: {json.dumps({'type': 'metadata', 'provider': provider, 'model': model})}\n\n"

    accumulated = ""
    reasoning_content = ""
    current_provider = provider
    current_model = model

    try:
        async for event in stream_llm(provider=provider, model=model, messages=messages, system_prompt=system_prompt):
            # v0.13.9: Defensive normalization for legacy string tokens
            if isinstance(event, str):
                # Legacy: raw string token -> normalize to dict
                event = {"type": "token", "text": event}
                logger.debug("[stream] Normalized legacy string token to dict event")
            
            if not isinstance(event, dict):
                logger.warning(f"[stream] Unknown event type from provider: {event!r}")
                continue

            event_type = event.get("type")

            if event_type == "token":
                # v0.13.9: Changed from "content" to "text" field
                content = event.get("text", "")
                accumulated += content
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            elif event_type == "reasoning" and enable_reasoning:
                # v0.13.9: Changed from "content" to "text" field
                reasoning_content = event.get("text", "")
                yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning_content})}\n\n"

            elif event_type == "metadata":
                current_provider = event.get("provider", provider)
                current_model = event.get("model", model)
                logger.debug(f"[stream] Metadata event: provider={current_provider}, model={current_model}")

            elif event_type == "error":
                # v0.13.9: Support both "message" and "error" fields
                error_msg = event.get("message") or event.get("error", "Unknown error")
                logger.error(f"[stream] Error event from provider: {error_msg}")
                yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
                return
            
            elif event_type == "done":
                # Optional done event from provider (not currently used)
                logger.debug("[stream] Done event received from provider")
                pass
            
            else:
                logger.warning(f"[stream] Unknown event type '{event_type}' from provider: {event}")

    except Exception as e:
        logger.exception("[stream] Stream failed: %s", e)
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    answer_content, extracted_reasoning = _parse_reasoning_tags(accumulated)
    if extracted_reasoning:
        reasoning_content = extracted_reasoning

    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id, role="user", content=message, provider="local"))
    memory_service.create_message(db, memory_schemas.MessageCreate(
        project_id=project_id, role="assistant", content=answer_content,
        provider=current_provider, model=current_model, reasoning=reasoning_content or None))

    yield f"data: {json.dumps({'type': 'done', 'provider': current_provider, 'model': current_model, 'total_length': len(answer_content)})}\n\n"


@router.post("/chat")
async def stream_chat(
    req: StreamChatRequest,
    db: Session = Depends(get_db),
    auth: AuthResult = Depends(require_auth),
):
    """Stream chat response using SSE."""
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")

    context_block = _build_context_block(db, req.project_id)
    semantic_context = _get_semantic_context(db, req.project_id, req.message) if req.use_semantic_search else ""
    doc_context = _build_document_context(db, req.project_id)
    
    full_context = ""
    if context_block:
        full_context += context_block + "\n\n"
    if semantic_context:
        full_context += semantic_context + "\n\n"
    if doc_context:
        full_context += "=== UPLOADED DOCUMENTS ===\n" + doc_context
    
    if full_context:
        print(f"[stream_chat] Built context: {len(full_context)} chars")

    if req.provider and req.model:
        provider = req.provider
        model = req.model
        print(f"[stream_chat] Using explicit provider/model: {provider}/{model}")
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        print(f"[stream_chat] Using explicit provider with default model: {provider}/{model}")
    else:
        job_type = _classify_job_type(req.message, req.job_type or "")
        provider, model = _select_provider_for_job_type(job_type)
        print(f"[stream_chat] Job-type routing: {job_type.value} -> {provider}/{model}")

    available = get_available_streaming_provider()
    if not available:
        raise HTTPException(status_code=503, detail="No LLM provider available")
    
    from .streaming import get_available_streaming_providers
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        print(f"[stream_chat] Provider {provider} not available, falling back to {available}")
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])

    messages: List[dict] = []
    if req.include_history:
        history = memory_service.list_messages(db, req.project_id, limit=req.history_limit)
        messages = [{"role": msg.role, "content": msg.content} for msg in history]
    messages.append({"role": "user", "content": req.message})

    system_prompt = f"Project: {project.name}."
    if project.description:
        system_prompt += f" {project.description}"
    
    if full_context:
        system_prompt += f"""

You have access to the following context from this project:

{full_context}

Use this context to answer the user's questions. If asked about people, documents, 
or information that appears in the context above, use that information to respond.
Do NOT claim you don't have information if it's present in the context."""

    print(f"[stream_chat] Starting stream: provider={provider}, model={model}, context_len={len(full_context)}")

    job_type_value = job_type.value if 'job_type' in locals() else None
    
    if (job_type_value and provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value)):
        print(f"[stream] High-stakes job '{job_type_value}' → critique pipeline → fake-streaming")
        
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id, message=req.message, provider=provider, model=model,
                system_prompt=system_prompt, messages=messages, full_context=full_context,
                job_type_str=job_type_value, db=db, enable_reasoning=req.enable_reasoning),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    
    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id, message=req.message, provider=provider, model=model,
            system_prompt=system_prompt, messages=messages, db=db, enable_reasoning=req.enable_reasoning),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/providers")
async def list_streaming_providers(db: Session = Depends(get_db), auth: AuthResult = Depends(require_auth)):
    """List available providers for streaming."""
    from .streaming import HAS_OPENAI, HAS_ANTHROPIC, HAS_GEMINI

    providers = {}

    if HAS_OPENAI and os.getenv("OPENAI_API_KEY"):
        providers["openai"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["openai"],
            "models": ["gpt-4.1-mini", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        }

    if HAS_ANTHROPIC and os.getenv("ANTHROPIC_API_KEY"):
        providers["anthropic"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["anthropic"],
            "models": ["claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022"],
        }

    if HAS_GEMINI and os.getenv("GOOGLE_API_KEY"):
        providers["gemini"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["gemini"],
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        }

    return {"providers": providers, "default": get_available_streaming_provider()}