# FILE: app/llm/stream_router.py
"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

v0.15.3: OPTIONAL IMPROVEMENT - PROVIDER METADATA EVENT HANDLING
- Added: explicit handling for provider event type "metadata" (captures provider/model/usage, no warning spam)
- Added: explicit handling for event type "usage" (captures usage without requiring terminal done)
- Change: treat "done" as terminal (break loop) after capturing final provider/model/usage

v0.15.2: AUDIT/TELEMETRY INSTRUMENTATION FOR STREAMING
- Added: AuditLogger trace + routing/model-call events for /stream/chat
- Added: TRACE_END for normal completion, errors, and client disconnects
- Ensures: /telemetry/* reflects streaming traffic as well as /chat

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
import uuid
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
from app.llm.schemas import JobType, RoutingConfig, LLMTask

from .streaming import stream_llm, get_available_streaming_provider

# Audit/telemetry
from app.llm.audit_logger import get_audit_logger, RoutingTrace

# Critique pipeline imports
from app.llm.router import (
    run_high_stakes_with_critique,
    synthesize_envelope_from_task,
    is_high_stakes_job,
    is_opus_model,
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
    thinking_match = re.search(r"<THINKING>([\s\S]*?)</THINKING>", raw, re.IGNORECASE)
    answer_match = re.search(r"<ANSWER>([\s\S]*?)</ANSWER>", raw, re.IGNORECASE)

    if thinking_match and answer_match:
        reasoning = thinking_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return answer, reasoning

    cleaned = re.sub(r"</?THINKING[^>]*>", "", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"</?ANSWER[^>]*>", "", cleaned, flags=re.IGNORECASE)
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

        recent_docs = (
            db.query(DocumentContent)
            .filter(DocumentContent.project_id == project_id)
            .order_by(DocumentContent.created_at.desc())
            .limit(5)
            .all()
        )

        if not recent_docs:
            return ""

        context_parts = []
        for doc in recent_docs:
            summary = doc.summary[:500] if doc.summary else ""
            raw_preview = doc.raw_text[:1000] if doc.raw_text else ""
            if summary or raw_preview:
                context_parts.append(
                    f"[{doc.filename}]:\nSummary: {summary}\nContent: {raw_preview}..."
                )

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
        "security review",
        "security audit",
        "security assessment",
        "penetration test",
        "pentest",
        "threat model",
        "threat modeling",
        "vulnerability",
        "vulnerabilities",
        "vulnerability assessment",
        "exploit",
        "attack vector",
        "attack surface",
        "sql injection",
        "xss",
        "csrf",
        "authentication bypass",
        "privilege escalation",
        "session fixation",
        "session hijacking",
        "security analysis",
        "security check",
        "encryption review",
        "key management",
        "secrets management",
        "authentication security",
        "authorization security",
        "security hardening",
        "security posture",
    ]

    arch_keywords = [
        "architect",
        "architecture",
        "design a system",
        "system design",
        "microservice",
        "micro-service",
        "infrastructure",
        "infra",
        "scalab",
        "database schema",
        "db schema",
        "api design",
        "high-level design",
        "hld",
        "distributed system",
        "design pattern",
        "tech stack",
    ]

    review_keywords = [
        "review this",
        "review my",
        "code review",
        "check this code",
        "find bugs",
        "audit this",
        "critique",
        "what's wrong with",
    ]

    code_keywords = [
        "write a function",
        "write code",
        "implement",
        "debug",
        "fix this code",
        "refactor",
        "def ",
        "function ",
        "```",
    ]

    language_keywords = [
        "python",
        "javascript",
        "typescript",
        "java",
        "c++",
        "rust",
        "react",
        "vue",
        "fastapi",
        "django",
    ]

    # PRIORITY 1: Security (explicit terms only)
    if any(kw in msg_lower for kw in security_keywords):
        print("[stream_router] Classified: SECURITY_REVIEW (explicit security keyword)")
        return JobType.SECURITY_REVIEW

    # PRIORITY 2: Architecture
    if any(kw in msg_lower for kw in arch_keywords):
        print("[stream_router] Classified: ARCHITECTURE_DESIGN")
        return JobType.ARCHITECTURE_DESIGN

    # PRIORITY 3: Code review
    if any(kw in msg_lower for kw in review_keywords):
        print("[stream_router] Classified: CODE_REVIEW")
        return JobType.CODE_REVIEW

    # PRIORITY 4: Code tasks
    is_code_related = any(kw in msg_lower for kw in code_keywords) or any(kw in msg_lower for kw in language_keywords)

    if is_code_related:
        complex_indicators = ["complex", "full file", "entire file", "production"]
        if any(x in msg_lower for x in complex_indicators):
            print("[stream_router] Classified: COMPLEX_CODE_CHANGE")
            return JobType.COMPLEX_CODE_CHANGE
        print("[stream_router] Classified: SIMPLE_CODE_CHANGE")
        return JobType.SIMPLE_CODE_CHANGE

    print("[stream_router] Classified: CASUAL_CHAT (default)")
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

    if job_type == JobType.DEEP_RESEARCH:
        return ("gemini", DEFAULT_MODELS["gemini"])

    provider_key = os.getenv("ORB_DEFAULT_PROVIDER", "anthropic")
    return (provider_key, DEFAULT_MODELS.get(provider_key, DEFAULT_MODELS["anthropic"]))


class StreamChatRequest(BaseModel):
    project_id: int
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None
    job_type: Optional[str] = None
    requested_type: Optional[str] = None
    include_history: bool = True
    history_limit: int = 20
    use_semantic_search: bool = False
    enable_reasoning: bool = False


def _make_session_id(auth: AuthResult) -> str:
    """Best-effort session id for audit correlation."""
    # AuthResult shape can vary; be defensive.
    for attr in ("session_id", "sid", "session", "session_token"):
        try:
            v = getattr(auth, attr, None)
            if v:
                return str(v)
        except Exception:
            pass
    try:
        user = getattr(auth, "user", None)
        if isinstance(user, dict):
            return str(user.get("id") or user.get("email") or user.get("username") or "")
    except Exception:
        pass
    return f"legacy-{uuid.uuid4()}"


def _coerce_int(v) -> int:
    try:
        if v is None:
            return 0
        return int(v)
    except Exception:
        return 0


def _extract_usage_tokens(usage_obj) -> tuple[int, int]:
    """Best-effort extraction of (prompt_tokens, completion_tokens) from a usage object."""
    if usage_obj is None:
        return (0, 0)

    # Common dict form: {prompt_tokens, completion_tokens, total_tokens}
    if isinstance(usage_obj, dict):
        pt = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or usage_obj.get("prompt")
        ct = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or usage_obj.get("completion")
        return (_coerce_int(pt), _coerce_int(ct))

    # Object form: .prompt_tokens/.completion_tokens OR .input_tokens/.output_tokens
    pt = getattr(usage_obj, "prompt_tokens", None)
    ct = getattr(usage_obj, "completion_tokens", None)
    if pt is None:
        pt = getattr(usage_obj, "input_tokens", None)
    if ct is None:
        ct = getattr(usage_obj, "output_tokens", None)
    return (_coerce_int(pt), _coerce_int(ct))


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
    trace: Optional[RoutingTrace] = None,
    enable_reasoning: bool = False,
):
    """
    Fake-streaming response generator for high-stakes critique pipeline.

    The critique pipeline requires non-streaming calls to:
    1. Get primary response from Opus
    2. Get critique from Sonnet
    3. Synthesize final response

    This function simulates streaming by chunking the final response.
    """
    import asyncio

    # Start timing for audit/telemetry
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    try:
        # Build task for critique pipeline
        task = LLMTask(
            project_id=project_id,
            user_message=message,
            system_prompt=system_prompt,
            messages=messages,
            full_context=full_context,
            job_type=job_type_str,
            enable_reasoning=enable_reasoning,
        )

        # Run critique pipeline (non-streaming)
        result = await run_high_stakes_with_critique(
            task=task,
            provider_id=provider,
            model_id=model,
            envelope=synthesize_envelope_from_task(task),
            job_type_str=job_type_str,
        )

        # Best-effort token usage extraction (critique pipeline result shape can vary)
        usage_obj = getattr(result, "usage", None)
        if usage_obj is None and isinstance(result, dict):
            usage_obj = result.get("usage")
        hs_prompt_tokens, hs_completion_tokens = _extract_usage_tokens(usage_obj)

        duration_ms = max(0, int(loop.time() * 1000) - started_ms)

        # Defensive error extraction
        error_message = getattr(result, "error_message", None)
        content = getattr(result, "content", None)
        if isinstance(result, dict):
            error_message = error_message or result.get("error_message") or result.get("error")
            content = content if content is not None else result.get("content")

        if error_message:
            if trace and not trace_finished:
                trace.log_model_call(
                    "primary",
                    provider,
                    model,
                    "primary",
                    hs_prompt_tokens,
                    hs_completion_tokens,
                    duration_ms,
                    success=False,
                    error=str(error_message),
                )
                trace.finalize(success=False, error_message=str(error_message))
                trace_finished = True
            yield f"data: {json.dumps({'type': 'error', 'error': str(error_message)})}\n\n"
            return

        content = content or ""

        # Parse reasoning tags if present
        final_answer, reasoning = _parse_reasoning_tags(content)

        # Emit audit/telemetry model + trace end
        if trace and not trace_finished:
            trace.log_model_call(
                "primary",
                provider,
                model,
                "primary",
                hs_prompt_tokens,
                hs_completion_tokens,
                duration_ms,
                success=True,
            )

        # Chunk response for fake streaming
        chunk_size = 50
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i:i + chunk_size]
            yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            await asyncio.sleep(0.01)  # simulate delay

        # Emit reasoning if requested
        if enable_reasoning and reasoning:
            yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"

        # Persist messages
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"),
        )
        memory_service.create_message(
            db,
            memory_schemas.MessageCreate(
                project_id=project_id,
                role="assistant",
                content=final_answer,
                provider=provider,
                model=model,
                reasoning=reasoning if reasoning else None,
            ),
        )

        # Final trace end
        if trace and not trace_finished:
            trace.finalize(success=True)
            trace_finished = True

        yield f"data: {json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(final_answer)})}\n\n"

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        logger.exception("[stream] High-stakes stream failed: %s", e)
        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", 0, 0, 0, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return


async def generate_sse_stream(
    project_id: int,
    message: str,
    provider: str,
    model: str,
    system_prompt: str,
    messages: List[dict],
    db: Session,
    trace: Optional[RoutingTrace] = None,
    enable_reasoning: bool = False,
):
    """
    Stream response via Server-Sent Events.

    v0.13.9: Normalizes streaming event format for all providers.
    v0.15.3: Handles provider "metadata" + "usage" events; treats "done" as terminal.
    """
    import asyncio

    accumulated = ""
    reasoning_content = ""
    current_provider = provider
    current_model = model

    final_usage = None  # set when provider emits usage/terminal events

    # For audit/telemetry duration
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    try:
        async for event in stream_llm(
            provider=provider,
            model=model,
            messages=messages,
            system_prompt=system_prompt,
        ):
            if isinstance(event, str):
                event = {"type": "token", "content": event}

            # Defensive: if a provider yields non-dict, coerce
            if not isinstance(event, dict):
                event = {"type": "token", "content": str(event)}

            event_type = event.get("type", "token")

            if event_type == "token":
                content = event.get("content") or event.get("text") or ""
                if content:
                    accumulated += content
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

            elif event_type == "reasoning":
                content = event.get("content") or ""
                if content:
                    reasoning_content += content
                    if enable_reasoning:
                        yield f"data: {json.dumps({'type': 'reasoning', 'content': content})}\n\n"

            elif event_type == "metadata":
                # Optional improvement: capture provider/model without warning spam
                maybe_provider = event.get("provider")
                maybe_model = event.get("model")
                if maybe_provider:
                    current_provider = str(maybe_provider)
                if maybe_model:
                    current_model = str(maybe_model)

                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u

                # Only forward metadata to client when debugging
                if os.getenv("ORB_ROUTER_DEBUG") == "1":
                    yield f"data: {json.dumps({'type': 'metadata', 'provider': current_provider, 'model': current_model})}\n\n"

            elif event_type == "usage":
                # Some providers may emit usage separately
                u = event.get("usage") or event.get("data")
                if isinstance(u, dict):
                    final_usage = u

            elif event_type == "error":
                error_msg = event.get("message") or event.get("error") or "Unknown error"
                logger.error(f"[stream] Provider error: {error_msg}")
                duration_ms = max(0, int(loop.time() * 1000) - started_ms)
                if trace and not trace_finished:
                    trace.log_model_call(
                        "primary",
                        current_provider,
                        current_model,
                        "primary",
                        0,
                        0,
                        duration_ms,
                        success=False,
                        error=str(error_msg),
                    )
                    trace.finalize(success=False, error_message=str(error_msg))
                    trace_finished = True
                yield f"data: {json.dumps({'type': 'error', 'error': str(error_msg)})}\n\n"
                return

            elif event_type == "done":
                # Provider terminal event. Capture usage/metadata if present, then finish.
                maybe_provider = event.get("provider")
                maybe_model = event.get("model")
                if maybe_provider:
                    current_provider = str(maybe_provider)
                if maybe_model:
                    current_model = str(maybe_model)

                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u

                logger.debug("[stream] Done event received from provider")
                break

            else:
                # Keep this warning (unknown future provider events are useful to see)
                logger.warning(f"[stream] Unknown event type '{event_type}' from provider: {event}")

    except asyncio.CancelledError:
        if trace and not trace_finished:
            trace.log_warning("STREAM", "client_disconnect")
            trace.finalize(success=False, error_message="client_disconnect")
            trace_finished = True
        raise
    except Exception as e:
        logger.exception("[stream] Stream failed: %s", e)
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)
        if trace and not trace_finished:
            trace.log_model_call("primary", current_provider, current_model, "primary", 0, 0, duration_ms, success=False, error=str(e))
            trace.finalize(success=False, error_message=str(e))
            trace_finished = True
        yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        return

    duration_ms = max(0, int(loop.time() * 1000) - started_ms)

    answer_content, extracted_reasoning = _parse_reasoning_tags(accumulated)
    if extracted_reasoning:
        reasoning_content = extracted_reasoning

    # Persist messages
    memory_service.create_message(
        db,
        memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"),
    )
    memory_service.create_message(
        db,
        memory_schemas.MessageCreate(
            project_id=project_id,
            role="assistant",
            content=answer_content,
            provider=current_provider,
            model=current_model,
            reasoning=reasoning_content or None,
        ),
    )

    # Emit audit/telemetry model + trace end (token counts best-effort from provider usage)
    if trace and not trace_finished:
        prompt_tokens, completion_tokens = _extract_usage_tokens(final_usage)
        trace.log_model_call(
            "primary",
            current_provider,
            current_model,
            "primary",
            prompt_tokens,
            completion_tokens,
            duration_ms,
            success=True,
        )
        trace.finalize(success=True)
        trace_finished = True

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

    # --- audit trace ---
    audit = get_audit_logger()
    trace: Optional[RoutingTrace] = None
    request_id = str(uuid.uuid4())
    if audit:
        trace = audit.start_trace(
            session_id=_make_session_id(auth),
            project_id=req.project_id,
            user_text=req.message,
            request_id=request_id,
        )

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

    # Always resolve a job_type (even if provider/model explicitly provided)
    job_type = _classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value

    # Decide provider/model
    if req.provider and req.model:
        provider = req.provider
        model = req.model
        routing_reason = "Explicit provider+model from request"
        print(f"[stream_chat] Using explicit provider/model: {provider}/{model}")
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = "Explicit provider from request (default model)"
        print(f"[stream_chat] Using explicit provider with default model: {provider}/{model}")
    else:
        provider, model = _select_provider_for_job_type(job_type)
        routing_reason = f"Job-type routing: {job_type_value} -> {provider}/{model}"
        print(f"[stream_chat] {routing_reason}")

    # Provider availability / fallback
    available = get_available_streaming_provider()
    if not available:
        if trace:
            trace.log_error("STREAM", "no_provider_available")
            trace.finalize(success=False, error_message="No LLM provider available")
        raise HTTPException(status_code=503, detail="No LLM provider available")

    from .streaming import get_available_streaming_providers

    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        print(f"[stream_chat] Provider {provider} not available, falling back to {available}")
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = f"{routing_reason} | fallback_to={provider}/{model}"

    # Emit request/routing audit events (streaming previously had none)
    if trace:
        trace.log_request_start(
            job_type=req.job_type or "",
            resolved_job_type=job_type_value,
            provider=provider,
            model=model,
            reason=routing_reason,
            frontier_override=False,
            file_map_injected=False,
            attachments=None,
        )
        trace.log_routing_decision(
            job_type=job_type_value,
            provider=provider,
            model=model,
            reason=routing_reason,
            frontier_override=False,
            file_map_injected=False,
        )

    # Build message list
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

    # High-stakes: route to critique pipeline (fake streaming)
    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
        print(f"[stream] High-stakes job '{job_type_value}' → critique pipeline → fake-streaming")
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str=job_type_value,
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id,
            message=req.message,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            db=db,
            trace=trace,
            enable_reasoning=req.enable_reasoning,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
            "models": [
                "claude-sonnet-4-5-20250929",
                "claude-sonnet-4-20250514",
                "claude-3-5-sonnet-20241022",
            ],
        }

    if HAS_GEMINI and os.getenv("GOOGLE_API_KEY"):
        providers["gemini"] = {
            "available": True,
            "default_model": DEFAULT_MODELS["gemini"],
            "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        }

    return {"providers": providers, "default": get_available_streaming_provider()}
