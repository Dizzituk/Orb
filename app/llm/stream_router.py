# FILE: app/llm/stream_router.py
r"""
Streaming endpoints for real-time LLM responses.
Uses Server-Sent Events (SSE).

LOCAL TOOL TRIGGER - ZOBIE MAP
- Runs mapper script locally (host) against sandbox controller API
- Produces evidence artifacts in ORB_ZOBIE_MAPPER_OUT_DIR

LOCAL TOOL TRIGGER - CREATE ARCHITECTURE MAP
- Runs mapper script locally (host) against sandbox controller API
- Generates a detailed architecture map *in sections* to avoid context overflows
- Writes section files + a stitched full file into ORB_ZOBIE_MAPPER_OUT_DIR
- Only streams short progress + final summary to the UI (no giant map dump)
- Optionally ingests section docs + full doc into Memory as DocumentContent

NOTE ABOUT OPENAI MODELS:
- Some newer OpenAI models reject 'max_tokens' and require 'max_completion_tokens'.
- To avoid provider-wrapper drift breaking this tool, the archmap tool includes a direct
  OpenAI call path that uses 'max_completion_tokens' when provider == "openai".
"""

import os
import re
import json
import uuid
import sys
import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

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
    # NOTE: stream_router.py reads OPENAI_DEFAULT_MODEL for DEFAULT_MODELS["openai"]
    "openai": os.getenv("OPENAI_DEFAULT_MODEL", "gpt-5-mini"),
    "anthropic": os.getenv("ANTHROPIC_SONNET_MODEL", "claude-sonnet-4-5-20250929"),
    "anthropic_opus": os.getenv("ANTHROPIC_OPUS_MODEL", "claude-opus-4-5-20251101"),
    "gemini": os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.0-flash"),
}

# =============================================================================
# LOCAL TOOL: ZOBIE MAP (host-side mapper -> sandbox controller)
# =============================================================================

_ZOBIE_TRIGGER_SET = {"zobie map", "zombie map", "zobie_map", "/zobie_map", "/zombie_map"}

ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL", "http://192.168.250.2:8765")
ZOBIE_MAPPER_SCRIPT = os.getenv("ORB_ZOBIE_MAPPER_SCRIPT", r"D:\tools\zobie_mapper\zobie_map.py")
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR", r"D:\tools\zobie_mapper\out")
ZOBIE_MAPPER_TIMEOUT_SEC = int(os.getenv("ORB_ZOBIE_MAPPER_TIMEOUT_SEC", "300"))

# OPTIONAL: extra mapper args (lets you run your “full” mode without changing code)
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "").strip()
ZOBIE_MAPPER_ARGS = ZOBIE_MAPPER_ARGS_RAW.split() if ZOBIE_MAPPER_ARGS_RAW else []

# =============================================================================
# LOCAL TOOL: CREATE ARCHITECTURE MAP (mapper -> evidence -> LLM -> files)
# =============================================================================

# LOCAL TOOL: CREATE ARCHITECTURE MAP (mapper -> evidence -> LLM -> files)
from app.llm.local_tools.archmap_helpers import (
    _ARCHMAP_TRIGGER_SET,
    ARCHMAP_PROVIDER,
    ARCHMAP_MODEL,
    ARCHMAP_TEMPLATE_PATH,
    ARCHMAP_MAX_FILES,
    ARCHMAP_MAX_CODE_FILES,
    ARCHMAP_MAX_CHARS_PER_FILE,
    ARCHMAP_MAX_ARTIFACT_CHARS,
    ARCHMAP_CONTROLLER_TIMEOUT_SEC,
    ARCHMAP_STREAM_TOKENS_TO_CLIENT,
    ARCHMAP_SECTION_MODE,
    ARCHMAP_SECTION_MAX_SCANNED_FILES,
    ARCHMAP_SECTION_MAX_CHARS,
    ARCHMAP_OPENAI_URL,
    ARCHMAP_OPENAI_TIMEOUT_SEC,
    ARCHMAP_OPENAI_MAX_COMPLETION_TOKENS,
    ARCHMAP_OPENAI_TEMPERATURE,
    ARCHMAP_INGEST_TO_MEMORY,
    ARCHMAP_DOC_RAW_MAX_CHARS,
    _DENY_FILE_PATTERNS,
    _is_denied_repo_path,
    _safe_read_text,
    _line_number,
    _extract_mapper_stamp,
    _next_archmap_version,
    _controller_http_json,
    _controller_fetch_file_content,
    _select_archmap_code_files,
    _ingest_generated_markdown_to_memory,
    _load_index_json,
    _fmt_scanned_file,
    _section_plan,
    _pick_scanned_files_for_section,
    _openai_chat_completion_nonstream,
)

# Local-tool streaming helpers (kept out of this file to keep it small)
from app.llm.local_tools.zobie_tools import (
    generate_local_architecture_map_stream,
    generate_local_zobie_map_stream,
)




def _is_zobie_map_trigger(msg: str) -> bool:
    s = (msg or "").strip().lower()
    return s in _ZOBIE_TRIGGER_SET


def _is_archmap_trigger(msg: str) -> bool:
    s = (msg or "").strip().lower()
    return s in _ARCHMAP_TRIGGER_SET


def _chunk_text(s: str, chunk_size: int = 120) -> List[str]:
    if not s:
        return []
    return [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]


def _cap_text(label: str, text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n<<truncated {label}: {len(text)} chars total>>\n"


def _parse_reasoning_tags(raw: str) -> Tuple[str, str]:
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


def _make_session_id(auth: AuthResult) -> str:
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


def _extract_usage_tokens(usage_obj) -> Tuple[int, int]:
    if usage_obj is None:
        return (0, 0)
    if isinstance(usage_obj, dict):
        pt = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or usage_obj.get("prompt")
        ct = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or usage_obj.get("completion")
        return (_coerce_int(pt), _coerce_int(ct))
    pt = getattr(usage_obj, "prompt_tokens", None) or getattr(usage_obj, "input_tokens", None)
    ct = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None)
    return (_coerce_int(pt), _coerce_int(ct))


def _build_context_block(db: Session, project_id: int) -> str:
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
                context_parts.append(f"[{doc.filename}]:\nSummary: {summary}\nContent: {raw_preview}...")

        return "\n\n".join(context_parts)
    except Exception as e:
        print(f"[stream_router] Error building document context: {e}")
        return ""


def _get_semantic_context(db: Session, project_id: int, query: str) -> str:
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
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            pass

    msg_lower = message.lower()
    print(f"[stream_router] Classifying message (first 200 chars): {repr(message[:200])}")

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

    if any(kw in msg_lower for kw in security_keywords):
        print("[stream_router] Classified: SECURITY_REVIEW (explicit security keyword)")
        return JobType.SECURITY_REVIEW

    if any(kw in msg_lower for kw in arch_keywords):
        print("[stream_router] Classified: ARCHITECTURE_DESIGN")
        return JobType.ARCHITECTURE_DESIGN

    if any(kw in msg_lower for kw in review_keywords):
        print("[stream_router] Classified: CODE_REVIEW")
        return JobType.CODE_REVIEW

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


def _select_provider_for_job_type(job_type: JobType) -> Tuple[str, str]:
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
    return (provider_key, DEFAULT_MODELS.get(provider_key, DEFAULT_MODELS["openai"]))


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
    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    try:
        task = LLMTask(
            project_id=project_id,
            user_message=message,
            system_prompt=system_prompt,
            messages=messages,
            full_context=full_context,
            job_type=job_type_str,
            enable_reasoning=enable_reasoning,
        )

        envelope = synthesize_envelope_from_task(task)

        # Spec Gate (only stage allowed to ask user questions)
        if (job_type_str or "").strip().lower() == "architecture_design":
            spec_gate_provider = os.getenv("SPEC_GATE_PROVIDER", "openai")
            spec_gate_model = os.getenv("OPENAI_SPEC_GATE_MODEL", DEFAULT_MODELS["openai"])

            spec_id, spec_hash, open_questions = await run_spec_gate(
                db,
                job_id=envelope.job_id,
                user_intent=message,
                provider_id=spec_gate_provider,
                model_id=spec_gate_model,
                constraints_hint={"stability_accuracy": "high", "allowed_tools": "free_only"},
            )

            if open_questions:
                yield "data: " + json.dumps(
                    {
                        "type": "pause",
                        "pause_state": "needs_spec_clarification",
                        "job_id": envelope.job_id,
                        "spec_id": spec_id,
                        "spec_hash": spec_hash,
                        "open_questions": open_questions,
                        "artifacts_written": f"jobs/{envelope.job_id}/spec/ + jobs/{envelope.job_id}/ledger/",
                    }
                ) + "\n\n"
                return

            system_prompt = system_prompt + (
                "\n\nYou MUST echo these identifiers at the TOP of your response exactly:\n"
                f"SPEC_ID: {spec_id}\nSPEC_HASH: {spec_hash}\n"
                "Do not ask the user any questions."
            )

        result = await run_high_stakes_with_critique(
            task=task,
            provider_id=provider,
            model_id=model,
            envelope=envelope,
            job_type_str=job_type_str,
        )

        usage_obj = getattr(result, "usage", None)
        if usage_obj is None and isinstance(result, dict):
            usage_obj = result.get("usage")
        hs_prompt_tokens, hs_completion_tokens = _extract_usage_tokens(usage_obj)
        duration_ms = max(0, int(loop.time() * 1000) - started_ms)

        error_message = getattr(result, "error_message", None)
        content = getattr(result, "content", None)
        if isinstance(result, dict):
            error_message = error_message or result.get("error_message") or result.get("error")
            content = content if content is not None else result.get("content")

        if error_message:
            if trace and not trace_finished:
                trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=False, error=str(error_message))
                trace.finalize(success=False, error_message=str(error_message))
                trace_finished = True
            yield "data: " + json.dumps({'type': 'error', 'error': str(error_message)}) + "\n\n"
            return

        content = content or ""
        final_answer, reasoning = _parse_reasoning_tags(content)

        # Hard enforcement: downstream stages must not ask user questions
        if (job_type_str or "").strip().lower() == "architecture_design":
            if detect_user_questions(final_answer or ""):
                from app.pot_spec.ledger import append_event
                from app.pot_spec.service import get_job_artifact_root

                job_root = get_job_artifact_root()
                append_event(
                    job_artifact_root=job_root,
                    job_id=envelope.job_id,
                    event={
                        "event": "POLICY_VIOLATION_STAGE_ASKED_QUESTIONS",
                        "job_id": envelope.job_id,
                        "stage": "HIGH_STAKES_PRIMARY",
                        "status": "rejected",
                    },
                )

                spec_gate_provider = os.getenv("SPEC_GATE_PROVIDER", "openai")
                spec_gate_model = os.getenv("OPENAI_SPEC_GATE_MODEL", DEFAULT_MODELS["openai"])

                spec_id, spec_hash, open_questions = await run_spec_gate(
                    db,
                    job_id=envelope.job_id,
                    user_intent=message,
                    provider_id=spec_gate_provider,
                    model_id=spec_gate_model,
                    reroute_reason="Downstream stage asked the user questions. Only Spec Gate may ask questions.",
                    downstream_output_excerpt=(final_answer or "")[:2000],
                )

                yield "data: " + json.dumps(
                    {
                        "type": "pause",
                        "pause_state": "needs_spec_clarification",
                        "job_id": envelope.job_id,
                        "spec_id": spec_id,
                        "spec_hash": spec_hash,
                        "open_questions": open_questions,
                        "artifacts_written": f"jobs/{envelope.job_id}/spec/ + jobs/{envelope.job_id}/ledger/",
                    }
                ) + "\n\n"
                return

        if trace and not trace_finished:
            trace.log_model_call("primary", provider, model, "primary", hs_prompt_tokens, hs_completion_tokens, duration_ms, success=True)

        chunk_size = 50
        for i in range(0, len(final_answer), chunk_size):
            chunk = final_answer[i : i + chunk_size]
            yield "data: " + json.dumps({'type': 'token', 'content': chunk}) + "\n\n"
            await asyncio.sleep(0.01)

        if enable_reasoning and reasoning:
            yield "data: " + json.dumps({'type': 'reasoning', 'content': reasoning}) + "\n\n"

        memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
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

        if trace and not trace_finished:
            trace.finalize(success=True)
            trace_finished = True

        yield "data: " + json.dumps({'type': 'done', 'provider': provider, 'model': model, 'total_length': len(final_answer)}) + "\n\n"

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
        yield "data: " + json.dumps({'type': 'error', 'error': str(e)}) + "\n\n"
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
    accumulated = ""
    reasoning_content = ""
    current_provider = provider
    current_model = model
    final_usage = None

    loop = asyncio.get_event_loop()
    started_ms = int(loop.time() * 1000)
    trace_finished = False

    try:
        async for event in stream_llm(provider=provider, model=model, messages=messages, system_prompt=system_prompt):
            if isinstance(event, str):
                event = {"type": "token", "content": event}
            if not isinstance(event, dict):
                event = {"type": "token", "content": str(event)}

            event_type = event.get("type", "token")

            if event_type == "token":
                content = event.get("content") or event.get("text") or ""
                if content:
                    accumulated += content
                    yield "data: " + json.dumps({'type': 'token', 'content': content}) + "\n\n"

            elif event_type == "reasoning":
                content = event.get("content") or ""
                if content:
                    reasoning_content += content
                    if enable_reasoning:
                        yield "data: " + json.dumps({'type': 'reasoning', 'content': content}) + "\n\n"

            elif event_type == "metadata":
                maybe_provider = event.get("provider")
                maybe_model = event.get("model")
                if maybe_provider:
                    current_provider = str(maybe_provider)
                if maybe_model:
                    current_model = str(maybe_model)
                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u
                if os.getenv("ORB_ROUTER_DEBUG") == "1":
                    yield "data: " + json.dumps({'type': 'metadata', 'provider': current_provider, 'model': current_model}) + "\n\n"

            elif event_type == "usage":
                u = event.get("usage") or event.get("data")
                if isinstance(u, dict):
                    final_usage = u

            elif event_type == "error":
                error_msg = event.get("message") or event.get("error") or "Unknown error"
                logger.error(f"[stream] Provider error: {error_msg}")
                duration_ms = max(0, int(loop.time() * 1000) - started_ms)
                if trace and not trace_finished:
                    trace.log_model_call("primary", current_provider, current_model, "primary", 0, 0, duration_ms, success=False, error=str(error_msg))
                    trace.finalize(success=False, error_message=str(error_msg))
                    trace_finished = True
                yield "data: " + json.dumps({'type': 'error', 'error': str(error_msg)}) + "\n\n"
                return

            elif event_type == "done":
                maybe_provider = event.get("provider")
                maybe_model = event.get("model")
                if maybe_provider:
                    current_provider = str(maybe_provider)
                if maybe_model:
                    current_model = str(maybe_model)
                u = event.get("usage")
                if isinstance(u, dict):
                    final_usage = u
                break

            else:
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
        yield "data: " + json.dumps({'type': 'error', 'error': str(e)}) + "\n\n"
        return

    duration_ms = max(0, int(loop.time() * 1000) - started_ms)

    answer_content, extracted_reasoning = _parse_reasoning_tags(accumulated)
    if extracted_reasoning:
        reasoning_content = extracted_reasoning

    memory_service.create_message(db, memory_schemas.MessageCreate(project_id=project_id, role="user", content=message, provider="local"))
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

    if trace and not trace_finished:
        prompt_tokens, completion_tokens = _extract_usage_tokens(final_usage)
        trace.log_model_call("primary", current_provider, current_model, "primary", prompt_tokens, completion_tokens, duration_ms, success=True)
        trace.finalize(success=True)
        trace_finished = True

    yield "data: " + json.dumps({'type': 'done', 'provider': current_provider, 'model': current_model, 'total_length': len(answer_content)}) + "\n\n"


@router.post("/chat")
async def stream_chat(req: StreamChatRequest, db: Session = Depends(get_db), auth: AuthResult = Depends(require_auth)):
    project = memory_service.get_project(db, req.project_id)
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {req.project_id} not found")

    audit = get_audit_logger()
    trace: Optional[RoutingTrace] = None
    request_id = str(uuid.uuid4())
    if audit:
        trace = audit.start_trace(session_id=_make_session_id(auth), project_id=req.project_id, user_text=req.message, request_id=request_id)

    if _is_archmap_trigger(req.message):
        routing_reason = "Local tool trigger: CREATE ARCHITECTURE MAP"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.architecture_map", provider=ARCHMAP_PROVIDER, model=ARCHMAP_MODEL, reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if _is_zobie_map_trigger(req.message):
        routing_reason = "Local tool trigger: ZOBIE MAP"
        if trace:
            trace.log_request_start(job_type=req.job_type or "", resolved_job_type="local.zobie_map", provider="local", model="zobie_mapper", reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
            trace.log_routing_decision(job_type="local.zobie_map", provider="local", model="zobie_mapper", reason=routing_reason, frontier_override=False, file_map_injected=False)

        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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

    job_type = _classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value

    if req.provider and req.model:
        provider = req.provider
        model = req.model
        routing_reason = "Explicit provider+model from request"
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = "Explicit provider from request (default model)"
    else:
        provider, model = _select_provider_for_job_type(job_type)
        routing_reason = f"Job-type routing: {job_type_value} -> {provider}/{model}"

    available = get_available_streaming_provider()
    if not available:
        if trace:
            trace.log_error("STREAM", "no_provider_available")
            trace.finalize(success=False, error_message="No LLM provider available")
        raise HTTPException(status_code=503, detail="No LLM provider available")

    from .streaming import get_available_streaming_providers
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
        routing_reason = f"{routing_reason} | fallback_to={provider}/{model}"

    if trace:
        trace.log_request_start(job_type=req.job_type or "", resolved_job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False, attachments=None)
        trace.log_routing_decision(job_type=job_type_value, provider=provider, model=model, reason=routing_reason, frontier_override=False, file_map_injected=False)

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

    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
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
