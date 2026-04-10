# FILE: app/llm/routing/chat_routing.py
"""
Chat and normal routing handlers for stream routing.

v15.0 (2026-04-06): MAJOR REFACTOR — modularised into three files:
    - chat_intent_detection.py: All regex patterns + detector functions
    - chat_model_selection.py:  Sticky cache, complexity tiers, provider fallback
    - chat_routing.py (this):   Handler orchestration only

v1.1 (2026-03-03): Integrated Grounding Gate
v1.0 (2026-01-20): Extracted from stream_router.py for modularity.

This module provides:
- `handle_chat_mode()`       — Chat routing with full capability layer
- `handle_normal_routing()`  — Standard job-type routing
- `handle_legacy_triggers()` — Fallback for when translation layer unavailable
"""

from __future__ import annotations

import logging
from typing import Optional, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.memory import service as memory_service
from app.llm.pipeline.high_stakes import is_high_stakes_job, is_opus_model
from app.llm.streaming import get_available_streaming_provider, get_available_streaming_providers

from app.llm.stream_utils import (
    DEFAULT_MODELS,
    classify_job_type,
    select_provider_for_job_type,
)

from app.llm.legacy_triggers import (
    is_zobie_map_trigger,
    is_archmap_trigger,
    is_update_arch_trigger,
    is_introspection_trigger,
    is_sandbox_trigger,
)

from .handler_registry import (
    _LOCAL_TOOLS_AVAILABLE,
    _SANDBOX_AVAILABLE,
    _INTROSPECTION_AVAILABLE,
    _RAG_STREAM_AVAILABLE,
    generate_sse_stream,
    generate_sandbox_stream,
    generate_introspection_stream,
    generate_local_architecture_map_stream,
    generate_local_zobie_map_stream,
    generate_update_architecture_stream,
    generate_rag_query_stream,
    generate_high_stakes_critique_stream,
)

from .prompt_builders import (
    build_system_prompt,
    build_messages,
    build_full_context,
)

from .rag_fallback import is_architecture_query

# Intent detection (extracted module)
from .chat_intent_detection import (
    detect_build_deploy_intent as _detect_build_deploy_intent,
    detect_file_creation_intent as _detect_file_creation_intent,
    detect_image_gen_intent as _detect_image_gen_intent,
    detect_image_refinement as _detect_image_refinement,
    detect_codebase_exploration as _detect_codebase_exploration,
    is_builds_context as _is_builds_context,
    last_assistant_was_image as _last_assistant_was_image,
)

# Model selection (extracted module)
from .chat_model_selection import (
    select_chat_model,
    get_sticky_model as _get_sticky_model,
    set_sticky_model as _set_sticky_model,
    ensure_provider_available,
    infer_sticky_from_history as _infer_sticky_from_history,
)

# Grounding Gate (v1.1)
try:
    from app.grounding.chat_integration import run_grounding_sync
    _GROUNDING_AVAILABLE = True
except ImportError:
    _GROUNDING_AVAILABLE = False
    run_grounding_sync = None
    logging.warning("[chat_routing] Grounding gate not available")

logger = logging.getLogger(__name__)


# =============================================================================
# CHAT MODE HANDLER
# =============================================================================

def handle_chat_mode(
    req: Any,
    project: Any,
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """Handle CHAT mode — full capability layer with tool access.

    Orchestrates: message persistence → file upload handling → model selection
    → context building → grounding gate → tool injection → SSE stream.
    """
    print(f"[CHAT_MODE] Handling chat for project={req.project_id}, message={req.message[:50]}...")

    # ── 1. Persist user message ──
    _persist_user_message(req, db)

    # ── 2. Build base context ──
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)

    # ── 3. Handle file uploads ──
    synthetic_attachments, is_image_upload, is_video_upload, full_context = (
        _process_file_uploads(req, full_context)
    )

    # ── 4. Prepare codebase context gatherer ──
    try:
        from app.llm.routing.chat_codebase_reader import gather_codebase_context
        _codebase_gather_pending = True
    except ImportError:
        gather_codebase_context = None
        _codebase_gather_pending = False

    # ── 5. Select model ──
    provider, model, extras = select_chat_model(
        req, db,
        synthetic_attachments=synthetic_attachments,
        is_image_upload=is_image_upload,
        is_video_upload=is_video_upload,
    )

    # Image route — early return to image pipeline
    if extras.get("image_route"):
        from app.llm.image_router import generate_image_stream
        return StreamingResponse(
            generate_image_stream(project_id=req.project_id, message=req.message, db=db),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Confirmation gate — early return with confirmation SSE
    if extras.get("confirmation_sse"):
        return StreamingResponse(
            extras["confirmation_sse"],
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── 6. Build messages ──
    messages = _build_chat_messages(req, db)

    # ── 7. Inject UI context / tab data ──
    ui_ctx = getattr(req, 'ui_context', None)
    full_context = _inject_tab_data(ui_ctx, full_context, db)

    # ── 8. Gather codebase context ──
    codebase_ctx = ""
    if _codebase_gather_pending and gather_codebase_context is not None:
        try:
            codebase_ctx = gather_codebase_context(message=req.message, model=model, db=db)
            if codebase_ctx:
                full_context += f"\n\n{codebase_ctx}"
                print(f"[CHAT_MODE] Codebase context injected: {len(codebase_ctx)} chars")
        except Exception as e:
            print(f"[CHAT_MODE] Codebase context failed (non-fatal): {e}")

    # ── 9. Build system prompt ──
    system_prompt = build_system_prompt(project, full_context, ui_context=ui_ctx)

    # ── 10. Grounding gate ──
    system_prompt = _run_grounding_gate(req, system_prompt, label="CHAT_MODE")

    # ── 11. Tool injection ──
    _chat_tools, system_prompt = _inject_tools(
        provider, model, req, system_prompt,
        codebase_gather_pending=_codebase_gather_pending,
        codebase_ctx=codebase_ctx,
    )

    # ── 12. Inject image into last user message for Gemini vision ──
    if is_image_upload and provider == "google":
        messages = _inject_image_into_messages(
            messages, getattr(req, "file_upload_local_path", None),
            getattr(req, "file_upload_mime", None) or "",
        )

    if ui_ctx:
        print(f"[CHAT_MODE] UI context: view={ui_ctx.view_type}, job={ui_ctx.job_type}, label={ui_ctx.label}")

    print(f"[CHAT_MODE] Calling generate_sse_stream: provider={provider}, model={model}, messages={len(messages)}")

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
            tools=_chat_tools,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# NORMAL ROUTING HANDLER
# =============================================================================

def handle_normal_routing(
    req: Any,
    project: Any,
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """Handle normal job-type routing with RAG fallback."""

    # RAG fallback for architecture queries
    if _RAG_STREAM_AVAILABLE and is_architecture_query(req.message):
        print(f"[NORMAL_ROUTING] RAG fallback: detected architecture query")
        return StreamingResponse(
            generate_rag_query_stream(
                project_id=req.project_id, message=req.message, db=db, trace=trace,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)

    # Job continuation
    if req.continue_job_id and req.job_state == "needs_spec_clarification":
        provider = "anthropic"
        model = DEFAULT_MODELS["anthropic_opus"]
        messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
        system_prompt = build_system_prompt(project, full_context)
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id, message=req.message,
                provider=provider, model=model,
                system_prompt=system_prompt, messages=messages,
                full_context=full_context, job_type_str="architecture_design",
                db=db, trace=trace, enable_reasoning=req.enable_reasoning,
                continue_job_id=req.continue_job_id,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # Normal job classification
    job_type = classify_job_type(req.message, req.job_type or "")
    job_type_value = job_type.value

    # Determine provider/model
    if req.provider and req.model:
        provider, model = req.provider, req.model
    elif req.provider:
        provider = req.provider
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    else:
        provider, model = select_provider_for_job_type(job_type)

    provider, model = ensure_provider_available(provider, model)

    messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)

    # Codebase context for trusted models
    _nr_codebase_ctx = ""
    try:
        from app.llm.routing.chat_codebase_reader import gather_codebase_context, is_trusted_model
        if is_trusted_model(model):
            _nr_codebase_ctx = gather_codebase_context(message=req.message, model=model, db=db)
            if _nr_codebase_ctx:
                full_context += f"\n\n{_nr_codebase_ctx}"
                print(f"[NORMAL_ROUTING] Codebase context injected: {len(_nr_codebase_ctx)} chars")
    except Exception as e:
        print(f"[NORMAL_ROUTING] Codebase context failed (non-fatal): {e}")

    system_prompt = build_system_prompt(project, full_context)

    # Grounding gate
    system_prompt = _run_grounding_gate(req, system_prompt, label="NORMAL_ROUTING")

    # Tool injection (same logic as chat mode)
    _nr_tools, system_prompt = _inject_tools(
        provider, model, req, system_prompt,
        codebase_gather_pending=bool(_nr_codebase_ctx),
        codebase_ctx=_nr_codebase_ctx,
    )

    # High-stakes routing
    if provider == "anthropic" and is_opus_model(model) and is_high_stakes_job(job_type_value):
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id, message=req.message,
                provider=provider, model=model,
                system_prompt=system_prompt, messages=messages,
                full_context=full_context, job_type_str=job_type_value,
                db=db, trace=trace, enable_reasoning=req.enable_reasoning,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return StreamingResponse(
        generate_sse_stream(
            project_id=req.project_id, message=req.message,
            provider=provider, model=model,
            system_prompt=system_prompt, messages=messages,
            db=db, trace=trace, enable_reasoning=req.enable_reasoning,
            tools=_nr_tools,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# LEGACY TRIGGERS HANDLER
# =============================================================================

def handle_legacy_triggers(
    req: Any,
    db: Session,
    trace: Any,
) -> Optional[StreamingResponse]:
    """Handle legacy triggers when translation layer unavailable."""
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

    if _SANDBOX_AVAILABLE and is_sandbox_trigger(req.message):
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream", headers=sse_headers,
        )

    if is_update_arch_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream", headers=sse_headers,
        )

    if is_archmap_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream", headers=sse_headers,
        )

    if is_zobie_map_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream", headers=sse_headers,
        )

    if _INTROSPECTION_AVAILABLE and is_introspection_trigger(req.message):
        return StreamingResponse(
            generate_introspection_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream", headers=sse_headers,
        )

    return None


# =============================================================================
# PRIVATE HELPERS (shared by handle_chat_mode and handle_normal_routing)
# =============================================================================

def _resolve_message_with_documents(req: Any) -> str:
    """Resolve req.message by inlining any attached document content.

    v7.1 (2026-04-09): When the frontend converts large pastes into structured
    documents, req.message is just '[Documents attached]'. This helper inlines
    the actual document content so it gets persisted to DB and is available
    to the Weaver and other downstream consumers that read conversation history.
    """
    user_content = req.message
    docs = getattr(req, 'documents', None)
    if docs and isinstance(docs, list) and len(docs) > 0:
        doc_blocks = []
        for doc in docs:
            label = doc.get('label', 'Document')
            body = doc.get('content', '')
            idx = doc.get('index', '')
            doc_blocks.append(
                f"--- Document {idx}: {label} ---\n{body}\n--- End Document {idx} ---"
            )
        doc_text = '\n\n'.join(doc_blocks)
        if user_content in ('[Documents attached]', ''):
            user_content = doc_text
        else:
            user_content = user_content + '\n\n' + doc_text
    return user_content


def _persist_user_message(req: Any, db: Session) -> None:
    """Persist the user's message before any routing decisions."""
    try:
        from app.memory import schemas as _mem_schemas
        memory_service.create_message(
            db,
            _mem_schemas.MessageCreate(
                project_id=req.project_id,
                role="user",
                # v7.1 (2026-04-09): Persist document content, not placeholder.
                # Large pastes arrive as req.documents but req.message is just
                # '[Documents attached]'. Persisting the placeholder means the
                # Weaver never sees the pasted content when it reads history.
                content=_resolve_message_with_documents(req),
                provider="system",
            ),
        )
    except Exception as e:
        print(f"[CHAT_MODE] Failed to persist user message: {e}")


def _process_file_uploads(req: Any, full_context: str) -> tuple[list, bool, bool, str]:
    """Handle file_upload_* fields on the request.

    Returns (synthetic_attachments, is_image_upload, is_video_upload, updated_full_context).
    """
    _file_local_path = getattr(req, "file_upload_local_path", None)
    _file_name = getattr(req, "file_upload_name", None)
    _file_mime = getattr(req, "file_upload_mime", None) or ""
    _file_gemini_name = getattr(req, "file_upload_gemini_name", None)
    _file_gemini_uri = getattr(req, "file_upload_uri", None)
    is_image_upload = _file_mime.startswith("image/")
    is_video_upload = _file_mime.startswith("video/")
    synthetic_attachments: list = []

    if not (_file_local_path and _file_name):
        return synthetic_attachments, is_image_upload, is_video_upload, full_context

    if is_image_upload or is_video_upload:
        synthetic_attachments.append({
            "path": _file_local_path,
            "name": _file_name,
            "mime": _file_mime,
            "gemini_name": _file_gemini_name,
            "gemini_uri": _file_gemini_uri,
        })
        print(f"[CHAT_MODE] Image/video upload detected: {_file_name} ({_file_mime})")
        print(f"[CHAT_MODE]   -> Gemini URI: {_file_gemini_uri}")
        print(f"[CHAT_MODE]   -> Gemini name: {_file_gemini_name}")
        print(f"[CHAT_MODE]   -> Local path: {_file_local_path}")
    else:
        try:
            from app.llm.file_analyzer import extract_text as _extract_text
            _file_text, _file_err = _extract_text(file_path=_file_local_path, filename=_file_name)
            if _file_text:
                _preview = _file_text[:50000]
                _upload_block = (
                    f"=== UPLOADED FILE: {_file_name} ===\n"
                    f"{_preview}\n"
                    f"=== END FILE ===\n\n"
                )
                full_context = _upload_block + (full_context or "")
                print(f"[CHAT_MODE] Injected uploaded file content: {_file_name} ({len(_file_text)} chars)")
            elif _file_err:
                print(f"[CHAT_MODE] File extraction failed for {_file_name}: {_file_err}")
        except Exception as _fex:
            print(f"[CHAT_MODE] File injection error: {_fex}")

    return synthetic_attachments, is_image_upload, is_video_upload, full_context


def _build_chat_messages(req: Any, db: Session) -> list:
    """Build message list — panel history if available, else DB history."""
    panel_hist = getattr(req, 'panel_history', None)
    if panel_hist and isinstance(panel_hist, list) and len(panel_hist) > 0:
        messages = []
        for entry in panel_hist[-20:]:
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})
        # v7.0 (2026-04-08): Inject pasted document content into user message.
        # The frontend converts large pastes (>3000 chars) into structured
        # document attachments. The content arrives in req.documents but was
        # never added to the LLM messages — the model only saw the placeholder
        # '[Documents attached]' and had to use file-search tools to find content.
        user_content = req.message
        _docs = getattr(req, 'documents', None)
        if _docs and isinstance(_docs, list) and len(_docs) > 0:
            _doc_blocks = []
            for _doc in _docs:
                _label = _doc.get('label', 'Document')
                _body = _doc.get('content', '')
                _idx = _doc.get('index', '')
                _doc_blocks.append(
                    f"--- Document {_idx}: {_label} ---\n{_body}\n--- End Document {_idx} ---"
                )
            _doc_text = '\n\n'.join(_doc_blocks)
            if user_content in ('[Documents attached]', ''):
                user_content = _doc_text
            else:
                user_content = user_content + '\n\n' + _doc_text
            print(
                f"[CHAT_MODE] Injected {len(_docs)} document(s) into user message "
                f"({len(_doc_text)} chars)"
            )
        messages.append({'role': 'user', 'content': user_content})
        print(f"[CHAT_MODE] Using panel history: {len(messages)-1} prior messages + current")
        return messages

    # v7.0: Also inject documents in the DB-history path
    _fb_message = req.message
    _fb_docs = getattr(req, 'documents', None)
    if _fb_docs and isinstance(_fb_docs, list) and len(_fb_docs) > 0:
        _fb_blocks = []
        for _d in _fb_docs:
            _fb_blocks.append(
                f"--- Document {_d.get('index', '')}: {_d.get('label', 'Document')} ---\n"
                f"{_d.get('content', '')}\n--- End Document {_d.get('index', '')} ---"
            )
        _fb_text = '\n\n'.join(_fb_blocks)
        _fb_message = _fb_text if _fb_message in ('[Documents attached]', '') else _fb_message + '\n\n' + _fb_text
        print(f"[CHAT_MODE] Injected {len(_fb_docs)} doc(s) into DB-history message ({len(_fb_text)} chars)")
    return build_messages(
        message=_fb_message,
        project_id=req.project_id,
        db=db,
        include_history=req.include_history,
        history_limit=req.history_limit,
    )


def _inject_tab_data(ui_ctx: Any, full_context: str, db: Session) -> str:
    """Inject live tab data (e.g. portfolio positions) into context."""
    if not (ui_ctx and getattr(ui_ctx, 'job_type', None)):
        return full_context
    try:
        from app.llm.routing.ui_context_data import fetch_tab_data
        tab_data = fetch_tab_data(ui_ctx.job_type, db)
        if tab_data:
            full_context += f"\n\n{tab_data}"
            print(f"[CHAT_MODE] Tab data injected for {ui_ctx.job_type}: {len(tab_data)} chars")
    except Exception as e:
        print(f"[CHAT_MODE] Tab data injection failed: {e}")
    return full_context


def _run_grounding_gate(req: Any, system_prompt: str, label: str = "CHAT_MODE") -> str:
    """Run the grounding gate if available.  Returns the (possibly modified) system prompt."""
    if not (_GROUNDING_AVAILABLE and run_grounding_sync is not None):
        return system_prompt
    try:
        system_prompt, _grounding_meta = run_grounding_sync(
            message=req.message,
            system_prompt=system_prompt,
            context={"user_id": getattr(req, "user_id", "default")},
        )
        if _grounding_meta.get("grounding_applied"):
            print(
                f"[{label}] Grounding gate ACTIVE: "
                f"category={_grounding_meta.get('category')}, "
                f"sources={_grounding_meta.get('source_count')}, "
                f"domain={_grounding_meta.get('domain_hint')}"
            )
        else:
            print(
                f"[{label}] Grounding gate: no grounding needed "
                f"(category={_grounding_meta.get('category', 'n/a')}, "
                f"reason={_grounding_meta.get('reason', 'personal')})"
            )
    except Exception as e:
        print(f"[{label}] Grounding gate error (non-fatal): {e}")
    return system_prompt


# Tool role prompt blocks (shared between chat and normal routing)
_TOOL_ROLE_BLOCK = (
    "\n\n## TOOL ACCESS\n"
    "You have tool access for exploring the codebase AND writing to user folders.\n\n"
    "CODEBASE TOOLS (read-only): read_file, list_files, search_files, read_logs, search_my_files, read_user_file\n"
    "USER FILE TOOLS (read+write): get_user_folders, write_user_file\n"
    "Use get_user_folders to discover real folder paths, then write_user_file to save files there.\n\n"
    "IMPORTANT: Actually USE the tools. Do not just say you will — call them.\n\n"
    "YOUR ROLE: You are a RESEARCHER, PLANNER, and ASSISTANT.\n"
    "- Explore files, read code, understand patterns, discover architecture\n"
    "- Report your findings as text in the chat — describe what you found\n"
    "- When the user asks you to create a file in their personal folders\n"
    "  (Documents, Pictures, Desktop, etc.), call get_user_folders then write_user_file.\n"
    "- When asked to plan or spec, USE tools first to inspect the codebase,\n"
    "  then produce a detailed implementation plan based on real file contents\n"
    "- Present file paths, structures, and what needs to change\n\n"
    "DO NOT:\n"
    "- Try to create, write, or modify ASTRA codebase files — those go through the sandbox\n"
    "- Dump raw file contents — summarise and highlight relevant patterns\n"
    "- Say you will do something without actually calling the tools to do it\n\n"
    "GOOD OUTPUT: Call tools to explore, then present findings and plans.\n"
    "BAD OUTPUT: Saying 'I will inspect...' without calling any tools.\n"
)

_WEB_SEARCH_PROMPT = (
    "\n\n## WEB SEARCH TOOL\n"
    "You have access to a web_search tool. Use it when the user asks you to\n"
    "research, look up, find pricing, get current information, or anything\n"
    "that requires knowledge you do not have.\n"
    "IMPORTANT: Actually CALL the web_search tool. Do not just say you will.\n"
    "Call it with a specific search query and use the results in your response.\n"
)

_CHAT_TOOLS_OVERRIDE = (
    "   - You CAN: read files, write files, execute code, explore directories\n"
)
_CHAT_TOOLS_REPLACEMENT = (
    "   - Codebase files have been PRE-LOADED into your context below.\n"
    "   - You do NOT have codebase tool access. Do NOT generate tool_call blocks for file operations.\n"
    "   - Do NOT call execute_command or shell commands.\n"
    "   - Reference the [CODEBASE CONTEXT] files directly in your response.\n"
)


def _inject_tools(
    provider: str,
    model: str,
    req: Any,
    system_prompt: str,
    codebase_gather_pending: bool = False,
    codebase_ctx: str = "",
) -> tuple[list | None, str]:
    """Inject tool definitions + prompt blocks.  Returns (tools_list, updated_system_prompt)."""
    _chat_tools = None

    try:
        from app.llm.chat_tool_loop import is_tool_eligible, get_chat_tools

        # If model lacks tool access but context needs it, swap to tool-capable model
        if not is_tool_eligible(provider, model):
            from app.memory.complexity import DEEP_KEYWORDS, _count_keyword_hits
            _deep_hits = _count_keyword_hits(req.message.lower(), DEEP_KEYWORDS)
            _needs_tools = (
                _is_builds_context(req)
                or _detect_codebase_exploration(req.message)
                or _deep_hits >= 1
            )
            if _needs_tools:
                import os as _os
                _tool_provider = _os.getenv("TOOL_CHAT_PROVIDER", "google")
                _tool_model = _os.getenv("TOOL_CHAT_MODEL", "gemini-3.1-pro-preview-customtools")
                if is_tool_eligible(_tool_provider, _tool_model):
                    print(f"[TOOLS] Context needs tools but {provider}/{model} has none — "
                          f"swapping to {_tool_provider}/{_tool_model}")
                    provider = _tool_provider
                    model = _tool_model
                    _set_sticky_model(req.project_id, provider, model)

        if is_tool_eligible(provider, model):
            _chat_tools = get_chat_tools()
            print(f"[TOOLS] Tool access ENABLED for {provider}/{model} ({len(_chat_tools)} tools)")
            system_prompt += _TOOL_ROLE_BLOCK
        else:
            if codebase_gather_pending and codebase_ctx:
                if _CHAT_TOOLS_OVERRIDE in system_prompt:
                    system_prompt = system_prompt.replace(_CHAT_TOOLS_OVERRIDE, _CHAT_TOOLS_REPLACEMENT)
                    print("[TOOLS] Capability layer overridden for non-trusted model")
    except ImportError:
        print("[TOOLS] chat_tool_loop not available")

    # Universal web search — ALL models get web_search as a tool
    try:
        from app.debug.tool_definitions import get_universal_tools
        from app.llm.chat_tool_loop import _to_anthropic_tool_format
        _universal = [_to_anthropic_tool_format(t) for t in get_universal_tools()]
        if _chat_tools is not None:
            _existing_names = {t.get("name") for t in _chat_tools}
            for ut in _universal:
                if ut["name"] not in _existing_names:
                    _chat_tools.append(ut)
            print(f"[TOOLS] Universal tools merged: {len(_chat_tools)} total")
        else:
            _chat_tools = _universal
            system_prompt += _WEB_SEARCH_PROMPT
            print(f"[TOOLS] Universal web_search tool injected for {provider}/{model}")
    except ImportError as _uie:
        print(f"[TOOLS] Universal tools not available: {_uie}")

    return _chat_tools, system_prompt


def _inject_image_into_messages(
    messages: list,
    file_local_path: str | None,
    file_mime: str,
) -> list:
    """Inject base64 image into the last user message for Gemini vision."""
    if not file_local_path:
        return messages
    try:
        import base64 as _b64
        with open(file_local_path, "rb") as _img_f:
            _img_bytes = _img_f.read()
        _img_b64 = _b64.b64encode(_img_bytes).decode("utf-8")
        for _i in range(len(messages) - 1, -1, -1):
            if messages[_i].get("role") == "user":
                _text_content = messages[_i].get("content", "")
                messages[_i]["content"] = [
                    {"type": "text", "text": _text_content},
                    {"type": "image", "mime_type": file_mime, "data": _img_b64},
                ]
                print(f"[CHAT_MODE] Injected image into user message [{_i}]: {file_mime}, {len(_img_bytes)} bytes")
                break
    except Exception as _img_err:
        print(f"[CHAT_MODE] Failed to inject image: {_img_err}")
    return messages


__all__ = [
    "handle_chat_mode",
    "handle_normal_routing",
    "handle_legacy_triggers",
    # Re-export for backward compatibility with external importers
    "_detect_build_deploy_intent",
    "_detect_file_creation_intent",
    "_detect_image_gen_intent",
    "_get_sticky_model",
    "_set_sticky_model",
]
