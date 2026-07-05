# FILE: app/llm/routing/chat_routing.py
# Purpose: Chat and normal routing handlers for stream routing (facade/shim; handlers retained here).
# Called-by: app.endpoints.chat_attachments, app.llm._stream_router_utils, app.llm.project_scoping_stream, app.llm.stream_router
# Depends-on: app.llm.routing.chat_request_prep, app.llm.routing.chat_prompt_tools, app.llm.routing.handler_registry (+ more)
# Last-renovated: 2026-06-21
"""
Chat and normal routing handlers for stream routing.

Split 2026-06-21 (BATCH 7): the cleanly-isolated helper bands were extracted:
    - chat_request_prep.py: message/file-upload prep (_resolve_message_with_documents,
      _persist_user_message, _process_file_uploads, _build_chat_messages, _inject_image_into_messages)
    - chat_prompt_tools.py:  prompt/tool decoration (_inject_tab_data, _run_grounding_gate,
      _inject_tools + the _TOOL_ROLE_BLOCK / _WEB_SEARCH_PROMPT / _CHAT_TOOLS_* prompt blocks)
The 3 handlers + _rag_llm_safety_net stay here. All public names re-exported -> importers unchanged.

RAG-as-context (2026-07-01, Lane A<->C contract): at both divert points
(deterministic is_architecture_query + LLM safety-net) we now try
codebase_context_bridge.try_build_codebase_context() first — when it returns
a grounded block, the message STAYS in chat and the normal chat model answers
in Astra's voice with the block injected into its context. Raw RAG output is
only streamed as the reply when the context build is unavailable (legacy
fallback — e.g. Lane C not merged, kill-switched, or errored).
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

# Helpers extracted 2026-06-21 (BATCH 7) — re-imported so the handlers below
# (and any external importer) resolve them through this module unchanged.
from .chat_request_prep import (
    _resolve_message_with_documents,
    _persist_user_message,
    _process_file_uploads,
    _build_chat_messages,
    _inject_image_into_messages,
)
from .chat_prompt_tools import (
    _inject_tab_data,
    _run_grounding_gate,
    _inject_tools,
)


# =============================================================================
# CHAT MODE HANDLER
# =============================================================================

def _rag_llm_safety_net(req: Any, db: Session, trace: Any, persist_user: bool = False):
    """LLM safety-net for codebase/feature questions that miss Tier-0 routing.

    Codebase questions whose feature/UI words aren't in the guard list (e.g.
    "what data does the chat window have", "how does the finance panel work")
    are classified as plain chat by Tier-0 and never reach RAG. Rather than
    broaden the brittle regex (false positives like "book me a table"), this
    fires ONLY for natural-shape questions without a guard keyword
    (needs_llm_codebase_check) and asks a lightweight model whether the question
    is about ASTRA's own code/features.

    v2026-07-01 (RAG-as-context): when the LLM says "codebase question", we
    first try to build a grounded context block and keep the turn IN chat —
    the chat model answers in Astra's voice. Only when the block is
    unavailable does the legacy RAG stream take over as the reply.
    Fail-closed and non-fatal: any failure returns (None, "") and chat
    proceeds.

    Returns (streaming_response_or_none, codebase_context_str).
    """
    if not _RAG_STREAM_AVAILABLE:
        return None, ""
    # Never hijack an in-flight job continuation (e.g. a spec-clarification reply
    # that happens to look like a feature question).
    if getattr(req, "continue_job_id", None):
        return None, ""
    try:
        from app.llm.routing.rag_fallback import needs_llm_codebase_check
        if not needs_llm_codebase_check(req.message):
            return None, ""
        # Identity/capability questions ("tell me about yourself", "what are
        # your abilities") are answered in chat via the capability manifest
        # (build_full_context). They match the broad "tell me about X" shape, so
        # exclude them here — they must NOT be diverted into code RAG.
        try:
            from app.astra_memory.topic_tagger import extract_tags
            if "identity_capability" in (extract_tags(req.message) or []):
                return None, ""
        except Exception:
            pass
        from app.llm.routing.codebase_intent_llm import is_codebase_question_llm
        if not is_codebase_question_llm(req.message):
            return None, ""
        # RAG-as-context first: stay in chat with grounded context injected.
        from .codebase_context_bridge import try_build_codebase_context
        _cb_ctx = try_build_codebase_context(req.message, db)
        if _cb_ctx:
            print("[RAG_SAFETY_NET] codebase question → grounded context injected, staying in chat")
            return None, _cb_ctx
        print("[RAG_SAFETY_NET] LLM classified message as a codebase question → RAG (legacy divert)")
        return StreamingResponse(
            generate_rag_query_stream(
                project_id=req.project_id, message=req.message, db=db, trace=trace,
                persist_user=persist_user,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        ), ""
    except Exception as e:
        print(f"[RAG_SAFETY_NET] failed (non-fatal): {e}")
        return None, ""


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
    _user_message_id = _persist_user_message(req, db)

    # ── 1b. RAG safety-net: codebase/feature questions that missed Tier-0
    # (e.g. "what data does the chat window have"). Placed AFTER persistence
    # so the user turn is always saved. v2026-07-01: prefers returning a
    # grounded context block (stay in chat) over diverting the reply to RAG.
    _rag_net, _codebase_ctx_block = _rag_llm_safety_net(req, db, trace)
    if _rag_net is not None:
        return _rag_net

    # ── 2. Build base context ──
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    if _codebase_ctx_block:
        full_context += f"\n\n{_codebase_ctx_block}"
        print(f"[CHAT_MODE] RAG-as-context block injected: {len(_codebase_ctx_block)} chars")

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

    # v16.0 (2026-05-01): Image gen no longer bypasses the chat LLM.
    # Previously this short-circuited to a fresh Gemini synth that lost
    # all conversation context. Now: chat LLM (gpt-5.4) handles the turn,
    # emits an [IMAGE_PROMPT]: marker, and the wrapper sends that prompt
    # straight to gpt-image-2. extras["image_route"] is now a *hint* that
    # tells the system prompt builder to include image-gen instructions
    # and tells the stream wrapper to scan for the marker.
    _image_intent_detected = bool(extras.get("image_route"))

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

    # ── 9. Build system prompt (with image-gen marker instructions if relevant) ──
    system_prompt = build_system_prompt(
        project, full_context, ui_context=ui_ctx,
        image_intent=_image_intent_detected,
    )

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

    _inner_stream = generate_sse_stream(
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
        # v2026-06-14: hand over the already-persisted user row so the stream
        # doesn't write a duplicate, and reports its id in the done event.
        user_message_id=_user_message_id,
        # v2026-06-24: routing provenance so an explicit desktop pin survives restart.
        model_source=extras.get("model_source"),
    )

    # v16.0: Wrap stream with image dispatcher when image intent was detected.
    # The chat LLM will emit [IMAGE_PROMPT]: <prompt> in its response; the
    # wrapper extracts it and fires gpt-image-2 directly. No Gemini synth
    # in between, full conversation context preserved.
    if _image_intent_detected:
        from app.llm.image_extractor import wrap_stream_with_image_dispatch
        print("[CHAT_MODE] Image intent detected — wrapping stream with image dispatcher")
        _inner_stream = wrap_stream_with_image_dispatch(
            _inner_stream, project_id=req.project_id, db=db,
        )

    return StreamingResponse(
        _inner_stream,
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
    """Handle normal job-type routing with RAG fallback (as-context first)."""

    # Architecture queries (deterministic regex + guard). v2026-07-01: try
    # RAG-as-context first — build a grounded block and let the NORMAL chat
    # model answer in Astra's voice. Legacy divert (raw RAG reply) only when
    # the block is unavailable (Lane C missing, kill-switched, or errored).
    _codebase_ctx_block = ""
    if _RAG_STREAM_AVAILABLE and is_architecture_query(req.message):
        from .codebase_context_bridge import try_build_codebase_context
        _codebase_ctx_block = try_build_codebase_context(req.message, db)
        if not _codebase_ctx_block:
            # persist_user=True: the normal-routing path doesn't persist the
            # user turn itself, so the RAG stream must save both turns.
            print(f"[NORMAL_ROUTING] RAG fallback: architecture query (legacy divert)")
            return StreamingResponse(
                generate_rag_query_stream(
                    project_id=req.project_id, message=req.message, db=db, trace=trace,
                    persist_user=True,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )
        print(f"[NORMAL_ROUTING] RAG-as-context: architecture query stays in chat")

    # LLM safety-net for natural-shape codebase questions without a guard keyword.
    if not _codebase_ctx_block:
        _rag_net, _codebase_ctx_block = _rag_llm_safety_net(req, db, trace, persist_user=True)
        if _rag_net is not None:
            return _rag_net

    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    if _codebase_ctx_block:
        full_context += f"\n\n{_codebase_ctx_block}"
        print(f"[NORMAL_ROUTING] RAG-as-context block injected: {len(_codebase_ctx_block)} chars")

    # Job continuation — LANE D: env-only role (JOB_CONTINUATION_* in .env,
    # seeded to the old DEFAULT_MODELS["anthropic_opus"] resolution).
    if req.continue_job_id and req.job_state == "needs_spec_clarification":
        from app.llm.frontier_models import get_role_model
        provider, model = get_role_model("JOB_CONTINUATION", "ARCHITECT")
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
