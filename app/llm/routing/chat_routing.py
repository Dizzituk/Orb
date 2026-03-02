# FILE: app/llm/routing/chat_routing.py
"""
Chat and normal routing handlers for stream routing.

v1.0 (2026-01-20): Extracted from stream_router.py for modularity.

This module provides:
- `handle_chat_mode()` - Lightweight chat routing
- `handle_normal_routing()` - Standard job-type routing
- `handle_legacy_triggers()` - Fallback for when translation layer unavailable
"""

from __future__ import annotations

import logging
from typing import List, Optional, Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.memory import service as memory_service
# Import from pipeline.high_stakes to avoid circular import with router.py
from app.llm.pipeline.high_stakes import is_high_stakes_job, is_opus_model
from app.llm.streaming import get_available_streaming_provider, get_available_streaming_providers

from app.llm.stream_utils import (
    DEFAULT_MODELS,
    classify_job_type,
    select_provider_for_job_type,
)

from app.memory.complexity import classify_complexity

from app.llm.legacy_triggers import (
    is_zobie_map_trigger,
    is_archmap_trigger,
    is_update_arch_trigger,
    is_introspection_trigger,
    is_sandbox_trigger,
)

from .handler_registry import (
    # Availability flags
    _LOCAL_TOOLS_AVAILABLE,
    _SANDBOX_AVAILABLE,
    _INTROSPECTION_AVAILABLE,
    _RAG_STREAM_AVAILABLE,
    # Handlers
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

logger = logging.getLogger(__name__)


# =============================================================================
# CHAT MODE HANDLER
# =============================================================================

def handle_chat_mode(
    req: Any,  # StreamRequest
    project: Any,  # Project model
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """
    Handle CHAT mode - lightweight model, no commands.
    
    v4.8: Uses stage_models for provider/model selection with debug logging.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        project: Project ORM object
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse for chat
    """
    print(f"[CHAT_MODE] Handling chat for project={req.project_id}, message={req.message[:50]}...")
    
    # v6.1: Always persist the user's message BEFORE any confirmation gate.
    # Without this, if the confirmation gate fires, the message is never saved
    # and downstream handlers (Weaver) can't find the conversation.
    try:
        from app.memory import schemas as _mem_schemas
        memory_service.create_message(
            db,
            _mem_schemas.MessageCreate(
                project_id=req.project_id,
                role="user",
                content=req.message,
                provider="system",
            ),
        )
    except Exception as e:
        print(f"[CHAT_MODE] Failed to persist user message: {e}")
    
    # Build context
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    
    # v7.0: Pre-gather codebase context for trusted models.
    # Reads files from the sandbox (read-only) via RAG-guided discovery.
    # This gives Opus/Gemini 3.1 actual codebase knowledge in standard chat.
    try:
        from app.llm.routing.chat_codebase_reader import (
            gather_codebase_context,
        )
        # We don't know the final model yet (complexity may upgrade it),
        # so we store the message and gather later, after model selection.
        _codebase_gather_pending = True
    except ImportError:
        _codebase_gather_pending = False
    
    # v5.5: Frontend model override (from model switcher dropdown)
    if req.provider and req.model:
        provider = req.provider
        model = req.model
        print(f"[CHAT_MODE] Using frontend override: provider={provider}, model={model}")
    else:
        # v5.7: Run complexity classifier to decide model tier.
        # Even in chat mode, complex messages deserve a better model.
        complexity = classify_complexity(
            query=req.message,
            intent=None,
            attachments=getattr(req, 'attachments', None),
        )
        print(f"[CHAT_MODE] Complexity: tier={complexity.tier}, target={complexity.model_target}, "
              f"confidence={complexity.confidence}, signals={complexity.signals}")
        
        # v3.2: Read chat provider/model from .env stage config.
        # All tiers use the configured CHAT provider by default.
        # Only "deep" escalates to Opus, and "multimodal" to Gemini vision.
        import os as _os
        _chat_provider = _os.getenv("CHAT_PROVIDER", "google")
        _chat_model = _os.getenv("CHAT_MODEL", "gemini-2.5-flash")

        # v6.0: Chat panel requests (with ui_context) skip the confirmation gate.
        _skip_confirm = getattr(req, 'ui_context', None) is not None

        if complexity.tier == "deep":
            # Deep/architectural queries → Opus (or env override)
            provider = _os.getenv("CHAT_DEEP_PROVIDER", "anthropic")
            model = _os.getenv("CHAT_DEEP_MODEL", "claude-opus-4-6")
            print(f"[CHAT_MODE] Complexity UPGRADE: deep -> {provider}/{model}")
            if not _skip_confirm:
                try:
                    from app.llm.routing.confirmation_gate import (
                        should_confirm_model_escalation,
                        format_confirmation_sse,
                    )
                    confirm_req = should_confirm_model_escalation(
                        from_tier="lookup", to_tier="deep",
                        confidence=complexity.confidence,
                        message=req.message,
                    )
                    if confirm_req:
                        async def _confirm_stream():
                            import json as _json
                            yield format_confirmation_sse(confirm_req)
                            yield f"data: {_json.dumps({'type': 'done', 'provider': 'local', 'model': 'confirmation_gate', 'total_length': 0})}\n\n"
                        return StreamingResponse(
                            _confirm_stream(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
                except ImportError:
                    pass
            else:
                print(f"[CHAT_MODE] Confirmation gate skipped (chat panel request)")
        elif complexity.tier == "reasoning":
            # v3.2: Reasoning uses configured chat provider, not hardcoded OpenAI
            provider = _chat_provider
            model = _chat_model
            print(f"[CHAT_MODE] Reasoning tier -> {provider}/{model}")
            if not _skip_confirm:
                try:
                    from app.llm.routing.confirmation_gate import (
                        should_confirm_model_escalation,
                        format_confirmation_sse,
                    )
                    confirm_req = should_confirm_model_escalation(
                        from_tier="lookup", to_tier="reasoning",
                        confidence=complexity.confidence,
                        message=req.message,
                    )
                    if confirm_req:
                        async def _confirm_stream():
                            import json as _json
                            yield format_confirmation_sse(confirm_req)
                            yield f"data: {_json.dumps({'type': 'done', 'provider': 'local', 'model': 'confirmation_gate', 'total_length': 0})}\n\n"
                        return StreamingResponse(
                            _confirm_stream(),
                            media_type="text/event-stream",
                            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                        )
                except ImportError:
                    pass
            else:
                print(f"[CHAT_MODE] Confirmation gate skipped (chat panel request)")
        elif complexity.tier == "multimodal":
            attachments = getattr(req, 'attachments', None) or []
            if len(attachments) >= 2:
                provider = "google"
                model = "gemini-3.1-pro-preview"
                print(f"[CHAT_MODE] Multimodal ({len(attachments)} files) -> Gemini 3.1 Pro")
            else:
                provider = "google"
                model = "gemini-2.5-flash"
                print(f"[CHAT_MODE] Multimodal (single) -> Gemini 2.5 Flash")
        else:
            # lookup or ping_pong — use configured chat model
            provider = _chat_provider
            model = _chat_model
            print(f"[CHAT_MODE] {complexity.tier} -> {provider}/{model}")
    
    # v3.2: Check provider availability — but DON'T silently swap provider
    # while keeping the original model (that causes openai+gemini mismatches).
    providers_available = get_available_streaming_providers()
    print(f"[CHAT_MODE] Provider availability: {providers_available}")
    
    if not providers_available.get(provider, False):
        # Provider key not available — try to find a working alternative
        available = get_available_streaming_provider()
        if available:
            print(f"[CHAT_MODE] Provider '{provider}' not available, falling back to {available} WITH its default model")
            provider = available
            # CRITICAL: Also switch the model to match the new provider.
            # Without this, we'd send e.g. 'gemini-2.5-flash' to the OpenAI API.
            import os as _os2
            if provider == "google":
                model = _os2.getenv("CHAT_MODEL", "gemini-2.5-flash")
            elif provider == "anthropic":
                model = _os2.getenv("ANTHROPIC_DEFAULT_MODEL", "claude-sonnet-4-6")
            elif provider == "openai":
                model = _os2.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
        else:
            print(f"[CHAT_MODE] WARNING: No providers available at all")
    
    # Build messages
    # v6.0: If chat panel sent its own conversation history, use that instead of DB history.
    # This prevents the panel from inheriting the main chat's conversation thread.
    panel_hist = getattr(req, 'panel_history', None)
    if panel_hist and isinstance(panel_hist, list) and len(panel_hist) > 0:
        # Use panel's local history — already [{role, content}] format
        messages = []
        for entry in panel_hist[-20:]:  # Cap at 20 messages
            role = entry.get('role', 'user')
            content = entry.get('content', '')
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})
        messages.append({'role': 'user', 'content': req.message})
        print(f"[CHAT_MODE] Using panel history: {len(messages)-1} prior messages + current")
    else:
        messages = build_messages(
            message=req.message,
            project_id=req.project_id,
            db=db,
            include_history=req.include_history,
            history_limit=req.history_limit,
        )
    
    # Build system prompt (includes capability layer + UI context)
    ui_ctx = getattr(req, 'ui_context', None)
    
    # v6.0: Inject live tab data (e.g. portfolio positions) into context
    if ui_ctx and getattr(ui_ctx, 'job_type', None):
        try:
            from app.llm.routing.ui_context_data import fetch_tab_data
            tab_data = fetch_tab_data(ui_ctx.job_type, db)
            if tab_data:
                full_context += f"\n\n{tab_data}"
                print(f"[CHAT_MODE] Tab data injected for {ui_ctx.job_type}: {len(tab_data)} chars")
        except Exception as e:
            print(f"[CHAT_MODE] Tab data injection failed: {e}")
    
    # v7.0: Gather codebase context for trusted models (sandbox read-only)
    codebase_ctx = ""
    if _codebase_gather_pending:
        try:
            codebase_ctx = gather_codebase_context(
                message=req.message, model=model, db=db,
            )
            if codebase_ctx:
                full_context += f"\n\n{codebase_ctx}"
                print(f"[CHAT_MODE] Codebase context injected: {len(codebase_ctx)} chars")
        except Exception as e:
            print(f"[CHAT_MODE] Codebase context failed (non-fatal): {e}")
    
    system_prompt = build_system_prompt(project, full_context, ui_context=ui_ctx)
    
    # v7.0: When codebase context was injected, override the capability layer
    # to prevent the model from trying to explore manually.
    # The capability layer says "you CAN execute code, explore directories"
    # which causes the model to hallucinate tool calls. Replace that section.
    if _codebase_gather_pending and codebase_ctx:
        _CHAT_TOOLS_OVERRIDE = (
            "   - You CAN: read files, write files, execute code, explore directories\n"
        )
        _CHAT_TOOLS_REPLACEMENT = (
            "   - Codebase files have been PRE-LOADED into your context below.\n"
            "   - You do NOT have tool access in chat mode. Do NOT generate tool_call blocks.\n"
            "   - Do NOT call execute_command or shell commands.\n"
            "   - Reference the [CODEBASE CONTEXT] files directly in your response.\n"
        )
        if _CHAT_TOOLS_OVERRIDE in system_prompt:
            system_prompt = system_prompt.replace(
                _CHAT_TOOLS_OVERRIDE, _CHAT_TOOLS_REPLACEMENT,
            )
            print("[CHAT_MODE] Capability layer overridden for codebase-aware chat")
    
    if ui_ctx:
        print(f"[CHAT_MODE] UI context injected: view={ui_ctx.view_type}, job={ui_ctx.job_type}, label={ui_ctx.label}")
    
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
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# =============================================================================
# NORMAL ROUTING HANDLER
# =============================================================================

def handle_normal_routing(
    req: Any,  # StreamRequest
    project: Any,  # Project model
    db: Session,
    trace: Any,
) -> StreamingResponse:
    """
    Handle normal job-type routing with RAG fallback.
    
    v4.12: Includes RAG fallback for architecture queries.
    
    Args:
        req: StreamRequest with project_id, message, etc.
        project: Project ORM object
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse for the routed job
    """
    
    # =========================================================================
    # RAG FALLBACK: Detect architecture questions when translation layer fails
    # =========================================================================
    if _RAG_STREAM_AVAILABLE and is_architecture_query(req.message):
        print(f"[NORMAL_ROUTING] RAG fallback: detected architecture query")
        print(f"[NORMAL_ROUTING]   message={req.message[:80]}...")
        return StreamingResponse(
            generate_rag_query_stream(
                project_id=req.project_id,
                message=req.message,
                db=db,
                trace=trace,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    
    # Build context
    full_context = build_full_context(db, req.project_id, req.message, req.use_semantic_search)
    
    # Job continuation
    if req.continue_job_id and req.job_state == "needs_spec_clarification":
        provider = "anthropic"
        model = DEFAULT_MODELS["anthropic_opus"]
        messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
        system_prompt = build_system_prompt(project, full_context)
        
        return StreamingResponse(
            generate_high_stakes_critique_stream(
                project_id=req.project_id,
                message=req.message,
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                messages=messages,
                full_context=full_context,
                job_type_str="architecture_design",
                db=db,
                trace=trace,
                enable_reasoning=req.enable_reasoning,
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
    
    # v3.2: Provider availability check with matched model fallback
    providers_available = get_available_streaming_providers()
    if not providers_available.get(provider, False):
        available = get_available_streaming_provider()
        if not available:
            raise HTTPException(status_code=503, detail="No LLM provider available")
        print(f"[NORMAL_ROUTING] Provider '{provider}' not available, falling back to {available}")
        provider = available
        model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS.get("google", "gemini-2.5-flash"))
    
    # Build messages and system prompt
    messages = build_messages(req.message, req.project_id, db, req.include_history, req.history_limit)
    
    # v7.0: Inject codebase context for trusted models (same as handle_chat_mode)
    _nr_codebase_ctx = ""
    try:
        from app.llm.routing.chat_codebase_reader import (
            gather_codebase_context, is_trusted_model,
        )
        if is_trusted_model(model):
            _nr_codebase_ctx = gather_codebase_context(
                message=req.message, model=model, db=db,
            )
            if _nr_codebase_ctx:
                full_context += f"\n\n{_nr_codebase_ctx}"
                print(f"[NORMAL_ROUTING] Codebase context injected: {len(_nr_codebase_ctx)} chars")
    except Exception as e:
        print(f"[NORMAL_ROUTING] Codebase context failed (non-fatal): {e}")
    
    system_prompt = build_system_prompt(project, full_context)
    
    # v7.0: Override capability layer when codebase context is present
    if _nr_codebase_ctx:
        _TOOLS_LINE = "   - You CAN: read files, write files, execute code, explore directories\n"
        _TOOLS_REPLACE = (
            "   - Codebase files have been PRE-LOADED into your context below.\n"
            "   - You do NOT have tool access in chat mode. Do NOT generate tool_call blocks.\n"
            "   - Do NOT call execute_command or shell commands.\n"
            "   - Reference the [CODEBASE CONTEXT] files directly in your response.\n"
        )
        if _TOOLS_LINE in system_prompt:
            system_prompt = system_prompt.replace(_TOOLS_LINE, _TOOLS_REPLACE)
            print("[NORMAL_ROUTING] Capability layer overridden for codebase-aware response")
    
    # High-stakes routing
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
    
    # Normal stream
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


# =============================================================================
# LEGACY TRIGGERS HANDLER
# =============================================================================

def handle_legacy_triggers(
    req: Any,  # StreamRequest
    db: Session,
    trace: Any,
) -> Optional[StreamingResponse]:
    """
    Handle legacy triggers when translation layer unavailable.
    
    Args:
        req: StreamRequest with project_id and message
        db: Database session
        trace: Audit trace
    
    Returns:
        StreamingResponse if trigger matched, None otherwise
    """
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    
    if _SANDBOX_AVAILABLE and is_sandbox_trigger(req.message):
        return StreamingResponse(
            generate_sandbox_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_update_arch_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_update_architecture_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_archmap_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_architecture_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if is_zobie_map_trigger(req.message) and _LOCAL_TOOLS_AVAILABLE:
        return StreamingResponse(
            generate_local_zobie_map_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    if _INTROSPECTION_AVAILABLE and is_introspection_trigger(req.message):
        return StreamingResponse(
            generate_introspection_stream(project_id=req.project_id, message=req.message, db=db, trace=trace),
            media_type="text/event-stream",
            headers=sse_headers,
        )
    
    return None


__all__ = [
    "handle_chat_mode",
    "handle_normal_routing",
    "handle_legacy_triggers",
]
