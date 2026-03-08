# FILE: app/debug/debug_chat.py
"""
Debug Chat Endpoint: SSE streaming endpoint for the Debug Assistant.

Manages the conversation loop, context assembly, model routing,
and agentic tool-use loop. Mirrors the pattern in stream_router.py
but specialised for debug queries.

Endpoint: POST /api/debug/chat (SSE stream)
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import require_auth
from app.auth.middleware import AuthResult

from app.debug.context_assembler import assemble_context
from app.debug.model_router import classify_query, DebugTier
from app.debug.system_prompt import build_debug_system_prompt
from app.debug.tool_definitions import get_tools_for_tier
# Phase 2: from app.debug.action_executor import execute_tool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/debug", tags=["Debug Assistant"])

# Maximum tool-use iterations per turn (prevents runaway loops)
MAX_TOOL_ITERATIONS = 10

# Conversation history per session (in-memory for now)
_conversations: Dict[str, List[dict]] = {}
MAX_CONVERSATION_LENGTH = 50


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class DebugChatRequest(BaseModel):
    """Incoming debug chat message."""
    message: str
    session_id: str = "default"
    # Optional overrides
    force_tier: Optional[str] = None  # "triage", "analysis", "agentic"
    include_context: bool = True


# =============================================================================
# SSE HELPERS
# =============================================================================

def _sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


# =============================================================================
# LLM CALL WITH TOOL LOOP
# =============================================================================

async def _call_llm_simple(
    provider: str,
    model: str,
    messages: List[dict],
    system_prompt: str,
    tools: List[dict],
    max_tokens: int = 4096,
) -> AsyncGenerator[str, None]:
    """
    Call the LLM via the registry (no tool loop).

    Phase 1: All tiers use this path. Tool definitions are included in the
    system prompt so the model can reference them, but actual tool execution
    is not performed. The model may suggest tool usage in its response text.

    Phase 2 will add _call_llm_with_tools() for the agentic tier, which
    bypasses the registry’s internal tool loop and manages tool calls
    directly against the provider API.

    Yields SSE events: token, error.
    """
    from app.providers.registry import llm_call

    try:
        # Inject available tool descriptions into system prompt so the
        # model is aware of what it *could* do (even if we don’t execute).
        enhanced_prompt = system_prompt
        if tools:
            tool_desc = "\n".join(
                f"- {t['name']}: {t.get('description', '')}" for t in tools
            )
            enhanced_prompt += (
                "\n\n## Available Tools (informational — Phase 1 read-only)\n"
                "You have awareness of these tools. In Phase 1 you cannot execute\n"
                "write tools, but you CAN execute read tools via the context that\n"
                "has already been assembled for you. If a fix is needed, describe\n"
                "the exact changes required so they can be applied manually or\n"
                "in Phase 2 when agentic write access is enabled.\n\n"
                f"{tool_desc}"
            )

        result = await llm_call(
            provider_id=provider,
            model_id=model,
            messages=messages,
            system_prompt=enhanced_prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            timeout_seconds=120,
            enable_tools=False,  # Phase 1: no tool execution via registry
        )

        if result.status.value != "success":
            yield _sse_event("error", {
                "error": f"LLM call failed: {result.status.value}",
                "detail": result.error_message or "",
            })
            return

        content = result.content or ""
        if content:
            yield _sse_event("token", {"content": content})

    except Exception as e:
        logger.error("[debug_chat] LLM call error: %s", e, exc_info=True)
        yield _sse_event("error", {"error": str(e)})
        return


# ---------------------------------------------------------------------------
# Phase 2 placeholder: agentic tool loop (not yet active)
# ---------------------------------------------------------------------------
# When Phase 2 is implemented, this will call the provider API directly
# (bypassing the registry’s internal tool loop) so we can:
# 1. Send debug tool schemas to the model
# 2. Intercept tool_use responses
# 3. Execute via action_executor
# 4. Stream tool_call / tool_result SSE events to the UI
# 5. Feed results back and continue the loop
# For now, agentic tier falls through to _call_llm_simple with a note
# that write tools are not yet active.
# ---------------------------------------------------------------------------


# =============================================================================
# MAIN ENDPOINT
# =============================================================================

@router.post("/chat")
async def debug_chat(
    req: DebugChatRequest,
    auth: AuthResult = Depends(require_auth),
):
    """
    Debug Assistant chat endpoint (SSE stream).

    Flow:
    1. Assemble ASTRA context
    2. Classify query → select model tier
    3. Build system prompt with context
    4. Stream LLM response (with optional tool loop)
    """

    async def generate() -> AsyncGenerator[str, None]:
        start_time = time.time()
        session_id = req.session_id

        # Get or create conversation history
        if session_id not in _conversations:
            _conversations[session_id] = []
        history = _conversations[session_id]

        # Trim if too long
        if len(history) > MAX_CONVERSATION_LENGTH * 2:
            history = history[-MAX_CONVERSATION_LENGTH:]
            _conversations[session_id] = history

        # 1. Assemble context
        context_xml = ""
        if req.include_context:
            try:
                ctx = await assemble_context()
                context_xml = ctx.xml
                logger.info(
                    "[debug_chat] Context assembled: %d tokens, %d sources, %dms",
                    ctx.total_tokens, len(ctx.sources_included), ctx.assembly_time_ms,
                )
            except Exception as e:
                logger.warning("[debug_chat] Context assembly failed: %s", e)
                context_xml = "<astra_context>\n  <error>Context assembly failed</error>\n</astra_context>"

        # 2. Route to model
        if req.force_tier:
            tier_map = {
                "triage": DebugTier.TRIAGE,
                "analysis": DebugTier.ANALYSIS,
                "agentic": DebugTier.AGENTIC,
            }
            from app.debug.model_router import TIER_MODELS, RoutingDecision
            tier = tier_map.get(req.force_tier, DebugTier.TRIAGE)
            cfg = TIER_MODELS[tier]
            routing = RoutingDecision(
                tier=tier,
                provider=cfg["provider"],
                model=cfg["model"],
                reason=f"Forced tier: {req.force_tier}",
                enable_tools=(tier == DebugTier.AGENTIC),
            )
        else:
            routing = classify_query(req.message, history)

        logger.info(
            "[debug_chat] Routed to %s (%s/%s): %s",
            routing.tier.value, routing.provider, routing.model, routing.reason,
        )

        # Emit metadata
        yield _sse_event("metadata", {
            "provider": routing.provider,
            "model": routing.model,
            "tier": routing.tier.value,
            "reason": routing.reason,
        })

        # 3. Build system prompt
        system_prompt = build_debug_system_prompt(context_xml)

        # 4. Build messages
        messages = list(history) + [{"role": "user", "content": req.message}]

        # Get tools for this tier
        tools = get_tools_for_tier(routing.tier.value)

        # 5. Stream response
        full_response = ""
        async for event in _call_llm_simple(
            provider=routing.provider,
            model=routing.model,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
        ):
            # Capture text tokens for history
            try:
                parsed = json.loads(event.split("data: ", 1)[1].split("\n")[0])
                if parsed.get("content"):
                    full_response += parsed["content"]
            except Exception:
                pass
            yield event

        # 6. Update history
        history.append({"role": "user", "content": req.message})
        if full_response:
            history.append({"role": "assistant", "content": full_response})
        _conversations[session_id] = history

        # 7. Done event
        elapsed_ms = int((time.time() - start_time) * 1000)
        yield _sse_event("done", {
            "total_length": len(full_response),
            "elapsed_ms": elapsed_ms,
            "tier": routing.tier.value,
            "provider": routing.provider,
            "model": routing.model,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{session_id}")
async def get_debug_history(
    session_id: str,
    auth: AuthResult = Depends(require_auth),
):
    """Get conversation history for a debug session."""
    history = _conversations.get(session_id, [])
    return {"session_id": session_id, "messages": history, "count": len(history)}


@router.delete("/history/{session_id}")
async def clear_debug_history(
    session_id: str,
    auth: AuthResult = Depends(require_auth),
):
    """Clear conversation history for a debug session."""
    _conversations.pop(session_id, None)
    return {"session_id": session_id, "cleared": True}


@router.get("/status")
async def debug_status(auth: AuthResult = Depends(require_auth)):
    """Get debug assistant status and configuration."""
    from app.debug.model_router import TIER_MODELS, get_tier_cost_estimate, DebugTier

    tiers = {}
    for tier in DebugTier:
        cfg = TIER_MODELS[tier]
        costs = get_tier_cost_estimate(tier)
        tiers[tier.value] = {
            "provider": cfg["provider"],
            "model": cfg["model"],
            **costs,
        }

    return {
        "status": "active",
        "active_sessions": len(_conversations),
        "max_conversation_length": MAX_CONVERSATION_LENGTH,
        "max_tool_iterations": MAX_TOOL_ITERATIONS,
        "tiers": tiers,
    }


# =============================================================================
# v2.1: DEBUG LOCK — Gemini + tools + RAG (sidebar context lock)
# =============================================================================

async def stream_debug_locked(
    db,
    project_id: int,
    message: str,
    panel_history: list,
    provider: str = "google",
    model: str = "gemini-3.1-pro-preview-customtools",
    debug_project_id: Optional[str] = None,
    video_file_uri: Optional[str] = None,
    video_mime_type: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """Stream handler for debug-locked sidebar chat.

    When the user locks the sidebar to Debug context, all messages route
    here. Gemini gets:
    - Full tool access (sandbox read/write/shell)
    - RAG codebase search
    - Panel conversation history
    - Build project context (if available)
    """
    import asyncio
    from app.debug.tool_definitions import get_tools_for_tier
    from app.debug.context_assembler import assemble_context
    from app.debug.system_prompt import build_debug_system_prompt

    def _sse(data: dict) -> bytes:
        return f"data: {json.dumps(data)}\n\n".encode("utf-8")

    try:
        # 1. Assemble context: RAG search + architecture
        context_block = ""
        try:
            ctx = await assemble_context()
            context_block = ctx.to_xml() if hasattr(ctx, 'to_xml') else str(ctx)
        except Exception as ctx_err:
            logger.warning("[debug_locked] Context assembly failed: %s", ctx_err)
            context_block = "<context>Context assembly unavailable.</context>"

        # Also do a RAG search for the user's query
        rag_context = ""
        try:
            from app.rag.service import search_rag
            rag_results = search_rag(db, message, limit=5)
            if rag_results:
                rag_context = "\n## RAG Search Results\n"
                for r in rag_results:
                    path = getattr(r, 'file_path', '') or r.get('file_path', '') if isinstance(r, dict) else ''
                    content = getattr(r, 'content', '') or r.get('content', '') if isinstance(r, dict) else str(r)
                    rag_context += f"### {path}\n{content[:500]}\n\n"
        except Exception as rag_err:
            logger.debug("[debug_locked] RAG search failed: %s", rag_err)

        # 2. Load build context from debug project (if pipeline-created)
        build_context = ""
        if debug_project_id:
            try:
                from app.debug.project_service import get_project as get_debug_project
                dp = get_debug_project(debug_project_id)
                if dp and dp.get("description"):
                    desc = dp["description"]
                    build_context = (
                        f"\n## Active Debug Project: {dp.get('title', '')}\n"
                        f"Status: {dp.get('status', '')}\n"
                    )
                    # Check if it contains a build report
                    if "--- BUILD REPORT ---" in desc:
                        report_part = desc.split("--- BUILD REPORT ---", 1)[1][:4000]
                        build_context += f"\n### Build Report (summary)\n{report_part}\n"
                    else:
                        build_context += f"\n### Description\n{desc[:2000]}\n"
                    if dp.get("error_summary"):
                        build_context += f"\n### Errors\n{dp['error_summary']}\n"
                    logger.info("[debug_locked] Loaded build context from project %s (%d chars)", debug_project_id, len(build_context))
            except Exception as bp_err:
                logger.debug("[debug_locked] Build context load failed: %s", bp_err)

        # 3. Build system prompt with tools
        system_prompt = (
            build_debug_system_prompt(context_block)
            + "\n\n## Debug Lock Mode\n"
            "You are in DEBUG LOCK mode. The user has locked their sidebar chat to Debug context.\n"
            "You have FULL tool access: read_file, write_file, run_shell, search_files, list_dir.\n"
            "Use tools to gather evidence when the user asks you to investigate or fix something.\n"
            "Be direct, technical, and action-oriented when asked to diagnose issues.\n"
            "Do NOT proactively scan logs or dump diagnostics — wait for the user to tell you what to look at.\n"
            "For greetings or casual messages, just respond naturally and briefly.\n"
            "Host files (D:/Orb, D:/orb-desktop) are READ ONLY. Sandbox writes go via write_file tool.\n"
        )

        # Add video analysis instruction if a screen recording is attached
        if video_file_uri:
            system_prompt += (
                "\n\n## Screen Recording Attached\n"
                "The user has recorded their screen to show you a bug or demonstrate something.\n"
                "Watch the video carefully, listen to their narration, and diagnose the issue.\n"
                "Reference specific things you see in the recording when responding.\n"
                "After viewing, use your tools (read_file, search_files, etc.) to investigate further.\n"
            )

        system_prompt += (
            f"\n## Codebase Context\n{context_block}\n"
            f"{rag_context}\n"
            f"{build_context}\n"
        )

        # 3. Build messages from panel history
        messages = []
        for msg in (panel_history or [])[-20:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        # 4. Emit metadata
        yield _sse({"type": "metadata", "provider": provider, "model": model})

        # 5. Track tool calls for streaming to frontend
        tool_log = []
        def _on_tool(name, args, result_preview):
            tool_log.append({"tool": name, "args": args, "preview": result_preview})

        # 6. Build multimodal content parts if video is attached
        extra_parts = None
        if video_file_uri:
            from app.debug.screen_capture import build_video_content_part
            video_part = build_video_content_part(
                file_uri=video_file_uri,
                mime_type=video_mime_type or "video/webm",
            )
            extra_parts = [video_part]
            logger.info("[debug_locked] Attached video part: %s", video_file_uri)

        # 7. Call Gemini with native function-calling tool loop
        from app.debug.gemini_tool_loop import run_gemini_tool_loop

        content = await run_gemini_tool_loop(
            system_prompt=system_prompt,
            messages=messages,
            model_id=model,
            temperature=0.2,
            max_tokens=8192,
            on_tool_call=_on_tool,
            content_parts=extra_parts,
        )

        # Stream tool call log before the response (if any tools were used)
        if tool_log:
            tool_summary = "\n".join(
                f"🔧 **{t['tool']}** → {t['preview'][:100]}" for t in tool_log
            )
            yield _sse({"type": "token", "content": tool_summary + "\n\n---\n\n"})

        # Stream response in chunks for a natural feel
        chunk_size = 40
        for i in range(0, len(content), chunk_size):
            yield _sse({"type": "token", "content": content[i:i + chunk_size]})

        yield _sse({
            "type": "done",
            "provider": provider,
            "model": model,
        })

    except Exception as e:
        logger.exception("[debug_locked] Stream error: %s", e)
        yield _sse({"type": "error", "error": str(e)})
