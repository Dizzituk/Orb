# FILE: app/debug/debug_chat_endpoint.py
# Purpose: Debug Chat Endpoint (JOB A) — /api/debug/chat SSE surface + history/status (split from debug_chat.py).
# Called-by: app.debug.debug_chat
# Depends-on: app.auth, app.auth.middleware, app.debug.context_assembler, app.debug.model_router, app.debug.system_prompt, app.debug.tool_definitions
# Last-renovated: 2026-07-02 (TIER_MODELS -> tier_model_config: call-time provider/model)
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
    reasoning: Optional[dict] = None,
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
            temperature=(1.0 if reasoning else 0.2),
            max_tokens=max_tokens,
            timeout_seconds=(300 if reasoning else 120),
            enable_tools=False,  # Phase 1: no tool execution via registry
            reasoning=reasoning,
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
            from app.debug.model_router import tier_model_config, RoutingDecision
            tier = tier_map.get(req.force_tier, DebugTier.TRIAGE)
            cfg = tier_model_config(tier)
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
            reasoning=routing.reasoning,
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
    from app.debug.model_router import tier_model_config, get_tier_cost_estimate, DebugTier

    tiers = {}
    for tier in DebugTier:
        cfg = tier_model_config(tier)
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
