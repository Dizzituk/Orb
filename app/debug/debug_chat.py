# FILE: app/debug/debug_chat.py
# Purpose: Debug Chat Endpoint: SSE streaming endpoint for the Debug Assistant.
# Called-by: app.llm.stream_router, main
# Depends-on: app.auth, app.auth.middleware, app.cost.cost_recorder, app.debug.context_assembler (+16 more)
# Last-renovated: 2026-06-11
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
    provider: str = "openai",
    model: Optional[str] = None,
    debug_project_id: Optional[str] = None,
    video_file_uri: Optional[str] = None,
    video_mime_type: Optional[str] = None,
    video_local_path: Optional[str] = None,
    file_upload_uri: Optional[str] = None,
    file_upload_mime: Optional[str] = None,
    file_upload_name: Optional[str] = None,
    file_upload_local_path: Optional[str] = None,
    file_upload_gemini_name: Optional[str] = None,
    documents: Optional[list] = None,
    reasoning_dial: Optional[str] = None,
) -> AsyncGenerator[bytes, None]:
    """Stream handler for debug-locked sidebar chat.

    v3.0 (2026-03-21): Hybrid architecture — Gemini vision + Claude Opus tools.

    Two-phase approach:
      Phase 1 (Gemini 3.1): If video/image is attached, send to Gemini for
        visual-only analysis. No tools, just observation. Returns structured
        text describing what was seen on screen.
      Phase 2 (Claude Opus 4.6): Full investigation with tool access. Gets
        Gemini's visual analysis as context, plus RAG, build context, and
        codebase access. Claude does the actual reasoning, diagnosis, and
        code changes with interleaved streaming.

    When no video/image is attached, skips Phase 1 and goes straight to
    Claude Opus with tools.
    """
    from app.debug.context_assembler import assemble_context
    from app.debug.system_prompt import build_debug_system_prompt
    from app.debug.gemini_vision import analyse_visual_content, cleanup_uploads
    from app.llm.chat_tool_loop import get_chat_tools, TOOL_TIER_WRITE

    def _sse(data: dict) -> bytes:
        return f"data: {json.dumps(data)}\n\n".encode("utf-8")

    try:
        # ═══════════════════════════════════════════════════════════════
        # Phase 1: Gemini Vision Analysis (only if video/image attached)
        # ═══════════════════════════════════════════════════════════════

        # --- AUTO-ROUTE: orchestrator vs chat ----------------------------
        try:
            from app.debug.orchestrator.router_classifier import classify_message
            from app.debug.orchestrator.chat_stream_bridge import (
                stream_orchestration_as_chat,
            )
            prior_assistant = ""
            for m in reversed(panel_history or []):
                if m.get("role") == "assistant":
                    prior_assistant = (m.get("content") or "")[:4000]
                    break
            lower_prior = prior_assistant.lower()
            history_has_plan = (
                ("plan" in lower_prior or "steps" in lower_prior
                 or "next i" in lower_prior or "going to" in lower_prior)
                and (
                    "1." in prior_assistant or "- " in prior_assistant
                    or "* " in prior_assistant
                )
            )
            decision = await classify_message(
                message=message, history_has_plan=history_has_plan,
            )
            logger.info(
                "[debug_locked] router decision=%s source=%s conf=%.2f reason=%s",
                decision.get("decision"), decision.get("source"),
                decision.get("confidence", 0.0), decision.get("reason", ""),
            )

            if decision.get("decision") == "orchestrate" and debug_project_id:
                from app.debug.project_service import get_project as _get_proj
                dp = _get_proj(debug_project_id)
                bug_source = ""
                if dp:
                    bug_source = (dp.get("description") or "").strip()
                    if not bug_source:
                        bug_source = dp.get("error_summary") or ""
                if len(message) > len(bug_source):
                    bug_source = message
                target_hint = None
                try:
                    meta = json.loads((dp or {}).get("metadata_json", "{}") or "{}")
                    target_hint = meta.get("build_target_id")
                except Exception:
                    pass

                logger.info(
                    "[debug_locked] ROUTING TO ORCHESTRATOR (project=%s, target=%s)",
                    debug_project_id, target_hint,
                )
                # Record user message in activity store
                try:
                    from app.debug.orchestrator.activity_store import record_user_message
                    record_user_message(debug_project_id=str(debug_project_id), message=message)
                except Exception as _act_err:
                    logger.debug("[debug_locked] activity log (user) failed: %s", _act_err)
                full_response = ""
                async for chunk in stream_orchestration_as_chat(
                    project_id=str(debug_project_id),
                    bug_list=bug_source,
                    target_project=target_hint,
                    max_iterations=2,
                    max_subagents_parallel=5,
                    enable_behaviour_verify=False,
                ):
                    try:
                        payload = json.loads(chunk.decode("utf-8").split("data: ", 1)[1].strip())
                        if payload.get("type") == "token" and payload.get("content"):
                            full_response += payload["content"]
                    except Exception:
                        pass
                    yield chunk

                try:
                    from app.memory import service as mem_svc, schemas as mem_schemas
                    mem_svc.create_message(db, mem_schemas.MessageCreate(
                        project_id=project_id, role="user", content=message,
                        provider="user", model="debug-input",
                    ))
                    if full_response.strip():
                        mem_svc.create_message(db, mem_schemas.MessageCreate(
                            project_id=project_id, role="assistant",
                            content=full_response, provider="orchestrator",
                            model="orchestrator",
                        ))
                except Exception as _persist_err:
                    logger.warning("[debug_locked] orchestrator persistence failed: %s", _persist_err)
                return
        except Exception as _route_err:
            logger.warning("[debug_locked] auto-router failed, continuing with chat: %s", _route_err)
        # --- END AUTO-ROUTE ---------------------------------------------

        vision_analysis = None
        has_visual = bool(video_file_uri) or bool(file_upload_uri)

        if has_visual:
            yield _sse({"type": "token", "content": "🔍 Analysing visual content with Gemini...\n"})

            vision_analysis = await analyse_visual_content(
                user_message=message,
                video_file_uri=video_file_uri,
                video_mime_type=video_mime_type,
                file_upload_uri=file_upload_uri,
                file_upload_mime=file_upload_mime,
                file_upload_name=file_upload_name,
            )

            if vision_analysis:
                # Stream a brief summary so the user knows vision analysis completed
                preview = vision_analysis[:150].replace('\n', ' ')
                yield _sse({"type": "token", "content": f"  → Visual analysis complete ({len(vision_analysis)} chars)\n\n"})
                logger.info("[debug_locked] Vision analysis: %d chars", len(vision_analysis))
            else:
                yield _sse({"type": "token", "content": "  → Visual analysis unavailable, proceeding with text context\n\n"})

        # ═══════════════════════════════════════════════════════════════
        # Context Assembly (shared by both phases)
        # ═══════════════════════════════════════════════════════════════

        # 1. Assemble context: RAG search + architecture
        context_block = ""
        try:
            ctx = await assemble_context()
            context_block = ctx.to_xml() if hasattr(ctx, 'to_xml') else str(ctx)
        except Exception as ctx_err:
            logger.warning("[debug_locked] Context assembly failed: %s", ctx_err)
            context_block = "<context>Context assembly unavailable.</context>"

        # RAG search for the user's query
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
                    try:
                        meta = json.loads(dp.get("metadata_json", "{}") or "{}")
                        target_id = meta.get("build_target_id")
                        if target_id:
                            from app.pipeline_v2.target_registry import get_profile
                            target_profile = get_profile(target_id)
                            if target_profile:
                                build_context += (
                                    f"\n### Build Target\n"
                                    f"Project: {target_profile.project_name}\n"
                                    f"Root: `{target_profile.project_root}`\n"
                                    f"Language: {target_profile.language} | Framework: {target_profile.framework}\n"
                                    f"Architecture: {target_profile.architecture_pattern}\n"
                                    f"Source root: `{target_profile.absolute_source_root}`\n"
                                    f"Package: `{target_profile.package_name}`\n"
                                )
                                if target_profile.key_directories:
                                    dirs = ", ".join(f"{k} (`{v}`)" for k, v in target_profile.key_directories.items())
                                    build_context += f"Key dirs: {dirs}\n"
                    except Exception as meta_err:
                        logger.debug("[debug_locked] Build target resolution failed: %s", meta_err)
                    if "--- BUILD REPORT ---" in desc:
                        report_part = desc.split("--- BUILD REPORT ---", 1)[1][:4000]
                        build_context += f"\n### Build Report (summary)\n{report_part}\n"
                    else:
                        build_context += f"\n### Description\n{desc[:2000]}\n"
                    if dp.get("error_summary"):
                        build_context += f"\n### Errors\n{dp['error_summary']}\n"
            except Exception as bp_err:
                logger.debug("[debug_locked] Build context load failed: %s", bp_err)

        # 2b. If no debug project, try resolving project from the message
        if not build_context:
            try:
                from app.pipeline_v2.target_registry import resolve_project_from_message
                resolved = resolve_project_from_message(message, panel_history)
                if resolved:
                    build_context = (
                        f"\n## Detected Project Context\n"
                        f"Project: {resolved.project_name}\n"
                        f"Root: `{resolved.project_root}`\n"
                        f"Language: {resolved.language} | Framework: {resolved.framework}\n"
                        f"Architecture: {resolved.architecture_pattern}\n"
                        f"Source root: `{resolved.absolute_source_root}`\n"
                        f"Package: `{resolved.package_name}`\n"
                    )
                    if resolved.key_directories:
                        dirs = ", ".join(f"{k} (`{v}`)" for k, v in resolved.key_directories.items())
                        build_context += f"Key dirs: {dirs}\n"
                    logger.info("[debug_locked] Resolved project from message: %s", resolved.project_name)
            except Exception as resolve_err:
                logger.debug("[debug_locked] Project resolution failed: %s", resolve_err)

        # ═══════════════════════════════════════════════════════════════
        # Phase 2: Claude Opus Investigation (with tools)
        # ═══════════════════════════════════════════════════════════════

        # 3. Build system prompt
        system_prompt = build_debug_system_prompt(context_block, reasoning_dial=reasoning_dial)
        system_prompt += (
            "\n\n## Debug Lock Mode — Claude Opus\n"
            "You are in DEBUG LOCK mode with FULL tool access.\n"
            "You have tools to read files, write files, edit files, run commands, "
            "search files, and list directories across the ASTRA codebase.\n"
            "Use tools to gather evidence when investigating issues.\n"
            "Be direct, technical, and action-oriented.\n"
            "For greetings or casual messages, just respond naturally and briefly.\n\n"
            "## How to Apply Changes\n"
            "Use your tools directly — do NOT paste code in chat and tell the user to copy it.\n"
            "read_file reads any file. write_file creates or overwrites. edit_file does find-and-replace.\n"
            "run_command runs PowerShell commands for testing and builds.\n"
            "search_files finds files by pattern. list_files lists directory contents.\n\n"
            "## Android Emulator Testing (CRITICAL)\n"
            "You have ADB tools to interact with the Android emulator directly.\n"
            "ALWAYS verify your changes by building, deploying, and testing:\n\n"
            "### Test-Iterate Loop:\n"
            "1. Make code changes with write_file/edit_file\n"
            "2. Run gradle_build to check compilation\n"
            "3. If build fails → read errors, fix, rebuild\n"
            "4. If build succeeds → run gradle_install to deploy to emulator\n"
            "5. Run app_restart to launch the new build\n"
            "6. Use emulator_screenshot to capture the screen\n"
            "7. Use emulator_ui_dump to inspect the UI hierarchy\n"
            "8. Use emulator_tap, emulator_type, emulator_key to interact\n"
            "9. If something is wrong → fix code → go to step 2\n"
            "10. If all looks correct → report success to user\n\n"
            "### BEFORE writing code that calls any function:\n"
            "ALWAYS read the target file first to verify the actual function signatures.\n"
            "Do NOT assume function names or parameter names. Read first, write second.\n\n"
            "### After ANY code change:\n"
            "ALWAYS run gradle_build to verify compilation before telling the user it is done.\n"
            "If the build fails, fix the errors and rebuild. Do not report success until BUILD SUCCESSFUL.\n\n"
            "### CRITICAL — Do NOT auto-build or auto-deploy\n"
            "Only run gradle_build, gradle_install, or any build/deploy tools when the user has "
            "EXPLICITLY asked you to build, compile, deploy, or install.\n"
            "If the user is asking a QUESTION about builds, discussing architecture, or talking "
            "about how something works — just answer the question. Do NOT trigger a build.\n"
            "Questions like 'how do we compile an APK?' or 'what does the build process do?' "
            "are NOT requests to build. Only imperative instructions like 'build it', 'make an APK', "
            "'deploy to the phone' are genuine build requests.\n\n"
            "You have full filesystem access via absolute paths (D:/Orb/..., D:/Astra Android Folder/...).\n"
            "\n## Testing by Environment\n"
            "Match your testing tools to the project you are working on:\n\n"
            "### Android projects (Astra Bridge, Driver CoPilot):\n"
            "- Use gradle_build / gradle_install to compile and deploy\n"
            "- Use app_restart to launch on the emulator\n"
            "- Use emulator_screenshot / emulator_ui_dump to verify\n"
            "- Use emulator_tap / emulator_type / emulator_key to interact\n"
            "- Use get_crash_log to diagnose crashes\n\n"
            "### Desktop projects (ASTRA Backend, ASTRA Desktop):\n"
            "- Use run_command to start/restart services\n"
            "- Use run_command to run Python/TypeScript syntax checks\n"
            "- Use run_command with curl to test API endpoints\n"
            "- For frontend: the user will verify visually or send screenshots\n\n"
            "### For ALL projects:\n"
            "- ALWAYS compile/build check after code changes\n"
            "- ALWAYS read function signatures before writing calls\n"
            "- NEVER report success without verifying the build passes\n"
            "- If you have emulator access and you changed Android code, USE IT to test\n"
        )

        # Inject Gemini's visual analysis as context
        if vision_analysis:
            system_prompt += (
                "\n\n## Visual Analysis (from screen recording / screenshot)\n"
                "Google Gemini has watched the user's screen recording or analysed their "
                "screenshot and produced the following detailed report. This is your primary "
                "evidence of what the user experienced. You HAVE this information — do not "
                "ask the user to show you anything or claim you cannot see the video.\n\n"
                "When you respond, begin by briefly confirming what Gemini observed, e.g.:\n"
                "\"Gemini has analysed your recording. Here is what I understand...\"\n"
                "Then proceed with your diagnosis based on the evidence below.\n\n"
                "--- START OF VISUAL ANALYSIS ---\n"

                f"{vision_analysis}\n"
                "--- END OF VISUAL ANALYSIS ---\n"
            )

        system_prompt += (
            f"\n## Codebase Context\n{context_block}\n"
            f"{rag_context}\n"
            f"{build_context}\n"
        )

        # 4. Build messages from panel history
        messages = []
        for msg in (panel_history or [])[-20:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        # 4b. Format structured documents (from large text pastes)
        if documents:
            doc_blocks = []
            for doc in documents:
                idx = doc.get("index", 0)
                label = doc.get("label", f"Document {idx}")
                content = doc.get("content", "")
                doc_blocks.append(
                    f'<document index="{idx}" label="{label}">\n{content}\n</document>'
                )
            doc_xml = "\n\n".join(doc_blocks)
            # Prepend documents to the user message so the model sees them as structured context
            messages[-1]["content"] = (
                f"<documents>\n{doc_xml}\n</documents>\n\n"
                + messages[-1]["content"]
            )
            logger.info("[debug_locked] Injected %d document(s) into user message", len(documents))

        # 4c. Load past user corrections for debug context
        try:
            from app.debug.feedback import get_relevant_corrections
            corrections_block = get_relevant_corrections(project_id=project_id)
            if corrections_block:
                system_prompt += "\n\n" + corrections_block
                logger.info("[debug_locked] Injected %d chars of past corrections", len(corrections_block))
        except Exception as corr_err:
            logger.debug("[debug_locked] Corrections load failed: %s", corr_err)

        # 5. Emit metadata - Debug brain model from ENV (DEBUG_CHAT_MODEL -> OPENAI_DEFAULT_MODEL); no hardcode (spec section 3)
        from app.debug.debug_model_config import get_debug_chat_model
        actual_provider = "openai"
        actual_model = (model or "").strip() or get_debug_chat_model()
        yield _sse({"type": "metadata", "provider": actual_provider, "model": actual_model})

        # 6. Get write-tier tools in Anthropic format
        tools = get_chat_tools(TOOL_TIER_WRITE)

        # 7. Stream Claude Opus with tool loop — interleaved reasoning + tool calls
        from app.llm.chat_tool_loop import stream_with_tools

        full_response = ""
        total_prompt_tokens = 0
        total_completion_tokens = 0

        async for chunk in stream_with_tools(
            messages=messages,
            system_prompt=system_prompt,
            provider=actual_provider,
            model=actual_model,
            tools=tools,
            enable_reasoning=True,  # Debug tab: always reason. Non-negotiable — the whole venue is for thinking work.
            max_tokens=24576,  # Room for reasoning tokens + visible output. Reasoning counts against this budget.
        ):
            chunk_type = chunk.get("type", "")

            if chunk_type == "token":
                text = chunk.get("text", "")
                full_response += text
                yield _sse({"type": "token", "content": text})

            elif chunk_type == "tool_call":
                yield _sse({
                    "type": "tool_call",
                    "name": chunk.get("name", ""),
                    "tool_use_id": chunk.get("tool_use_id", ""),
                    "input": chunk.get("input", {}),
                })

            elif chunk_type == "tool_result":
                yield _sse({
                    "type": "tool_result",
                    "name": chunk.get("name", ""),
                    "result_preview": chunk.get("result_preview", ""),
                })

            elif chunk_type == "reasoning":
                yield _sse({"type": "reasoning", "content": chunk.get("text", "")})

            elif chunk_type == "narration" or chunk_type.startswith("subagent"):
                # Phase-level narration (intent / reflection lines, debug-visibility
                # spec) + sub-agent fan-out activity from spawn_agents (subagent_spawn /
                # _start / _progress / _complete / _spawn_complete) -- forward verbatim
                # so the frontend renders the plan lines + the live fan-out box.
                yield _sse(chunk)

            elif chunk_type == "done":
                # Extract usage if present
                usage = chunk.get("usage", {})
                if usage:
                    total_prompt_tokens += usage.get("prompt_tokens", 0)
                    total_completion_tokens += usage.get("completion_tokens", 0)
                yield _sse({
                    "type": "done",
                    "provider": actual_provider,
                    "model": actual_model,
                })

            elif chunk_type == "error":
                yield _sse({"type": "error", "error": chunk.get("error", "Unknown error")})

        # ═══════════════════════════════════════════════════════════════
        # Record API cost
        # ═══════════════════════════════════════════════════════════════

        try:
            from app.cost.cost_recorder import record_llm_cost
            cost = record_llm_cost(
                provider=actual_provider,
                model=actual_model,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                stage="debug_locked",
            )
            if cost > 0.01:
                yield _sse({"type": "token", "content": f"\n\n---\n*Cost: ${cost:.4f} ({total_prompt_tokens}+{total_completion_tokens} tokens)*\n"})
        except Exception as cost_err:
            logger.debug("[debug_locked] Cost recording failed: %s", cost_err)

        # ═══════════════════════════════════════════════════════════════
        # Persist messages to database (survives restart)
        # ═══════════════════════════════════════════════════════════════

        try:
            from app.memory import service as mem_svc, schemas as mem_schemas
            # Save user message
            mem_svc.create_message(db, mem_schemas.MessageCreate(
                project_id=project_id,
                role="user",
                content=message,
                provider="user",
                model="debug-input",
            ))
            # Save assistant response
            if full_response.strip():
                mem_svc.create_message(db, mem_schemas.MessageCreate(
                    project_id=project_id,
                    role="assistant",
                    content=full_response,
                    provider=actual_provider,
                    model=actual_model,
                ))
            logger.info("[debug_locked] Persisted messages to project %d", project_id)
        except Exception as persist_err:
            logger.warning("[debug_locked] Message persistence failed: %s", persist_err)

        # ═══════════════════════════════════════════════════════════════
        # Cleanup: delete recordings/uploads from Gemini Files API + disk
        # ═══════════════════════════════════════════════════════════════

        await cleanup_uploads(
            video_file_uri=video_file_uri,
            video_local_path=video_local_path,
            file_upload_gemini_name=file_upload_gemini_name,
            file_upload_local_path=file_upload_local_path,
        )

    except Exception as e:
        logger.exception("[debug_locked] Stream error: %s", e)
        yield _sse({"type": "error", "error": str(e)})