# FILE: app/bridge/capability_layer.py
"""
Shared capability layer for bridge chat.

This module provides:
- `run_astra_chat()`    — collect full response (for /bridge/chat)
- `stream_astra_chat()` — yield text tokens as they arrive (for /bridge/chat-and-speak)

Both use the same intelligence path the desktop uses:
  Model selection → context → grounding gate → tool injection → tool loop

Created 2026-04-06 as Phase 1+3 of the "Make AstraBridge a True Bridge" job.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncGenerator, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API
# =============================================================================

async def run_astra_chat(
    message: str,
    project_id: int,
    history: list[dict],
    db: Session,
    source: str = "bridge",
    domain_context: str = "",
    translation_result: Any = None,
    web_search_context: str = "",
    search_executed: bool = False,
    search_succeeded: bool = False,
) -> dict:
    """Run the full ASTRA capability layer and return collected results.

    Returns:
        {"reply": str, "provider": str, "model": str, "tools_used": list}
    """
    prep = await _prepare_astra_chat(
        message, project_id, history, db, source,
        domain_context, translation_result,
        web_search_context, search_executed, search_succeeded,
    )

    # Image route — no LLM call needed
    if prep.get("image_route"):
        return {
            "reply": "Image generation is available on the desktop app. "
                     "From the phone, I can describe images or help plan visual content.",
            "provider": "bridge",
            "model": "capability-gate",
            "tools_used": [],
        }

    provider = prep["provider"]
    model = prep["model"]
    system_prompt = prep["system_prompt"]
    messages = prep["messages"]
    chat_tools = prep["chat_tools"]

    full_response = ""
    tools_used: list[str] = []

    try:
        stream = _create_llm_stream(messages, system_prompt, provider, model, chat_tools)
        async for chunk in stream:
            if isinstance(chunk, dict):
                ctype = chunk.get("type", "")
                if ctype == "token":
                    full_response += chunk.get("text", "")
                elif ctype == "tool_call":
                    tools_used.append(chunk.get("name", "unknown"))
                    print(f"[{source}] Tool call: {chunk.get('name')}")
                elif ctype == "error":
                    err = chunk.get("message", "Unknown error")
                    logger.error("[%s] LLM stream error: %s", source, err)
                    if not full_response:
                        full_response = f"I encountered an error: {err}"
            elif isinstance(chunk, str):
                full_response += chunk
    except Exception as e:
        logger.error("[%s] LLM call failed (%s/%s): %s", source, provider, model, e)
        if not full_response:
            full_response = f"Error generating response: {e}"

    if not full_response:
        full_response = "I wasn't able to generate a response. Please try again."

    print(f"[{source}] Response: {len(full_response)} chars, {len(tools_used)} tool calls")

    return {
        "reply": full_response,
        "provider": provider,
        "model": model,
        "tools_used": tools_used,
    }


async def stream_astra_chat(
    message: str,
    project_id: int,
    history: list[dict],
    db: Session,
    source: str = "bridge-tts",
    domain_context: str = "",
    translation_result: Any = None,
    web_search_context: str = "",
    search_executed: bool = False,
    search_succeeded: bool = False,
) -> AsyncGenerator[dict, None]:
    """Stream text tokens from the full ASTRA capability layer.

    Yields dicts:
        {"type": "token",    "text": "..."}     — text chunk
        {"type": "tool_call","name": "..."}      — tool was invoked (informational)
        {"type": "metadata", "provider": "...", "model": "..."}  — first yield
        {"type": "done",     "full_text": "..."} — final yield with complete text

    Callers (e.g. chat-and-speak) can accumulate tokens into sentences
    and pipe each sentence to TTS as soon as it's complete — while the
    tool loop is still running for subsequent tool calls.
    """
    prep = await _prepare_astra_chat(
        message, project_id, history, db, source,
        domain_context, translation_result,
        web_search_context, search_executed, search_succeeded,
    )

    if prep.get("image_route"):
        yield {"type": "token", "text": "Image generation is available on the desktop app."}
        yield {"type": "done", "full_text": "Image generation is available on the desktop app.", "provider": "bridge", "model": "capability-gate"}
        return

    provider = prep["provider"]
    model = prep["model"]
    system_prompt = prep["system_prompt"]
    messages = prep["messages"]
    chat_tools = prep["chat_tools"]

    yield {"type": "metadata", "provider": provider, "model": model}

    full_text = ""
    try:
        stream = _create_llm_stream(messages, system_prompt, provider, model, chat_tools)
        async for chunk in stream:
            if isinstance(chunk, dict):
                ctype = chunk.get("type", "")
                if ctype == "token":
                    text = chunk.get("text", "")
                    full_text += text
                    yield {"type": "token", "text": text}
                elif ctype == "tool_call":
                    print(f"[{source}] Tool call: {chunk.get('name')}")
                    yield {"type": "tool_call", "name": chunk.get("name", "")}
                elif ctype == "error":
                    err = chunk.get("message", "Unknown error")
                    logger.error("[%s] Stream error: %s", source, err)
                    if not full_text:
                        full_text = f"I encountered an error: {err}"
                        yield {"type": "token", "text": full_text}
            elif isinstance(chunk, str):
                full_text += chunk
                yield {"type": "token", "text": chunk}
    except Exception as e:
        logger.error("[%s] LLM call failed (%s/%s): %s", source, provider, model, e)
        if not full_text:
            full_text = f"Error generating response: {e}"
            yield {"type": "token", "text": full_text}

    if not full_text:
        full_text = "I wasn't able to generate a response. Please try again."
        yield {"type": "token", "text": full_text}

    print(f"[{source}] Stream complete: {len(full_text)} chars")
    yield {"type": "done", "full_text": full_text, "provider": provider, "model": model}


# =============================================================================
# SHARED SETUP (used by both run_ and stream_)
# =============================================================================

async def _prepare_astra_chat(
    message: str,
    project_id: int,
    history: list[dict],
    db: Session,
    source: str,
    domain_context: str,
    translation_result: Any,
    web_search_context: str,
    search_executed: bool,
    search_succeeded: bool,
) -> dict:
    """Prepare everything needed for an ASTRA chat call.

    Returns a dict with keys:
        provider, model, system_prompt, messages, chat_tools
        image_route (bool, if image gen was detected)
    """
    from app.llm.routing.prompt_builders import build_system_prompt, build_full_context
    from app.llm.routing.chat_model_selection import select_chat_model
    from app.llm.routing.chat_intent_detection import detect_codebase_exploration
    from app.memory import service as memory_service

    # ── 1. Build context ──
    full_context = build_full_context(db, project_id, message, use_semantic_search=True)
    if domain_context:
        full_context += f"\n\n{domain_context}"

    # ── 2. Model selection ──
    _bridge_req = _BridgeRequest(message=message, project_id=project_id)
    provider, model, extras = select_chat_model(_bridge_req, db)

    if extras.get("image_route"):
        return {"image_route": True}

    print(f"[{source}] Model selected: {provider}/{model}")

    # ── 3. Build messages ──
    messages = []
    for entry in history[-20:]:
        role = entry.get("role", "user")
        content = entry.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    if not messages or messages[-1].get("content") != message:
        messages.append({"role": "user", "content": message})

    # ── 4. Codebase context ──
    codebase_ctx = ""
    try:
        from app.llm.routing.chat_codebase_reader import gather_codebase_context
        codebase_ctx = gather_codebase_context(message=message, model=model, db=db)
        if codebase_ctx:
            full_context += f"\n\n{codebase_ctx}"
            print(f"[{source}] Codebase context: {len(codebase_ctx)} chars")
    except Exception as e:
        print(f"[{source}] Codebase context failed (non-fatal): {e}")

    # ── 5. System prompt ──
    try:
        project = memory_service.get_project(db, project_id)
    except Exception:
        project = None

    system_prompt = build_system_prompt(project, full_context, ui_context=None)
    system_prompt += (
        "\n\nCONTEXT: The user is on the Astra Bridge mobile app. "
        "They may be driving or on the move. Keep responses concise but complete. "
        "You have the same capabilities as the desktop — use tools when needed.\n"
    )

    # ── 6. Grounding gate ──
    try:
        from app.grounding.chat_integration import run_grounding_sync
        system_prompt, _meta = run_grounding_sync(
            message=message, system_prompt=system_prompt,
            context={"user_id": "bridge"},
        )
        if _meta.get("grounding_applied"):
            print(f"[{source}] Grounding gate ACTIVE: {_meta.get('category')}")
    except Exception as e:
        print(f"[{source}] Grounding gate error (non-fatal): {e}")

    # ── 7. Web search + honesty ──
    if web_search_context:
        system_prompt += "\n\n" + web_search_context
    try:
        from app.bridge.capability_honesty import build_honesty_system_addendum
        _intent = (
            translation_result.resolved_intent.value
            if translation_result and translation_result.resolved_intent else None
        )
        addendum = build_honesty_system_addendum(_intent, search_executed, search_succeeded)
        if addendum:
            system_prompt += addendum
    except Exception:
        pass

    # ── 8. Tool injection ──
    chat_tools = _inject_chat_tools(provider, model, message, source, system_prompt)
    # _inject_chat_tools may modify system_prompt, so we capture its return
    chat_tools, system_prompt = chat_tools

    return {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "messages": messages,
        "chat_tools": chat_tools,
    }


def _inject_chat_tools(
    provider: str,
    model: str,
    message: str,
    source: str,
    system_prompt: str,
) -> tuple[list | None, str]:
    """Inject tool definitions and update system prompt.  Returns (tools, updated_prompt)."""
    from app.llm.routing.chat_intent_detection import detect_codebase_exploration
    from app.llm.routing.chat_model_selection import set_sticky_model

    chat_tools = None
    try:
        from app.llm.chat_tool_loop import is_tool_eligible, get_chat_tools

        if not is_tool_eligible(provider, model):
            from app.memory.complexity import DEEP_KEYWORDS, _count_keyword_hits
            _deep_hits = _count_keyword_hits(message.lower(), DEEP_KEYWORDS)
            if detect_codebase_exploration(message) or _deep_hits >= 1:
                _tp = os.getenv("TOOL_CHAT_PROVIDER", "google")
                _tm = os.getenv("TOOL_CHAT_MODEL", "gemini-3.1-pro-preview-customtools")
                if is_tool_eligible(_tp, _tm):
                    print(f"[{source}] Swapping to tool-capable: {_tp}/{_tm}")
                    provider = _tp
                    model = _tm

        if is_tool_eligible(provider, model):
            chat_tools = get_chat_tools()
            print(f"[{source}] Tool access ENABLED: {provider}/{model} ({len(chat_tools)} tools)")
            system_prompt += (
                "\n\n## TOOL ACCESS\n"
                "You have tool access for exploring the codebase AND writing to user folders.\n\n"
                "CODEBASE TOOLS (read-only): read_file, list_files, search_files, read_logs, search_my_files, read_user_file\n"
                "USER FILE TOOLS (read+write): get_user_folders, write_user_file\n"
                "CLOUD STORAGE TOOLS: cloud_upload (upload local file to Google Drive), cloud_list (list Drive contents)\n"
                "Use get_user_folders to discover real folder paths, then write_user_file to save files there.\n\n"
                "IMPORTANT: Actually USE the tools. Do not just say you will — call them.\n"
            )
    except ImportError:
        print(f"[{source}] chat_tool_loop not available")

    # Universal web search
    try:
        from app.debug.tool_definitions import get_universal_tools
        from app.llm.chat_tool_loop import _to_anthropic_tool_format
        _universal = [_to_anthropic_tool_format(t) for t in get_universal_tools()]
        if chat_tools is not None:
            _existing = {t.get("name") for t in chat_tools}
            for ut in _universal:
                if ut["name"] not in _existing:
                    chat_tools.append(ut)
        else:
            chat_tools = _universal
            system_prompt += (
                "\n\n## WEB SEARCH TOOL\n"
                "You have access to a web_search tool. Use it when you need current information.\n"
                "IMPORTANT: Actually CALL the tool. Do not just say you will.\n"
            )

        # Memory tools addendum — always appended, whether or not other
        # tools were available. The rule against confabulated memory
        # writes is load-bearing: without it, the model will claim to
        # have saved things when no write happened.
        system_prompt += _build_memory_tools_addendum()
    except ImportError:
        pass

    return chat_tools, system_prompt


def _build_memory_tools_addendum() -> str:
    """System-prompt addendum describing the memory tools and enforcing
    the no-confabulation rule. Kept here rather than in the tool
    definition itself so we can tune it centrally."""
    return (
        "\n\n## MEMORY TOOLS — CRITICAL RULES\n"
        "You have tools to write to ASTRA's tiered memory system: "
        "save_to_memory, update_memory, forget_memory, search_memory, save_residence.\n\n"
        "**HARD RULE**: NEVER claim to have saved, remembered, stored, "
        "noted, or recorded anything unless you have actually called "
        "save_to_memory (or save_residence / update_memory) in THIS turn "
        "AND received a result with saved=true. If you did not call the "
        "tool, you did not save anything — do not tell the user you did.\n\n"
        "**When to save proactively**:\n"
        "- The user states a durable biographical fact (where they've lived, "
        "what they do, family, background)\n"
        "- The user states a preference they want to persist ('I prefer X', "
        "'I always want Y', 'from now on Z')\n"
        "- The user explicitly says 'remember this' / 'don't forget' / "
        "'save that' — use weight=5 and permanence=permanent for these\n"
        "- The user corrects a previously-stored fact — use update_memory, "
        "not save_to_memory\n\n"
        "**When NOT to save**:\n"
        "- Aspirations stated in passing ('I might go to Spain one day')\n"
        "- Current-session context that won't matter tomorrow\n"
        "- Questions the user is asking you — those aren't facts\n"
        "- API keys, passwords, secrets — tools refuse these anyway\n\n"
        "**Residence history**: When the user mentions a PAST place they "
        "lived ('I grew up in X', 'I lived in Y for N years'), use "
        "save_residence, not save_to_memory. For where they live NOW, use "
        "save_to_memory with key='current_location' and permanence=permanent.\n\n"
        "**Answering 'what do you know about me' questions**: Call "
        "search_memory first to get the real state, then answer from that. "
        "Do not guess from conversation context alone.\n"
    )


def _create_llm_stream(messages, system_prompt, provider, model, chat_tools):
    """Create the appropriate LLM stream (with or without tools)."""
    if chat_tools:
        from app.llm.chat_tool_loop import stream_with_tools
        return stream_with_tools(
            messages=messages,
            system_prompt=system_prompt,
            provider=provider,
            model=model,
            tools=chat_tools,
            enable_reasoning=False,
            max_tokens=16384,
        )
    else:
        from app.llm.streaming import stream_llm
        return stream_llm(
            provider=provider,
            model=model,
            messages=messages,
            system_prompt=system_prompt,
        )


class _BridgeRequest:
    """Minimal request-like object for select_chat_model compatibility."""

    def __init__(self, message: str, project_id: int, provider=None, model=None):
        self.message = message
        self.project_id = project_id
        self.provider = provider
        self.model = model
        self.ui_context = None
        self.attachments = None
        self.file_upload_local_path = None
        self.file_upload_name = None
        self.file_upload_mime = None
        self.file_upload_gemini_name = None
        self.file_upload_uri = None
