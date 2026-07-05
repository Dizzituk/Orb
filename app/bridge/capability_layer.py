# FILE: app/bridge/capability_layer.py
"""
Shared capability layer for bridge chat.

This module provides:
- `run_astra_chat()`    — collect full response (used by /bridge/chat AND
  /bridge/chat-and-speak, which then pipes the collected reply to TTS)
- `stream_astra_chat()` — token-streaming variant (currently unused/reserved)

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
# CODEBASE-QUESTION DETECTION (phone/voice parity with desktop RAG, 2026-06-14)
# =============================================================================

async def _detect_codebase_question(
    message: str, translation_result: Any, db: Session
) -> Optional[str]:
    """Return the question if it's about ASTRA's OWN code/features (so the phone
    answers from the RAG index, like the desktop does), else None.

    Mirrors the desktop chain: Tier-0 RAG intent → deterministic regex+guard →
    LLM safety-net. Identity/capability questions ("what are your abilities") are
    excluded — those are answered in normal chat via the capability manifest.
    All failures are non-fatal (→ None → normal chat).
    """
    # Tier-0 already resolved it to a codebase query.
    try:
        intent = (
            translation_result.resolved_intent.value
            if translation_result and getattr(translation_result, "resolved_intent", None)
            else None
        )
        if intent == "RAG_CODEBASE_QUERY":
            return message
    except Exception:
        pass

    try:
        from app.llm.routing.rag_fallback import (
            is_architecture_query, needs_llm_codebase_check,
        )
        if is_architecture_query(message):
            return message
        if needs_llm_codebase_check(message):
            try:
                from app.astra_memory.topic_tagger import extract_tags
                if "identity_capability" in (extract_tags(message) or []):
                    return None
            except Exception:
                pass
            from app.llm.routing.codebase_intent_llm import is_codebase_question_llm_async
            if await is_codebase_question_llm_async(message):
                return message
    except Exception as e:
        logger.warning("[bridge] codebase-question detection failed (non-fatal): %s", e)
    return None


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
    raw_message: Optional[str] = None,
    client_request_id: Optional[str] = None,
) -> dict:
    """Run the full ASTRA capability layer and return collected results.

    `client_request_id` (optional) is the phone's X-Idempotency-Key, threaded
    through so the per-turn image guard can dedupe a retried image request to
    a single paid generation. When absent the guard falls back to a hash of
    (project_id + message), so the text path is covered either way.

    Returns:
        {"reply": str, "provider": str, "model": str, "tools_used": list}
    """
    # Surface tag (2026-07-01): registry tools executed inside this turn's
    # tool loop read the surface via app.llm.turn_surface (the chat tool loop
    # itself passes no context down). Tools that produce a phone-deliverable
    # file register it as a pending artifact; markers are appended after the
    # stream finishes — the model never has to echo a marker itself.
    from app.llm.turn_surface import begin_turn, drain_turn_artifacts, format_artifact_markers
    begin_turn("bridge")

    prep = await _prepare_astra_chat(
        message, project_id, history, db, source,
        domain_context, translation_result,
        web_search_context, search_executed, search_succeeded,
        raw_message=raw_message,
    )

    # RAG route — codebase/feature question: answer from the architecture index
    # (parity with the desktop RAG path). Returns plain text for the phone.
    if prep.get("rag_route"):
        from app.rag.answerer import ask_architecture_async
        try:
            ans = await ask_architecture_async(
                db=db, question=prep["rag_question"], project_id=str(project_id),
            )
            reply = ans.answer or "I couldn't find anything about that in my codebase index."
            return {
                "reply": reply,
                "provider": "rag",
                "model": ans.model_used or "rag_answerer",
                "tools_used": ["rag_codebase"],
            }
        except Exception as e:
            logger.error("[%s] RAG route failed: %s", source, e)
            return {
                "reply": "I tried to search my own codebase but the lookup failed. Try again in a moment.",
                "provider": "rag",
                "model": "rag_answerer-failed",
                "tools_used": ["rag_codebase"],
            }

    # Image route — bypass the LLM call and run the same image-gen
    # pipeline the desktop uses (classification, refinement detection,
    # prompt synthesis, primary + fallback provider). Result file lands
    # in IMAGE_OUTPUT_DIR. Reply text is TTS-friendly (no markdown URL)
    # and ends with an [ASTRA_ARTIFACT:image:<filename>] marker that the
    # bridge router strips before sending to the phone and uses to build
    # the X-Artifacts header / attachments field. Marker stays in the DB
    # so historical messages can re-emit the chip on reload.
    if prep.get("image_route"):
        from app.llm.image_router import generate_image_core
        from app.llm.image_turn_guard import run_guarded_image
        # Per-turn cost guard (J2): a slow image request delivered 3 times
        # (phone retries on timeout) must bill ONE paid generation, not three.
        # The guard caches the first artifact for a stable key (client id, else
        # project+message hash) and replays it / joins an in-flight gen on
        # repeats within the window. generate_image_core is wrapped as a
        # zero-arg coroutine factory so the guard owns the single call.
        try:
            result = await run_guarded_image(
                project_id=project_id,
                message=message,
                generator=lambda: generate_image_core(project_id, message, db),
                client_request_id=client_request_id,
                source=source,
            )
        except Exception as e:
            logger.error("[%s] Image generation crashed: %s", source, e)
            result = None

        if not result:
            return {
                "reply": (
                    "I tried to make the image but the generation failed. "
                    "Want me to try again, or describe it instead?"
                ),
                "provider": "bridge",
                "model": "image-gen-failed",
                "tools_used": ["image_generation"],
            }

        provider_name = result.get("provider", "image-gen")
        model_name = result.get("model", "unknown")
        filename = result.get("filename", "")
        # TTS-friendly reply. The artifact marker is appended *after* the
        # user-facing sentence so the phone-side stripper produces clean
        # text for both the chat bubble and the spoken audio.
        reply = (
            f"I've made the image for you. "
            f"[ASTRA_ARTIFACT:image:{filename}]"
        )
        return {
            "reply": reply,
            "provider": provider_name,
            "model": model_name,
            "tools_used": ["image_generation"],
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

    # Append markers for artifacts registered by tools during this turn
    # (e.g. a rendered report document). Same send-side handling as image
    # markers: stored in the DB, stripped for display/TTS, chips built from
    # the refs. See app/llm/turn_surface.py.
    try:
        pending = drain_turn_artifacts()
        if pending:
            full_response = f"{full_response} {format_artifact_markers(pending)}"
    except Exception as e:
        logger.warning("[%s] artifact marker append failed: %s", source, e)

    print(f"[{source}] Response: {len(full_response)} chars, {len(tools_used)} tool calls")

    return {
        "reply": full_response,
        "provider": provider,
        "model": model,
        "model_source": prep.get("model_source"),  # v2026-06-24: routing provenance
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
    raw_message: Optional[str] = None,
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
        raw_message=raw_message,
    )

    if prep.get("rag_route"):
        from app.rag.answerer import ask_architecture_async
        yield {"type": "metadata", "provider": "rag", "model": "rag_answerer"}
        try:
            ans = await ask_architecture_async(
                db=db, question=prep["rag_question"], project_id=str(project_id),
            )
            text = ans.answer or "I couldn't find anything about that in my codebase index."
        except Exception as e:
            logger.error("[%s] RAG route failed: %s", source, e)
            text = "I tried to search my own codebase but the lookup failed. Try again in a moment."
        yield {"type": "token", "text": text}
        yield {"type": "done", "full_text": text, "provider": "rag", "model": "rag_answerer"}
        return

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
    raw_message: Optional[str] = None,
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

    # v1.0 (2026-05-24): Bind project_id into the working-set context
    # var so Bridge tool calls auto-register the files they touch.
    try:
        from app.memory.working_set import set_current as _ws_set_current
        _ws_set_current(project_id, "bridge")
    except Exception:
        pass

    # v1.1 (2026-05-24): Bootstrap working set from explicit file
    # references in the user's message.  Same rules as desktop:
    # exactly-one match per reference or skip silently.
    try:
        from app.memory._working_set_bootstrap import bootstrap_from_message
        from app.memory import service as _mem_svc
        _proj_name = None
        try:
            _proj = _mem_svc.get_project(db, project_id)
            _proj_name = getattr(_proj, "name", None)
        except Exception:
            pass
        bootstrap_from_message(
            project_id=project_id,
            project_name=_proj_name,
            message=message,
        )
    except Exception:
        pass

    # ── 0. Codebase-question short-circuit (phone/voice parity with desktop) ──
    # If the user is asking about ASTRA's own code/features, answer from the RAG
    # index rather than the generic chat brain. run_/stream_astra_chat handle the
    # rag_route by calling ask_architecture_async — the bridge returns text, not
    # an SSE stream, so we can't reuse the desktop generate_rag_query_stream.
    # Detect on the RAW user message: the augmented `message` may be prefixed
    # with an attachment description ("[Image attached: ...]") which defeats the
    # start-anchored detection regexes.
    _rag_q = await _detect_codebase_question(raw_message or message, translation_result, db)
    if _rag_q is not None:
        print(f"[{source}] RAG route: codebase question → architecture index")
        return {"rag_route": True, "rag_question": _rag_q}

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
    # STUDY FLOW (2026-07-03): hands-free study turns must act-then-answer,
    # never announce. Bridge sources only — desktop/room prompts untouched.
    from app.bridge.study_flow_prompt import study_flow_block
    system_prompt += study_flow_block(source)

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
        "model_source": extras.get("model_source"),  # v2026-06-24: routing provenance
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
                "You have tool access for exploring the codebase AND reading/writing the user's files.\n\n"
                "CODEBASE TOOLS (read-only): read_file, list_files, search_files, read_logs\n"
                "USER FILE TOOLS: search_my_files, read_user_file, write_user_file, get_user_folders, list_files\n"
                "CLOUD STORAGE TOOLS: cloud_upload (upload local file to Google Drive), cloud_list (list Drive contents).\n\n"
                "## Updating existing user files (READ-MODIFY-WRITE pattern)\n"
                "When the user asks you to ADD TO, UPDATE, or MODIFY an existing file\n"
                "(dashboard.html, log.html, a spreadsheet, etc.), follow this sequence:\n"
                "  1. list_files on the target folder (e.g. C:/Users/dizzi/OneDrive/Documents/Work/)\n"
                "     to confirm what is on disk. The conversation may not mention it,\n"
                "     but the file usually exists already.\n"
                "  2. read_user_file on the existing file to load its current contents.\n"
                "  3. Construct the new full contents in your head (existing contents plus your change).\n"
                "  4. write_user_file with the FULL new contents. write_user_file OVERWRITES\n"
                "     the whole file, so you must include every existing line you want kept.\n"
                "NEVER call write_user_file with only the new section. That deletes the rest.\n"
                "NEVER ask the user 'Documents or Desktop?' — check the filesystem first with list_files.\n\n"
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
