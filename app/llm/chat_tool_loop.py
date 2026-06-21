# FILE: app/llm/chat_tool_loop.py
# Purpose: Chat Tool Loop — gives trusted models real tool access in chat mode.
# Called-by: app.bridge.capability_layer, app.debug.debug_chat, app.endpoints._video_code_tools, app.llm.routing.chat_routing (+1 more)
# Depends-on: app.debug.action_executor, app.debug.gemini_expense_tools, app.debug.gemini_finance_tools, app.debug.tool_definitions (+2 more)
# Last-renovated: 2026-06-11
"""
Chat Tool Loop — gives trusted models real tool access in chat mode.

v1.0 (2026-03-03): Initial implementation.
When a trusted model (Opus, Gemini Pro) is used in chat mode, this module
provides a tool execution loop. The model can call read_file, list_files,
search_files, run_command etc. and get real results back.

This replaces the pre-loading approach (gather_codebase_context) which
only found files by keyword matching and couldn't do iterative exploration.
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

CHAT_TOOL_LOOP_BUILD_ID = "2026-03-03-v1.4-gemini-thought-signatures"

# Max tool call rounds to prevent infinite loops
# v1.1: Bumped from 15 to 30 - Opus needs rounds for exploration AND response
MAX_TOOL_ROUNDS = 30

# ── Re-export surface (SPLIT BATCH 9, 2026-06-21) ───────────────────────────
# Eligibility gate + tool-declaration assembly moved to chat_tool_registry.py;
# the name->executor dispatcher moved to chat_tool_dispatch.py. Re-exported here
# so every importer of app.llm.chat_tool_loop resolves the full public surface
# unchanged. The streaming loops below call execute_chat_tool through this
# module's namespace (rebindable for monkeypatching on this module).
from app.llm.chat_tool_registry import (  # noqa: F401
    _TRUSTED_MODELS_ANTHROPIC,
    _TRUSTED_MODELS_GOOGLE,
    _TRUSTED_MODELS_OPENAI,
    is_tool_eligible,
    _to_anthropic_tool_format,
    TOOL_TIER_READ,
    TOOL_TIER_WRITE,
    get_chat_tools,
)
from app.llm.chat_tool_dispatch import execute_chat_tool  # noqa: F401


async def _stream_with_tools_anthropic(
    messages: List[Dict],
    system_prompt: str,
    provider: str,
    model: str,
    tools: List[Dict],
    enable_reasoning: bool = False,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Anthropic tool loop: rebuild messages each round (stateless API)."""
    from app.llm._streaming_utils_3 import stream_llm

    current_messages = list(messages)
    rounds = 0

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        pending_tool_calls: List[Dict] = []
        text_chunks: List[str] = []
        done_chunk = None

        async for chunk in stream_llm(
            provider=provider,
            model=model,
            messages=current_messages,
            system_prompt=system_prompt,
            tools=tools,
            enable_reasoning=enable_reasoning,
            max_tokens=max_tokens,
        ):
            if chunk.get("type") == "tool_use":
                pending_tool_calls.append(chunk)
                yield {
                    "type": "tool_call",
                    "name": chunk["name"],
                    "input": chunk.get("input", {}),
                }
            elif chunk.get("type") == "done":
                done_chunk = chunk
            elif chunk.get("type") == "token":
                text_chunks.append(chunk.get("text", ""))
                yield chunk
            else:
                yield chunk

        if not pending_tool_calls:
            if done_chunk:
                yield done_chunk
            return

        logger.info(
            "[chat_tool_loop] Round %d: executing %d tool call(s)",
            rounds, len(pending_tool_calls),
        )

        assistant_content = []
        if text_chunks:
            assistant_content.append({
                "type": "text",
                "text": "".join(text_chunks),
            })
        for tc in pending_tool_calls:
            assistant_content.append({
                "type": "tool_use",
                "id": tc["tool_use_id"],
                "name": tc["name"],
                "input": tc.get("input", {}),
            })

        current_messages.append({
            "role": "assistant",
            "content": assistant_content,
        })

        tool_results = []
        for tc in pending_tool_calls:
            result_str = await execute_chat_tool(
                tc["name"], tc.get("input", {}),
            )
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc["tool_use_id"],
                "content": result_str,
            })
            yield {
                "type": "tool_result",
                "name": tc["name"],
                "result_preview": result_str[:200],
            }

        current_messages.append({
            "role": "user",
            "content": tool_results,
        })

        logger.info(
            "[chat_tool_loop] Round %d complete, continuing...", rounds,
        )

    logger.warning("[chat_tool_loop] Hit MAX_TOOL_ROUNDS (%d)", MAX_TOOL_ROUNDS)
    yield {"type": "token", "text": "\n\n[Tool loop reached maximum rounds]"}
    yield {"type": "done", "provider": provider, "model": model}


async def _stream_with_tools_gemini(
    messages: List[Dict],
    system_prompt: str,
    provider: str,
    model: str,
    tools: List[Dict],
    enable_reasoning: bool = False,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Gemini tool loop: uses SDK ChatSession to preserve thought signatures.

    v1.4: Gemini 3+ models require thought_signature on function call parts
    when replaying history.  The SDK's ChatSession handles this automatically
    when we use chat.send_message() and let it manage its own history.
    """
    import os
    try:
        import google.generativeai as genai
        from google.generativeai import types as _gtypes
        from google.ai import generativelanguage as glm
        from google.protobuf import struct_pb2
    except ImportError:
        yield {"type": "error", "message": "google.generativeai not installed"}
        return

    from app.llm._streaming_utils_3 import (
        _convert_tools_to_gemini, _build_gemini_parts, _int_env,
    )
    from app.llm.streaming import enhance_system_prompt_with_reasoning

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "GOOGLE_API_KEY not set"}
        return

    genai.configure(api_key=api_key)

    enhanced_prompt = enhance_system_prompt_with_reasoning(
        system_prompt, enable_reasoning,
    )

    # Convert tools to Gemini format
    gemini_tools = _convert_tools_to_gemini(tools) if tools else None
    logger.info(
        "[chat_tool_loop_gemini] Tools: %d -> %d functions",
        len(tools),
        len(gemini_tools[0].function_declarations) if gemini_tools else 0,
    )

    # Build model with tools + system prompt
    model_kwargs: Dict = {"model_name": model}
    if enhanced_prompt and enhanced_prompt.strip():
        model_kwargs["system_instruction"] = enhanced_prompt
    if gemini_tools:
        model_kwargs["tools"] = gemini_tools
    gemini_model = genai.GenerativeModel(**model_kwargs)

    # Build history from all messages except the last
    clean_messages = [m for m in messages if m.get("role") != "system"]
    if not clean_messages:
        yield {"type": "error", "message": "No messages to send"}
        return

    history = []
    for msg in clean_messages[:-1]:
        parts = _build_gemini_parts(msg)
        if not parts:
            continue
        role = "user" if msg["role"] == "user" else "model"
        history.append({"role": role, "parts": parts})

    # Create a persistent ChatSession — SDK manages thought signatures
    chat = gemini_model.start_chat(history=history)

    yield {"type": "metadata", "provider": "gemini", "model": model}

    # Send initial message
    last_parts = _build_gemini_parts(clean_messages[-1])
    if not last_parts:
        yield {"type": "error", "message": "Final message is empty"}
        return

    rounds = 0
    next_send = last_parts

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        pending_tool_calls: List[Dict] = []
        text_chunks: List[str] = []

        # Send message through the persistent chat session (not stream_llm)
        try:
            response = chat.send_message(next_send, stream=True)
        except Exception as e:
            yield {"type": "error", "message": f"Gemini API error: {e}"}
            return

        for chunk in response:
            try:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if not hasattr(candidate, 'content') or not candidate.content:
                            continue
                        for part in candidate.content.parts:
                            fc = getattr(part, 'function_call', None)
                            if fc and fc.name:
                                args_dict = dict(fc.args) if fc.args else {}
                                import uuid as _uuid
                                tool_use_id = f"gemini-fc-{_uuid.uuid4().hex[:12]}"
                                pending_tool_calls.append({
                                    "type": "tool_use",
                                    "tool_use_id": tool_use_id,
                                    "name": fc.name,
                                    "input": args_dict,
                                })
                                yield {
                                    "type": "tool_call",
                                    "name": fc.name,
                                    "input": args_dict,
                                }
                                continue
                            text = getattr(part, 'text', None)
                            if text:
                                text_chunks.append(text)
                                yield {"type": "token", "text": text}
                else:
                    text = getattr(chunk, "text", None)
                    if text:
                        text_chunks.append(text)
                        yield {"type": "token", "text": text}
            except ValueError:
                pass

        if not pending_tool_calls:
            # No tool calls — extract usage and finish
            usage_md = getattr(response, "usage_metadata", None)
            done_event: Dict = {"type": "done", "provider": "gemini", "model": model}
            if usage_md:
                pt = getattr(usage_md, "prompt_token_count", None)
                ct = getattr(usage_md, "candidates_token_count", None)
                tt = getattr(usage_md, "total_token_count", None)
                done_event["usage"] = {
                    "prompt_tokens": int(pt or 0),
                    "completion_tokens": int(ct or 0),
                    "total_tokens": int(tt or (int(pt or 0) + int(ct or 0))),
                }
            yield done_event
            return

        # Execute tool calls
        logger.info(
            "[chat_tool_loop_gemini] Round %d: executing %d tool call(s)",
            rounds, len(pending_tool_calls),
        )

        # Build function_response parts to send back via chat.send_message
        # The SDK automatically includes thought signatures from the previous
        # model response in the chat history, so we just send function results.
        response_parts = []
        for tc in pending_tool_calls:
            result_str = await execute_chat_tool(
                tc["name"], tc.get("input", {}),
            )
            resp_struct = struct_pb2.Struct()
            resp_struct.update({"result": str(result_str)})
            response_parts.append(glm.Part(
                function_response=glm.FunctionResponse(
                    name=tc["name"],
                    response=resp_struct,
                )
            ))
            yield {
                "type": "tool_result",
                "name": tc["name"],
                "result_preview": result_str[:200],
            }

        # Next round: send function responses through the same chat session
        next_send = response_parts

        logger.info(
            "[chat_tool_loop_gemini] Round %d complete, continuing...", rounds,
        )

    logger.warning("[chat_tool_loop_gemini] Hit MAX_TOOL_ROUNDS (%d)", MAX_TOOL_ROUNDS)
    yield {"type": "token", "text": "\n\n[Tool loop reached maximum rounds]"}
    yield {"type": "done", "provider": "gemini", "model": model}


async def _stream_with_tools_openai(
    messages: List[Dict],
    system_prompt: str,
    provider: str,
    model: str,
    tools: List[Dict],
    enable_reasoning: bool = False,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """OpenAI tool loop: accumulates streamed function args, executes, feeds back.

    OpenAI tool call format differs from Anthropic:
    - Tool calls come as streaming deltas with partial JSON arguments
    - Tool results are sent as {role: "tool", tool_call_id: ..., content: ...}
    - The assistant message must include the full tool_calls array
    """
    from app.llm._streaming_utils_3 import stream_llm

    current_messages = [{"role": "system", "content": system_prompt}] + list(messages)
    rounds = 0
    # Reflection gate (phase-narration spec): once a phase has returned this turn, the
    # next phase dispatch must carry narration.reflection. Tracked across rounds here.
    any_phase_dispatched = False
    from app.llm.loop_guard import LoopGuard, stream_loop_wrapup  # loop circuit-breaker
    guard = LoopGuard()
    loop_stop_reason = None

    while rounds < MAX_TOOL_ROUNDS:
        rounds += 1
        # Accumulators for streamed tool calls
        tool_call_accum: Dict[int, Dict] = {}  # index -> {id, name, arguments_json}
        text_chunks: List[str] = []
        done_chunk = None

        async for chunk in stream_llm(
            provider=provider,
            model=model,
            messages=current_messages[1:],  # strip system (stream_openai adds it)
            system_prompt=system_prompt,
            tools=tools,
            enable_reasoning=enable_reasoning,
            max_tokens=max_tokens,
        ):
            chunk_type = chunk.get("type", "")

            if chunk_type == "tool_use":
                # New tool call starting — OpenAI sends id + name on first delta
                idx = chunk.get("index", len(tool_call_accum))
                tool_call_accum[idx] = {
                    "id": chunk.get("id", ""),
                    "name": chunk.get("name", ""),
                    "arguments_json": chunk.get("input_json", ""),
                }
            elif chunk_type == "tool_use_delta":
                # Continuation — append argument JSON fragment
                # Find the last tool call being accumulated
                if tool_call_accum:
                    last_idx = max(tool_call_accum.keys())
                    tool_call_accum[last_idx]["arguments_json"] += chunk.get("arguments", "")
            elif chunk_type == "done":
                done_chunk = chunk
            elif chunk_type == "token":
                text_chunks.append(chunk.get("text", ""))
                yield chunk
            else:
                yield chunk

        # No tool calls — we're done
        if not tool_call_accum:
            if done_chunk:
                yield done_chunk
            return

        # Parse accumulated tool calls
        parsed_calls = []
        for idx in sorted(tool_call_accum.keys()):
            tc = tool_call_accum[idx]
            try:
                input_dict = json.loads(tc["arguments_json"]) if tc["arguments_json"] else {}
            except json.JSONDecodeError:
                input_dict = {"raw": tc["arguments_json"]}
            parsed_calls.append({
                "id": tc["id"],
                "name": tc["name"],
                "input": input_dict,
            })
            yield {
                "type": "tool_call",
                "name": tc["name"],
                "tool_use_id": tc["id"],
                "input": input_dict,
            }

        logger.info(
            "[chat_tool_loop] Round %d: executing %d OpenAI tool call(s)",
            rounds, len(parsed_calls),
        )

        # Build assistant message with tool_calls (OpenAI format)
        assistant_msg: Dict[str, Any] = {"role": "assistant"}
        if text_chunks:
            assistant_msg["content"] = "".join(text_chunks)
        else:
            assistant_msg["content"] = None
        assistant_msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["input"]),
                },
            }
            for tc in parsed_calls
        ]
        current_messages.append(assistant_msg)

        # Execute tools (phase-aware). run_openai_tool_call validates + narrates phase
        # moves (spawn_agents / sandbox boot), streams sub-agent activity, runs leaf
        # tools, and returns each result via a TOOL_MESSAGE sentinel. Reflection gate:
        # a 2nd+ phase dispatch this turn must carry a reflection (require_reflection).
        from app.debug.orchestrator.phase_narration import (
            run_openai_tool_call, TOOL_MESSAGE,
        )
        require_reflection = any_phase_dispatched
        for tc in parsed_calls:
            async for ev in run_openai_tool_call(tc, require_reflection=require_reflection):
                if ev.get("type") == TOOL_MESSAGE:
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": (ev.get("content") or "")[:30000],
                    })
                    guard.record(tc["name"], ev.get("signature") or tc["name"], bool(ev.get("failed")))
                    if ev.get("dispatched_phase"):
                        any_phase_dispatched = True
                else:
                    yield ev

        logger.info("[chat_tool_loop] Round %d complete, continuing...", rounds)

        loop_stop_reason = guard.tripped()
        if loop_stop_reason:
            break

    logger.warning("[chat_tool_loop_openai] stop: %s", loop_stop_reason or "MAX_TOOL_ROUNDS")
    async for ev in stream_loop_wrapup(  # graceful tool-free summary; never dead-ends
        current_messages, provider, model,
        reason=loop_stop_reason, rounds=rounds,
        enable_reasoning=enable_reasoning, max_tokens=max_tokens,
    ):
        yield ev


async def stream_with_tools(
    messages: List[Dict],
    system_prompt: str,
    provider: str,
    model: str,
    tools: List[Dict],
    enable_reasoning: bool = False,
    max_tokens: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream LLM with tool execution loop.

    v1.4: Routes to provider-specific tool loop:
    - Anthropic: stateless message rebuilding (no thought signatures needed)
    - Gemini: persistent ChatSession (SDK handles thought signatures automatically)

    Yields the same chunk types as stream_llm (token, metadata, done, etc.)
    plus tool_call/tool_result events for the UI.
    """
    if provider in ("google", "gemini"):
        async for chunk in _stream_with_tools_gemini(
            messages, system_prompt, provider, model, tools,
            enable_reasoning, max_tokens,
        ):
            yield chunk
    elif provider == "openai":
        async for chunk in _stream_with_tools_openai(
            messages, system_prompt, provider, model, tools,
            enable_reasoning, max_tokens,
        ):
            yield chunk
    else:
        async for chunk in _stream_with_tools_anthropic(
            messages, system_prompt, provider, model, tools,
            enable_reasoning, max_tokens,
        ):
            yield chunk
