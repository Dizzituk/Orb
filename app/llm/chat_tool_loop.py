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

import asyncio
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional

logger = logging.getLogger(__name__)

CHAT_TOOL_LOOP_BUILD_ID = "2026-03-03-v1.4-gemini-thought-signatures"

# Max tool call rounds to prevent infinite loops
# v1.1: Bumped from 15 to 30 - Opus needs rounds for exploration AND response
MAX_TOOL_ROUNDS = 30

# Models that get tool access in chat mode
# v1.2: Gemini models listed for future tool loop support.
# Currently only Anthropic models run the full tool loop (stream_anthropic
# accepts tools). Gemini models get codebase context via chat_codebase_reader
# pre-loading instead. When stream_gemini gains tool support, flip the
# provider gate below.
_TRUSTED_MODELS_ANTHROPIC = {
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
}

# Gemini models that WOULD get tool access once stream_gemini supports it.
# For now these use pre-loaded codebase context (chat_codebase_reader).
_TRUSTED_MODELS_GOOGLE = {
    "gemini-3.1-pro-preview",
    "gemini-3.1-pro-preview-customtools",
}

# v11.0: OpenAI models — now tool-eligible via _stream_with_tools_openai.
# v12.0 (2026-06-17): the GPT-5 family is tool-eligible by PREFIX (see
# is_tool_eligible), not just this set. The set stays for documentation; the
# prefix guard means an ENV model swap to any new gpt-5* id (e.g. gpt-5.5) can
# never silently strip tools on the phone/bridge or normal-chat paths (spec 3.2).
# NOTE: the Debug /stream/chat path does not gate on this -- it loads write-tier
# tools directly -- so this guard protects the OTHER OpenAI tool paths.
_TRUSTED_MODELS_OPENAI = {
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-turbo",
    "gpt-5.5",
    "gpt-5.5-mini",
}


def is_tool_eligible(provider: str, model: str) -> bool:
    """Check if this provider/model combo supports the tool execution loop.

    v1.3: Gemini 3.1 Pro customtools now supported via stream_gemini tools param.
    v11.0: OpenAI models now tool-eligible (tool loop + streaming delta accumulation).
    """
    if provider == "anthropic":
        return model in _TRUSTED_MODELS_ANTHROPIC
    if provider in ("google", "gemini"):
        return model in _TRUSTED_MODELS_GOOGLE
    if provider == "openai":
        # Prefix guard: any gpt-5* OpenAI model is tool-eligible, so an ENV model
        # change can never silently disable tools (spec 3.2 offender #3).
        return model in _TRUSTED_MODELS_OPENAI or str(model).lower().startswith("gpt-5")
    return False


def _to_anthropic_tool_format(tool: Dict) -> Dict:
    """Convert internal tool format to Anthropic API format.

    Internal:  {"name", "description", "parameters": {"type": "object", "properties": ...}}
    Anthropic: {"name", "description", "input_schema": {"type": "object", "properties": ...}}
    """
    return {
        "name": tool["name"],
        "description": tool.get("description", ""),
        "input_schema": tool.get("parameters", tool.get("input_schema", {"type": "object", "properties": {}})),
    }


# Tool tiers for different contexts
TOOL_TIER_READ = "read"       # Chat mode: read_file, list_files, search_files, read_logs, read_pipeline_state
TOOL_TIER_WRITE = "write"     # Debug/agentic: adds write_file, edit_file, run_command


def get_chat_tools(tier: str = TOOL_TIER_READ) -> List[Dict]:
    """Get tool definitions in Anthropic API format for the given tier.

    Tiers:
        read  — Chat mode. Can explore codebase but not modify anything.
        write — Debug/agentic. Full access including writes and commands.

    Returns tools in Anthropic API format (input_schema, not parameters).
    """
    from app.debug.tool_definitions import get_phase1_tools, get_phase2_tools

    if tier == TOOL_TIER_WRITE:
        raw_tools = get_phase2_tools()
    else:
        raw_tools = get_phase1_tools()

    tools = [_to_anthropic_tool_format(t) for t in raw_tools]
    # Work-day ledger tools - let any chat start/finish/reconcile a delivery
    # day straight to the DB (the 'Work' tab), via the same declarations the
    # image loop uses. Without these, "starting work, 11, 1500 miles" has no
    # tool to call and the model wrongly reaches for write_file / edit_file.
    try:
        from app.debug.gemini_finance_tools import FINANCE_TOOL_DECLARATIONS
        tools.extend(_to_anthropic_tool_format(t) for t in FINANCE_TOOL_DECLARATIONS)
    except Exception as exc:
        logger.warning("[chat_tool_loop] finance tools unavailable: %s", exc)
    # Expense logging tool ("log £120 fuel") - its own module, same wiring.
    try:
        from app.debug.gemini_expense_tools import EXPENSE_TOOL_DECLARATIONS
        tools.extend(_to_anthropic_tool_format(t) for t in EXPENSE_TOOL_DECLARATIONS)
    except Exception as exc:
        logger.warning("[chat_tool_loop] expense tools unavailable: %s", exc)
    # Central-registry tools (2026-06-12): the Session-7 families (documents/
    # editor, podcasts, email, streams, movie night, displays) plus sentinel/
    # vehicle register in app.tools.registry, which this loop never read.
    # Without this merge the chat model has no editor access and answers
    # "upload it here" while the document sits open in the pane.
    try:
        from app.tools.chat_bridge import get_registry_chat_declarations
        _existing = {t.get("name") for t in tools}
        tools.extend(get_registry_chat_declarations(exclude_names=_existing))
    except Exception as exc:
        logger.warning("[chat_tool_loop] registry tools unavailable: %s", exc)
    # Debug orchestrator: ONLY the write tier (the Debug-tab brain) gets the
    # spawn_agents delegation tool, so ordinary chat / sub-agents never see it.
    if tier == TOOL_TIER_WRITE:
        try:
            from app.debug.orchestrator.spawn_tool import SPAWN_AGENTS_TOOL
            tools.append(_to_anthropic_tool_format(SPAWN_AGENTS_TOOL))
        except Exception as exc:
            logger.warning("[chat_tool_loop] spawn_agents tool unavailable: %s", exc)
        # Sandbox boot/self-heal: start/stop/status the clone + read its boot log.
        # These act on the SANDBOX CLONE only (never the host) -- the host-boot gate
        # in filesystem.py refuses host boots, so this is the sanctioned boot path.
        try:
            from app.sandbox.tools import SANDBOX_TOOLS
            from app.debug.sandbox_boot_tool import (
                READ_SANDBOX_BOOT_TOOL, INSPECT_SANDBOX_BOOT_TOOL,
            )
            from app.debug.sandbox_selfheal import SELFHEAL_SANDBOX_BOOT_TOOL
            for _sbx_tool in SANDBOX_TOOLS:
                tools.append(_to_anthropic_tool_format(_sbx_tool))
            tools.append(_to_anthropic_tool_format(READ_SANDBOX_BOOT_TOOL))
            # inspect_sandbox_boot = the FULL visual loop (backend + screenshot/OCR).
            tools.append(_to_anthropic_tool_format(INSPECT_SANDBOX_BOOT_TOOL))
            # selfheal_sandbox_boot = autonomous fix->reboot->re-verify loop (Job 03).
            tools.append(_to_anthropic_tool_format(SELFHEAL_SANDBOX_BOOT_TOOL))
        except Exception as exc:
            logger.warning("[chat_tool_loop] sandbox tools unavailable: %s", exc)
    # Phase-level narration (debug-visibility spec): attach the required `narration`
    # object to the phase-dispatch tools (spawn_agents + sandbox boot/inspect) so the
    # brain must declare its intent (and reflect) at the move boundary. Leaf tools are
    # untouched. Done here -- one place -- so the protected sandbox schemas stay intact.
    try:
        from app.debug.orchestrator.phase_narration import inject_narration_schema
        tools = [inject_narration_schema(t) for t in tools]
    except Exception as exc:
        logger.warning("[chat_tool_loop] narration schema inject unavailable: %s", exc)
    return tools


async def execute_chat_tool(name: str, params: Dict) -> str:
    """Execute a tool call and return the result string."""
    # spawn_agents: non-streaming fallback (the Debug OpenAI loop special-cases it
    # to stream live sub-agent activity; this path serves any other caller).
    if name == "spawn_agents":
        from app.debug.orchestrator.spawn_tool import run_spawn_agents_blocking
        return await run_spawn_agents_blocking(params or {})
    # Sandbox clone control (start/stop/status + read boot log). Sync + can block
    # (start waits for boot; read_sandbox_boot can poll), so run off the event loop.
    if name in ("start_sandbox_clone", "stop_sandbox_clone", "check_sandbox_status"):
        from app.sandbox.tools import execute_sandbox_tool
        return await asyncio.to_thread(execute_sandbox_tool, name, params or {})
    if name == "read_sandbox_boot":
        from app.debug.sandbox_boot_tool import read_sandbox_boot
        return await asyncio.to_thread(read_sandbox_boot, params or {})
    # inspect_sandbox_boot: full visual loop (backend readout + sandbox screenshot +
    # vision/OCR). Sync + blocks (polls boot, runs a sandbox screenshot, calls vision),
    # so run it off the event loop just like read_sandbox_boot.
    if name == "inspect_sandbox_boot":
        from app.debug.sandbox_boot_tool import inspect_sandbox_boot
        return await asyncio.to_thread(inspect_sandbox_boot, params or {})
    # selfheal_sandbox_boot: autonomous fix->reboot->re-verify loop; sync + long-blocking.
    if name == "selfheal_sandbox_boot":
        from app.debug.sandbox_selfheal import selfheal_sandbox_boot
        return await asyncio.to_thread(selfheal_sandbox_boot, params or {})
    # Work-day ledger tools run via their own adapters (central-registry
    # handlers), not the debug action_executor - same path as the image loop.
    try:
        from app.debug.gemini_finance_tools import FINANCE_TOOL_EXECUTORS
        if name in FINANCE_TOOL_EXECUTORS:
            return await FINANCE_TOOL_EXECUTORS[name](params or {})
    except Exception as exc:
        logger.error("[chat_tool_loop] finance tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"
    try:
        from app.debug.gemini_expense_tools import EXPENSE_TOOL_EXECUTORS
        if name in EXPENSE_TOOL_EXECUTORS:
            return await EXPENSE_TOOL_EXECUTORS[name](params or {})
    except Exception as exc:
        logger.error("[chat_tool_loop] expense tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"
    # Central-registry tools (documents/editor, podcasts, email, streams,
    # movies, displays, ...). action_executor keeps precedence for every
    # name it owns; only names it does NOT know are tried against the
    # registry, so read_file/web_search/etc. behave exactly as before.
    try:
        from app.debug.action_executor import TOOL_HANDLERS as _AE_HANDLERS
    except Exception:
        _AE_HANDLERS = {}
    if name not in _AE_HANDLERS:
        try:
            from app.tools.chat_bridge import (
                is_registry_chat_tool, execute_registry_chat_tool,
            )
            if is_registry_chat_tool(name):
                return await execute_registry_chat_tool(name, params or {})
        except Exception as exc:
            logger.error("[chat_tool_loop] registry tool %s failed: %s", name, exc)
            return f"Tool error: {exc}"
    from app.debug.action_executor import execute_tool
    try:
        result = await execute_tool(name, params)
        cap = 128000 if name == "read_file" else 30000
        if len(result) > cap:
            result = result[:cap] + f"\n\n... [TRUNCATED — {len(result)} chars total]"
        return result
    except Exception as exc:
        logger.error("[chat_tool_loop] Tool %s failed: %s", name, exc)
        return f"Tool error: {exc}"


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