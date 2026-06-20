# Purpose: streaming utils 3
# Called-by: app.endpoints._video_code_tools, app.llm.chat_tool_loop, app.llm.streaming, app.memory.document_knowledge_promoter (+2 more)
# Depends-on: app.llm._streaming_utils_2, app.llm.streaming, app.llm.streaming_ollama
# Last-renovated: 2026-06-11
from __future__ import annotations
import json
import logging
import os
from app.llm._streaming_utils_2 import _openai_text_nonstream, _should_retry_stream_error, get_default_provider
from typing import Any, AsyncGenerator, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
HAS_ANTHROPIC = True
HAS_GEMINI = True


def _provider_key(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("gemini", "google"):
        return "google"
    return p

def _int_env(name: str) -> Optional[int]:
    v = os.getenv(name, "").strip()
    if not v:
        return None
    try:
        return int(v)
    except Exception:
        return None

def _openai_needs_max_completion_tokens(model: str) -> bool:
    m = (model or "").strip().lower()
    # Covers gpt-5.* and typical "o-series" reasoning models if you add them later.
    return m.startswith("gpt-5") or m.startswith("o1") or m.startswith("o3")

async def stream_anthropic(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from Anthropic using async client.
    
    Args:
        max_tokens: Override max output tokens (default: ANTHROPIC_MAX_TOKENS env or 4096)
        timeout_seconds: Request timeout (currently logged but not enforced at HTTP level)
    """
    from .streaming import anthropic, enhance_system_prompt_with_reasoning, get_default_model, get_model_for_route
    print(f"[STREAM_ANTHROPIC] Called: model={model}, route={route}, max_tokens={max_tokens}, timeout={timeout_seconds}s")

    if not HAS_ANTHROPIC:
        yield {"type": "error", "message": "anthropic package not installed"}
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY not set"}
        return

    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("anthropic", route)
    else:
        use_model = get_default_model("anthropic")

    yield {"type": "metadata", "provider": "anthropic", "model": use_model}

    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # v0.19: Apply timeout — default 300s for long architecture docs, env override available
    effective_timeout = timeout_seconds or _int_env("ANTHROPIC_TIMEOUT_SECONDS") or 300
    client = anthropic.AsyncAnthropic(
        api_key=api_key,
        timeout=effective_timeout,
    )

    prompt_tokens = None
    completion_tokens = None
    _current_tool_id = ""
    _current_tool_name = ""
    _current_tool_input_json = ""
    _stop_reason = None

    # Determine max tokens: explicit param > env var > default
    effective_max_tokens = max_tokens or _int_env("ANTHROPIC_MAX_TOKENS") or 4096
    print(f"[STREAM_ANTHROPIC] Effective max_tokens={effective_max_tokens}, timeout={effective_timeout}s")
    
    # Anthropic Messages API does not accept role="system" in messages —
    # system content must go via the top-level `system` parameter.
    # Filter out any system messages that may have been injected upstream.
    clean_messages = [m for m in messages if m.get("role") != "system"]
    
    try:
        create_kwargs = dict(
            model=use_model,
            max_tokens=effective_max_tokens,
            system=enhanced_prompt,
            messages=clean_messages,
            stream=True,
        )
        if tools:
            create_kwargs["tools"] = tools

        resp = await client.messages.create(**create_kwargs)

        async for event in resp:
            if event.type == "content_block_start":
                block = getattr(event, "content_block", None)
                if block and getattr(block, "type", None) == "tool_use":
                    _current_tool_id = getattr(block, "id", "")
                    _current_tool_name = getattr(block, "name", "")
                    _current_tool_input_json = ""

            if event.type == "content_block_delta":
                delta = event.delta
                delta_text = getattr(delta, "text", None)
                if delta_text:
                    yield {"type": "token", "text": delta_text}
                # Accumulate tool input JSON
                delta_json = getattr(delta, "partial_json", None)
                if delta_json:
                    _current_tool_input_json += delta_json

            if event.type == "content_block_stop":
                if _current_tool_name:
                    import json as _json
                    try:
                        _tool_input = _json.loads(_current_tool_input_json) if _current_tool_input_json else {}
                    except _json.JSONDecodeError:
                        _tool_input = {}
                    yield {
                        "type": "tool_use",
                        "tool_use_id": _current_tool_id,
                        "name": _current_tool_name,
                        "input": _tool_input,
                    }
                    _current_tool_id = ""
                    _current_tool_name = ""
                    _current_tool_input_json = ""

            if event.type == "message_delta":
                _stop = getattr(event.delta, "stop_reason", None)
                if _stop:
                    _stop_reason = _stop
                usage = getattr(event, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "input_tokens", None)
                    completion_tokens = getattr(usage, "output_tokens", None)

        if prompt_tokens is not None or completion_tokens is not None:
            total_tokens = (int(prompt_tokens or 0) + int(completion_tokens or 0))
            yield {
                "type": "done",
                "provider": "anthropic",
                "model": use_model,
                "stop_reason": _stop_reason,
                "usage": {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens),
                },
            }
        else:
            yield {"type": "done", "provider": "anthropic", "model": use_model, "stop_reason": _stop_reason}

    except Exception as e:
        yield {"type": "error", "message": str(e)}

# =========================================================================
# Gemini tool format helpers
# =========================================================================

def _convert_tools_to_gemini(tools: List[Dict]) -> List:
    """Convert tools from Anthropic/internal format to Gemini FunctionDeclaration format.

    Internal format (same as Anthropic input_schema):
        {"name": "read_file", "description": "...", "input_schema": {"type": "object", ...}}
    or:
        {"name": "read_file", "description": "...", "parameters": {"type": "object", ...}}

    Gemini format:
        genai.types.Tool(function_declarations=[FunctionDeclaration(name, description, parameters)])
    """
    from google.generativeai import types as _gtypes

    func_decls = []
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        params = tool.get("input_schema") or tool.get("parameters") or {}
        # Gemini FunctionDeclaration expects a JSON schema dict for parameters
        func_decls.append(_gtypes.FunctionDeclaration(
            name=name,
            description=desc,
            parameters=params if params.get("properties") else None,
        ))

    return [_gtypes.Tool(function_declarations=func_decls)] if func_decls else []


def _build_gemini_parts(msg: Dict) -> List:
    """Convert an internal message dict to a list of Gemini-compatible parts.

    Handles:
    - Plain text messages: {"role": "user", "content": "hello"}
    - Assistant messages with tool_use blocks (Anthropic format):
        {"role": "assistant", "content": [{"type": "text", ...}, {"type": "tool_use", ...}]}
    - User messages with tool_result blocks (Anthropic format):
        {"role": "user", "content": [{"type": "tool_result", ...}]}
    """
    try:
        from google.ai import generativelanguage as glm
        from google.protobuf import struct_pb2
    except ImportError:
        # Fallback: just return text
        content = msg.get("content", "")
        return [content] if content else []

    content = msg.get("content", "")

    # Simple text message
    if isinstance(content, str):
        return [content] if content else []

    # Structured content blocks (Anthropic tool loop format)
    if isinstance(content, list):
        parts = []
        for block in content:
            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    parts.append(text)

            elif block_type == "image":
                # v14.0: Image upload — convert base64 to Gemini inline_data blob
                import base64 as _b64_mod
                _img_mime = block.get("mime_type", "image/png")
                _img_data_b64 = block.get("data", "")
                if _img_data_b64:
                    _img_bytes = _b64_mod.b64decode(_img_data_b64)
                    parts.append(glm.Part(
                        inline_data=glm.Blob(
                            mime_type=_img_mime,
                            data=_img_bytes,
                        )
                    ))
                    print(f"[GEMINI_PARTS] Injected inline image: {_img_mime}, {len(_img_bytes)} bytes")

            elif block_type == "tool_use":
                # Assistant wants to call a tool -> function_call Part
                args_struct = struct_pb2.Struct()
                tool_input = block.get("input", {})
                if tool_input:
                    args_struct.update(tool_input)
                parts.append(glm.Part(
                    function_call=glm.FunctionCall(
                        name=block.get("name", ""),
                        args=args_struct,
                    )
                ))

            elif block_type == "tool_result":
                # User sends tool result back -> function_response Part
                result_content = block.get("content", "")
                if isinstance(result_content, list):
                    # Sometimes tool results are wrapped: [{"type": "text", "text": "..."}]
                    result_content = " ".join(
                        item.get("text", "") for item in result_content
                        if isinstance(item, dict)
                    )
                resp_struct = struct_pb2.Struct()
                resp_struct.update({"result": str(result_content)})
                # tool_result blocks carry tool_use_id; name comes from
                # the matching tool_use block.  The chat_tool_loop stores
                # the name on the block when it builds tool_result entries.
                tool_name = block.get("name", block.get("tool_use_id", "unknown"))
                parts.append(glm.Part(
                    function_response=glm.FunctionResponse(
                        name=tool_name,
                        response=resp_struct,
                    )
                ))

        return parts if parts else []

    return []


async def stream_gemini(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from Gemini (Google Generative AI).

    v9.0: Added tools parameter for function calling (Gemini 3.1 Pro customtools).
    Tools are converted from Anthropic/internal format to Gemini FunctionDeclaration format.
    """
    from .streaming import enhance_system_prompt_with_reasoning, genai, get_default_model, get_model_for_route
    if not HAS_GEMINI:
        yield {"type": "error", "message": "google.generativeai package not installed"}
        return

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "GOOGLE_API_KEY not set"}
        return

    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("google", route)
    else:
        use_model = get_default_model("google")

    genai.configure(api_key=api_key)

    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    yield {"type": "metadata", "provider": "gemini", "model": use_model}

    prompt_tokens = None
    completion_tokens = None
    _current_tool_id = ""
    _current_tool_name = ""
    _current_tool_input_json = ""
    _stop_reason = None
    total_tokens = None

    def _uget(obj, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    try:
        # v9.0: Convert tools from internal/Anthropic format to Gemini format
        gemini_tools = None
        if tools:
            gemini_tools = _convert_tools_to_gemini(tools)
            print(f"[STREAM_GEMINI] Tools registered: {len(tools)} -> {len(gemini_tools[0].function_declarations) if gemini_tools else 0} functions")

        # v3.2: Gemini 3 rejects empty system_instruction — only pass if non-empty
        model_kwargs = {"model_name": use_model}
        if enhanced_prompt and enhanced_prompt.strip():
            model_kwargs["system_instruction"] = enhanced_prompt
        if gemini_tools:
            model_kwargs["tools"] = gemini_tools
        gemini_model = genai.GenerativeModel(**model_kwargs)

        # v3.2: Filter out system messages — system content goes via system_instruction.
        # Gemini 3 models reject history starting with a 'model' turn, and system
        # messages were being miscast as model turns.
        clean_messages = [m for m in messages if m.get("role") != "system"]

        if not clean_messages:
            yield {"type": "error", "message": "No non-system messages to send to Gemini"}
            return

        # v3.2 debug: log message structure to diagnose empty content errors
        print(f"[STREAM_GEMINI] Messages: {len(messages)} total, {len(clean_messages)} after system filter")
        for i, m in enumerate(clean_messages):
            c = m.get('content', '')
            print(f"[STREAM_GEMINI]   [{i}] role={m.get('role')}, content_len={len(c) if c else 0}")

        # v9.0: Build history, handling both text and tool-call/result messages
        history = []
        for msg in clean_messages[:-1]:
            parts = _build_gemini_parts(msg)
            if not parts:
                print(f"[STREAM_GEMINI] WARNING: Skipping empty {msg.get('role')} message in history")
                continue
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": parts})

        chat = gemini_model.start_chat(history=history)

        # v9.0: Build parts for the final message (may contain tool results)
        last_msg = clean_messages[-1] if clean_messages else {}
        last_parts = _build_gemini_parts(last_msg)
        if not last_parts:
            yield {"type": "error", "message": "Final message content is empty - nothing to send to Gemini"}
            return

        response = chat.send_message(last_parts, stream=True)

        last_chunk = None
        for chunk in response:
            last_chunk = chunk
            try:
                # v9.0: Check for function calls in the response
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if not hasattr(candidate, 'content') or not candidate.content:
                            continue
                        for part in candidate.content.parts:
                            # Function call from the model
                            fc = getattr(part, 'function_call', None)
                            if fc and fc.name:
                                # Convert args from proto Struct to dict
                                args_dict = dict(fc.args) if fc.args else {}
                                # Generate a unique tool_use_id (Gemini doesn't have one)
                                import uuid as _uuid
                                tool_use_id = f"gemini-fc-{_uuid.uuid4().hex[:12]}"
                                yield {
                                    "type": "tool_use",
                                    "tool_use_id": tool_use_id,
                                    "name": fc.name,
                                    "input": args_dict,
                                }
                                continue
                            # Normal text
                            text = getattr(part, 'text', None)
                            if text:
                                yield {"type": "token", "text": text}
                else:
                    text = getattr(chunk, "text", None)
                    if text:
                        yield {"type": "token", "text": text}
            except ValueError:
                # Gemini returns empty Part for refused/empty responses
                pass

        usage_md = _uget(response, "usage_metadata") or _uget(last_chunk, "usage_metadata")
        if usage_md:
            pt = _uget(usage_md, "prompt_token_count")
            ct = _uget(usage_md, "candidates_token_count")
            tt = _uget(usage_md, "total_token_count")
            if pt is not None:
                prompt_tokens = int(pt)
            if ct is not None:
                completion_tokens = int(ct)
            if tt is not None:
                total_tokens = int(tt)

        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            usage = {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0))),
            }
            yield {"type": "done", "provider": "gemini", "model": use_model, "usage": usage}
        else:
            yield {"type": "done", "provider": "gemini", "model": use_model}

    except Exception as e:
        yield {"type": "error", "message": str(e)}

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    tools: Optional[List[Dict]] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from specified LLM provider.
    
    Args:
        max_tokens: Override max output tokens (provider-specific defaults otherwise)
        timeout_seconds: Request timeout hint (implementation varies by provider)
    """
    from .streaming import stream_openai
    print(f"[STREAM_LLM] Called: provider={provider}, model={model}, messages={len(messages)}, max_tokens={max_tokens}")

    if not provider:
        provider = get_default_provider()
        print(f"[STREAM_LLM] No provider specified, using default: {provider}")

    if not provider:
        print("[STREAM_LLM] ERROR: No LLM providers available!")
        yield {"type": "error", "message": "No LLM providers available"}
        return

    provider = provider.lower()
    print(f"[STREAM_LLM] Routing to provider: {provider}, model: {model}")

    if provider == "openai":
        async for event in stream_openai(
            messages, system_prompt, model, enable_reasoning, route,
            tools=tools, max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        ):
            yield event
    elif provider == "anthropic":
        async for event in stream_anthropic(
            messages, system_prompt, model, enable_reasoning, route,
            max_tokens=max_tokens, timeout_seconds=timeout_seconds,
            tools=tools,
        ):
            yield event
    elif provider in ("gemini", "google"):
        async for event in stream_gemini(
            messages, system_prompt, model, enable_reasoning, route,
            tools=tools, max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        ):
            yield event
    elif provider in ("ollama", "local"):
        from app.llm.streaming_ollama import stream_ollama
        async for event in stream_ollama(
            messages, system_prompt, model, enable_reasoning, route,
            tools=tools, max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        ):
            yield event
    elif provider in ("vllm", "vllm_local"):
        # Local vLLM OpenAI-compatible server — serves the agent worker "Nat"
        # (and, later, the parallel 26B fleet). OpenAI-shaped, so tools +
        # tool_use deltas flow through exactly like the openai branch.
        from app.llm.streaming_vllm import stream_vllm
        async for event in stream_vllm(
            messages, system_prompt, model, enable_reasoning, route,
            tools=tools, max_tokens=max_tokens, timeout_seconds=timeout_seconds,
        ):
            yield event
    else:
        print(f"[STREAM_LLM] ERROR: Unknown provider '{provider}'")
        yield {"type": "error", "message": f"Unknown provider '{provider}'"}

async def call_llm_text(
    provider: str,
    model: Optional[str],
    system_prompt: str,
    user_prompt: str,
    *,
    messages: Optional[List[Dict[str, str]]] = None,
    repo_snapshot: Optional[Dict[str, Any]] = None,
    constraints_hint: Optional[Any] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> str:
    """
    Convenience helper for callers that want a single final string.

    - Uses stream_llm under the hood.
    - Collects {"type":"token"} chunks into one string.
    - Raises RuntimeError on {"type":"error"}.
    - Retries once for transient stream disconnects; for OpenAI it falls back to a non-stream call.
    """
    from .streaming import enhance_system_prompt_with_reasoning, get_model_for_route
    if not provider:
        raise ValueError("call_llm_text: provider is required")

    if not model:
        model = get_model_for_route(provider, route)

    # Build augmented system prompt (kept small-ish and structured)
    sys = system_prompt or ""
    if constraints_hint is not None:
        try:
            ch = json.dumps(constraints_hint, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            ch = str(constraints_hint)
        sys += "\n\n[CONSTRAINTS_HINT]\n" + ch

    if repo_snapshot is not None:
        try:
            rs = json.dumps(repo_snapshot, ensure_ascii=False, separators=(",", ":"), default=str)
        except Exception:
            rs = str(repo_snapshot)
        sys += "\n\n[REPO_SNAPSHOT]\n" + rs

    enhanced_sys = enhance_system_prompt_with_reasoning(sys, enable_reasoning)

    use_messages = messages if messages is not None else [{"role": "user", "content": user_prompt}]

    async def _collect_via_stream() -> str:
        out: List[str] = []
        async for event in stream_llm(
            use_messages,
            system_prompt=enhanced_sys,
            provider=provider,
            model=model,
            enable_reasoning=enable_reasoning,
            route=route,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        ):
            et = event.get("type")
            if et == "token":
                t = event.get("text", "")
                if t:
                    out.append(t)
            elif et == "error":
                raise RuntimeError(event.get("message", "LLM error"))
            else:
                pass
        return "".join(out).strip()

    try:
        return await _collect_via_stream()
    except RuntimeError as e:
        if not _should_retry_stream_error(str(e)):
            raise

        pk = _provider_key(provider)
        logger.warning(f"[call_llm_text] transient stream error, retrying via fallback: {e}")

        if pk == "openai":
            full_messages = [{"role": "system", "content": enhanced_sys}] + use_messages
            return await _openai_text_nonstream(messages=full_messages, system_prompt=enhanced_sys, model=model)

        # For other providers: one more stream attempt
        return await _collect_via_stream()
