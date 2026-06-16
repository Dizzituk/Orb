# FILE: app/llm/streaming.py
# Purpose: Orb Streaming LLM Module
# Called-by: app.bridge.capability_layer, app.llm._streaming_utils_2, app.llm._streaming_utils_3, app.llm._weaver_stream_utils_14 (+12 more)
# Depends-on: app.llm._streaming_utils_2, app.llm._streaming_utils_3
# Last-renovated: 2026-06-14 (cap OpenAI tools at 128 to prevent array_above_max_length 400)
"""
Orb Streaming LLM Module

Provides streaming interface for different LLM providers.
All streaming providers must follow the canonical event schema.

v0.16.0: Added reasoning support and metadata emission.
v0.17.0: Debug logging
v0.17.1: ENV-driven model selection (no hardcoded model IDs)
v0.17.2: Added call_llm_text helper for non-stream callers (Spec Gate, etc.)
v0.17.3: call_llm_text retry + OpenAI non-stream fallback for transient stream disconnects
v0.18.0: Added max_tokens and timeout_seconds params to stream_llm/stream_anthropic for archmap

CANONICAL EVENT SCHEMA:
All streaming functions must yield dict events with this structure:

{"type": "metadata", "provider": "...", "model": "..."}
    - Sent once at the start to indicate the provider and model being used

{"type": "token", "text": "<chunk of answer>"}
    - Streaming chunks of the response

{"type": "reasoning", "text": "<hidden chain-of-thought>"}
    - Optional reasoning/thinking content

{"type": "error", "message": "..."}
    - Error message if something goes wrong

{"type": "done"}
    - Optional final event indicating completion

The SSE layer (stream_router.py) handles JSON serialization.

NOTE (OpenAI token param drift):
Some newer OpenAI chat models (e.g. gpt-5.*) reject `max_tokens` and require
`max_completion_tokens`. This module sets the correct param based on model.
"""

from __future__ import annotations

import os
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any
from app.llm._streaming_utils_2 import DEFAULT_MODEL_ENV, ROUTE_MODEL_ENV, _env_model, _openai_text_nonstream, _should_retry_stream_error, get_available_streaming_provider, get_available_streaming_providers, get_default_provider
from app.llm._streaming_utils_3 import _int_env, _openai_needs_max_completion_tokens, _provider_key, call_llm_text, stream_anthropic, stream_gemini, stream_llm

logger = logging.getLogger(__name__)

# Import provider packages conditionally
try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


# =========================
# ENV-driven model selection
# =========================

# Provider -> ENV var that MUST define its default model (if the provider is used without explicit model)

# Provider -> Route -> ENV var that can override the model for that route.
# If the route env var is missing, we fall back to the provider default env var.


def enhance_system_prompt_with_reasoning(prompt: str, enable: bool = True) -> str:
    """Add reasoning instruction to system prompt if enabled."""
    if not enable:
        return prompt

    reasoning_instruction = """
When responding, use the following format:

<THINKING>
Your internal reasoning, step-by-step analysis, and thought process goes here.
</THINKING>

<ANSWER>
Your final response to the user goes here.
</ANSWER>

The THINKING section will be hidden from the user.
"""
    return (prompt or "") + "\n\n" + reasoning_instruction


def get_default_model(provider: str) -> str:
    """Get default model for provider (ENV-driven, no hardcoded IDs)."""
    pk = _provider_key(provider)
    env_var = DEFAULT_MODEL_ENV.get(pk)
    if not env_var:
        raise ValueError(f"No DEFAULT_MODEL_ENV mapping for provider '{provider}'")
    model = _env_model(env_var)
    if not model:
        raise ValueError(f"{env_var} is not set (required to choose default model for provider '{provider}')")
    return model


def get_model_for_route(provider: str, route: Optional[str] = None) -> str:
    """
    Get model for provider+route using ENV.
    Falls back to provider default ENV var if a route-specific ENV var is missing.
    """
    pk = _provider_key(provider)
    if not route:
        return get_default_model(pk)

    route_map = ROUTE_MODEL_ENV.get(pk, {})
    route_env = route_map.get(route)
    if route_env:
        m = _env_model(route_env)
        if m:
            return m

    # fallback to default
    return get_default_model(pk)


# =========================
# STREAMING GENERATORS
# =========================

async def stream_openai(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
    tools: Optional[List[Dict]] = None,
    max_tokens: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from OpenAI using async client.
    """
    print(f"[STREAM_OPENAI] Called: model={model}, route={route}")

    if not HAS_OPENAI:
        print("[STREAM_OPENAI] ERROR: openai package not installed")
        yield {"type": "error", "message": "openai package not installed"}
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[STREAM_OPENAI] ERROR: OPENAI_API_KEY not set")
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    # Use provided model, route-based model, or default
    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("openai", route)
    else:
        use_model = get_default_model("openai")

    print(f"[STREAM_OPENAI] Using model: {use_model}")

    # v0.20: Reasoning effort for gpt-5.x / o-series models.
    # Controlled by OPENAI_REASONING_EFFORT env var (minimal | low | medium |
    # high). Only applied to models that accept it (the same gate used for
    # max_completion_tokens). Lets the user trade latency/cost for careful
    # multi-step reasoning — specifically valuable for web automation flows
    # where one wrong click breaks the whole sequence.
    #
    # v0.20.2 (2026-04-26): OpenAI rejects `tools` + `reasoning_effort` on
    # the entire gpt-5.x family in /v1/chat/completions with HTTP 400 —
    # not just mini. Confirmed: gpt-5.4-mini AND gpt-5.4 both fail with
    # the same error. The only way to use reasoning_effort alongside tools
    # is to switch to /v1/responses (different API shape — not done here).
    # Practical effect: this env var is currently inert for chat (tools are
    # always loaded). Agentic-tab routing still gives us full gpt-5.4 over
    # mini, which is the main capability win; explicit reasoning_effort is
    # an additional knob we can't use on this endpoint without restructuring.
    reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "").strip().lower()
    apply_reasoning = (
        reasoning_effort in ("minimal", "low", "medium", "high")
        and _openai_needs_max_completion_tokens(use_model)
        and not bool(tools)
    )
    if apply_reasoning:
        print(f"[STREAM_OPENAI] Reasoning effort: {reasoning_effort}")
    elif reasoning_effort and bool(tools) and _openai_needs_max_completion_tokens(use_model):
        print(f"[STREAM_OPENAI] Reasoning effort skipped: {use_model} + tools not supported on /v1/chat/completions (use /v1/responses for tools+reasoning)")

    client = AsyncOpenAI(api_key=api_key)

    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    full_messages = [{"role": "system", "content": enhanced_prompt}]
    full_messages.extend(messages)

    yield {"type": "metadata", "provider": "openai", "model": use_model}

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    max_completion_tokens = (
        _int_env("OPENAI_MAX_COMPLETION_TOKENS")
        or _int_env("OPENAI_STREAM_MAX_COMPLETION_TOKENS")
    )
    legacy_max_tokens = (
        _int_env("OPENAI_MAX_TOKENS")
        or _int_env("OPENAI_STREAM_MAX_TOKENS")
    )

    try:
        create_kwargs: Dict[str, Any] = {
            "model": use_model,
            "messages": full_messages,
            "stream": True,
        }


        # Convert Anthropic-format tools to OpenAI format if provided
        if tools:
            openai_tools = []
            for t in tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", t.get("parameters", {})),
                    },
                })
            # v2026-06-14 (hotfix): OpenAI hard-limits the tools array to 128.
            # Sending more 400s the ENTIRE request ("array_above_max_length"),
            # which silently kills chat (no tokens, no reply). Cap defensively so a
            # growing tool registry can never break the OpenAI path. Other providers
            # (Anthropic/Gemini) have their own limits and are untouched. The proper
            # fix is curating the tool set below 128 / prioritising per-context
            # — flagged for follow-up; this keeps chat working meanwhile.
            _OPENAI_MAX_TOOLS = 128
            if len(openai_tools) > _OPENAI_MAX_TOOLS:
                # Plain tail-truncation silently dropped tools the product can't
                # afford to lose: the food-logging / recipe tools (save_recipe,
                # log_recipe, get_health_history, ...) sit late in the assembled
                # list, so they were the ones cut (live 2026-06-15) — leaving
                # save_recipe/log_recipe unreachable on the OpenAI path. Move an
                # essentials allowlist to the front so they always survive the cap;
                # the dropped tail then comes from genuinely lower-priority tools.
                # OpenAI ignores tool order, so this changes only WHICH tools are
                # cut, never behaviour. (128 is OpenAI's hard array limit.)
                _ESSENTIAL_TOOLS = {
                    "log_nutrition", "resolve_nutrition", "save_recipe", "log_recipe",
                    "get_recent_nutrition", "get_today_summary", "get_health_history",
                    "log_weight", "log_workout", "log_exercise",
                }
                _essential = [t for t in openai_tools
                              if t["function"]["name"] in _ESSENTIAL_TOOLS]
                _rest = [t for t in openai_tools
                         if t["function"]["name"] not in _ESSENTIAL_TOOLS]
                _ordered = _essential + _rest
                _dropped = [t["function"]["name"] for t in _ordered[_OPENAI_MAX_TOOLS:]]
                print(f"[STREAM_OPENAI] Tools {len(openai_tools)} > {_OPENAI_MAX_TOOLS} max — capping; kept {len(_essential)} essential; dropped: {_dropped}")
                openai_tools = _ordered[:_OPENAI_MAX_TOOLS]
            create_kwargs["tools"] = openai_tools

        # Override max_tokens if explicitly provided
        if max_tokens:
            if _openai_needs_max_completion_tokens(use_model):
                create_kwargs["max_completion_tokens"] = int(max_tokens)
            else:
                create_kwargs["max_tokens"] = int(max_tokens)
        elif _openai_needs_max_completion_tokens(use_model):
            create_kwargs["max_completion_tokens"] = int(max_completion_tokens or 32768)
        elif legacy_max_tokens is not None:
            create_kwargs["max_tokens"] = int(legacy_max_tokens)

        # v0.20: pass reasoning_effort through to the API for gpt-5.x / o-series.
        # Unknown params are stripped silently by older SDK versions, so this
        # is safe to include unconditionally when the gate matched above.
        if apply_reasoning:
            create_kwargs["reasoning_effort"] = reasoning_effort

        try:
            stream = await client.chat.completions.create(
                **create_kwargs,
                stream_options={"include_usage": True},
            )
        except TypeError:
            stream = await client.chat.completions.create(**create_kwargs)

        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                total_tokens = getattr(usage, "total_tokens", None)

            choices = getattr(chunk, "choices", None)
            if not choices:
                continue

            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue

            # Handle tool calls from streaming deltas
            tool_calls_delta = getattr(delta, "tool_calls", None)
            if tool_calls_delta:
                for tc in tool_calls_delta:
                    func = getattr(tc, "function", None)
                    if func:
                        tc_id = getattr(tc, "id", None)
                        tc_name = getattr(func, "name", None)
                        tc_args = getattr(func, "arguments", None) or ""
                        if tc_id and tc_name:
                            yield {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input_json": tc_args,
                                "index": getattr(tc, "index", 0),
                            }
                        elif tc_args:
                            yield {
                                "type": "tool_use_delta",
                                "arguments": tc_args,
                                "index": getattr(tc, "index", 0),
                            }

            content = getattr(delta, "content", None)
            if content:
                yield {"type": "token", "text": content}

        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            yield {
                "type": "done",
                "provider": "openai",
                "model": use_model,
                "usage": {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0))),
                },
            }
        else:
            yield {"type": "done", "provider": "openai", "model": use_model}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


# =========================
# MAIN STREAMING FUNCTION
# =========================


# =========================
# NON-STREAM HELPERS
# =========================