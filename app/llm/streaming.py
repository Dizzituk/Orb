# FILE: app/llm/streaming.py
"""
Orb Streaming LLM Module

Provides streaming interface for different LLM providers.
All streaming providers must follow the canonical event schema.

v0.16.0: Added reasoning support and metadata emission.
v0.17.0: Debug logging
v0.17.1: ENV-driven model selection (no hardcoded model IDs)
v0.17.2: Added call_llm_text helper for non-stream callers (Spec Gate, etc.)
v0.17.3: call_llm_text retry + OpenAI non-stream fallback for transient stream disconnects

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
import json
import logging
from typing import AsyncGenerator, Dict, List, Optional, Any
import re

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
DEFAULT_MODEL_ENV = {
    "openai": "OPENAI_DEFAULT_MODEL",
    "anthropic": "ANTHROPIC_DEFAULT_MODEL",
    "google": "GOOGLE_DEFAULT_MODEL",  # for gemini/google routes
}

# Provider -> Route -> ENV var that can override the model for that route.
# If the route env var is missing, we fall back to the provider default env var.
ROUTE_MODEL_ENV = {
    "openai": {
        "default": "OPENAI_DEFAULT_MODEL",
        "high_stakes": "OPENAI_HIGH_STAKES_MODEL",
        "budget": "OPENAI_BUDGET_MODEL",
        "spec_gate": "OPENAI_SPEC_GATE_MODEL",
    },
    "anthropic": {
        "default": "ANTHROPIC_DEFAULT_MODEL",
        "high_stakes": "ANTHROPIC_HIGH_STAKES_MODEL",
        "budget": "ANTHROPIC_BUDGET_MODEL",
        "spec_gate": "ANTHROPIC_SPEC_GATE_MODEL",
    },
    "google": {
        "default": "GOOGLE_DEFAULT_MODEL",
        "high_stakes": "GOOGLE_HIGH_STAKES_MODEL",
        "budget": "GOOGLE_BUDGET_MODEL",
        "spec_gate": "GOOGLE_SPEC_GATE_MODEL",
    },
}


def _provider_key(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in ("gemini", "google"):
        return "google"
    return p


def _env_model(var_name: str) -> Optional[str]:
    v = os.getenv(var_name, "").strip()
    return v or None


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


def get_available_streaming_providers() -> Dict[str, bool]:
    """Get dict of available streaming providers."""
    return {
        "openai": HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY")),
        "gemini": HAS_GEMINI and bool(os.getenv("GOOGLE_API_KEY")),
    }


def get_available_streaming_provider() -> Optional[str]:
    """Get the first available provider name."""
    providers = get_available_streaming_providers()
    for name, available in providers.items():
        if available:
            return name
    return None


def get_default_provider() -> Optional[str]:
    """Get the first available provider."""
    return get_available_streaming_provider()


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


def _should_retry_stream_error(message: str) -> bool:
    s = (message or "").lower()
    needles = [
        "incomplete chunked read",
        "peer closed connection",
        "server disconnected",
        "connection reset",
        "readerror",
        "timeout",
    ]
    return any(n in s for n in needles)


# =========================
# STREAMING GENERATORS
# =========================

async def stream_openai(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
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

        if _openai_needs_max_completion_tokens(use_model):
            create_kwargs["max_completion_tokens"] = int(max_completion_tokens or 8192)
        elif legacy_max_tokens is not None:
            create_kwargs["max_tokens"] = int(legacy_max_tokens)

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


async def stream_anthropic(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from Anthropic using async client."""
    print(f"[STREAM_ANTHROPIC] Called: model={model}, route={route}")

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

    client = anthropic.AsyncAnthropic(api_key=api_key)

    prompt_tokens = None
    completion_tokens = None

    try:
        resp = await client.messages.create(
            model=use_model,
            max_tokens=_int_env("ANTHROPIC_MAX_TOKENS") or 4096,
            system=enhanced_prompt,
            messages=messages,
            stream=True,
        )

        async for event in resp:
            if event.type == "content_block_delta":
                delta_text = getattr(event.delta, "text", None)
                if delta_text:
                    yield {"type": "token", "text": delta_text}

            if event.type == "message_delta":
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
                "usage": {
                    "prompt_tokens": int(prompt_tokens or 0),
                    "completion_tokens": int(completion_tokens or 0),
                    "total_tokens": int(total_tokens),
                },
            }
        else:
            yield {"type": "done", "provider": "anthropic", "model": use_model}

    except Exception as e:
        yield {"type": "error", "message": str(e)}


async def stream_gemini(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from Gemini (Google Generative AI)."""
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
    total_tokens = None

    def _uget(obj, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    try:
        gemini_model = genai.GenerativeModel(
            model_name=use_model,
            system_instruction=enhanced_prompt,
        )

        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=history)

        last_msg = messages[-1]["content"] if messages else ""
        response = chat.send_message(last_msg, stream=True)

        last_chunk = None
        for chunk in response:
            last_chunk = chunk
            if getattr(chunk, "text", None):
                yield {"type": "token", "text": chunk.text}

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


# =========================
# MAIN STREAMING FUNCTION
# =========================

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """Stream from specified LLM provider."""
    print(f"[STREAM_LLM] Called: provider={provider}, model={model}, messages={len(messages)}")

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
        async for event in stream_openai(messages, system_prompt, model, enable_reasoning, route):
            yield event
    elif provider == "anthropic":
        async for event in stream_anthropic(messages, system_prompt, model, enable_reasoning, route):
            yield event
    elif provider in ("gemini", "google"):
        async for event in stream_gemini(messages, system_prompt, model, enable_reasoning, route):
            yield event
    else:
        print(f"[STREAM_LLM] ERROR: Unknown provider '{provider}'")
        yield {"type": "error", "message": f"Unknown provider '{provider}'"}


# =========================
# NON-STREAM HELPERS
# =========================

async def _openai_text_nonstream(
    *,
    messages: List[Dict[str, str]],
    system_prompt: str,
    model: str,
) -> str:
    """
    Non-stream OpenAI call used as a fallback for transient stream disconnects.
    """
    if not HAS_OPENAI:
        raise RuntimeError("openai package not installed")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = AsyncOpenAI(api_key=api_key)

    # Ensure a system message exists
    full_messages: List[Dict[str, str]] = []
    if messages and messages[0].get("role") == "system":
        full_messages = messages
    else:
        full_messages = [{"role": "system", "content": system_prompt or ""}] + (messages or [])

    max_completion_tokens = (
        _int_env("OPENAI_MAX_COMPLETION_TOKENS")
        or _int_env("OPENAI_STREAM_MAX_COMPLETION_TOKENS")
        or 8192
    )
    legacy_max_tokens = (
        _int_env("OPENAI_MAX_TOKENS")
        or _int_env("OPENAI_STREAM_MAX_TOKENS")
    )

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": full_messages,
    }

    if _openai_needs_max_completion_tokens(model):
        create_kwargs["max_completion_tokens"] = int(max_completion_tokens)
    elif legacy_max_tokens is not None:
        create_kwargs["max_tokens"] = int(legacy_max_tokens)

    resp = await client.chat.completions.create(**create_kwargs)
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    msg = getattr(choices[0], "message", None)
    return (getattr(msg, "content", None) or "").strip()


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
) -> str:
    """
    Convenience helper for callers that want a single final string.

    - Uses stream_llm under the hood.
    - Collects {"type":"token"} chunks into one string.
    - Raises RuntimeError on {"type":"error"}.
    - Retries once for transient stream disconnects; for OpenAI it falls back to a non-stream call.
    """
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
