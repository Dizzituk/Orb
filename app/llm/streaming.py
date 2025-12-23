# FILE: app/llm/streaming.py
"""
Orb Streaming LLM Module

Provides streaming interface for different LLM providers.
All streaming providers must follow the canonical event schema.

v0.16.0: Added reasoning support and metadata emission.

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

import os
import logging
from typing import AsyncGenerator, Dict, List, Optional
from datetime import datetime
import asyncio
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

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-3-5-sonnet-20240620",
    "google": "gemini-2.0-flash",
}

# Route-based model overrides
ROUTE_MODELS = {
    "openai": {
        "default": "gpt-4.1-mini",
        "high_stakes": "gpt-4.1",
        "budget": "gpt-4.1-mini",
    },
    "anthropic": {
        "default": "claude-3-5-sonnet-20240620",
        "high_stakes": "claude-3-5-sonnet-20240620",
        "budget": "claude-3-5-haiku-20240307",
    },
    "google": {
        "default": "gemini-2.0-flash",
        "high_stakes": "gemini-2.5-pro",
        "budget": "gemini-2.0-flash",
    },
}


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
    return prompt + "\n\n" + reasoning_instruction


def get_model_for_route(provider: str, route: Optional[str] = None) -> str:
    """Get appropriate model for provider and route."""
    if not route:
        return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])

    provider_routes = ROUTE_MODELS.get(provider, {})
    return provider_routes.get(route, provider_routes.get("default", DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])))


def get_default_model(provider: str) -> str:
    """Get default model for provider."""
    return DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])


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


# ============ STREAMING GENERATORS ============

async def stream_openai(
    messages: List[Dict],
    system_prompt: str = "",
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from OpenAI using async client.

    Yields dict events following canonical schema:
    - {"type": "metadata", "provider": "openai", "model": "..."}
    - {"type": "token", "text": "..."}
    - {"type": "error", "message": "..."}
    - {"type": "done", "usage": {...}}  (when available)
    """
    if not HAS_OPENAI:
        yield {"type": "error", "message": "openai package not installed"}
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "OPENAI_API_KEY not set"}
        return

    # Use provided model, route-based model, or default
    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("openai", route)
    else:
        use_model = get_default_model("openai")

    client = AsyncOpenAI(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    full_messages = [{"role": "system", "content": enhanced_prompt}]
    full_messages.extend(messages)

    # Emit metadata first
    yield {"type": "metadata", "provider": "openai", "model": use_model}

    def _uget(obj, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    prompt_tokens = None
    completion_tokens = None
    total_tokens = None

    # Optional token caps (env-tunable). For gpt-5.* prefer max_completion_tokens.
    max_completion_tokens = (
        _int_env("OPENAI_MAX_COMPLETION_TOKENS")
        or _int_env("OPENAI_STREAM_MAX_COMPLETION_TOKENS")
    )
    legacy_max_tokens = (
        _int_env("OPENAI_MAX_TOKENS")
        or _int_env("OPENAI_STREAM_MAX_TOKENS")
    )

    try:
        create_kwargs = {
            "model": use_model,
            "messages": full_messages,
            "stream": True,
        }

        if _openai_needs_max_completion_tokens(use_model):
            create_kwargs["max_completion_tokens"] = int(max_completion_tokens or 8192)
        elif legacy_max_tokens is not None:
            create_kwargs["max_tokens"] = int(legacy_max_tokens)

        # Prefer include_usage if supported by installed OpenAI SDK.
        try:
            stream = await client.chat.completions.create(
                **create_kwargs,
                stream_options={"include_usage": True},
            )
        except TypeError:
            # Older SDKs may not accept stream_options.
            stream = await client.chat.completions.create(**create_kwargs)

        async for chunk in stream:
            # Usage is typically attached to the final chunk when include_usage is enabled.
            u = _uget(chunk, "usage")
            if u:
                pt = _uget(u, "prompt_tokens")
                ct = _uget(u, "completion_tokens")
                tt = _uget(u, "total_tokens")
                if pt is not None:
                    prompt_tokens = int(pt)
                if ct is not None:
                    completion_tokens = int(ct)
                if tt is not None:
                    total_tokens = int(tt)

            try:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield {"type": "token", "text": chunk.choices[0].delta.content}
            except Exception:
                continue

        # Emit done with usage if available
        if prompt_tokens is not None or completion_tokens is not None or total_tokens is not None:
            if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            usage = {
                "prompt_tokens": int(prompt_tokens or 0),
                "completion_tokens": int(completion_tokens or 0),
                "total_tokens": int(total_tokens or (int(prompt_tokens or 0) + int(completion_tokens or 0))),
            }
            yield {"type": "done", "provider": "openai", "model": use_model, "usage": usage}
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
    """
    Stream from Anthropic using async client.

    Yields dict events following canonical schema:
    - {"type": "metadata", "provider": "anthropic", "model": "..."}
    - {"type": "token", "text": "..."}
    - {"type": "error", "message": "..."}
    - {"type": "done", "usage": {...}}  (when available)
    """
    if not HAS_ANTHROPIC:
        yield {"type": "error", "message": "anthropic package not installed"}
        return

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY not set"}
        return

    # Use provided model, route-based model, or default
    if model:
        use_model = model
    elif route:
        use_model = get_model_for_route("anthropic", route)
    else:
        use_model = get_default_model("anthropic")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Enhance system prompt
    enhanced_prompt = enhance_system_prompt_with_reasoning(system_prompt, enable_reasoning)

    # Emit metadata first
    yield {"type": "metadata", "provider": "anthropic", "model": use_model}

    input_tokens = None
    output_tokens = None

    def _uget(obj, key: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    try:
        async with client.messages.stream(
            model=use_model,
            max_tokens=4096,
            system=enhanced_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield {"type": "token", "text": text}

            final_msg = None
            try:
                maybe = stream.get_final_message()
                if asyncio.iscoroutine(maybe):
                    final_msg = await maybe
                else:
                    final_msg = maybe
            except Exception:
                final_msg = None

            if final_msg is not None:
                u = _uget(final_msg, "usage")
                if u:
                    it = _uget(u, "input_tokens")
                    ot = _uget(u, "output_tokens")
                    if it is not None:
                        input_tokens = int(it)
                    if ot is not None:
                        output_tokens = int(ot)

        if input_tokens is not None or output_tokens is not None:
            usage = {
                "prompt_tokens": int(input_tokens or 0),
                "completion_tokens": int(output_tokens or 0),
                "total_tokens": int((input_tokens or 0) + (output_tokens or 0)),
            }
            yield {"type": "done", "provider": "anthropic", "model": use_model, "usage": usage}
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
    """
    Stream from Gemini.

    Yields dict events following canonical schema:
    - {"type": "metadata", "provider": "gemini", "model": "..."}
    - {"type": "token", "text": "..."}
    - {"type": "error", "message": "..."}
    - {"type": "done", "usage": {...}}  (when available)
    """
    if not HAS_GEMINI:
        yield {"type": "error", "message": "google-generativeai package not installed"}
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


# ============ MAIN STREAMING FUNCTION ============

async def stream_llm(
    messages: List[Dict],
    system_prompt: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    enable_reasoning: bool = False,
    route: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Stream from specified LLM provider.

    Args:
        messages: List of message dicts with role and content
        system_prompt: System prompt string
        provider: Provider name ("openai", "anthropic", "gemini")
        model: Specific model name (optional)
        enable_reasoning: If True, instruct LLM to use THINKING/ANSWER tags
        route: Route name for model selection (optional)

    Yields:
        Dict events following canonical schema:
        - {"type": "metadata", "provider": "...", "model": "..."}
        - {"type": "token", "text": "..."}
        - {"type": "reasoning", "text": "..."} (optional)
        - {"type": "error", "message": "..."}
    """
    if not provider:
        provider = get_default_provider()

    if not provider:
        yield {"type": "error", "message": "No LLM providers available"}
        return

    provider = provider.lower()

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
        yield {"type": "error", "message": f"Unknown provider '{provider}'"}
