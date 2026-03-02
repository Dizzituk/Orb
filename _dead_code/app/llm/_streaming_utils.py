import os
from openai import AsyncOpenAI
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_ENV = {
    "openai": "OPENAI_DEFAULT_MODEL",
    "anthropic": "ANTHROPIC_DEFAULT_MODEL",
    "google": "GOOGLE_DEFAULT_MODEL",  # for gemini/google routes
}

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

def _env_model(var_name: str) -> Optional[str]:
    v = os.getenv(var_name, "").strip()
    return v or None

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

async def _openai_text_nonstream(
    *,
    messages: List[Dict[str, str]],
    system_prompt: str,
    model: str,
) -> str:
    """
    Non-stream OpenAI call used as a fallback for transient stream disconnects.
    """
    from .streaming import AsyncOpenAI, _int_env, _openai_needs_max_completion_tokens
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
