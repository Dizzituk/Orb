# Purpose: streaming utils 2
# Called-by: app.llm._streaming_utils_3, app.llm.streaming
# Depends-on: app.llm.streaming
# Last-renovated: 2026-06-11
from __future__ import annotations
import os
from typing import Any, Dict, List, Optional
HAS_OPENAI = True
HAS_ANTHROPIC = True
HAS_GEMINI = True


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
    """Get dict of available streaming providers.
    
    v3.3: Added ollama/local provider — always available if OLLAMA_BASE_URL is set.
    v3.2: Uses 'google' as key (matching provider name used in routing)
    instead of legacy 'gemini' key.
    """
    # Check if Ollama is configured (no API key needed, just a URL)
    ollama_url = os.getenv("OLLAMA_BASE_URL", "")
    ollama_available = bool(ollama_url)

    result = {
        "openai": HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY")),
        "google": HAS_GEMINI and bool(os.getenv("GOOGLE_API_KEY")),
        "ollama": ollama_available,
        "local": ollama_available,  # Alias for ollama
    }
    # v3.2 debug: log key lengths so we can diagnose sync issues
    _oai = os.getenv("OPENAI_API_KEY", "")
    _ant = os.getenv("ANTHROPIC_API_KEY", "")
    _goo = os.getenv("GOOGLE_API_KEY", "")
    if not all([result["openai"], result["anthropic"], result["google"]]):
        print(f"[PROVIDER_AVAIL] Key lengths: OAI={len(_oai)}, ANT={len(_ant)}, GOO={len(_goo)}")
    if ollama_available:
        print(f"[PROVIDER_AVAIL] Ollama: configured at {ollama_url}")
    return result

def get_available_streaming_provider() -> Optional[str]:
    """Get the first available provider name.
    
    v3.2: Prioritises google > anthropic > openai to match current config.
    """
    providers = get_available_streaming_providers()
    # Check in priority order matching current pipeline config
    for name in ("google", "anthropic", "openai"):
        if providers.get(name, False):
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
