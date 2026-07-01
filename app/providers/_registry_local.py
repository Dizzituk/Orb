# FILE: app/providers/_registry_local.py
# Purpose: Local OpenAI-compatible provider lane + background-local locality guard for llm_call.
# Called-by: app.providers.registry, app.providers._registry_utils_3
# Depends-on: openai SDK (AsyncOpenAI at LOCAL_LLM_BASE_URL), app.providers._registry_utils_3/_4
# Last-renovated: 2026-07-01
"""
Local provider lane for the non-streaming registry (provider id "local").

Serves Nat (vLLM, OpenAI-compatible, port 8003) today; the Gemma 27B
workers on the RTX 3090 later. Migration = repoint LOCAL_LLM_* env vars.

Env resolution (no literal model IDs or endpoints in code):
    LOCAL_LLM_BASE_URL      -> falls back to VLLM_BASE_URL (Nat's endpoint)
    LOCAL_LLM_DEFAULT_MODEL -> falls back to NAT_MODEL
    LOCAL_LLM_API_KEY       -> falls back to VLLM_API_KEY; dummy accepted
                               ("EMPTY" is the vLLM convention, mirrors
                               app/llm/streaming_vllm.py)

Locality guard (mirrors the enable_web_search precedent in registry.py —
flagged and refused at the registry, never silently rerouted):
    llm_call(..., execution_context="background_local") refuses every cloud
    chat/completion provider (openai/anthropic/google/gemini) with a
    LOCALITY_REFUSED result; provider_id=None resolves to "local".

Whitelisted background exceptions — verified NOT to ride llm_call, so the
guard cannot touch them (and must not grow to):
    - Embedding calls: app/embeddings/service.py + Gemini multimodal embeds
      call their SDKs directly, never the chat registry.
    - Brave Search: plain HTTPS in app/tools/registry.py:_brave_search.

Naming note: the STREAMING side has its own local lanes (streaming_vllm.py
provider "vllm_local", streaming_ollama.py) — intentionally untouched; this
is the non-streaming counterpart.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from app.providers._registry_utils_3 import LlmUsage, _normalize_messages_for_openai
from app.providers._registry_utils_4 import LlmCallResult, LlmCallStatus

logger = logging.getLogger(__name__)

# The execution_context value that activates the locality guard.
LOCALITY_BACKGROUND_LOCAL = "background_local"

# Cloud chat/completion providers refused under background_local.
CLOUD_CHAT_PROVIDERS = frozenset({"openai", "anthropic", "google", "gemini"})


def local_base_url() -> str:
    return (os.getenv("LOCAL_LLM_BASE_URL") or os.getenv("VLLM_BASE_URL") or "").strip()


def local_api_key() -> str:
    # Local servers accept a dummy key; "EMPTY" is the vLLM convention.
    return (os.getenv("LOCAL_LLM_API_KEY") or os.getenv("VLLM_API_KEY") or "EMPTY").strip()


def local_default_model() -> str:
    return (os.getenv("LOCAL_LLM_DEFAULT_MODEL") or os.getenv("NAT_MODEL") or "").strip()


def is_local_available() -> bool:
    """Available when a base URL is configured and the openai SDK imports."""
    if not local_base_url():
        return False
    try:
        from openai import AsyncOpenAI  # noqa: F401
    except Exception:
        return False
    return True


def locality_refusal(execution_context: Optional[str], provider_id: Optional[str]) -> Optional[str]:
    """Return a refusal message if this (context, provider) pair is forbidden.

    Only execution_context == "background_local" constrains anything; the
    provider must then be non-cloud. Returns None when the call may proceed.
    """
    if execution_context != LOCALITY_BACKGROUND_LOCAL:
        return None
    if provider_id in CLOUD_CHAT_PROVIDERS:
        return (
            f"Locality refused: execution_context={LOCALITY_BACKGROUND_LOCAL!r} forbids "
            f"cloud chat provider {provider_id!r}. Background work runs on the local "
            f"lane only (provider_id='local'); embeddings and Brave Search are the "
            f"whitelisted non-chat exceptions."
        )
    return None


async def call_local(
    model_id: str,
    messages: List[dict],
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    tool_defs: Optional[List[dict]] = None,
) -> LlmCallResult:
    """Non-streaming chat completion against the local OpenAI-compatible server.

    The registry's tool loop is not wired for the local lane yet — background
    tasks use plain completions. Tools requested here are flagged and ignored.
    """
    from openai import AsyncOpenAI

    if tool_defs:
        logger.warning("[registry.local] tool loop not supported on the local lane yet; ignoring %d tools.", len(tool_defs))

    client = AsyncOpenAI(base_url=local_base_url(), api_key=local_api_key(), timeout=timeout_seconds)
    oai_messages = _normalize_messages_for_openai(messages, system_prompt)

    create_kwargs = dict(
        model=model_id,
        messages=oai_messages,
        stream=False,
        max_tokens=int(max_tokens),  # vLLM/llama.cpp/Ollama all take plain max_tokens
    )
    try:
        t = float(temperature) if temperature is not None else None
    except Exception:
        t = None
    if t is not None:
        create_kwargs["temperature"] = t

    resp = await client.chat.completions.create(**create_kwargs)

    content = (resp.choices[0].message.content or "") if resp.choices else ""
    usage = LlmUsage(
        prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) if getattr(resp, "usage", None) else 0,
        completion_tokens=getattr(resp.usage, "completion_tokens", 0) if getattr(resp, "usage", None) else 0,
    )
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

    raw = None
    try:
        raw = resp.model_dump() if hasattr(resp, "model_dump") else None
    except Exception:
        raw = None

    return LlmCallResult(
        status=LlmCallStatus.SUCCESS,
        provider_id="local",
        model_id=model_id,
        content=content,
        usage=usage,
        raw_response=raw,
    )
