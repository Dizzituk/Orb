# FILE: app/llm/clients.py
"""
LLM Client Helpers (Phase 4, Unified Provider Path)

IMPORTANT:
- This module NO LONGER makes raw chat/completions API calls.
- All LLM calls go through app.providers.registry.llm_call (single path).
- Embeddings remain a separate concern and use OpenAI directly.

PHASE 4 FIXES:
- Fixed JobBudget field names (max_tokens, max_cost_estimate, max_wall_time_seconds)
- Fixed modalities → modalities_in
- Fixed needs_tools type (bool → list[str])

This module provides:
- Message formatting helpers
- Provider availability checks
- Backward-compatible wrapper functions for existing code

Async-first:
- async_call_openai()
- async_call_anthropic()
- async_call_google()

Sync wrappers (CLI/testing only — DO NOT use in FastAPI handlers):
- call_openai()
- call_anthropic()
- call_google()

Embeddings:
- get_embeddings()
- generate_embedding()  (single-vector convenience wrapper)

Provider info:
- check_provider_availability()
- list_available_providers()
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from app.jobs.schemas import (
    JobEnvelope,
    JobType,
    Importance,
    DataSensitivity,
    Modality,
    JobBudget,
    OutputContract,
    validate_job_envelope,
    ValidationError,
)
from app.providers.registry import (
    llm_call as registry_llm_call,
    is_provider_available,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _build_basic_envelope(
    *,
    provider_id: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    job_type: JobType = JobType.CHAT_SIMPLE,
    importance: Importance = Importance.MEDIUM,
    max_tokens: int = 8000,
    max_cost_usd: float = 1.0,
    timeout_seconds: int = 60,
    project_id: int = 1,
) -> JobEnvelope:
    """
    Build a minimal JobEnvelope for simple chat-style calls.

    This is used by async_call_openai/anthropic/google and provides a
    consistent Phase 4 envelope even for legacy call sites.
    """
    # Ensure system prompt is included as first message (if provided)
    msg_list: List[Dict[str, Any]] = []
    if system_prompt:
        msg_list.append({"role": "system", "content": system_prompt})
    msg_list.extend(messages)

    # FIXED: Use correct JobBudget field names
    budget = JobBudget(
        max_tokens=max_tokens,
        max_cost_estimate=max_cost_usd,
        max_wall_time_seconds=timeout_seconds,
    )

    envelope = JobEnvelope(
        job_id=str(uuid4()),
        session_id=f"clients-{provider_id}-{uuid4()}",
        project_id=project_id,
        job_type=job_type,
        importance=importance,
        data_sensitivity=DataSensitivity.INTERNAL,
        modalities_in=[Modality.TEXT],  # FIXED: was 'modalities'
        budget=budget,
        output_contract=OutputContract.TEXT_RESPONSE,
        messages=msg_list,
        metadata={
            "source": "clients",
            "provider_hint": provider_id,
            "model_hint": model_id,
        },
        allow_multi_model_review=False,
        needs_tools=[],  # FIXED: was False (bool), must be list[str]
    )

    # Validate according to Phase 4 rules
    validate_job_envelope(envelope)
    return envelope


def _llm_call_and_unpack(
    *,
    provider_id: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str],
    job_type: JobType,
) -> Tuple[str, Dict[str, Any]]:
    """
    Internal sync helper:
    - Builds envelope
    - Calls registry.llm_call (async, via asyncio.run)
    - Returns (content, usage_dict)
    """
    import asyncio

    async def _runner() -> Tuple[str, Dict[str, Any]]:
        try:
            envelope = _build_basic_envelope(
                provider_id=provider_id,
                model_id=model_id,
                messages=messages,
                system_prompt=system_prompt,
                job_type=job_type,
            )
        except ValidationError as ve:
            logger.warning("[clients] Envelope validation failed: %s", ve)
            return (
                f"Envelope validation failed: {ve}",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "error": "validation_error",
                },
            )
        except Exception as ve:
            # Catch Pydantic ValidationError as well
            logger.warning("[clients] Envelope validation failed: %s", ve)
            return (
                f"Envelope validation failed: {ve}",
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "error": "validation_error",
                },
            )

        try:
            result = await registry_llm_call(
                provider_id=provider_id,
                model_id=model_id,
                messages=envelope.messages,
                system_prompt=None,  # already baked into messages
                job_envelope=envelope,
            )
        except Exception as exc:
            logger.exception("[clients] llm_call failed: %s", exc)
            return (
                str(exc),
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "error": "exception",
                },
            )

        usage = {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
            "cost_usd": result.usage.cost_estimate,
        }

        if not result.is_success():
            logger.error(
                "[clients] LLM call error (provider=%s, model=%s): %s",
                provider_id,
                model_id,
                result.error_message,
            )
            usage["error"] = result.error_code or "error"
            return result.error_message or "", usage

        return result.content or "", usage

    return asyncio.run(_runner())


async def _llm_call_and_unpack_async(
    *,
    provider_id: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    system_prompt: Optional[str],
    job_type: JobType,
) -> Tuple[str, Dict[str, Any]]:
    """
    Internal async helper (FastAPI-safe version of _llm_call_and_unpack).
    """
    try:
        envelope = _build_basic_envelope(
            provider_id=provider_id,
            model_id=model_id,
            messages=messages,
            system_prompt=system_prompt,
            job_type=job_type,
        )
    except ValidationError as ve:
        logger.warning("[clients] Envelope validation failed: %s", ve)
        return (
            f"Envelope validation failed: {ve}",
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "error": "validation_error",
            },
        )
    except Exception as ve:
        # Catch Pydantic ValidationError as well
        logger.warning("[clients] Envelope validation failed: %s", ve)
        return (
            f"Envelope validation failed: {ve}",
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "error": "validation_error",
            },
        )

    try:
        result = await registry_llm_call(
            provider_id=provider_id,
            model_id=model_id,
            messages=envelope.messages,
            system_prompt=None,
            job_envelope=envelope,
        )
    except Exception as exc:
        logger.exception("[clients] llm_call failed: %s", exc)
        return (
            str(exc),
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "error": "exception",
            },
        )

    usage = {
        "prompt_tokens": result.usage.prompt_tokens,
        "completion_tokens": result.usage.completion_tokens,
        "total_tokens": result.usage.total_tokens,
        "cost_usd": result.usage.cost_estimate,
    }

    if not result.is_success():
        logger.error(
            "[clients] LLM call error (provider=%s, model=%s): %s",
            provider_id,
            model_id,
            result.error_message,
        )
        usage["error"] = result.error_code or "error"
        return result.error_message or "", usage

    return result.content or "", usage


# =============================================================================
# PROVIDER AVAILABILITY / INFO
# =============================================================================


def check_provider_availability() -> Dict[str, bool]:
    """
    Check which providers have API keys configured and available.

    Returns:
        Dict of provider_id -> is_available
    """
    return {
        "openai": is_provider_available("openai"),
        "anthropic": is_provider_available("anthropic"),
        "google": is_provider_available("google"),
    }


def list_available_providers() -> List[str]:
    """
    Backward-compatible helper expected by app.llm.__init__.

    Returns:
        List of provider IDs that are currently available.
    """
    availability = check_provider_availability()
    return [pid for pid, ok in availability.items() if ok]


# =============================================================================
# ASYNC ENTRY POINTS (Primary — for FastAPI / internal async code)
# =============================================================================


async def async_call_openai(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Async helper for calling OpenAI chat models via provider registry.

    Note: temperature is accepted for API compatibility but not yet
    wired through to the registry. All calls use default temperature.

    Returns:
        (content, usage_dict)
    """
    model_id = model or os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
    return await _llm_call_and_unpack_async(
        provider_id="openai",
        model_id=model_id,
        messages=messages,
        system_prompt=system_prompt,
        job_type=JobType.CHAT_SIMPLE,
    )


async def async_call_anthropic(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Async helper for calling Anthropic models via provider registry.

    Note: temperature is accepted for API compatibility but not yet
    wired through to the registry. All calls use default temperature.

    Returns:
        (content, usage_dict)
    """
    model_id = model or os.getenv(
        "ANTHROPIC_DEFAULT_MODEL", "claude-3-5-sonnet-latest"
    )
    return await _llm_call_and_unpack_async(
        provider_id="anthropic",
        model_id=model_id,
        messages=messages,
        system_prompt=system_prompt,
        job_type=JobType.CHAT_SIMPLE,
    )


async def async_call_google(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Async helper for calling Google Gemini models via provider registry.

    Note: temperature is accepted for API compatibility but not yet
    wired through to the registry. All calls use default temperature.

    Returns:
        (content, usage_dict)
    """
    model_id = model or os.getenv(
        "GOOGLE_DEFAULT_MODEL", os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
    )
    return await _llm_call_and_unpack_async(
        provider_id="google",
        model_id=model_id,
        messages=messages,
        system_prompt=system_prompt,
        job_type=JobType.CHAT_SIMPLE,
    )


# =============================================================================
# SYNC WRAPPERS (CLI / streaming / tests ONLY)
# =============================================================================


def call_openai(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Sync wrapper for async_call_openai.

    WARNING:
        Uses asyncio.run(). Intended ONLY for:
        - CLI tools
        - Local scripts
        - Legacy streaming code

        DO NOT use this directly inside FastAPI request handlers.
    """
    import asyncio

    return asyncio.run(
        async_call_openai(
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            model=model,
        )
    )


def call_anthropic(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Sync wrapper for async_call_anthropic.

    WARNING:
        Uses asyncio.run(). Intended ONLY for CLI/streaming/testing.
    """
    import asyncio

    return asyncio.run(
        async_call_anthropic(
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            model=model,
        )
    )


def call_google(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.7,
    model: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Sync wrapper for async_call_google.

    WARNING:
        Uses asyncio.run(). Intended ONLY for CLI/streaming/testing.
    """
    import asyncio

    return asyncio.run(
        async_call_google(
            system_prompt=system_prompt,
            messages=messages,
            temperature=temperature,
            model=model,
        )
    )


# =============================================================================
# EMBEDDINGS (Separate from provider registry for now)
# =============================================================================


def get_embeddings(
    texts: Union[str, List[str]],
    model: Optional[str] = None,
) -> Union[List[float], List[List[float]]]:
    """
    Generate embedding vectors for the given text(s) using OpenAI.

    Backward-compatible helper expected by some parts of the codebase.

    Args:
        texts: Single string or list of strings.
        model: Optional model override.

    Returns:
        - If input is a single string: List[float]
        - If input is a list of strings: List[List[float]]
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set; cannot generate embeddings."
        )

    client = OpenAI(api_key=api_key)
    embedding_model = model or os.getenv(
        "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
    )

    # Normalize input
    is_single = isinstance(texts, str)
    input_data: List[str] = [texts] if is_single else list(texts)

    resp = client.embeddings.create(
        model=embedding_model,
        input=input_data,
    )
    vectors: List[List[float]] = [item.embedding for item in resp.data]

    if is_single:
        return vectors[0]
    return vectors


def generate_embedding(
    text: str,
    model: Optional[str] = None,
) -> List[float]:
    """
    Convenience wrapper around get_embeddings() for a single string.

    Returns:
        List[float] embedding vector.
    """
    vec = get_embeddings(texts=text, model=model)
    # get_embeddings(str) guarantees a single vector (List[float])
    return vec  # type: ignore[return-value]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Async chat helpers
    "async_call_openai",
    "async_call_anthropic",
    "async_call_google",
    # Sync wrappers
    "call_openai",
    "call_anthropic",
    "call_google",
    # Embeddings
    "get_embeddings",
    "generate_embedding",
    # Provider info
    "check_provider_availability",
    "list_available_providers",
]