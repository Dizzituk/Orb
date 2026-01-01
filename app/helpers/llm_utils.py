# FILE: app/helpers/llm_utils.py
"""
LLM utility functions for Orb endpoints.

Extracted from main.py for better organization.
"""

import asyncio
import inspect
from typing import Optional

from app.llm import LLMResult, JobType
from app.llm.clients import call_openai
from app.auth.middleware import AuthResult


def simple_llm_call(prompt: str) -> str:
    """Quick LLM call for analysis tasks."""
    try:
        content, _ = call_openai(
            system_prompt="You are a helpful assistant. Respond with only what is asked, no extra text.",
            messages=[{"role": "user", "content": prompt}],
        )
        return content
    except Exception as e:
        print(f"[simple_llm_call] Error: {e}")
        return ""


def make_session_id(auth: AuthResult) -> str:
    """Best-effort stable session id for audit correlation."""
    sid = getattr(auth, "session_id", None)
    if sid:
        return str(sid)
    uid = getattr(auth, "user_id", None)
    if uid:
        return str(uid)
    return "unknown"


def sync_await(maybe_awaitable):
    """
    Resolve an awaitable returned from a sync call path.
    
    Fixes cases where call_llm (or downstream) returns a coroutine.
    In normal operation call_llm is sync and this is a no-op.
    """
    if inspect.isawaitable(maybe_awaitable):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(maybe_awaitable)
        raise RuntimeError(
            "call_llm returned an awaitable on a running event loop. "
            "Convert the caller to async and await the result."
        )
    return maybe_awaitable


def extract_provider_value(result: LLMResult) -> str:
    """Extract provider string from LLMResult."""
    if result.provider is None:
        return "unknown"
    if hasattr(result.provider, 'value'):
        return result.provider.value
    return str(result.provider)


def extract_model_value(result: LLMResult) -> Optional[str]:
    """Extract model string from LLMResult."""
    return result.model if hasattr(result, 'model') else None


def classify_job_type(message: str, requested_type: str) -> JobType:
    """
    Simple job type passthrough - lets router.py handle all classification.
    
    The router's classify_and_route() does the real classification using job_classifier.
    This just validates the requested type or defaults to CHAT_LIGHT.
    """
    if requested_type and requested_type != "casual_chat":
        try:
            return JobType(requested_type)
        except ValueError:
            print(f"[classify] Invalid job_type '{requested_type}', defaulting to CHAT_LIGHT")
            return JobType.CHAT_LIGHT
    
    print(f"[classify] Defaulting to CHAT_LIGHT")
    return JobType.CHAT_LIGHT


def map_model_to_vision_tier(model: str) -> str:
    """
    Map job_classifier's model selection to gemini_vision tier.
    
    Tier mapping:
    - gemini-2.0-flash → "fast"
    - gemini-2.5-pro → "complex" (for IMAGE_COMPLEX)
    - gemini-3.0-pro-preview or gemini-3-pro → "video_heavy"
    - default → "fast"
    """
    if not model:
        return "fast"
    
    model_lower = model.lower()
    
    if "flash" in model_lower or "2.0" in model_lower:
        return "fast"
    elif "3.0" in model_lower or "3-pro" in model_lower or "3.0-pro" in model_lower:
        return "video_heavy"
    elif "2.5-pro" in model_lower:
        return "complex"
    else:
        return "fast"
