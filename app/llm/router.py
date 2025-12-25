# FILE: app/llm/router.py
"""Public LLM routing API.

This module is intentionally small. The implementation lives in:
- app/llm/routing/core.py

Keeping this wrapper thin makes the routing surface easier to audit and refactor
without breaking imports across the codebase.
"""

from __future__ import annotations

# Implementation module (moved)
from app.llm.routing.core import (
    # Async implementations (still available)
    call_llm_async,
    quick_chat_async,
    request_code_async,
    review_work_async,

    # Compatibility names expected elsewhere
    analyze_with_vision,
    web_search_query,
    list_job_types,
    get_routing_info,
    is_policy_routing_enabled,
    enable_policy_routing,

    # Critical pipeline helpers used by stream_router and others
    run_high_stakes_with_critique,
    synthesize_envelope_from_task,
    is_high_stakes_job,
    is_opus_model,
)

# Backwards-compatible sync names (the rest of the app expects these)
call_llm = call_llm_async
quick_chat = quick_chat_async
request_code = request_code_async
review_work = review_work_async

__all__ = [
    # Primary API
    "call_llm",
    "quick_chat",
    "request_code",
    "review_work",

    # Async equivalents (keep exported for any internal callers)
    "call_llm_async",
    "quick_chat_async",
    "request_code_async",
    "review_work_async",

    # Tooling / extra endpoints
    "analyze_with_vision",
    "web_search_query",

    # Routing meta
    "list_job_types",
    "get_routing_info",
    "is_policy_routing_enabled",
    "enable_policy_routing",

    # Critical pipeline helpers
    "run_high_stakes_with_critique",
    "synthesize_envelope_from_task",
    "is_high_stakes_job",
    "is_opus_model",
]
