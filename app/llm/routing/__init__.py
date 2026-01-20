# FILE: app/llm/routing/__init__.py
"""
Routing subpackage for ASTRA stream routing.

v2.0 (2026-01-20): Major refactor - modularized stream_router.py

This package provides:
- handler_registry: Centralized handler imports and availability flags
- command_dispatch: Intent → handler dispatch for commands
- chat_routing: Chat mode, normal routing, legacy triggers
- prompt_builders: System prompt and message construction
- rag_fallback: Architecture query detection patterns

The main public API remains `app.llm.stream_router`. Modules in here are
internal organization for maintainability.

Existing modules (unchanged):
- core: Non-streaming LLM routing (from router.py)
- job_routing: Job classification helpers
- envelope: JobEnvelope construction
- local_actions: ZOBIE MAP, ARCH QUERY handlers
- video_code_debug: Video+code debug pipeline
- memory_injection: Memory injection helpers
- routing_persistence: Routing state persistence

NOTE: To avoid circular imports, most re-exports are removed.
Import directly from the submodules when needed.
"""

# Only export from modules that don't have circular dependencies
from .rag_fallback import is_architecture_query

# These are safe to export from core (no circular deps)
from .core import (
    call_llm_async,
    call_llm,
    quick_chat_async,
    quick_chat,
    request_code_async,
    request_code,
    classify_and_route,
)

__all__ = [
    # RAG fallback
    "is_architecture_query",
    # Core exports
    "call_llm_async",
    "call_llm",
    "quick_chat_async",
    "quick_chat",
    "request_code_async",
    "request_code",
    "classify_and_route",
]
