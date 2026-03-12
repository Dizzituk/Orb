# FILE: app/intelligent_memory/__init__.py
"""
Intelligent Memory Architecture.

Three-layer memory with structured retrieval:
  Layer 1 — Hot Cache: in-memory dict, preloaded on startup (<50KB)
  Layer 2 — Knowledge Graph: NetworkX entity-relationship graph
  Layer 3 — Deep Store: existing RAG vector search (improved)

Plus:
  Asset Index — tracks everything ASTRA has created/downloaded
  Retrieval Router — cascading query through layers
  Knowledge Extraction — conversation→memory pipeline

Queries hit layers in order: Hot→Graph→Deep. Common operations
never touch the slow vector path.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-MEM-001.
"""
from app.intelligent_memory.hot_cache import HotCache, get_hot_cache
from app.intelligent_memory.retrieval_router import RetrievalRouter, get_retrieval_router

__all__ = [
    "HotCache",
    "get_hot_cache",
    "RetrievalRouter",
    "get_retrieval_router",
]
