# FILE: app/embeddings/__init__.py
"""
Semantic search module for Orb.
Provides embedding generation, storage, and vector similarity search.
"""

import os

from .service import (
    generate_embedding,
    store_embedding,
    search_embeddings,
    index_project,
    reindex_project,
    delete_embeddings_for_source,
    index_note,
    index_message,
    index_document,
)

from .models import Embedding

from .schemas import (
    EmbeddingOut,
    SearchRequest,
    SearchResult,
    SearchResponse,
    IndexRequest,
    IndexResponse,
)


def auto_index_enabled() -> bool:
    """
    Check if automatic embedding indexing is enabled for notes/messages.
    Controlled by ORB_AUTO_INDEX env var. Default: enabled.
    """
    return os.getenv("ORB_AUTO_INDEX", "true").lower() in {"1", "true", "yes"}


__all__ = [
    # Service functions
    "generate_embedding",
    "store_embedding",
    "search_embeddings",
    "index_project",
    "reindex_project",
    "delete_embeddings_for_source",
    "index_note",
    "index_message",
    "index_document",
    # Feature flag
    "auto_index_enabled",
    # Model
    "Embedding",
    # Schemas
    "EmbeddingOut",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "IndexRequest",
    "IndexResponse",
]