# FILE: app/embeddings/__init__.py
# Purpose: Semantic search module for Orb (Gemini Embedding 2).
# Called-by: app.db, app.drive.content_indexer, app.embeddings.router, app.endpoints.chat (+5 more)
# Depends-on: app.embeddings.gemini_provider, app.embeddings.models, app.embeddings.schemas, app.embeddings.service
# Last-renovated: 2026-06-11
"""
Semantic search module for Orb (Gemini Embedding 2).
Provides text + multimodal embedding generation, storage, and vector similarity search.
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

# LANE E (2026-07-02): multimodal + batch surfaces ride the provider router
# (gemini or local per EMBEDDINGS_* env). TaskType stays the shared interface.
from .provider_router import (
    generate_image_embedding,
    generate_multimodal_embedding,
    generate_embeddings_batch,
)
from .gemini_provider import TaskType

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
    # Gemini multimodal
    "generate_image_embedding",
    "generate_multimodal_embedding",
    "generate_embeddings_batch",
    "TaskType",
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