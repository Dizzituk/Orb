# FILE: app/embeddings/schemas.py
"""
Pydantic schemas for embedding endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict


class EmbeddingOut(BaseModel):
    """Output schema for embedding records."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    project_id: int
    source_type: str
    source_id: int
    chunk_index: int
    content: str
    created_at: datetime


class SearchRequest(BaseModel):
    """Request schema for semantic search."""
    project_id: int
    query: str
    top_k: int = 5
    source_types: Optional[List[str]] = None  # Filter by type: ["note", "message", "file"]


class SearchResult(BaseModel):
    """Single search result with similarity score."""
    source_type: str
    source_id: int
    chunk_index: int
    content: str
    similarity: float  # Cosine similarity (0-1)
    

class SearchResponse(BaseModel):
    """Response schema for semantic search."""
    query: str
    results: List[SearchResult]
    total_searched: int


class IndexRequest(BaseModel):
    """Request schema for indexing."""
    project_id: int
    source_types: Optional[List[str]] = None  # Which types to index


class IndexResponse(BaseModel):
    """Response schema for indexing operations."""
    project_id: int
    indexed_count: int
    skipped_count: int
    error_count: int
    details: Optional[str] = None