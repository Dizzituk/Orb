# Purpose: schemas
# Called-by: app.shared_context, app.shared_context.builder, app.shared_context.router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from pydantic import BaseModel, Field
from typing import List


class SharedContextItem(BaseModel):
    """Single retrieved context item with provenance."""
    source_type: str = Field(..., description="Type of source: 'note', 'message', or 'file'")
    source_id: int = Field(..., description="ID in source table")
    chunk_index: int = Field(..., description="Which chunk of the source")
    similarity: float = Field(..., description="Cosine similarity (0-1)")
    content: str = Field(..., description="The actual text chunk")
    provenance: str = Field(..., description="Human-readable provenance string (e.g., 'note:123 chunk=0 sim=0.87')")
    pointer: str = Field(..., description="Deep-link pointer (e.g., '/memory/note/123')")


class SharedContextPackage(BaseModel):
    """Complete context package ready for injection."""
    query: str = Field(..., description="Original query")
    items: List[SharedContextItem] = Field(default_factory=list, description="Retrieved context items")
    formatted_block: str = Field(..., description="Ready-to-inject text block")
    truncated: bool = Field(default=False, description="True if budget forced truncation")
    budget_chars: int = Field(..., description="Character budget used")
    # Phase 7 audit (2026-05-02): builder.py also reports retrieval-stage
    # counts. Adding here so the schema matches what's actually produced;
    # downstream consumers can ignore them if they don't care.
    total_retrieved: int = Field(default=0, description="Total items returned by retrieval before selection")
    total_searched: int = Field(default=0, description="Total items the embedding search scanned")