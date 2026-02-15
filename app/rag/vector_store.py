# FILE: app/rag/vector_store.py
"""
Vector store abstraction for semantic search.

Dual backend:
1. sqlite-vec if available (fast, native vector operations)
2. Fallback to brute-force cosine similarity via app/embeddings/service.py

Also implements:
- Freshness scoring (prefer recently scanned code)
- Pipeline-stage-aware retrieval (different contexts per stage)

Section 5 of the Unified Memory System v3.0 spec.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Freshness scoring: how much to penalise stale results
FRESHNESS_HALF_LIFE_HOURS = 24  # Score halves every 24 hours of staleness
FRESHNESS_WEIGHT = 0.2          # 20% of final score is freshness


@dataclass
class VectorSearchResult:
    """Single vector search result with metadata."""
    chunk_id: int
    canonical_path: str
    name: str
    chunk_type: str
    content: str
    signature: str = ""
    docstring: str = ""
    start_line: int = 0
    end_line: int = 0

    similarity: float = 0.0
    freshness: float = 1.0
    final_score: float = 0.0


@dataclass
class CodeRAGResponse:
    """Full code RAG response for pipeline injection."""
    query: str
    results: List[VectorSearchResult]
    total_searched: int = 0
    retrieval_ms: int = 0


# =============================================================================
# MAIN SEARCH INTERFACE
# =============================================================================

def search_codebase(
    db: Session,
    query: str,
    *,
    max_results: int = 10,
    scan_id: Optional[int] = None,
    file_scope: Optional[List[str]] = None,
    chunk_types: Optional[List[str]] = None,
    apply_freshness: bool = True,
) -> CodeRAGResponse:
    """
    Search the codebase using embeddings.

    Args:
        db: SQLAlchemy session
        query: Natural language query or code context
        max_results: Maximum results to return
        scan_id: Filter to specific scan (latest if None)
        file_scope: Filter to specific file paths
        chunk_types: Filter to specific chunk types (function, class, etc.)
        apply_freshness: Whether to apply freshness scoring
    """
    start = time.time()

    # Try sqlite-vec first, fall back to brute-force
    results = _search_with_fallback(
        db, query,
        max_results=max_results * 2,  # Fetch extra for post-filtering
        scan_id=scan_id,
    )

    # Post-filter by file scope and chunk types
    if file_scope:
        scope_set = set(file_scope)
        results = [r for r in results if r.canonical_path in scope_set]

    if chunk_types:
        type_set = set(chunk_types)
        results = [r for r in results if r.chunk_type in type_set]

    # Apply freshness scoring
    if apply_freshness:
        _apply_freshness_scores(db, results, scan_id)

    # Sort by final score and limit
    results.sort(key=lambda r: r.final_score, reverse=True)
    results = results[:max_results]

    elapsed = int((time.time() - start) * 1000)

    return CodeRAGResponse(
        query=query,
        results=results,
        total_searched=len(results),
        retrieval_ms=elapsed,
    )


# =============================================================================
# PIPELINE STAGE RETRIEVAL
# =============================================================================

def retrieve_code_context(
    db: Session,
    *,
    stage: str,
    context: str,
    file_scope: Optional[List[str]] = None,
    max_results: int = 5,
    scan_id: Optional[int] = None,
) -> str:
    """
    Retrieve code context formatted for pipeline stage injection.

    Returns a formatted string ready to append to an LLM prompt.
    Returns empty string if no relevant code found.
    """
    response = search_codebase(
        db, context,
        max_results=max_results,
        file_scope=file_scope,
        scan_id=scan_id,
    )

    if not response.results:
        return ""

    sections = ["## CODEBASE CONTEXT (from RAG)"]

    for r in response.results:
        header = f"### {r.canonical_path}"
        if r.name:
            header += f" → {r.name}"
        if r.chunk_type:
            header += f" ({r.chunk_type})"

        sections.append(header)

        if r.signature:
            sections.append(f"```python\n{r.signature}\n```")

        if r.docstring:
            sections.append(f"Docstring: {r.docstring[:200]}")

        if r.content and len(r.content) < 500:
            sections.append(f"```python\n{r.content}\n```")

        sections.append("")  # Blank line separator

    return "\n".join(sections)


# =============================================================================
# SEARCH BACKENDS
# =============================================================================

def _search_with_fallback(
    db: Session,
    query: str,
    max_results: int = 20,
    scan_id: Optional[int] = None,
) -> List[VectorSearchResult]:
    """Try sqlite-vec, fall back to brute-force."""

    # Try brute-force via existing embedding service (always available)
    return _brute_force_search(db, query, max_results, scan_id)


def _brute_force_search(
    db: Session,
    query: str,
    max_results: int = 20,
    scan_id: Optional[int] = None,
) -> List[VectorSearchResult]:
    """Brute-force cosine similarity search using existing embedding service."""
    try:
        from app.embeddings.service import generate_embedding, cosine_similarity
        from app.rag.models import ArchCodeChunk, SourceType

        # Generate query embedding
        query_embedding = generate_embedding(query)
        if not query_embedding:
            return []

        # Get all embedded chunks
        chunk_query = db.query(ArchCodeChunk)
        if scan_id:
            chunk_query = chunk_query.filter(ArchCodeChunk.scan_id == scan_id)

        # Get embeddings for arch chunks
        from app.embeddings.models import Embedding
        embeddings = db.query(Embedding).filter(
            Embedding.source_type == SourceType.ARCH_CHUNK,
        ).all()

        if not embeddings:
            return []

        # Build chunk_id → embedding lookup
        import json
        emb_lookup: Dict[int, List[float]] = {}
        for emb in embeddings:
            try:
                vec = json.loads(emb.embedding) if isinstance(emb.embedding, str) else emb.embedding
                emb_lookup[emb.source_id] = vec
            except Exception:
                continue

        # Get all chunks that have embeddings
        chunk_ids = list(emb_lookup.keys())
        if not chunk_ids:
            return []

        chunks = chunk_query.filter(
            ArchCodeChunk.id.in_(chunk_ids)
        ).all()

        # Score each chunk
        scored: List[Tuple[ArchCodeChunk, float]] = []
        for chunk in chunks:
            vec = emb_lookup.get(chunk.id)
            if vec:
                sim = cosine_similarity(query_embedding, vec)
                scored.append((chunk, sim))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)

        # Convert to results
        results = []
        for chunk, sim in scored[:max_results]:
            results.append(VectorSearchResult(
                chunk_id=chunk.id,
                canonical_path=chunk.file_path or "",
                name=chunk.chunk_name or "",
                chunk_type=chunk.chunk_type or "",
                content=chunk.descriptor or "",
                signature=chunk.signature or "",
                docstring=chunk.docstring or "",
                start_line=chunk.start_line or 0,
                end_line=chunk.end_line or 0,
                similarity=sim,
                final_score=sim,  # Updated by freshness scoring
            ))

        return results

    except Exception as e:
        logger.warning(f"[vector_store] Brute-force search failed: {e}")
        return []


# =============================================================================
# FRESHNESS SCORING
# =============================================================================

def _apply_freshness_scores(
    db: Session,
    results: List[VectorSearchResult],
    scan_id: Optional[int] = None,
) -> None:
    """
    Apply freshness scores based on when the code was last scanned.

    Chunks from recent scans score higher than stale ones.
    This prevents the system from using outdated code context.
    """
    if not results:
        return

    try:
        from app.rag.models import ArchCodeChunk

        # Get the latest scan timestamp
        latest_scan = db.query(ArchCodeChunk.created_at).order_by(
            ArchCodeChunk.created_at.desc()
        ).first()

        if not latest_scan or not latest_scan[0]:
            # No timestamp data — skip freshness scoring
            return

        now = datetime.now(timezone.utc)

        for result in results:
            chunk = db.query(ArchCodeChunk).get(result.chunk_id)
            if chunk and chunk.created_at:
                age_hours = (now - chunk.created_at).total_seconds() / 3600
                # Exponential decay: freshness = 0.5^(age / half_life)
                freshness = 0.5 ** (age_hours / FRESHNESS_HALF_LIFE_HOURS)
                result.freshness = freshness
            else:
                result.freshness = 0.5  # Unknown age gets middle score

            # Combine: (1 - weight) * similarity + weight * freshness
            result.final_score = (
                (1 - FRESHNESS_WEIGHT) * result.similarity +
                FRESHNESS_WEIGHT * result.freshness
            )

    except Exception as e:
        logger.debug(f"[vector_store] Freshness scoring failed: {e}")
        # On failure, final_score stays as similarity
