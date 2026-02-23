# FILE: app/memory/domains/architecture.py
"""
Architecture domain store.

Wraps the existing arch_code_chunks table and lifecycle operations
(app/rag/lifecycle.py) behind the DomainStore interface.

This store is READ-HEAVY. Architecture entries come from the archmap
scanner — you don't manually store() into this domain. The store()
method is a no-op that raises NotImplementedError.

Query strategy:
    Uses keyword matching on chunk_name, qualified_name, signature,
    docstring, descriptor, and file_path. Relevance is scored by
    how many fields match.

    Embedding-based semantic search can be added later by importing
    the embedding service — this initial version uses text matching
    so we don't add a dependency on the embedding pipeline.
"""

import logging
from typing import Optional

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.rag.models import ArchCodeChunk
from app.rag.lifecycle import (
    quarantine_file,
    purge_quarantined as lifecycle_purge,
)
from app.memory.schemas_unified import (
    MemoryResult,
    StoreRequest,
    DomainStats,
)

logger = logging.getLogger(__name__)


class ArchitectureStore:
    """
    DomainStore implementation for architecture memory.

    Searches arch_code_chunks for code signatures, docstrings,
    descriptors, and file paths. Returns normalised MemoryResult objects.
    """

    @property
    def domain_name(self) -> str:
        return "architecture"

    # -----------------------------------------------------------------
    # Query
    # -----------------------------------------------------------------

    def query(
        self,
        text: str,
        project_id: str = "astra-core",
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """
        Search architecture entries by keyword matching.

        Splits the query text into keywords and searches across
        chunk_name, qualified_name, signature, docstring, descriptor,
        and file_path. Results are scored by match count.
        """
        db = get_db_session()
        try:
            return self._search(db, text, limit, min_relevance)
        finally:
            db.close()

    def _search(
        self,
        db: Session,
        text: str,
        limit: int,
        min_relevance: float,
    ) -> list[MemoryResult]:
        """Run the actual search against arch_code_chunks."""
        keywords = _extract_keywords(text)
        if not keywords:
            return []

        # Base filter: active entries only
        base_filter = ArchCodeChunk.status == "active"

        # Build keyword match conditions across searchable fields
        conditions = []
        for kw in keywords:
            pattern = f"%{kw}%"
            conditions.append(
                or_(
                    ArchCodeChunk.chunk_name.ilike(pattern),
                    ArchCodeChunk.qualified_name.ilike(pattern),
                    ArchCodeChunk.signature.ilike(pattern),
                    ArchCodeChunk.docstring.ilike(pattern),
                    ArchCodeChunk.descriptor.ilike(pattern),
                    ArchCodeChunk.file_path.ilike(pattern),
                )
            )

        # Require at least one keyword to match
        query = db.query(ArchCodeChunk).filter(
            and_(base_filter, or_(*conditions))
        )

        # Fetch candidates (overfetch for scoring)
        candidates = query.limit(limit * 3).all()

        # Score each candidate by keyword match count
        scored = []
        for chunk in candidates:
            score = _score_chunk(chunk, keywords)
            if score >= min_relevance:
                scored.append((chunk, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Convert to MemoryResult
        results = []
        for chunk, score in scored[:limit]:
            results.append(_chunk_to_result(chunk, score))

        return results

    # -----------------------------------------------------------------
    # Store (not supported — entries come from archmap scanner)
    # -----------------------------------------------------------------

    def store(self, request: StoreRequest) -> int:
        raise NotImplementedError(
            "Architecture entries are created by the archmap scanner. "
            "Use the scan pipeline to add architecture data."
        )

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def quarantine(
        self,
        file_path: str,
        project_id: str = "astra-core",
    ) -> int:
        """Quarantine all chunks for a file path."""
        db = get_db_session()
        try:
            # lifecycle.quarantine_file needs a refactor_job_id
            # Use a synthetic one for manual quarantine
            count = quarantine_file(
                db=db,
                file_path=file_path,
                refactor_job_id=f"manual-quarantine-{file_path}",
            )
            return count
        finally:
            db.close()

    def purge_quarantined(self, project_id: str = "astra-core") -> int:
        """Purge all quarantined architecture entries."""
        db = get_db_session()
        try:
            return lifecycle_purge(db=db)
        finally:
            db.close()

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def count(self, project_id: str = "astra-core") -> int:
        """Count active architecture chunks."""
        db = get_db_session()
        try:
            return db.query(func.count(ArchCodeChunk.id)).filter(
                ArchCodeChunk.status == "active"
            ).scalar() or 0
        finally:
            db.close()

    def get_stats(self, project_id: str = "astra-core") -> DomainStats:
        """Get architecture domain statistics."""
        db = get_db_session()
        try:
            total = db.query(func.count(ArchCodeChunk.id)).scalar() or 0
            active = db.query(func.count(ArchCodeChunk.id)).filter(
                ArchCodeChunk.status == "active"
            ).scalar() or 0
            quarantined = db.query(func.count(ArchCodeChunk.id)).filter(
                ArchCodeChunk.status == "quarantined"
            ).scalar() or 0
            embedded = db.query(func.count(ArchCodeChunk.id)).filter(
                ArchCodeChunk.embedded == True  # noqa: E712
            ).scalar() or 0

            return DomainStats(
                domain="architecture",
                total_entries=total,
                active_entries=active,
                quarantined_entries=quarantined,
                embedded_entries=embedded,
            )
        finally:
            db.close()


# =========================================================================
# Helpers (private)
# =========================================================================

def _extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from query text.

    Strips common stop words and short tokens.
    """
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "not", "with", "from", "by",
        "it", "this", "that", "how", "what", "where", "when", "who",
        "does", "do", "can", "will", "about", "me", "my", "show",
    }
    words = text.lower().split()
    return [w for w in words if len(w) >= 2 and w not in stop_words]


def _score_chunk(chunk: ArchCodeChunk, keywords: list[str]) -> float:
    """
    Score a chunk by how many keywords match across its fields.

    Returns a float between 0.0 and 1.0.
    """
    if not keywords:
        return 0.0

    searchable = " ".join(filter(None, [
        (chunk.chunk_name or "").lower(),
        (chunk.qualified_name or "").lower(),
        (chunk.signature or "").lower(),
        (chunk.docstring or "").lower(),
        (chunk.descriptor or "").lower(),
        (chunk.file_path or "").lower(),
    ]))

    matches = sum(1 for kw in keywords if kw in searchable)
    return matches / len(keywords)


def _chunk_to_result(chunk: ArchCodeChunk, score: float) -> MemoryResult:
    """Convert an ArchCodeChunk to a normalised MemoryResult."""
    # Build content from the most useful fields
    parts = []
    if chunk.qualified_name:
        parts.append(chunk.qualified_name)
    elif chunk.chunk_name:
        parts.append(chunk.chunk_name)
    if chunk.signature:
        parts.append(chunk.signature)
    if chunk.docstring:
        parts.append(chunk.docstring[:200])
    if chunk.descriptor:
        parts.append(chunk.descriptor[:300])

    content = "\n".join(parts) if parts else chunk.chunk_name or ""

    return MemoryResult(
        id=chunk.id,
        domain="architecture",
        content=content,
        project_id="astra-core",
        relevance=score,
        file_path=chunk.file_path,
        source_table="arch_code_chunks",
        status=chunk.status or "active",
        metadata={
            "chunk_type": chunk.chunk_type,
            "chunk_name": chunk.chunk_name,
            "qualified_name": chunk.qualified_name,
            "symbol_type": chunk.chunk_type,
            "package_role": chunk.package_role,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "embedded": chunk.embedded,
        },
    )
