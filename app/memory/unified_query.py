# FILE: app/memory/unified_query.py
# DEPRECATED: Functionality moved to app/memory/domains/knowledge.py (KnowledgeStore).
# Kept for reference. The KnowledgeStore wraps rag_entries behind the
# DomainStore protocol and is registered with the MemoryRouter singleton.
"""
Unified RAG query across all backing stores.

Provides a single search surface that queries:
  - arch_code_chunks (architecture domain)
  - rag_entries (knowledge, context, decision, ingested domains)

This module is used by the MemoryRouter for cross-domain queries
and can also be called directly for lower-level access.

Semantic search (embedding-based) can be layered on top later.
This initial version uses keyword matching for both stores.
"""

import logging
from typing import Optional

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry
from app.memory.schemas_unified import MemoryResult

logger = logging.getLogger(__name__)


def search_rag_entries(
    text: str,
    project_id: str = "astra-core",
    domains: Optional[list[str]] = None,
    limit: int = 10,
    min_relevance: float = 0.0,
) -> list[MemoryResult]:
    """
    Search the rag_entries table by keyword matching.

    Args:
        text: Search query
        project_id: Project scope
        domains: Filter to specific domains (None = all)
        limit: Max results
        min_relevance: Minimum relevance score (0.0 to 1.0)

    Returns:
        List of MemoryResult sorted by relevance descending.
    """
    db = get_db_session()
    try:
        return _search(db, text, project_id, domains, limit, min_relevance)
    finally:
        db.close()


def _search(
    db: Session,
    text: str,
    project_id: str,
    domains: Optional[list[str]],
    limit: int,
    min_relevance: float,
) -> list[MemoryResult]:
    """Run keyword search against rag_entries."""
    keywords = _extract_keywords(text)
    if not keywords:
        return []

    # Base filters
    filters = [
        RAGEntry.status == "ACTIVE",
        RAGEntry.project_id == project_id,
    ]

    if domains:
        filters.append(RAGEntry.domain.in_(domains))

    # Keyword match on chunk_text and file_path
    kw_conditions = []
    for kw in keywords:
        pattern = f"%{kw}%"
        kw_conditions.append(
            or_(
                RAGEntry.chunk_text.ilike(pattern),
                RAGEntry.file_path.ilike(pattern),
            )
        )

    query = db.query(RAGEntry).filter(
        and_(*filters, or_(*kw_conditions))
    )

    candidates = query.limit(limit * 3).all()

    # Score and filter
    scored = []
    for entry in candidates:
        score = _score_entry(entry, keywords)
        if score >= min_relevance:
            scored.append((entry, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [_entry_to_result(entry, score) for entry, score in scored[:limit]]


def count_rag_entries(
    project_id: str = "astra-core",
    domain: Optional[str] = None,
) -> int:
    """Count active rag_entries, optionally filtered by domain."""
    db = get_db_session()
    try:
        query = db.query(func.count(RAGEntry.id)).filter(
            RAGEntry.status == "ACTIVE",
            RAGEntry.project_id == project_id,
        )
        if domain:
            query = query.filter(RAGEntry.domain == domain)
        return query.scalar() or 0
    finally:
        db.close()


# =========================================================================
# Helpers
# =========================================================================

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from query text."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "not", "with", "from", "by",
        "it", "this", "that", "how", "what", "where", "when", "who",
        "does", "do", "can", "will", "about", "me", "my", "show",
    }
    words = text.lower().split()
    return [w for w in words if len(w) >= 2 and w not in stop_words]


def _score_entry(entry: RAGEntry, keywords: list[str]) -> float:
    """Score a rag_entry by keyword match ratio."""
    if not keywords:
        return 0.0

    searchable = " ".join(filter(None, [
        (entry.chunk_text or "").lower(),
        (entry.file_path or "").lower(),
    ]))

    matches = sum(1 for kw in keywords if kw in searchable)
    return matches / len(keywords)


def _entry_to_result(entry: RAGEntry, score: float) -> MemoryResult:
    """Convert a RAGEntry to a normalised MemoryResult."""
    return MemoryResult(
        id=entry.id,
        domain=entry.domain,
        content=entry.chunk_text,
        project_id=entry.project_id,
        relevance=score,
        file_path=entry.file_path,
        source_table="rag_entries",
        status=entry.status,
        metadata={
            "ingest_source": entry.ingest_source,
            "package_role": entry.package_role,
            "source_monolith": entry.source_monolith,
            "refactor_job_id": entry.refactor_job_id,
        },
    )
