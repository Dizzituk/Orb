# FILE: app/memory/domains/knowledge.py
# Purpose: Domain Knowledge store (Spec Section 4.6).
# Called-by: app.memory.domains, app.memory.startup
# Depends-on: app.db, app.memory.rag_entries_model, app.memory.schemas_unified
# Last-renovated: 2026-06-11
"""
Domain Knowledge store (Spec Section 4.6).

Stores factual knowledge specific to ASTRA's application domains.
Long-term memory about the user's professional expertise and the
domains ASTRA operates in.

Examples:
  - PT programme templates, exercise preferences
  - Catering management workflows, supplier info
  - Content calendar structures, blog voice
  - Client lists, financial workflows

Each entry is tagged with:
  - domain: fitness, catering, content, development, finance, general
  - source: user_stated, researched, inferred
  - Inferred entries have lower retrieval priority

Backed by rag_entries table with domain='knowledge'.
Uses semantic search (keyword matching now, embeddings later).

Usage:
    store = KnowledgeStore()
    store.add("fitness", "Client X prefers 4-day upper/lower split", "user_stated")
    store.add("catering", "Standard portion: 200g protein per cover", "user_stated")

    results = store.query_domain("fitness", "training split")
"""

import logging
from typing import Optional

from sqlalchemy import and_, or_, func
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry
from app.memory.schemas_unified import MemoryResult, StoreRequest, DomainStats

logger = logging.getLogger(__name__)

DOMAIN = "knowledge"

# Valid knowledge subdomains (mirrors preference registry)
VALID_SUBDOMAINS = {
    "development", "content", "video", "fitness",
    "catering", "finance", "general",
}

# Source types with retrieval priority weights
SOURCE_PRIORITY = {
    "user_stated": 1.0,   # User explicitly told ASTRA
    "researched": 0.8,    # ASTRA researched and user confirmed
    "inferred": 0.5,      # Inferred from patterns — lower priority
}


class KnowledgeStore:
    """
    DomainStore for per-domain factual knowledge.

    Long-term factual memory that accumulates over time.
    Entries are never replaced wholesale — they accumulate,
    and the source tag determines retrieval priority.
    """

    @property
    def domain_name(self) -> str:
        return DOMAIN

    # -----------------------------------------------------------------
    # Add / query knowledge
    # -----------------------------------------------------------------

    def add(
        self,
        subdomain: str,
        content: str,
        source: str = "user_stated",
        project_id: str = "astra-core",
        file_path: Optional[str] = None,
    ) -> int:
        """
        Add a knowledge entry.

        Args:
            subdomain: Knowledge domain (fitness, catering, etc.)
            content: The factual content
            source: How this was acquired (user_stated, researched, inferred)
            project_id: Project scope
            file_path: Optional source file reference

        Returns:
            Entry ID.
        """
        if subdomain not in VALID_SUBDOMAINS:
            logger.warning(
                "[knowledge] Unknown subdomain '%s', using 'general'",
                subdomain,
            )
            subdomain = "general"

        if source not in SOURCE_PRIORITY:
            source = "user_stated"

        db = get_db_session()
        try:
            entry = RAGEntry(
                project_id=project_id,
                domain=DOMAIN,
                chunk_text=f"[{subdomain}] {content}",
                file_path=file_path,
                status="ACTIVE",
                ingest_source=source,
                package_role=subdomain,  # Store subdomain for filtering
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)

            logger.debug(
                "[knowledge] Added %s/%s: %s",
                subdomain, source, content[:60],
            )
            return entry.id
        finally:
            db.close()

    def query_domain(
        self,
        subdomain: str,
        text: str,
        project_id: str = "astra-core",
        limit: int = 10,
    ) -> list[MemoryResult]:
        """
        Search knowledge within a specific subdomain.

        Results are weighted by source priority:
        user_stated > researched > inferred.
        """
        db = get_db_session()
        try:
            keywords = _extract_keywords(text)
            base_filter = and_(
                RAGEntry.domain == DOMAIN,
                RAGEntry.project_id == project_id,
                RAGEntry.status == "ACTIVE",
                RAGEntry.package_role == subdomain,
            )

            if not keywords:
                entries = (
                    db.query(RAGEntry)
                    .filter(base_filter)
                    .order_by(RAGEntry.indexed_at.desc())
                    .limit(limit)
                    .all()
                )
                return [_entry_to_result(e, 0.3) for e in entries]

            conditions = []
            for kw in keywords:
                conditions.append(RAGEntry.chunk_text.ilike(f"%{kw}%"))

            entries = (
                db.query(RAGEntry)
                .filter(base_filter, or_(*conditions))
                .limit(limit * 2)
                .all()
            )

            scored = []
            for entry in entries:
                kw_score = _keyword_score(entry.chunk_text, keywords)
                source_weight = SOURCE_PRIORITY.get(entry.ingest_source, 0.5)
                combined = (kw_score * 0.7) + (source_weight * 0.3)
                scored.append((entry, combined))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [_entry_to_result(e, s) for e, s in scored[:limit]]
        finally:
            db.close()

    # -----------------------------------------------------------------
    # DomainStore Protocol: query
    # -----------------------------------------------------------------

    def query(
        self,
        text: str,
        project_id: str = "astra-core",
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """
        Search knowledge across ALL subdomains.

        For domain-specific queries, use query_domain() instead.
        """
        db = get_db_session()
        try:
            keywords = _extract_keywords(text)

            base_filter = and_(
                RAGEntry.domain == DOMAIN,
                RAGEntry.project_id == project_id,
                RAGEntry.status == "ACTIVE",
            )

            if not keywords:
                entries = (
                    db.query(RAGEntry)
                    .filter(base_filter)
                    .order_by(RAGEntry.indexed_at.desc())
                    .limit(limit)
                    .all()
                )
                return [
                    _entry_to_result(e, 0.3) for e in entries
                    if 0.3 >= min_relevance
                ]

            conditions = []
            for kw in keywords:
                conditions.append(RAGEntry.chunk_text.ilike(f"%{kw}%"))

            entries = (
                db.query(RAGEntry)
                .filter(base_filter, or_(*conditions))
                .limit(limit * 3)
                .all()
            )

            scored = []
            for entry in entries:
                kw_score = _keyword_score(entry.chunk_text, keywords)
                source_weight = SOURCE_PRIORITY.get(entry.ingest_source, 0.5)
                combined = (kw_score * 0.7) + (source_weight * 0.3)
                if combined >= min_relevance:
                    scored.append((entry, combined))

            scored.sort(key=lambda x: x[1], reverse=True)
            return [_entry_to_result(e, s) for e, s in scored[:limit]]
        finally:
            db.close()

    # -----------------------------------------------------------------
    # DomainStore Protocol: store
    # -----------------------------------------------------------------

    def store(self, request: StoreRequest) -> int:
        """Store a knowledge entry via the generic interface."""
        subdomain = request.metadata.get("subdomain", "general")
        source = request.metadata.get("source", "user_stated")
        return self.add(
            subdomain=subdomain,
            content=request.content,
            source=source,
            project_id=request.project_id,
            file_path=request.file_path,
        )

    # -----------------------------------------------------------------
    # DomainStore Protocol: quarantine / count / stats
    # -----------------------------------------------------------------

    def quarantine(self, file_path: str, project_id: str = "astra-core") -> int:
        """Quarantine knowledge entries linked to a file."""
        db = get_db_session()
        try:
            entries = (
                db.query(RAGEntry)
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.file_path == file_path,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                )
                .all()
            )
            for e in entries:
                e.status = "QUARANTINED"
                e.quarantined_at = __import__("datetime").datetime.utcnow()
            if entries:
                db.commit()
            return len(entries)
        finally:
            db.close()

    def count(self, project_id: str = "astra-core") -> int:
        db = get_db_session()
        try:
            return (
                db.query(func.count(RAGEntry.id))
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                )
                .scalar() or 0
            )
        finally:
            db.close()

    def get_stats(self, project_id: str = "astra-core") -> DomainStats:
        db = get_db_session()
        try:
            total = (
                db.query(func.count(RAGEntry.id))
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                )
                .scalar() or 0
            )
            active = self.count(project_id)
            quarantined = (
                db.query(func.count(RAGEntry.id))
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "QUARANTINED",
                )
                .scalar() or 0
            )
            return DomainStats(
                domain=DOMAIN,
                total_entries=total,
                active_entries=active,
                quarantined_entries=quarantined,
            )
        finally:
            db.close()

    # -----------------------------------------------------------------
    # Subdomain listing
    # -----------------------------------------------------------------

    def get_subdomain_counts(
        self, project_id: str = "astra-core",
    ) -> dict[str, int]:
        """Get entry counts per subdomain."""
        db = get_db_session()
        try:
            rows = (
                db.query(RAGEntry.package_role, func.count(RAGEntry.id))
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                )
                .group_by(RAGEntry.package_role)
                .all()
            )
            return {role or "unknown": count for role, count in rows}
        finally:
            db.close()


# =========================================================================
# Helpers
# =========================================================================

def _extract_keywords(text: str) -> list[str]:
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "not", "with", "from", "by",
        "it", "this", "that", "how", "what", "where", "when", "who",
        "does", "do", "can", "will", "about", "me", "my", "show",
    }
    words = text.lower().split()
    return [w for w in words if len(w) >= 2 and w not in stop_words]


def _keyword_score(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lower = text.lower()
    matches = sum(1 for kw in keywords if kw in lower)
    return matches / len(keywords)


def _entry_to_result(entry: RAGEntry, score: float) -> MemoryResult:
    subdomain = entry.package_role or "general"
    return MemoryResult(
        id=entry.id,
        domain=DOMAIN,
        content=entry.chunk_text,
        project_id=entry.project_id,
        relevance=score,
        file_path=entry.file_path,
        source_table="rag_entries",
        status=entry.status,
        metadata={
            "subdomain": subdomain,
            "source": entry.ingest_source,
            "source_priority": SOURCE_PRIORITY.get(entry.ingest_source, 0.5),
        },
    )
