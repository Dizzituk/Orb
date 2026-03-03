# FILE: app/memory/memory_router.py
# DEPRECATED: Use app/memory/router.py instead.
# This file is kept for reference only. The protocol-based MemoryRouter
# in router.py with domain store registration replaced this implementation.
"""
MemoryRouter — Programmatic interface to the unified memory system.

Single entry point for all memory operations:
  - query(): Search across all domains (architecture, knowledge, context, etc.)
  - store(): Add entries to the appropriate domain store
  - quarantine_status(): Get lifecycle counts
  - resolve_redirects(): Handle queries for refactored files

This class unifies the arch_code_chunks (architecture) and rag_entries
(all other domains) backing stores behind a single query interface.

Domain stores handle domain-specific logic; MemoryRouter orchestrates.

Usage:
    from app.memory.memory_router import MemoryRouter

    router = MemoryRouter()
    results = router.query("how does the refactor loop work")
    router.store(domain="knowledge", content="...", file_path="...")
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry
from app.memory.schemas_unified import (
    MemoryResult,
    StoreRequest,
    QueryRequest,
    DomainStats,
)
from app.memory.unified_query import search_rag_entries, count_rag_entries

logger = logging.getLogger(__name__)


class MemoryRouter:
    """
    Unified programmatic interface to ASTRA's memory system.

    Bridges across backing stores:
      - arch_code_chunks → architecture domain
      - rag_entries → knowledge, context, decision, ingested, redirect domains
    """

    def __init__(self, project_id: str = "astra-core"):
        self.project_id = project_id

    # ─── Query ──────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        domains: Optional[list[str]] = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """
        Search across all memory domains.

        If a result is a redirect entry (domain='redirect'), it is
        resolved into a pointer to the new location instead of
        returning stale content.

        Args:
            text: Natural language query
            domains: Filter to specific domains (None = all)
            limit: Max results
            min_relevance: Minimum relevance score

        Returns:
            List of MemoryResult, sorted by relevance descending.
        """
        results = []

        # Search rag_entries (knowledge, context, decision, ingested)
        rag_results = search_rag_entries(
            text=text,
            project_id=self.project_id,
            domains=domains,
            limit=limit,
            min_relevance=min_relevance,
        )
        results.extend(rag_results)

        # Search architecture domain if requested or no filter
        if domains is None or "architecture" in domains:
            arch_results = self._search_architecture(text, limit)
            results.extend(arch_results)

        # Resolve redirects
        results = self._resolve_redirects(results)

        # Sort by relevance, trim to limit
        results.sort(key=lambda r: r.relevance, reverse=True)
        return results[:limit]

    # ─── Store ──────────────────────────────────────────────────────

    def store(
        self,
        domain: str,
        content: str,
        file_path: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> int:
        """
        Store a new memory entry.

        Args:
            domain: Target domain (knowledge, context, decision, ingested)
            content: The text content to store
            file_path: Optional file path association
            metadata: Optional metadata dict

        Returns:
            ID of the created entry.
        """
        meta = metadata or {}
        db = get_db_session()
        try:
            entry = RAGEntry(
                project_id=self.project_id,
                domain=domain,
                chunk_text=content,
                file_path=file_path,
                status="ACTIVE",
                ingest_source=meta.get("ingest_source", "manual"),
                source_monolith=meta.get("source_monolith"),
                refactor_job_id=meta.get("refactor_job_id"),
                package_role=meta.get("package_role"),
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)
            entry_id = entry.id
            return entry_id
        finally:
            db.close()

    # ─── Stats ──────────────────────────────────────────────────────

    def domain_stats(self) -> list[DomainStats]:
        """Get entry counts per domain."""
        db = get_db_session()
        try:
            from sqlalchemy import func

            rows = db.query(
                RAGEntry.domain,
                RAGEntry.status,
                func.count(RAGEntry.id),
            ).filter(
                RAGEntry.project_id == self.project_id,
            ).group_by(
                RAGEntry.domain, RAGEntry.status,
            ).all()

            stats_map: dict[str, DomainStats] = {}
            for domain, status, count in rows:
                if domain not in stats_map:
                    stats_map[domain] = DomainStats(domain=domain)
                ds = stats_map[domain]
                ds.total_entries += count
                if status == "ACTIVE":
                    ds.active_entries = count
                elif status == "QUARANTINED":
                    ds.quarantined_entries = count

            return list(stats_map.values())
        finally:
            db.close()

    # ─── Redirect Resolution ────────────────────────────────────────

    def _resolve_redirects(
        self,
        results: list[MemoryResult],
    ) -> list[MemoryResult]:
        """
        Replace redirect entries with informative pointers.

        When a file has been refactored, its old RAG entries get
        quarantined and a redirect entry is created. If a query
        matches the redirect, we rewrite the content to point
        the caller to the new location.
        """
        resolved = []
        for r in results:
            if r.domain == "redirect":
                # Rewrite as a helpful pointer
                r.content = (
                    f"[REDIRECT] {r.content} "
                    f"Search for the new path to find current content."
                )
                r.metadata["is_redirect"] = True
            resolved.append(r)
        return resolved

    # ─── Architecture Search ────────────────────────────────────────

    def _search_architecture(
        self,
        text: str,
        limit: int,
    ) -> list[MemoryResult]:
        """
        Search arch_code_chunks for architecture domain results.

        Uses keyword matching against symbol names and file paths.
        """
        from app.rag.models import ArchCodeChunk
        from sqlalchemy import or_

        db = get_db_session()
        try:
            keywords = text.lower().split()
            if not keywords:
                return []

            # v2.1: Cap keywords to prevent SQLite expression tree overflow
            if len(keywords) > 20:
                keywords = keywords[:20]

            conditions = []
            for kw in keywords:
                if len(kw) < 2:
                    continue
                pattern = f"%{kw}%"
                conditions.append(
                    or_(
                        ArchCodeChunk.symbol_name.ilike(pattern),
                        ArchCodeChunk.file_path.ilike(pattern),
                        ArchCodeChunk.chunk_text.ilike(pattern),
                    )
                )

            if not conditions:
                return []

            chunks = db.query(ArchCodeChunk).filter(
                ArchCodeChunk.status == "active",
                or_(*conditions),
            ).limit(limit).all()

            return [
                MemoryResult(
                    id=c.id,
                    domain="architecture",
                    content=c.chunk_text or "",
                    project_id=self.project_id,
                    relevance=0.5,  # Base score, can be refined later
                    file_path=c.file_path,
                    source_table="arch_code_chunks",
                    status=c.status,
                    metadata={
                        "symbol_name": c.symbol_name,
                        "symbol_type": getattr(c, "symbol_type", None),
                        "source_monolith": c.source_monolith,
                        "refactor_job_id": c.refactor_job_id,
                        "package_role": c.package_role,
                    },
                )
                for c in chunks
            ]
        finally:
            db.close()
