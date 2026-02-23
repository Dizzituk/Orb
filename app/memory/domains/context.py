# FILE: app/memory/domains/context.py
"""
Conversation Context domain store (Spec Section 4.4).

Ephemeral working memory for the current task or session.
Solves the pronoun problem: when the user says "write that up"
or "send it to the pipeline," the system knows what "that" and
"it" refer to.

NOT a chat log. Stores extracted facts and decisions:
  "User decided to refactor conduct_policy first"
  "Current task: building the RAG memory layer"
  "Working file: app/memory/domains/context.py"

Implementation:
  - Lightweight key-value entries in rag_entries (domain='context')
  - Each entry has a TTL (default 24 hours)
  - Recency-based retrieval, no embeddings needed
  - Garbage collection for expired entries
  - No quarantine lifecycle — entries simply expire

Usage:
    store = ContextStore()
    store.set_fact("current_task", "Building memory layer")
    store.set_fact("working_file", "app/memory/domains/context.py")
    store.set_decision("Refactor conduct_policy first")

    facts = store.get_recent(limit=10)
    task = store.get_fact("current_task")
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.memory.rag_entries_model import RAGEntry
from app.memory.schemas_unified import MemoryResult, StoreRequest, DomainStats

logger = logging.getLogger(__name__)

DOMAIN = "context"
DEFAULT_TTL_HOURS = 24


class ContextStore:
    """
    DomainStore for ephemeral conversation context.

    Entries are key-value facts and decisions extracted from
    conversation exchanges. They expire after TTL and are
    garbage collected automatically on read operations.
    """

    @property
    def domain_name(self) -> str:
        return DOMAIN

    # -----------------------------------------------------------------
    # Set / Get facts
    # -----------------------------------------------------------------

    def set_fact(
        self,
        key: str,
        value: str,
        project_id: str = "astra-core",
        ttl_hours: float = DEFAULT_TTL_HOURS,
    ) -> int:
        """
        Store or update a context fact.

        If a fact with this key already exists, it is replaced
        (old entry marked PURGED, new one created).

        Args:
            key: Fact identifier (e.g. "current_task", "working_file")
            value: The fact content
            project_id: Project scope
            ttl_hours: Time-to-live in hours

        Returns:
            Entry ID of the new fact.
        """
        db = get_db_session()
        try:
            # Expire old entry with same key
            self._expire_key(db, key, project_id)

            # Build metadata with TTL
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            metadata = json.dumps({
                "context_key": key,
                "ttl_hours": ttl_hours,
                "expires_at": expires_at.isoformat(),
                "entry_type": "fact",
            })

            entry = RAGEntry(
                project_id=project_id,
                domain=DOMAIN,
                chunk_text=f"[{key}] {value}",
                status="ACTIVE",
                ingest_source="context_extract",
                package_role=metadata,  # Reusing field for metadata
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)

            logger.debug("[context] Set fact: %s = %s", key, value[:80])
            return entry.id
        finally:
            db.close()

    def set_decision(
        self,
        decision: str,
        project_id: str = "astra-core",
        ttl_hours: float = DEFAULT_TTL_HOURS,
    ) -> int:
        """
        Store a session decision.

        Decisions don't have keys — they accumulate chronologically.

        Args:
            decision: The decision text
            project_id: Project scope
            ttl_hours: Time-to-live in hours

        Returns:
            Entry ID.
        """
        db = get_db_session()
        try:
            expires_at = datetime.utcnow() + timedelta(hours=ttl_hours)
            metadata = json.dumps({
                "ttl_hours": ttl_hours,
                "expires_at": expires_at.isoformat(),
                "entry_type": "decision",
            })

            entry = RAGEntry(
                project_id=project_id,
                domain=DOMAIN,
                chunk_text=f"[DECISION] {decision}",
                status="ACTIVE",
                ingest_source="context_extract",
                package_role=metadata,
            )
            db.add(entry)
            db.commit()
            db.refresh(entry)

            logger.debug("[context] Set decision: %s", decision[:80])
            return entry.id
        finally:
            db.close()

    def get_fact(
        self,
        key: str,
        project_id: str = "astra-core",
    ) -> Optional[str]:
        """
        Get the latest value for a context fact key.

        Returns None if not found or expired.
        """
        db = get_db_session()
        try:
            self._gc(db, project_id)

            entry = (
                db.query(RAGEntry)
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                    RAGEntry.chunk_text.like(f"[{key}]%"),
                )
                .order_by(RAGEntry.indexed_at.desc())
                .first()
            )

            if not entry:
                return None

            # Strip the key prefix
            prefix = f"[{key}] "
            text = entry.chunk_text
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        finally:
            db.close()

    def get_recent(
        self,
        limit: int = 20,
        project_id: str = "astra-core",
    ) -> list[dict]:
        """
        Get recent context entries, newest first.

        Returns list of dicts with id, text, type, timestamp.
        """
        db = get_db_session()
        try:
            self._gc(db, project_id)

            entries = (
                db.query(RAGEntry)
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                )
                .order_by(RAGEntry.indexed_at.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": e.id,
                    "text": e.chunk_text,
                    "type": _parse_entry_type(e),
                    "timestamp": e.indexed_at.isoformat() if e.indexed_at else None,
                }
                for e in entries
            ]
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
        Search context entries by keyword matching.

        Context uses recency-weighted scoring — newer entries
        score higher than older ones, even with fewer keyword matches.
        """
        db = get_db_session()
        try:
            self._gc(db, project_id)
            return self._search(db, text, project_id, limit, min_relevance)
        finally:
            db.close()

    def _search(
        self,
        db: Session,
        text: str,
        project_id: str,
        limit: int,
        min_relevance: float,
    ) -> list[MemoryResult]:
        """Keyword search with recency weighting."""
        keywords = _extract_keywords(text)
        if not keywords:
            # No keywords — return most recent entries
            entries = (
                db.query(RAGEntry)
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                )
                .order_by(RAGEntry.indexed_at.desc())
                .limit(limit)
                .all()
            )
            return [
                _entry_to_result(e, 0.3)  # Base recency score
                for e in entries
            ]

        from sqlalchemy import or_

        conditions = []
        for kw in keywords:
            conditions.append(RAGEntry.chunk_text.ilike(f"%{kw}%"))

        entries = (
            db.query(RAGEntry)
            .filter(
                RAGEntry.domain == DOMAIN,
                RAGEntry.project_id == project_id,
                RAGEntry.status == "ACTIVE",
                or_(*conditions),
            )
            .order_by(RAGEntry.indexed_at.desc())
            .limit(limit * 2)
            .all()
        )

        scored = []
        now = datetime.utcnow()
        for entry in entries:
            kw_score = _keyword_score(entry.chunk_text, keywords)
            recency_score = _recency_score(entry.indexed_at, now)
            combined = (kw_score * 0.6) + (recency_score * 0.4)
            if combined >= min_relevance:
                scored.append((entry, combined))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [_entry_to_result(e, s) for e, s in scored[:limit]]

    # -----------------------------------------------------------------
    # DomainStore Protocol: store
    # -----------------------------------------------------------------

    def store(self, request: StoreRequest) -> int:
        """Store a context entry via the generic StoreRequest interface."""
        entry_type = request.metadata.get("entry_type", "fact")
        key = request.metadata.get("context_key")
        ttl = request.metadata.get("ttl_hours", DEFAULT_TTL_HOURS)

        if entry_type == "decision":
            return self.set_decision(
                request.content, request.project_id, ttl
            )
        elif key:
            return self.set_fact(
                key, request.content, request.project_id, ttl
            )
        else:
            return self.set_fact(
                "note", request.content, request.project_id, ttl
            )

    # -----------------------------------------------------------------
    # DomainStore Protocol: quarantine / count / stats
    # -----------------------------------------------------------------

    def quarantine(self, file_path: str, project_id: str = "astra-core") -> int:
        """Context entries don't quarantine — they expire."""
        return 0

    def count(self, project_id: str = "astra-core") -> int:
        db = get_db_session()
        try:
            self._gc(db, project_id)
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
                .filter(RAGEntry.domain == DOMAIN)
                .scalar() or 0
            )
            active = self.count(project_id)
            return DomainStats(
                domain=DOMAIN,
                total_entries=total,
                active_entries=active,
            )
        finally:
            db.close()

    # -----------------------------------------------------------------
    # Garbage collection
    # -----------------------------------------------------------------

    def _gc(self, db: Session, project_id: str) -> int:
        """
        Garbage collect expired context entries.

        Called automatically on read operations. Marks expired
        entries as PURGED so they stop appearing in queries.
        """
        now = datetime.utcnow()
        entries = (
            db.query(RAGEntry)
            .filter(
                RAGEntry.domain == DOMAIN,
                RAGEntry.project_id == project_id,
                RAGEntry.status == "ACTIVE",
            )
            .all()
        )

        expired = 0
        for entry in entries:
            expires_at = _get_expires_at(entry)
            if expires_at and now > expires_at:
                entry.status = "PURGED"
                expired += 1

        if expired > 0:
            db.commit()
            logger.debug("[context] GC expired %d entries", expired)

        return expired

    def _expire_key(self, db: Session, key: str, project_id: str) -> int:
        """Mark all active entries with this key as PURGED."""
        entries = (
            db.query(RAGEntry)
            .filter(
                RAGEntry.domain == DOMAIN,
                RAGEntry.project_id == project_id,
                RAGEntry.status == "ACTIVE",
                RAGEntry.chunk_text.like(f"[{key}]%"),
            )
            .all()
        )
        for e in entries:
            e.status = "PURGED"
        if entries:
            db.commit()
        return len(entries)

    def clear_all(self, project_id: str = "astra-core") -> int:
        """Purge all context entries for a project."""
        db = get_db_session()
        try:
            entries = (
                db.query(RAGEntry)
                .filter(
                    RAGEntry.domain == DOMAIN,
                    RAGEntry.project_id == project_id,
                    RAGEntry.status == "ACTIVE",
                )
                .all()
            )
            for e in entries:
                e.status = "PURGED"
            if entries:
                db.commit()
            return len(entries)
        finally:
            db.close()


# =========================================================================
# Helpers
# =========================================================================

def _get_expires_at(entry: RAGEntry) -> Optional[datetime]:
    """Extract expires_at from entry metadata stored in package_role."""
    try:
        if not entry.package_role:
            # No metadata — use default TTL from indexed_at
            if entry.indexed_at:
                return entry.indexed_at + timedelta(hours=DEFAULT_TTL_HOURS)
            return None
        meta = json.loads(entry.package_role)
        exp_str = meta.get("expires_at")
        if exp_str:
            return datetime.fromisoformat(exp_str)
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_entry_type(entry: RAGEntry) -> str:
    """Parse entry type from metadata or content."""
    try:
        if entry.package_role:
            meta = json.loads(entry.package_role)
            return meta.get("entry_type", "fact")
    except (json.JSONDecodeError, ValueError):
        pass
    if entry.chunk_text.startswith("[DECISION]"):
        return "decision"
    return "fact"


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


def _recency_score(indexed_at: Optional[datetime], now: datetime) -> float:
    """Score 1.0 for just created, decaying toward 0.0 over 24 hours."""
    if not indexed_at:
        return 0.0
    age_hours = (now - indexed_at).total_seconds() / 3600.0
    if age_hours <= 0:
        return 1.0
    if age_hours >= DEFAULT_TTL_HOURS:
        return 0.0
    return 1.0 - (age_hours / DEFAULT_TTL_HOURS)


def _entry_to_result(entry: RAGEntry, score: float) -> MemoryResult:
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
            "entry_type": _parse_entry_type(entry),
            "ingest_source": entry.ingest_source,
        },
    )
