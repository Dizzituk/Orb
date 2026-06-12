# FILE: app/memory/domains/preferences.py
# Purpose: Preference domain store.
# Called-by: app.memory.domains, app.memory.startup
# Depends-on: app.astra_memory.preference_models, app.astra_memory.preference_service, app.db, app.memory.domains.preference_registry (+1 more)
# Last-renovated: 2026-06-11
"""
Preference domain store.

Wraps the existing app/astra_memory/ preference system behind the
DomainStore interface so MemoryRouter can query preferences alongside
architecture, knowledge, and other domains.

This store BRIDGES — it does not duplicate or replace the preference
infrastructure in app/astra_memory/. All actual CRUD is delegated
to preference_service.py.

Query strategy:
    Keyword matching across preference_key and preference_value.
    Returns normalised MemoryResult objects for MemoryRouter.

Storage strategy:
    Delegates to preference_service.create_preference() or
    update_preference_value(). Maps the spec's (domain, category, key)
    triple to (namespace, applies_to, preference_key).
"""

import json
import logging
from typing import Any, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db import get_db_session
from app.astra_memory.preference_models import (
    PreferenceRecord,
    PreferenceEvidence,
    PreferenceStrength,
    RecordStatus,
)
from app.astra_memory.preference_service import (
    create_preference,
    create_hard_rule,
    update_preference_value,
    get_preference,
    get_preference_value,
    get_preferences_for_component,
)
from app.memory.schemas_unified import MemoryResult, DomainStats
from app.memory.domains.preference_registry import (
    is_valid_domain,
    validate_domain,
)

logger = logging.getLogger(__name__)


class PreferenceStore:
    """
    DomainStore implementation for user preferences.

    Maps the spec's (domain, category, key) interface to the
    existing astra_preferences table's (namespace, applies_to,
    preference_key) columns.
    """

    @property
    def domain_name(self) -> str:
        return "preference"

    # -----------------------------------------------------------------
    # Get / Set (Spec Section 12.3)
    # -----------------------------------------------------------------

    def get(
        self,
        domain: str,
        category: Optional[str],
        key: str,
    ) -> Optional[PreferenceRecord]:
        """
        Get a single preference by domain + key.

        Args:
            domain: Preference namespace (development, content, etc.)
            category: Component scope (all, specgate, etc.) — used for
                      filtering but not part of the lookup key.
            key: The preference_key to look up.

        Returns:
            PreferenceRecord or None.
        """
        validate_domain(domain)
        db = get_db_session()
        try:
            query = db.query(PreferenceRecord).filter(
                PreferenceRecord.preference_key == key,
                PreferenceRecord.namespace == domain,
                PreferenceRecord.status == RecordStatus.ACTIVE,
            )
            if category:
                query = query.filter(
                    or_(
                        PreferenceRecord.applies_to == category,
                        PreferenceRecord.applies_to == "all",
                        PreferenceRecord.applies_to.is_(None),
                    )
                )
            return query.first()
        finally:
            db.close()

    def set(
        self,
        domain: str,
        category: Optional[str],
        key: str,
        value: Any,
        source: str = "user_declared",
        strength: PreferenceStrength = PreferenceStrength.DEFAULT,
    ) -> int:
        """
        Create or update a preference.

        Args:
            domain: Preference namespace.
            category: Component scope (maps to applies_to).
            key: Preference key.
            value: Preference value (any JSON-serialisable type).
            source: How this preference was captured.
            strength: Enforcement level.

        Returns:
            The preference record ID.
        """
        validate_domain(domain)
        db = get_db_session()
        try:
            pref = create_preference(
                db=db,
                preference_key=key,
                preference_value=value,
                strength=strength,
                source=source,
                applies_to=category,
                namespace=domain,
            )
            return pref.id
        finally:
            db.close()

    def get_history(
        self,
        domain: str,
        key: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Get the evidence history for a preference.

        Returns the append-only evidence ledger entries showing
        how the preference's confidence has evolved.

        Args:
            domain: Preference namespace.
            key: Preference key.
            limit: Max evidence entries to return.

        Returns:
            List of dicts with evidence details.
        """
        validate_domain(domain)
        db = get_db_session()
        try:
            # Verify the preference exists in this domain
            pref = db.query(PreferenceRecord).filter(
                PreferenceRecord.preference_key == key,
                PreferenceRecord.namespace == domain,
            ).first()

            if not pref:
                return []

            entries = (
                db.query(PreferenceEvidence)
                .filter(PreferenceEvidence.preference_key == key)
                .order_by(PreferenceEvidence.timestamp.desc())
                .limit(limit)
                .all()
            )

            return [
                {
                    "id": e.id,
                    "signal_type": e.signal_type.value if e.signal_type else None,
                    "weight": e.weight,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "context_pointer": e.context_pointer,
                    "details": e.details,
                }
                for e in entries
            ]
        finally:
            db.close()

    # -----------------------------------------------------------------
    # Query (for MemoryRouter integration)
    # -----------------------------------------------------------------

    def query(
        self,
        text: str,
        project_id: str = "astra-core",
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryResult]:
        """
        Search preferences by keyword matching.

        Searches across preference_key and preference_value fields.
        Returns normalised MemoryResult for MemoryRouter.
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
        """Run keyword search against astra_preferences.

        v2.1: Cap keywords at 20 to prevent SQLite expression tree overflow.
        Each keyword generates 3 LIKE conditions (key, namespace, applies_to),
        so 20 keywords = 60 conditions — well within SQLite's 1000 depth limit.
        Long messages (e.g. voice-to-text build descriptions) were generating
        900+ conditions and hitting 'Expression tree is too large'.
        """
        keywords = _extract_keywords(text)
        if not keywords:
            return []

        # v2.1: Cap at 20 keywords to prevent SQLite expression tree overflow
        if len(keywords) > 20:
            keywords = keywords[:20]

        # Build keyword match conditions
        # v0.14.0: Now also searches preference_value — critical for finding
        # biographical facts by content (e.g. "Redruth" matches location value).
        # Cap at 15 keywords since we now have 4 LIKE conditions per keyword.
        if len(keywords) > 15:
            keywords = keywords[:15]
        conditions = []
        for kw in keywords:
            pattern = f"%{kw}%"
            conditions.append(
                or_(
                    PreferenceRecord.preference_key.ilike(pattern),
                    PreferenceRecord.preference_value.ilike(pattern),
                    PreferenceRecord.namespace.ilike(pattern),
                    PreferenceRecord.applies_to.ilike(pattern),
                )
            )

        candidates = (
            db.query(PreferenceRecord)
            .filter(
                PreferenceRecord.status == RecordStatus.ACTIVE,
                or_(*conditions),
            )
            .limit(limit * 3)
            .all()
        )

        # Score and filter
        scored = []
        for pref in candidates:
            score = _score_preference(pref, keywords)
            if score >= min_relevance:
                scored.append((pref, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            _pref_to_result(pref, score)
            for pref, score in scored[:limit]
        ]

    # -----------------------------------------------------------------
    # Domain-scoped queries
    # -----------------------------------------------------------------

    def get_domain_preferences(
        self,
        domain: str,
        min_confidence: Optional[float] = None,
    ) -> list[PreferenceRecord]:
        """
        Get all active preferences for a domain.

        Args:
            domain: Preference namespace to query.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of PreferenceRecord, sorted by confidence desc.
        """
        validate_domain(domain)
        db = get_db_session()
        try:
            query = db.query(PreferenceRecord).filter(
                PreferenceRecord.namespace == domain,
                PreferenceRecord.status == RecordStatus.ACTIVE,
            )
            if min_confidence is not None:
                query = query.filter(
                    PreferenceRecord.confidence >= min_confidence
                )
            return query.order_by(
                PreferenceRecord.confidence.desc()
            ).all()
        finally:
            db.close()

    # -----------------------------------------------------------------
    # Stats
    # -----------------------------------------------------------------

    def get_stats(self, project_id: str = "astra-core") -> DomainStats:
        """Get preference domain statistics."""
        from sqlalchemy import func

        db = get_db_session()
        try:
            total = (
                db.query(func.count(PreferenceRecord.id)).scalar() or 0
            )
            active = (
                db.query(func.count(PreferenceRecord.id))
                .filter(PreferenceRecord.status == RecordStatus.ACTIVE)
                .scalar() or 0
            )

            return DomainStats(
                domain="preference",
                total_entries=total,
                active_entries=active,
            )
        finally:
            db.close()


# =========================================================================
# Helpers (private)
# =========================================================================

def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from query text."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "not", "with", "from", "by",
        "it", "this", "that", "how", "what", "where", "when", "who",
        "does", "do", "can", "will", "about", "me", "my", "show",
        "preference", "prefer", "setting",
    }
    words = text.lower().split()
    return [w for w in words if len(w) >= 2 and w not in stop_words]


def _score_preference(
    pref: PreferenceRecord,
    keywords: list[str],
) -> float:
    """Score a preference by keyword match ratio."""
    if not keywords:
        return 0.0

    # Build searchable text from key, namespace, applies_to, and value
    value_str = ""
    if isinstance(pref.preference_value, str):
        value_str = pref.preference_value
    elif isinstance(pref.preference_value, (dict, list)):
        value_str = json.dumps(pref.preference_value)
    elif pref.preference_value is not None:
        value_str = str(pref.preference_value)

    searchable = " ".join(filter(None, [
        (pref.preference_key or "").lower(),
        (pref.namespace or "").lower(),
        (pref.applies_to or "").lower(),
        value_str.lower(),
    ]))

    # Replace underscores with spaces for matching
    searchable = searchable.replace("_", " ")

    matches = sum(1 for kw in keywords if kw in searchable)
    return matches / len(keywords)


def _pref_to_result(
    pref: PreferenceRecord,
    score: float,
) -> MemoryResult:
    """Convert a PreferenceRecord to a normalised MemoryResult."""
    # Build human-readable content
    value_display = pref.preference_value
    if isinstance(value_display, bool):
        value_display = "yes" if value_display else "no"

    content = (
        f"[{pref.namespace}] {pref.preference_key} = {value_display}"
    )
    if pref.applies_to:
        content += f" (applies to: {pref.applies_to})"

    return MemoryResult(
        id=pref.id,
        domain="preference",
        content=content,
        project_id="astra-core",
        relevance=score,
        file_path=None,
        source_table="astra_preferences",
        status=pref.status.value if pref.status else "active",
        metadata={
            "preference_key": pref.preference_key,
            "preference_value": pref.preference_value,
            "namespace": pref.namespace,
            "applies_to": pref.applies_to,
            "strength": pref.strength.value if pref.strength else None,
            "confidence": pref.confidence,
            "source": pref.source,
            "evidence_count": pref.evidence_count,
        },
    )
