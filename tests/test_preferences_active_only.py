# FILE: tests/test_preferences_active_only.py
# Purpose: Lock in that get_applicable_preferences returns ACTIVE records only
#          (SUPERSEDED / EXPIRED are stale and must never be injected). Regression
#          guard for the per-turn memory-block bloat fix (2026-06-25).
# Called-by: pytest discovery
# Depends-on: app.astra_memory.preference_service, app.astra_memory.preference_models,
#             app.astra_memory._retrieval_utils
# Last-renovated: 2026-06-25 (new)
"""
get_applicable_preferences ACTIVE-only filter.

The injection front door pulls "applicable" preferences via
get_applicable_preferences. The old filter was `status != DISPUTED`, which also
admitted SUPERSEDED records (a newer ACTIVE value already replaced them — see
document_knowledge_promoter._write_facts_to_memory) and EXPIRED records (decayed
below belief). Live that was 777 stale SUPERSEDED rows injected every turn.

These tests prove only ACTIVE records are returned by default, and that
include_disputed widens to ACTIVE + DISPUTED while still excluding the stale
SUPERSEDED/EXPIRED records.

Run with: pytest tests/test_preferences_active_only.py -v
"""

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:", echo=False)
    # Register the preference models with Base before create_all.
    from app.astra_memory.models import AstraJob  # noqa: F401
    from app.astra_memory.preference_models import (  # noqa: F401
        PreferenceRecord, PreferenceEvidence, HotIndex,
        SummaryPyramid, MemoryRecordConfidence,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def _make(db, key, status):
    """Create a confidence=1.0, applies_to=None preference then set its status."""
    from app.astra_memory.preference_service import create_preference
    from app.astra_memory.preference_models import PreferenceStrength, RecordStatus
    pref = create_preference(
        db, preference_key=key, preference_value="v",
        strength=PreferenceStrength.HARD_RULE,
    )
    pref.status = getattr(RecordStatus, status)
    db.commit()
    return pref


def _keys(records):
    return {r.preference_key for r in records}


def test_only_active_returned_by_default(db):
    from app.astra_memory._retrieval_utils import get_applicable_preferences
    _make(db, "active_pref", "ACTIVE")
    _make(db, "superseded_pref", "SUPERSEDED")
    _make(db, "expired_pref", "EXPIRED")
    _make(db, "disputed_pref", "DISPUTED")

    keys = _keys(get_applicable_preferences(db, "llm_router"))
    assert keys == {"active_pref"}
    # The stale rows that were the bloat are gone.
    assert "superseded_pref" not in keys
    assert "expired_pref" not in keys


def test_include_disputed_adds_disputed_only(db):
    from app.astra_memory._retrieval_utils import get_applicable_preferences
    _make(db, "active_pref", "ACTIVE")
    _make(db, "superseded_pref", "SUPERSEDED")
    _make(db, "expired_pref", "EXPIRED")
    _make(db, "disputed_pref", "DISPUTED")

    keys = _keys(get_applicable_preferences(db, "llm_router", include_disputed=True))
    assert keys == {"active_pref", "disputed_pref"}
    # SUPERSEDED / EXPIRED stay excluded even with include_disputed.
    assert "superseded_pref" not in keys
    assert "expired_pref" not in keys
