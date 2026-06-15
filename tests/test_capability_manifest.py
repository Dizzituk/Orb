# FILE: tests/test_capability_manifest.py
# Purpose: Job 4 (2026-06-12) — capability manifest: generated from live
#          registries, stored as a high-priority memory record, retrieved by
#          identity/capability queries, regenerated on registry hash change.
# Called-by: pytest
# Depends-on: app.self_model.capability_manifest, app.astra_memory
# Last-renovated: 2026-06-12
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.astra_memory.preference_models import (
    HotIndex, SummaryPyramid, PreferenceRecord, IntentDepth,
)
from app.self_model import capability_manifest as cm

_TABLES = [
    HotIndex.__table__,
    SummaryPyramid.__table__,
    PreferenceRecord.__table__,
]


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def tmp_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "MANIFEST_PATH", tmp_path / "manifest.md")
    monkeypatch.setattr(cm, "HASH_PATH", tmp_path / "manifest.hash")
    return tmp_path


# =========================================================================
# Generation from live registries
# =========================================================================

def test_manifest_has_required_sections(db):
    text = cm.generate_manifest(db)
    for section in ("## Identity", "## Memory architecture",
                    "## Autonomous build pipeline", "## Tools",
                    "## Providers & models",
                    "## How ASTRA differs from a stock assistant"):
        assert section in text, f"missing section: {section}"
    assert "Weaver" in text and "SpecGate" in text


def test_manifest_reflects_live_tool_registry(db, monkeypatch):
    """No hardcoded capability list — a smaller registry yields a smaller
    manifest (the future restricted-profile property)."""
    full = cm.generate_manifest(db)

    monkeypatch.setattr(cm, "_collect_tools", lambda: [
        {"name": "only_tool", "version": "v1", "description": "The one tool."},
    ])
    restricted = cm.generate_manifest(db)

    assert "only_tool" in restricted
    assert "1 registered tools" in restricted
    assert len(restricted) < len(full)


def test_registry_hash_changes_with_capabilities(monkeypatch):
    base = cm.registry_hash()
    assert base == cm.registry_hash()  # stable

    monkeypatch.setattr(cm, "_collect_tools", lambda: [
        {"name": "new_tool", "version": "v1", "description": "x"},
    ])
    assert cm.registry_hash() != base


# =========================================================================
# Storage + regeneration
# =========================================================================

def test_regenerate_writes_file_and_hot_row(db, tmp_manifest):
    assert cm.regenerate_if_stale(db) is True
    assert cm.MANIFEST_PATH.exists()
    assert cm.HASH_PATH.exists()

    hot = db.query(HotIndex).filter(
        HotIndex.record_type == cm.HOT_RECORD_TYPE,
        HotIndex.record_id == cm.HOT_RECORD_ID,
    ).first()
    assert hot is not None
    assert hot.retrieval_priority >= 0.85, "manifest must rank above chatter"
    assert "identity_capability" in (hot.tags or [])
    assert "ASTRA" in (hot.entities or [])

    # Unchanged registry → no second regeneration
    assert cm.regenerate_if_stale(db) is False


def test_regenerate_fires_on_hash_change(db, tmp_manifest, monkeypatch):
    assert cm.regenerate_if_stale(db) is True
    monkeypatch.setattr(cm, "_collect_tools", lambda: [
        {"name": "brand_new_tool", "version": "v1", "description": "y"},
    ])
    assert cm.regenerate_if_stale(db) is True
    assert "brand_new_tool" in cm.MANIFEST_PATH.read_text(encoding="utf-8")


# =========================================================================
# Retrieval: identity/capability queries reach the manifest
# =========================================================================

@pytest.mark.parametrize("query", [
    "What are you, exactly?",
    "what are your capabilities?",
    "How are you different from other AI apps?",
])
def test_identity_queries_tag_identity_capability(query):
    from app.astra_memory.topic_tagger import extract_tags
    assert "identity_capability" in extract_tags(query)


def test_identity_query_retrieves_manifest(db, tmp_manifest):
    cm.regenerate_if_stale(db)

    from app.astra_memory.topic_tagger import extract_tags
    from app.astra_memory.retrieval import retrieve_for_query

    query = "What are you and what can you do?"
    tags = [t for t in extract_tags(query) if t != "general"]
    result = retrieve_for_query(
        db=db, user_message=query, query_tags=tags or None,
        depth_override=IntentDepth.D2,
    )
    manifest_hits = [r for r in result.records if r.record_type == "manifest"]
    assert manifest_hits, "manifest must enter the candidate set"
    top = manifest_hits[0]
    # D2 expands through the cold fetcher (content is depth-truncated to
    # ~2K chars, so assert on the document head, not the tail section)
    assert "Capability Manifest" in top.content
    assert "## Memory architecture" in top.content
    assert top.summary_level >= 2, "must be cold-expanded, not a hot one-liner"


def test_manifest_reaches_context_injection(db, tmp_manifest):
    cm.regenerate_if_stale(db)
    from app.llm.routing import memory_injection

    ctx = memory_injection.build_memory_context(
        db=db,
        messages=[{"role": "user",
                   "content": "How are you different from other AI apps?"}],
    )
    formatted = ctx.format_for_system_prompt()
    assert "ASTRA Capability Manifest" in formatted or "ASTRA self-description" in formatted


def test_casual_query_does_not_surface_manifest(db, tmp_manifest):
    cm.regenerate_if_stale(db)

    from app.astra_memory.topic_tagger import extract_tags
    from app.astra_memory.retrieval import retrieve_for_query

    query = "fancy a pizza tonight mate?"
    tags = [t for t in extract_tags(query) if t != "general"]
    result = retrieve_for_query(
        db=db, user_message=query, query_tags=tags or None,
        depth_override=IntentDepth.D1,
    )
    # With no tag filter the manifest may appear as a generic high-priority
    # hot row, but it must never be tag-matched for casual chat
    assert "identity_capability" not in (tags or [])
