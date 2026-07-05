# FILE: tests/test_enriched_cards.py
# Purpose: Enriched architecture cards — ingest idempotency, tier mapping, broad-question injection (2026-07-02).
# Called-by: pytest
# Depends-on: app.rag.enriched_cards, app.rag.jobs._embedding_priority, app.llm.routing.memory_injection_sources
# Last-renovated: 2026-07-02
"""
The enriched cards close the retrieval-altitude gap: broad self-knowledge
questions used to surface one random function-level chunk. These tests pin:
one-chunk-per-card ingest (front matter parsed, body whole), idempotency,
change -> re-embed, removal -> quarantine, kind->tier mapping, the broad
architecture-question gate, and the card collector's ordering + budget.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.rag import enriched_cards
from app.rag.enriched_cards import ingest_enriched_cards
from app.rag.models import ArchCodeChunk, ArchScanRun, ChunkType


CARD_TEXT = """---
card_id: system-overview
kind: overview
repo: cross
source: host
generated_at: 2026-07-02
source_paths: [D:\\Orb\\main.py]
---

# ASTRA — System Overview

Backend on :8000, desktop is the keymaster, phone rides /bridge/*.
"""

LEDGER_TEXT = """---
card_id: capability-ledger-core
kind: capability_ledger
repo: cross
source: host
generated_at: 2026-07-02
source_paths: [D:\\Orb\\IDLE-AGENTS-REPORT.md]
---

# Capability Ledger

EXISTS: idle governor. PLANNED: 3090 worker fleet.
"""


@pytest.fixture
def db(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        bind=engine, tables=[ArchCodeChunk.__table__, ArchScanRun.__table__]
    )
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = maker()
    session.add(ArchScanRun(status="complete"))
    session.commit()
    monkeypatch.setattr(enriched_cards, "ENRICHED_CARDS_DIR", tmp_path)
    yield session
    session.close()
    engine.dispose()


def _write_cards(tmp_path):
    (tmp_path / "overview__system.md").write_text(CARD_TEXT, encoding="utf-8")
    (tmp_path / "capability_ledger__core.md").write_text(LEDGER_TEXT, encoding="utf-8")


def test_ingest_one_chunk_per_card_front_matter_parsed(db, tmp_path):
    _write_cards(tmp_path)
    stats = ingest_enriched_cards(db)
    assert stats["added"] == 2 and stats["errors"] == 0

    rows = db.query(ArchCodeChunk).all()
    assert len(rows) == 2
    overview = next(c for c in rows if c.chunk_name == "system-overview")
    assert overview.chunk_type == ChunkType.ARCHITECTURE_CARD
    assert overview.chunk_type in ChunkType.EMBEDDABLE
    assert "kind=overview" in overview.signature
    # Body stored whole, front matter stripped — never shredded mid-card.
    assert overview.docstring.startswith("# ASTRA — System Overview")
    assert "card_id:" not in overview.docstring
    assert overview.descriptor.startswith("ARCHITECTURE CARD [overview] system-overview")
    assert overview.embedded is False and overview.status == "active"


def test_ingest_idempotent_then_reembeds_on_change(db, tmp_path):
    _write_cards(tmp_path)
    ingest_enriched_cards(db)
    for c in db.query(ArchCodeChunk).all():
        c.embedded = True
    db.commit()

    stats = ingest_enriched_cards(db)
    assert stats["unchanged"] == 2 and stats["added"] == 0 and stats["updated"] == 0
    assert all(c.embedded for c in db.query(ArchCodeChunk).all())

    (tmp_path / "overview__system.md").write_text(
        CARD_TEXT + "\nNew paragraph.\n", encoding="utf-8"
    )
    stats = ingest_enriched_cards(db)
    assert stats["updated"] == 1 and stats["unchanged"] == 1
    overview = db.query(ArchCodeChunk).filter_by(chunk_name="system-overview").one()
    assert overview.embedded is False and "New paragraph." in overview.docstring


def test_removed_card_quarantined(db, tmp_path):
    _write_cards(tmp_path)
    ingest_enriched_cards(db)
    (tmp_path / "capability_ledger__core.md").unlink()
    stats = ingest_enriched_cards(db)
    assert stats["removed"] == 1
    ledger = db.query(ArchCodeChunk).filter_by(chunk_name="capability-ledger-core").one()
    assert ledger.status == "quarantined"


def test_kind_to_tier_mapping():
    from app.rag.jobs._embedding_job_utils_7 import classify_chunk_priority
    from app.rag.jobs._embedding_job_utils_8 import EmbeddingPriority

    t1 = r"D:\Orb\.architecture\enriched\overview__system.md"
    t1b = r"D:\Orb\.architecture\enriched\capability_ledger__core.md"
    t2 = r"D:\Orb\.architecture\enriched\subsystem__rag-embeddings.md"
    t2b = r"D:\Orb\.architecture\enriched\endpoint_map__bridge.md"
    assert classify_chunk_priority(t1) == EmbeddingPriority.TIER1_CRITICAL
    assert classify_chunk_priority(t1b) == EmbeddingPriority.TIER1_CRITICAL
    assert classify_chunk_priority(t2) == EmbeddingPriority.TIER2_HIGH
    assert classify_chunk_priority(t2b) == EmbeddingPriority.TIER2_HIGH
    # Code chunks unaffected
    assert classify_chunk_priority(r"D:\Orb\main.py") == EmbeddingPriority.TIER1_CRITICAL


def test_broad_architecture_question_gate():
    from app.llm.routing.memory_injection_sources import is_broad_architecture_question as broad

    assert broad("what background tasks could run during idle time and what already exists for that")
    assert broad("how does a message from the phone app reach the backend and come back as speech")
    assert broad("what would need building for Astra to track land prices daily")
    assert broad("give me an overview of the system")
    assert not broad("what's the weather like today")
    assert not broad("thanks, that worked")


def test_collect_architecture_cards_order_and_budget(db, tmp_path, monkeypatch):
    _write_cards(tmp_path)
    ingest_enriched_cards(db)
    from app.llm.routing import memory_injection_sources as src

    block = src.collect_architecture_cards(db, "what already exists for idle background tasks")
    assert "[ARCHITECTURE CARD: system-overview" in block
    assert "[ARCHITECTURE CARD: capability-ledger-core" in block
    assert block.index("system-overview") < block.index("capability-ledger-core")

    # Non-broad question -> nothing injected
    assert src.collect_architecture_cards(db, "fix the bug in parser.py") == ""

    # Budget respected
    monkeypatch.setenv("ASTRA_ARCH_CARDS_MAX_CHARS", "500")
    trimmed = src.collect_architecture_cards(db, "what already exists for idle background tasks")
    assert len(trimmed) <= 600  # 500 + marker slack
