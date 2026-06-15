# FILE: tests/test_repo_chat_retrieval.py
# Purpose: Job 2 (2026-06-12) — repo-scan chunks reach conversational memory:
#          code questions retrieve chunks, casual chat does not, and
#          sandbox-sourced context carries the offline/snapshot label.
# Called-by: pytest
# Depends-on: app.rag.retrieval.chat_injection, app.llm.routing.memory_injection, app.rag.models
# Last-renovated: 2026-06-12
from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.embeddings.models import Embedding
from app.astra_memory.preference_models import HotIndex, SummaryPyramid, PreferenceRecord
from app.memory.architecture_models import ArchitectureScanRun, ArchitectureFileIndex
from app.rag.models import ArchScanRun, ArchCodeChunk, SourceType
from app.rag.retrieval import chat_injection

_TABLES = [
    ArchScanRun.__table__,
    ArchCodeChunk.__table__,
    Embedding.__table__,
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
        chat_injection.reset_probe_cache()


def _seed_scan(db, source: str, scanned: datetime) -> ArchScanRun:
    run = ArchScanRun(status="complete", source=source,
                      started_at=scanned, completed_at=scanned,
                      signatures_file="", index_file="")
    db.add(run)
    db.flush()
    chunks = [
        ArchCodeChunk(
            scan_id=run.id, source=source,
            file_path="app/astra_memory/retrieval.py",
            file_abs_path=r"D:\Orb\app\astra_memory\retrieval.py",
            chunk_type="function", chunk_name="retrieve_for_query",
            qualified_name="app/astra_memory/retrieval.py::retrieve_for_query",
            start_line=467, end_line=542,
            signature="def retrieve_for_query(db, user_message, ...)",
            docstring="Main retrieval entry point for conversational memory.",
            embedded=True, status="active",
        ),
        ArchCodeChunk(
            scan_id=run.id, source=source,
            file_path="app/briefing/briefing_compiler.py",
            file_abs_path=r"D:\Orb\app\briefing\briefing_compiler.py",
            chunk_type="class", chunk_name="BriefingCompiler",
            qualified_name="app/briefing/briefing_compiler.py::BriefingCompiler",
            start_line=10, end_line=200,
            signature="class BriefingCompiler",
            docstring="Compiles the morning briefing from section providers.",
            embedded=True, status="active",
        ),
    ]
    db.add_all(chunks)
    db.commit()
    return run


# =========================================================================
# Constant alignment (the silent-zero-results bug)
# =========================================================================

def test_source_type_constant_matches_stored_rows():
    # The writer (app/rag/jobs/_embedding_batch.py) stores "arch_code_chunk";
    # readers filter through SourceType.ARCH_CHUNK. They must agree.
    assert SourceType.ARCH_CHUNK == "arch_code_chunk"


# =========================================================================
# Gate: code questions in, casual chat out
# =========================================================================

def test_gate_accepts_natural_codebase_question():
    assert chat_injection.is_codebase_query(
        "Where is the memory retrieval implemented in the codebase?"
    )


def test_gate_accepts_topic_tagger_code_tags():
    assert chat_injection.is_codebase_query("anything", query_tags=["code"])
    assert chat_injection.is_codebase_query("anything", query_tags=["architecture"])


def test_gate_rejects_casual_chat():
    assert not chat_injection.is_codebase_query("Fancy a pizza tonight mate?")
    assert not chat_injection.is_codebase_query(
        "Remind me what's on the shopping list", query_tags=["general"]
    )


# =========================================================================
# Retrieval: code question surfaces chunks (keyword fallback path — the
# embedding channel degrades gracefully with no API key in tests)
# =========================================================================

def test_code_question_retrieves_repo_chunks(db, monkeypatch):
    _seed_scan(db, "host", datetime(2026, 6, 9, 7, 37))
    context = chat_injection.build_repo_context(
        db, "Where is retrieve_for_query implemented?", query_tags=["code"],
    )
    assert context
    assert "retrieve_for_query" in context
    assert "app/astra_memory/retrieval.py" in context
    assert "CODEBASE SNAPSHOT" in context


def test_non_code_question_returns_nothing(db):
    _seed_scan(db, "host", datetime(2026, 6, 9, 7, 37))
    context = chat_injection.build_repo_context(
        db, "Fancy a pizza tonight mate?", query_tags=["general"],
    )
    assert context == ""


def test_quarantined_chunks_excluded(db):
    run = _seed_scan(db, "host", datetime(2026, 6, 9, 7, 37))
    db.query(ArchCodeChunk).update({"status": "quarantined"})
    db.commit()
    context = chat_injection.build_repo_context(
        db, "Where is retrieve_for_query implemented?", query_tags=["code"],
    )
    assert context == ""


# =========================================================================
# Staleness labels
# =========================================================================

def test_host_snapshot_label_carries_scan_date(db):
    _seed_scan(db, "host", datetime(2026, 6, 9, 7, 37))
    context = chat_injection.build_repo_context(
        db, "Where is retrieve_for_query implemented?", query_tags=["code"],
    )
    assert "host repo scan dated 2026-06-09" in context


def test_sandbox_offline_label_when_unreachable(db, monkeypatch):
    _seed_scan(db, "sandbox", datetime(2026, 6, 8, 22, 15))
    monkeypatch.setattr(chat_injection, "check_sandbox_reachable",
                        lambda *a, **k: False)
    context = chat_injection.build_repo_context(
        db, "Where is retrieve_for_query implemented?", query_tags=["code"],
    )
    assert "sandbox currently OFFLINE" in context
    assert "2026-06-08" in context
    assert "may have changed" in context


def test_sandbox_online_label_when_reachable(db, monkeypatch):
    _seed_scan(db, "sandbox", datetime(2026, 6, 8, 22, 15))
    monkeypatch.setattr(chat_injection, "check_sandbox_reachable",
                        lambda *a, **k: True)
    context = chat_injection.build_repo_context(
        db, "Where is retrieve_for_query implemented?", query_tags=["code"],
    )
    assert "sandbox is online" in context
    assert "OFFLINE" not in context


def test_reachability_probe_returns_bool_and_caches():
    chat_injection.reset_probe_cache()
    value = chat_injection.check_sandbox_reachable(timeout_seconds=0.6, force=True)
    assert isinstance(value, bool)
    # Second call inside the TTL window must come from cache (same answer)
    assert chat_injection.check_sandbox_reachable(timeout_seconds=0.6) == value


# =========================================================================
# End-to-end through memory injection
# =========================================================================

def test_memory_injection_carries_codebase_block(db, monkeypatch):
    _seed_scan(db, "host", datetime(2026, 6, 9, 7, 37))
    from app.llm.routing import memory_injection

    ctx = memory_injection.build_memory_context(
        db=db,
        messages=[{"role": "user",
                   "content": "Where is retrieve_for_query implemented in the codebase?"}],
    )
    assert ctx.repo_context
    formatted = ctx.format_for_system_prompt()
    assert "<codebase_memory>" in formatted
    assert "retrieve_for_query" in formatted


def test_memory_injection_no_codebase_block_for_casual_chat(db):
    _seed_scan(db, "host", datetime(2026, 6, 9, 7, 37))
    from app.llm.routing import memory_injection

    ctx = memory_injection.build_memory_context(
        db=db,
        messages=[{"role": "user", "content": "Fancy a pizza tonight mate?"}],
    )
    assert ctx.repo_context == ""
    assert "<codebase_memory>" not in ctx.format_for_system_prompt()
