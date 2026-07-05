# FILE: tests/test_lane_e_reembed_and_visual.py
# Purpose: LANE E — reembed_batch gating/drain/checkpoint, visual queue enqueue+drain writeback, parity fingerprint.
# Called-by: pytest
# Depends-on: app.idle.tasks_reembed, app.embeddings.visual_queue
# Last-renovated: 2026-07-02

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.embeddings import provider_router, service
from app.embeddings.models import Embedding
from app.embeddings.visual_queue import (
    STATUS_DONE,
    STATUS_PENDING,
    VisualEmbedItem,
    drain_pending,
    enqueue_visual_item,
    pending_count,
)
from app.idle import tasks_reembed
from app.idle.models import IdleTaskRecord
from app.idle.router import TaskContext
from app.memory.rag_entries_model import RAGEntry

GEMINI_LABEL = provider_router.LEGACY_TEXT_LABEL
LOCAL_LABEL = "qwen3-embedding-0.6b"


@pytest.fixture(autouse=True, scope="module")
def _throwaway_master_key():
    """Embedding.content is an encrypted type — init crypto with a throwaway
    key (smoke-harness approach), restoring pristine global state after."""
    import base64
    import os
    from app.crypto import encryption
    saved_env = os.environ.get("ORB_MASTER_KEY")
    saved_mgr = encryption._encryption_manager
    saved_flag = encryption._master_key_initialized
    if not encryption.is_master_key_initialized():
        os.environ["ORB_MASTER_KEY"] = base64.urlsafe_b64encode(
            os.urandom(32)
        ).decode()
        encryption.init_master_key_from_env()
    yield
    encryption._encryption_manager = saved_mgr
    encryption._master_key_initialized = saved_flag
    if saved_env is None:
        os.environ.pop("ORB_MASTER_KEY", None)
    else:
        os.environ["ORB_MASTER_KEY"] = saved_env


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[
        Embedding.__table__,
        IdleTaskRecord.__table__,
        VisualEmbedItem.__table__,
    ])
    # rag_entries declares ix_rag_entries_status twice (column index=True +
    # explicit Index) — harmless on the live DB (pre-exists, checkfirst),
    # dupe-error on a fresh in-memory DB. Table lands before the index blows.
    try:
        Base.metadata.create_all(bind=engine, tables=[RAGEntry.__table__])
    except Exception:
        pass
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    yield maker
    engine.dispose()


@pytest.fixture(autouse=True)
def fresh_caches():
    service.reset_model_count_cache()
    yield
    service.reset_model_count_cache()


def _set_local(monkeypatch, cutover: bool):
    monkeypatch.setenv("EMBEDDINGS_TEXT_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDINGS_TEXT_MODEL", LOCAL_LABEL)
    monkeypatch.setenv("EMBEDDINGS_TEXT_DIMS", "3")
    monkeypatch.setenv("EMBEDDINGS_TEXT_QUERY_CUTOVER", "1" if cutover else "0")


def _plant_legacy(db, source_id, source_type="note", label=GEMINI_LABEL):
    row = Embedding(
        project_id=1, source_type=source_type, source_id=source_id, chunk_index=0,
        content=f"text {source_id}", embedding=json.dumps([1.0] * 1536),
        embedding_model=label, dims=1536 if label else None,
    )
    db.add(row)
    db.commit()
    return row


def _ctx(session_factory, should_yield=lambda: False):
    db = session_factory()
    rec = IdleTaskRecord(task_type=tasks_reembed.REEMBED_TASK, task_key="k", params_json="{}")
    db.add(rec)
    db.commit()
    return TaskContext(
        db=db, record=rec, params={}, fingerprint=None,
        should_yield=should_yield, session_factory=session_factory,
    ), db


def _fake_local_batch(monkeypatch, dim=3):
    def fake(texts, *, task_type=None, purpose=None, spec=None):
        return [[0.5] * dim if t else None for t in texts]
    monkeypatch.setattr(provider_router, "embed_text_batch", fake)


# ── gating ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reembed_gated_without_cutover(session_factory, monkeypatch):
    _set_local(monkeypatch, cutover=False)
    ctx, db = _ctx(session_factory)
    seed = session_factory()
    _plant_legacy(seed, 1)
    seed.close()

    outcome = await tasks_reembed.reembed_handler(ctx)
    assert outcome.status == "completed"
    assert "gated" in outcome.summary
    check = session_factory()
    assert check.query(Embedding).filter(
        Embedding.embedding_model == GEMINI_LABEL
    ).count() == 1  # nothing touched
    check.close()
    db.close()


@pytest.mark.asyncio
async def test_reembed_drains_legacy_rows_hot_first(session_factory, monkeypatch):
    _set_local(monkeypatch, cutover=True)
    monkeypatch.setattr(
        "app.embeddings.local_provider.text_available", lambda *a, **k: True
    )
    _fake_local_batch(monkeypatch)
    seed = session_factory()
    _plant_legacy(seed, 1, "note")
    _plant_legacy(seed, 2, "file")
    _plant_legacy(seed, 3, "custom_type")   # caught by the "rest" unit
    seed.close()

    ctx, db = _ctx(session_factory)
    outcome = await tasks_reembed.reembed_handler(ctx)
    assert outcome.status == "completed"

    check = session_factory()
    remaining = check.query(Embedding).filter(
        Embedding.embedding_model == GEMINI_LABEL
    ).count()
    restamped = check.query(Embedding).filter(
        Embedding.embedding_model == LOCAL_LABEL
    ).all()
    assert remaining == 0
    assert len(restamped) == 3
    assert all(r.dims == 3 for r in restamped)
    assert all(json.loads(r.embedding) == [0.5] * 3 for r in restamped)
    assert "legacy remaining=0" in outcome.summary
    check.close()
    db.close()


@pytest.mark.asyncio
async def test_reembed_heals_null_stamped_rows(session_factory, monkeypatch):
    """Boot-kill hotfix: rows the chunked backfill skipped (NULL stamp,
    damaged pages) are part of the legacy population and get restamped."""
    _set_local(monkeypatch, cutover=True)
    monkeypatch.setattr(
        "app.embeddings.local_provider.text_available", lambda *a, **k: True
    )
    _fake_local_batch(monkeypatch)
    seed = session_factory()
    _plant_legacy(seed, 1, "note", label=None)      # unstamped corrupt-window row
    _plant_legacy(seed, 2, "note")                  # normal legacy row
    seed.close()

    ctx, db = _ctx(session_factory)
    outcome = await tasks_reembed.reembed_handler(ctx)
    assert outcome.status == "completed"
    check = session_factory()
    assert check.query(Embedding).filter(
        Embedding.embedding_model.is_(None)
    ).count() == 0
    assert check.query(Embedding).filter(
        Embedding.embedding_model == LOCAL_LABEL
    ).count() == 2
    check.close()
    db.close()


@pytest.mark.asyncio
async def test_reembed_checkpoints_on_yield_and_resumes(session_factory, monkeypatch):
    _set_local(monkeypatch, cutover=True)
    monkeypatch.setattr(
        "app.embeddings.local_provider.text_available", lambda *a, **k: True
    )
    monkeypatch.setenv("ASTRA_REEMBED_BATCH_SIZE", "1")
    _fake_local_batch(monkeypatch)
    seed = session_factory()
    for i in range(3):
        _plant_legacy(seed, i + 1, "note")
    seed.close()

    # Yield after the first batch
    yields = iter([False, True, True, True, True])
    ctx, db = _ctx(session_factory, should_yield=lambda: next(yields))
    outcome = await tasks_reembed.reembed_handler(ctx)
    assert outcome.status == "paused"
    progress = json.loads(ctx.record.progress_json)
    assert progress["counts"]["hot"] == 1
    db.close()

    # Resume with no interruptions — finishes the remaining two
    ctx2, db2 = _ctx(session_factory)
    ctx2.record.progress_json = json.dumps(progress)
    outcome2 = await tasks_reembed.reembed_handler(ctx2)
    assert outcome2.status == "completed"
    check = session_factory()
    assert check.query(Embedding).filter(
        Embedding.embedding_model == GEMINI_LABEL
    ).count() == 0
    check.close()
    db2.close()


def test_reembed_fingerprint_tracks_gate_and_remaining(monkeypatch, session_factory):
    _set_local(monkeypatch, cutover=False)
    monkeypatch.setattr("app.db.SessionLocal", session_factory)
    fp1 = tasks_reembed.reembed_fingerprint({})
    assert "local|False" in fp1
    _set_local(monkeypatch, cutover=True)
    fp2 = tasks_reembed.reembed_fingerprint({})
    assert fp1 != fp2


def test_parity_fingerprint_one_run_per_pair(monkeypatch):
    _set_local(monkeypatch, cutover=False)
    fp = tasks_reembed.parity_fingerprint({})
    assert LOCAL_LABEL in fp and "cutover:False" in fp


# ── visual queue ─────────────────────────────────────────────────


def test_enqueue_dedupes_pending_items(session_factory):
    db = session_factory()
    a = enqueue_visual_item(db, "video_asset", ref_id=5, text="clip")
    b = enqueue_visual_item(db, "video_asset", ref_id=5, text="clip again")
    assert a.id == b.id
    assert pending_count(db) == 1
    db.close()


def test_drain_writes_vector_and_stamp_back_to_rag_entry(session_factory, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MULTIMODAL_PROVIDER", "gemini")
    monkeypatch.setattr(
        provider_router, "embed_multimodal",
        lambda **kw: [0.25, 0.75],
    )
    db = session_factory()
    entry = RAGEntry(project_id="astra-core", domain="video_asset", chunk_text="{}")
    db.add(entry)
    db.commit()
    enqueue_visual_item(db, "video_asset", ref_id=entry.id, text="a red car")

    stats = drain_pending(db)
    assert stats["embedded"] == 1 and stats["failed"] == 0
    db.refresh(entry)
    assert entry.embedding is not None
    assert entry.embedding_model == GEMINI_LABEL
    assert entry.embedding_dims == 2
    item = db.query(VisualEmbedItem).first()
    assert item.status == STATUS_DONE
    db.close()


def test_drain_skips_when_multimodal_unavailable(session_factory, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MULTIMODAL_PROVIDER", "local")
    monkeypatch.setattr(
        "app.embeddings.local_provider.multimodal_available", lambda *a, **k: False
    )
    db = session_factory()
    enqueue_visual_item(db, "video_asset", ref_id=1, text="x")
    stats = drain_pending(db)
    assert stats["skipped_unavailable"] == 1
    assert pending_count(db) == 1  # untouched, still pending
    db.close()
