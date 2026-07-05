# FILE: tests/test_lane_e_router_and_search.py
# Purpose: LANE E — provider router semantics, row stamping, model-filtered search (mixed-model structurally impossible), dual-read, migrations backfill.
# Called-by: pytest
# Depends-on: app.embeddings.*
# Last-renovated: 2026-07-02

import json

import pytest
from sqlalchemy import create_engine, text as _sql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.embeddings import local_provider, provider_router, service
from app.embeddings.models import Embedding

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
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[Embedding.__table__])
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = maker()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(autouse=True)
def fresh_caches():
    service.reset_model_count_cache()
    yield
    service.reset_model_count_cache()


def _set_local(monkeypatch, cutover: bool):
    monkeypatch.setenv("EMBEDDINGS_TEXT_PROVIDER", "local")
    monkeypatch.setenv("EMBEDDINGS_TEXT_MODEL", LOCAL_LABEL)
    monkeypatch.setenv("EMBEDDINGS_TEXT_DIMS", "4")
    monkeypatch.setenv("EMBEDDINGS_TEXT_QUERY_CUTOVER", "1" if cutover else "0")


def _set_gemini(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_TEXT_PROVIDER", "gemini")
    monkeypatch.setenv("EMBEDDINGS_TEXT_QUERY_CUTOVER", "0")


def _fake_spaces(monkeypatch, gemini_vec, local_vec):
    """Both providers return fixed vectors — search must pick per space."""
    monkeypatch.setattr(
        "app.embeddings.gemini_provider.generate_embedding",
        lambda text, task_type=None: list(gemini_vec),
    )
    monkeypatch.setattr(
        "app.embeddings.local_provider.generate_embedding",
        lambda text, task_type=None: list(local_vec),
    )


def _plant(db, source_id, vec, label):
    row = Embedding(
        project_id=1, source_type="note", source_id=source_id, chunk_index=0,
        content=f"content {source_id}", embedding=json.dumps(vec),
        embedding_model=label, dims=len(vec),
    )
    db.add(row)
    db.commit()
    return row


# ── router semantics ─────────────────────────────────────────────


def test_write_spec_defaults_to_gemini(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_TEXT_PROVIDER", raising=False)
    spec = provider_router.text_write_spec()
    assert spec.provider == "gemini"
    assert spec.label == GEMINI_LABEL
    assert spec.dims == 1536


def test_query_stays_gemini_until_cutover(monkeypatch):
    _set_local(monkeypatch, cutover=False)
    assert provider_router.text_write_spec().provider == "local"
    assert provider_router.text_query_spec().provider == "gemini"
    assert [s.provider for s in provider_router.text_query_spaces()] == ["gemini"]


def test_cutover_enables_dual_read_spaces(monkeypatch):
    _set_local(monkeypatch, cutover=True)
    spaces = provider_router.text_query_spaces()
    assert [s.provider for s in spaces] == ["local", "gemini"]
    assert spaces[0].label == LOCAL_LABEL


def test_rollback_is_env_flip(monkeypatch):
    _set_local(monkeypatch, cutover=True)
    assert provider_router.text_write_spec().provider == "local"
    _set_gemini(monkeypatch)
    assert provider_router.text_write_spec().provider == "gemini"
    assert [s.provider for s in provider_router.text_query_spaces()] == ["gemini"]


# ── stamping ─────────────────────────────────────────────────────


def test_store_embedding_stamps_model_and_dims(db, monkeypatch):
    _set_local(monkeypatch, cutover=False)
    row = service.store_embedding(db, 1, "note", 7, "hello", [0.1, 0.2, 0.3])
    assert row.embedding_model == LOCAL_LABEL
    assert row.dims == 3


def test_store_embedding_gemini_stamp(db, monkeypatch):
    _set_gemini(monkeypatch)
    row = service.store_embedding(db, 1, "note", 8, "hello", [0.1] * 5)
    assert row.embedding_model == GEMINI_LABEL
    assert row.dims == 5


# ── the hard rule: mixed-model cosine structurally impossible ────


def test_search_scores_only_active_model_rows(db, monkeypatch):
    """Acceptance 3: plant rows of both models; search must only score the
    active space. gemini-only mode → the local row can never appear."""
    _set_gemini(monkeypatch)
    _fake_spaces(monkeypatch, gemini_vec=[1.0, 0.0], local_vec=[9.9, 9.9])
    _plant(db, 1, [1.0, 0.0], GEMINI_LABEL)          # gemini row, perfect match
    _plant(db, 2, [1.0, 0.0], LOCAL_LABEL)           # local row, same vector!
    results, total = service.search_embeddings(db, 1, "q", top_k=10)
    assert total == 1
    assert [r.source_id for r in results] == [1]


def test_search_pre_cutover_ignores_local_rows(db, monkeypatch):
    """Writes local + cutover off: new local rows are invisible (accepted gap
    until parity sign-off) and NEVER scored against the gemini query vector."""
    _set_local(monkeypatch, cutover=False)
    _fake_spaces(monkeypatch, gemini_vec=[0.0, 1.0], local_vec=[1.0, 0.0])
    _plant(db, 1, [0.0, 1.0], GEMINI_LABEL)
    _plant(db, 2, [1.0, 0.0], LOCAL_LABEL)
    results, total = service.search_embeddings(db, 1, "q", top_k=10)
    assert total == 1
    assert [r.source_id for r in results] == [1]


def test_dual_read_scores_each_space_with_its_own_vector(db, monkeypatch):
    _set_local(monkeypatch, cutover=True)
    # Query vectors deliberately differ per space; each row matches ITS space.
    _fake_spaces(monkeypatch, gemini_vec=[1.0, 0.0], local_vec=[0.0, 1.0, 0.0])
    _plant(db, 1, [1.0, 0.0], GEMINI_LABEL)          # 2-d gemini space
    _plant(db, 2, [0.0, 1.0, 0.0], LOCAL_LABEL)      # 3-d local space
    results, total = service.search_embeddings(db, 1, "q", top_k=10)
    assert total == 2
    assert {r.source_id for r in results} == {1, 2}
    # Per-space normalisation puts both perfect matches at 1.0
    assert all(abs(r.similarity - 1.0) < 1e-6 for r in results)


def test_gemini_drops_out_at_zero_rows(db, monkeypatch):
    """Migration complete: no gemini rows left → the gemini provider must not
    be called at all (acceptance 1's mechanism)."""
    _set_local(monkeypatch, cutover=True)
    calls = {"gemini": 0}

    def gemini_embed(text, task_type=None):
        calls["gemini"] += 1
        return [1.0, 0.0]

    monkeypatch.setattr("app.embeddings.gemini_provider.generate_embedding", gemini_embed)
    monkeypatch.setattr(
        "app.embeddings.local_provider.generate_embedding",
        lambda text, task_type=None: [0.0, 1.0],
    )
    _plant(db, 2, [0.0, 1.0], LOCAL_LABEL)
    results, total = service.search_embeddings(db, 1, "q", top_k=10)
    assert [r.source_id for r in results] == [2]
    assert calls["gemini"] == 0


def test_null_stamped_rows_read_as_legacy_gemini(db, monkeypatch):
    """Boot-kill hotfix (2026-07-02): rows the chunked backfill could not
    stamp (damaged btree pages) must stay searchable in the gemini space —
    and must never leak into the local space."""
    _set_gemini(monkeypatch)
    _fake_spaces(monkeypatch, gemini_vec=[1.0, 0.0], local_vec=[0.0, 0.0])
    _plant(db, 1, [1.0, 0.0], GEMINI_LABEL)
    _plant(db, 2, [1.0, 0.0], None)          # unstamped — corrupt-window row
    results, total = service.search_embeddings(db, 1, "q", top_k=10)
    assert total == 2
    assert {r.source_id for r in results} == {1, 2}

    # Local space must NOT include the NULL row
    _set_local(monkeypatch, cutover=True)
    _fake_spaces(monkeypatch, gemini_vec=[1.0, 0.0], local_vec=[0.0, 1.0, 0.0])
    _plant(db, 3, [0.0, 1.0, 0.0], LOCAL_LABEL)
    service.reset_model_count_cache()
    results, _ = service.search_embeddings(db, 1, "q", top_k=10)
    by_id = {r.source_id for r in results}
    assert 3 in by_id and 1 in by_id and 2 in by_id  # dual-read: all spaces
    # ...but the NULL row scored ONLY against the gemini query vector:
    # plant a local-dims NULL row — it must never be scored in local space
    _plant(db, 4, [0.0, 1.0, 0.0], None)
    service.reset_model_count_cache()
    results, _ = service.search_embeddings(db, 1, "q", top_k=10)
    null_local = [r for r in results if r.source_id == 4]
    # It rides the gemini space (len mismatch vs gemini qvec → cosine 0.0),
    # so it can appear but never with a local-space score of 1.0 at the top
    assert not null_local or null_local[0].similarity < 0.99


def test_single_space_scores_are_raw_cosine(db, monkeypatch):
    """Pre-LANE-E behaviour preserved: one active space → no normalisation."""
    _set_gemini(monkeypatch)
    _fake_spaces(monkeypatch, gemini_vec=[1.0, 0.0], local_vec=[0.0, 0.0])
    _plant(db, 1, [1.0, 1.0], GEMINI_LABEL)  # cos = 1/sqrt(2)
    results, _ = service.search_embeddings(db, 1, "q", top_k=1)
    assert abs(results[0].similarity - 0.7071) < 0.001


# ── local provider instruction mapping ───────────────────────────


def test_local_provider_applies_query_instruction(monkeypatch):
    captured = {}

    def fake_post(base_url, model, inputs):
        captured["inputs"] = list(inputs)
        return [[0.1, 0.2] for _ in inputs]

    monkeypatch.setattr(local_provider, "_post_embeddings", fake_post)
    local_provider.generate_embedding("find my notes", task_type="RETRIEVAL_QUERY")
    assert captured["inputs"][0].startswith("Instruct: ")
    assert captured["inputs"][0].endswith("Query: find my notes")

    local_provider.generate_embedding("a document body", task_type="RETRIEVAL_DOCUMENT")
    assert captured["inputs"][0] == "a document body"


def test_local_provider_failure_returns_none(monkeypatch):
    monkeypatch.setattr(local_provider, "_post_embeddings", lambda *a: None)
    assert local_provider.generate_embedding("x") is None
    assert local_provider.generate_embeddings_batch(["x", "y"]) == [None, None]


# ── migrations backfill ──────────────────────────────────────────


def test_migration_adds_columns_and_backfills():
    from app.embeddings.migrations import migrate_embeddings_schema

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as conn:
        conn.execute(_sql(
            "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, project_id INTEGER, "
            "source_type VARCHAR(20), source_id INTEGER, chunk_index INTEGER, "
            "content TEXT, embedding TEXT, created_at DATETIME)"
        ))
        conn.execute(_sql(
            "INSERT INTO embeddings (project_id, source_type, source_id, chunk_index, "
            "content, embedding) VALUES (1, 'note', 1, 0, 'c', '[0.1]')"
        ))
        conn.execute(_sql(
            "CREATE TABLE rag_entries (id INTEGER PRIMARY KEY, embedding BLOB)"
        ))
        conn.execute(_sql("INSERT INTO rag_entries (embedding) VALUES (x'00000000')"))
        conn.execute(_sql("INSERT INTO rag_entries (embedding) VALUES (NULL)"))

    migrate_embeddings_schema(engine)
    migrate_embeddings_schema(engine)  # idempotent re-run

    with engine.connect() as conn:
        row = conn.execute(_sql(
            "SELECT embedding_model, dims FROM embeddings"
        )).fetchone()
        assert row[0] == GEMINI_LABEL and row[1] == 1536
        stamped = conn.execute(_sql(
            "SELECT COUNT(*) FROM rag_entries WHERE embedding_model IS NOT NULL"
        )).scalar()
        assert stamped == 1  # only the row that HAS a vector
    engine.dispose()
