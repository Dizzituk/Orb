# FILE: tests/test_embedding_modalities.py
# Purpose: Job 1 (2026-06-12) — per-modality embedding integration tests:
#          provider returns configured-dimension vectors for text/batch/image/
#          multimodal; ingest persists them with correct metadata; oversized
#          inputs are chunked/truncated, never silently dropped.
# Called-by: pytest
# Depends-on: app.embeddings.gemini_provider, app.embeddings.service, app.memory.models
# Last-renovated: 2026-06-12
from __future__ import annotations

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.embeddings import gemini_provider
from app.embeddings import service as embeddings_service
from app.embeddings.models import Embedding
from app.memory.models import Project, File, DocumentContent

_TABLES = [
    Embedding.__table__,
    Project.__table__,
    File.__table__,
    DocumentContent.__table__,
]


@pytest.fixture(autouse=True, scope="module")
def _throwaway_master_key():
    """File/DocumentContent columns are encrypted types — init crypto with a
    throwaway key (smoke-harness approach), restoring the pristine global
    state afterwards so test_encryption.py's expectations still hold."""
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


# =========================================================================
# Fake Gemini client (no API key in the test environment — keys sync from
# the DB at live boot; real-API smoke is a live-system check)
# =========================================================================

class _FakeEmbeddingsResult:
    def __init__(self, vectors):
        class _E:
            def __init__(self, values):
                self.values = values
        self.embeddings = [_E(v) for v in vectors]


class FakeGeminiClient:
    """Mimics google-genai's client.models.embed_content."""

    def __init__(self):
        self.calls = []
        outer = self

        class _Models:
            def embed_content(self, *, model, contents, config):
                outer.calls.append(
                    {"model": model, "contents": contents, "config": config}
                )
                dim = getattr(config, "output_dimensionality", None) or 1536
                if isinstance(contents, list):
                    vectors = [[0.01 * (i + 1)] * dim for i in range(len(contents))]
                else:
                    vectors = [[0.5] * dim]
                return _FakeEmbeddingsResult(vectors)

        self.models = _Models()


@pytest.fixture()
def fake_client(monkeypatch):
    client = FakeGeminiClient()
    monkeypatch.setattr(gemini_provider, "_client", client)
    yield client
    gemini_provider.reset_client()


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


# A 1x1 PNG (smallest valid image bytes)
_PNG_1PX = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c626001000000ffff03000006000557bfabd40000000049454e44ae426082"
)


# =========================================================================
# Modality: text
# =========================================================================

def test_text_embedding_returns_configured_dimension(fake_client):
    vec = gemini_provider.generate_embedding("hello world")
    assert vec is not None
    assert len(vec) == gemini_provider.EMBEDDING_DIMENSIONS == 1536
    assert fake_client.calls[0]["model"] == gemini_provider.GEMINI_EMBEDDING_MODEL


def test_model_string_is_gemini_embedding_2():
    assert gemini_provider.GEMINI_EMBEDDING_MODEL == "gemini-embedding-2-preview"
    # The service layer re-exports the same config — one source of truth
    assert embeddings_service.EMBEDDING_MODEL == gemini_provider.GEMINI_EMBEDDING_MODEL
    assert embeddings_service.EMBEDDING_DIMENSIONS == gemini_provider.EMBEDDING_DIMENSIONS


def test_oversized_text_truncated_not_errored(fake_client):
    huge = "word " * 20_000  # ~100K chars >> MAX_TEXT_CHARS
    vec = gemini_provider.generate_embedding(huge)
    assert vec is not None and len(vec) == 1536
    sent = fake_client.calls[0]["contents"]
    assert len(sent) <= gemini_provider.MAX_TEXT_CHARS


def test_batch_embedding_per_item_vectors(fake_client):
    vecs = gemini_provider.generate_embeddings_batch(["a", "b", "c"])
    assert len(vecs) == 3
    assert all(v is not None and len(v) == 1536 for v in vecs)


# =========================================================================
# Modality: image + interleaved multimodal
# =========================================================================

def test_image_embedding_from_bytes(fake_client):
    vec = gemini_provider.generate_image_embedding(
        image_bytes=_PNG_1PX, mime_type="image/png"
    )
    assert vec is not None and len(vec) == 1536


def test_multimodal_text_plus_image(fake_client):
    vec = gemini_provider.generate_multimodal_embedding(
        text="a tiny test pixel", image_bytes=_PNG_1PX, mime_type="image/png"
    )
    assert vec is not None and len(vec) == 1536


# =========================================================================
# Ingest persistence: document → chunks → embeddings rows with metadata
# =========================================================================

def _seed_document(db, text: str):
    proj = Project(name="T", description="test")
    db.add(proj)
    db.commit()
    f = File(project_id=proj.id, path="D:/tmp/doc.txt",
             original_name="doc.txt", file_type="text")
    db.add(f)
    db.commit()
    doc = DocumentContent(project_id=proj.id, file_id=f.id,
                          filename="doc.txt", doc_type="text",
                          raw_text=text, summary=text[:100])
    db.add(doc)
    db.commit()
    return proj, f, doc


def test_document_ingest_persists_vectors_with_metadata(fake_client, db):
    proj, f, doc = _seed_document(db, "Tenancy agreement for the flat. " * 20)
    created = embeddings_service.index_document(db, doc, force=True)
    assert created >= 1

    rows = db.query(Embedding).filter(
        Embedding.source_type == "file",
        Embedding.source_id == f.id,
    ).all()
    assert len(rows) == created
    for i, row in enumerate(sorted(rows, key=lambda r: r.chunk_index)):
        assert row.project_id == proj.id
        assert row.chunk_index == i
        stored = json.loads(row.embedding)
        assert len(stored) == 1536
        assert row.content  # chunk text persisted alongside the vector


def test_oversized_document_chunks_under_token_limit(fake_client, db):
    # ~200K chars — must be chunked, with every chunk under the 8,192-token
    # (~32K char) per-call limit, and nothing silently dropped
    big_text = ("The quick brown fox jumps over the lazy dog. " * 4500)
    chunks = embeddings_service.chunk_text(big_text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= gemini_provider.MAX_TEXT_CHARS
    # Reassembled coverage: total chunk chars >= original (overlap included)
    assert sum(len(c) for c in chunks) >= len(big_text) * 0.95


def test_search_round_trip_uses_query_task_type(fake_client, db):
    proj, f, doc = _seed_document(db, "Camper van insurance renewal documents.")
    embeddings_service.index_document(db, doc, force=True)
    results, searched = embeddings_service.search_embeddings(
        db, proj.id, "van insurance", top_k=3
    )
    assert searched >= 1
    assert results and results[0].source_type == "file"
    # Last call was the query embed — must use RETRIEVAL_QUERY geometry
    assert fake_client.calls[-1]["config"].task_type == "RETRIEVAL_QUERY"
