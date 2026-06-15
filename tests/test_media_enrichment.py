# FILE: tests/test_media_enrichment.py
# Purpose: Job 3 (2026-06-12) — enrichment layer: ID3 metadata for music,
#          vision captions for images (mocked), document hot-indexing, and the
#          "we have a document about this — here it is" retrieval behaviour.
# Called-by: pytest
# Depends-on: app.memory.enrichment, app.astra_memory.retrieval, app.memory.models
# Last-renovated: 2026-06-12
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.astra_memory.preference_models import (
    HotIndex, SummaryPyramid, PreferenceRecord, IntentDepth,
)
from app.embeddings.models import Embedding
from app.memory.models import Project, File, DocumentContent
from app.memory.enrichment import id3_reader
from app.memory.enrichment import media_enricher

_TABLES = [
    Project.__table__,
    File.__table__,
    DocumentContent.__table__,
    HotIndex.__table__,
    SummaryPyramid.__table__,
    PreferenceRecord.__table__,
    Embedding.__table__,
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


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


# =========================================================================
# ID3 reader (zero-dep)
# =========================================================================

def _make_id3v23_mp3(tmp_path, title: str, artist: str):
    """Build a minimal valid ID3v2.3 header with TIT2/TPE1 frames."""
    def frame(fid: bytes, text: str) -> bytes:
        payload = b"\x00" + text.encode("latin-1")  # encoding 0 = latin-1
        return fid + len(payload).to_bytes(4, "big") + b"\x00\x00" + payload

    frames = frame(b"TIT2", title) + frame(b"TPE1", artist)
    size = len(frames)
    syncsafe = bytes([
        (size >> 21) & 0x7F, (size >> 14) & 0x7F,
        (size >> 7) & 0x7F, size & 0x7F,
    ])
    header = b"ID3" + b"\x03\x00" + b"\x00" + syncsafe
    path = tmp_path / "track.mp3"
    path.write_bytes(header + frames + b"\xff\xfb" + b"\x00" * 64)
    return path


def test_id3v2_tags_read(tmp_path):
    path = _make_id3v23_mp3(tmp_path, "Deep Hypnotic", "Dizzit")
    tags = id3_reader.read_audio_tags(path)
    assert tags["title"] == "Deep Hypnotic"
    assert tags["artist"] == "Dizzit"
    assert tags["album"] == "unknown"  # absent frame → unknown is acceptable


def test_untagged_audio_returns_unknowns(tmp_path):
    path = tmp_path / "raw.mp3"
    path.write_bytes(b"\xff\xfb" + b"\x00" * 256)
    tags = id3_reader.read_audio_tags(path)
    assert tags == {"title": "unknown", "artist": "unknown",
                    "album": "unknown", "genre": "unknown"}


def test_enrich_music_never_calls_llm(tmp_path, monkeypatch):
    # Poison every LLM entry point — metadata-only is a hard policy
    import app.llm  # noqa: F401
    monkeypatch.setattr(
        media_enricher, "llm_keys_available", lambda: True
    )
    path = _make_id3v23_mp3(tmp_path, "Night Drive", "Taz")
    info = media_enricher.enrich_music(str(path))
    assert info["enriched"] is True
    assert "Night Drive" in info["description"]
    assert "Taz" in info["description"]
    assert "music" in info["tags"]


# =========================================================================
# Image enrichment (vision mocked; keyless placeholder path)
# =========================================================================

def test_enrich_image_uses_vision_caption(monkeypatch, tmp_path):
    img = tmp_path / "van.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    import app.llm.gemini_vision as vision
    monkeypatch.setattr(
        vision, "analyze_image",
        lambda *a, **k: {"summary": "A white camper van parked on a drive.",
                         "tags": ["photo", "vehicle"], "type": "photo"},
    )
    info = media_enricher.enrich_image(str(img))
    assert info["enriched"] is True
    assert "camper van" in info["description"]
    assert "image" in info["tags"]


def test_enrich_image_placeholder_when_vision_unavailable(monkeypatch, tmp_path):
    img = tmp_path / "shot.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    import app.llm.gemini_vision as vision
    monkeypatch.setattr(
        vision, "analyze_image",
        lambda *a, **k: {"summary": "x", "error": "no key"},
    )
    info = media_enricher.enrich_image(str(img))
    assert info["enriched"] is False
    assert "shot.png" in info["description"]


# =========================================================================
# Storage: File + DocumentContent + hot index, dedup, description injected
# =========================================================================

def test_store_media_document_creates_pipeline_records(db):
    file_id = media_enricher.store_media_document(
        db, path=r"D:\Media\van.png", filename="van.png", kind="image",
        description="A white camper van parked on a drive.",
        tags=["image", "photo"],
    )
    assert file_id

    doc = db.query(DocumentContent).filter(
        DocumentContent.file_id == file_id
    ).first()
    assert doc is not None
    assert doc.summary == "A white camper van parked on a drive."

    hot = db.query(HotIndex).filter(
        HotIndex.record_type == "document",
        HotIndex.record_id == str(file_id),
    ).first()
    assert hot is not None
    assert "camper van" in hot.one_liner
    assert hot.cold_storage_path == r"D:\Media\van.png"
    assert "document" in (hot.tags or [])


def test_store_media_document_dedupes_by_filename_and_kind(db):
    first = media_enricher.store_media_document(
        db, path=r"D:\Media\track.mp3", filename="track.mp3", kind="audio",
        description="Music file: Night Drive by Taz",
    )
    second = media_enricher.store_media_document(
        db, path=r"D:\Media\track.mp3", filename="track.mp3", kind="audio",
        description="Music file: Night Drive by Taz (genre: techno)",
    )
    assert first == second
    count = db.query(DocumentContent).filter(
        DocumentContent.filename == "track.mp3"
    ).count()
    assert count == 1


# =========================================================================
# Retrieval: "we have a document about this — here it is"
# =========================================================================

def test_document_topic_question_surfaces_description_and_path(db):
    media_enricher.store_media_document(
        db, path=r"D:\Documents\tenancy_agreement_2026.pdf",
        filename="tenancy_agreement_2026.pdf", kind="pdf",
        description="Tenancy agreement for the Manchester flat, signed "
                    "March 2026, twelve-month term.",
        raw_text="Tenancy agreement for the Manchester flat. Landlord: ... "
                 "Term: 12 months from March 2026. Rent: ...",
    )

    from app.astra_memory.retrieval import retrieve_for_query
    result = retrieve_for_query(
        db=db,
        user_message="what have we got about the tenancy agreement?",
        depth_override=IntentDepth.D2,
    )
    docs = [r for r in result.records if r.record_type == "document"]
    assert docs, "document record must enter the conversational candidate set"
    top = docs[0]
    assert "tenancy" in top.content.lower()
    assert "D:\\Documents\\tenancy_agreement_2026.pdf" in top.content


def test_seeded_image_surfaces_by_description(db):
    media_enricher.store_media_document(
        db, path=r"D:\Media\van.png", filename="van.png", kind="image",
        description="A white camper van parked outside the house.",
    )
    from app.astra_memory.retrieval import retrieve_for_query
    result = retrieve_for_query(
        db=db,
        user_message="have we got a photo of the camper van?",
        depth_override=IntentDepth.D1,
    )
    hits = [r for r in result.records
            if r.record_type == "document" and "camper van" in (r.content or "")]
    assert hits, "image description must be retrievable at hot (D1) depth"


# =========================================================================
# Backfill: resumable cursor + hot-index phase
# =========================================================================

def test_backfill_documents_hot_index_resumes(db, tmp_path, monkeypatch):
    from app.memory.enrichment import backfill_job

    monkeypatch.setattr(backfill_job, "STATE_DIR", tmp_path)
    monkeypatch.setattr(backfill_job, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(backfill_job, "ERROR_LOG", tmp_path / "errors.log")
    monkeypatch.setattr(backfill_job, "BATCH_SLEEP_SECONDS", 0)

    proj = Project(name="T", description="d")
    db.add(proj)
    db.commit()
    for i in range(3):
        f = File(project_id=proj.id, path=f"D:/docs/d{i}.txt",
                 original_name=f"d{i}.txt", file_type="text")
        db.add(f)
        db.commit()
        db.add(DocumentContent(project_id=proj.id, file_id=f.id,
                               filename=f"d{i}.txt", doc_type="text",
                               raw_text=f"content {i}", summary=f"doc {i}"))
        db.commit()

    state = backfill_job.run_backfill(
        phases=["documents_hot_index"], limit=2, db=db,
    )
    assert state["documents_hot_index"]["done"] == 2

    state = backfill_job.run_backfill(
        phases=["documents_hot_index"], limit=10, db=db,
    )
    assert state["documents_hot_index"]["done"] == 3  # resumed, not redone

    hot_count = db.query(HotIndex).filter(
        HotIndex.record_type == "document"
    ).count()
    assert hot_count == 3
