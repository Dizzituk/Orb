# tests/test_rag_arch_embedder.py
"""Tests for architecture embedder."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine, Column, Integer, String, Text, text, event
from sqlalchemy.orm import sessionmaker

from app.db import Base

# Register stub models for FK resolution BEFORE importing RAG models
if "ArchitectureScanRun" not in Base.registry._class_registry:
    class ArchitectureScanRun(Base):
        __tablename__ = "architecture_scan_runs"
        id = Column(Integer, primary_key=True)

if "ArchitectureFileIndex" not in Base.registry._class_registry:
    class ArchitectureFileIndex(Base):
        __tablename__ = "architecture_file_index"
        id = Column(Integer, primary_key=True)

# Import after stub models - the real Embedding model will be imported via arch_embedder
from app.rag.embeddings.arch_embedder import (
    ArchitectureEmbedder,
    ARCH_PROJECT_ID,
)
from app.rag.models import ArchScanRun, ArchDirectoryIndex, ArchCodeChunk, ChunkType


@pytest.fixture
def db_session():
    """Create in-memory database for testing."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed FK targets
    session.execute(text("INSERT INTO architecture_scan_runs (id) VALUES (1)"))
    for i in range(1, 11):
        session.execute(text(f"INSERT INTO architecture_file_index (id) VALUES ({i})"))
    session.commit()

    yield session
    session.close()


class TestArchitectureEmbedder:
    @patch('app.rag.embeddings.arch_embedder.generate_embedding')
    @patch('app.rag.embeddings.arch_embedder.store_embedding')
    def test_embed_directories(self, mock_store, mock_generate, db_session):
        # Mock embedding
        mock_generate.return_value = [0.1] * 1536
        
        # Create test data
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        directory = ArchDirectoryIndex(
            scan_id=scan.id,
            canonical_path="sandbox:d-drive/Orb/app",
            name="app",
            root_alias="d-drive",
            root_kind="sandbox",
            summary="Test summary",
        )
        db_session.add(directory)
        db_session.commit()
        
        # Embed
        embedder = ArchitectureEmbedder(db_session, scan.id)
        stats = embedder._embed_directories()
        
        assert stats["embedded"] == 1
        mock_generate.assert_called_once()
        mock_store.assert_called_once()
    
    @patch('app.rag.embeddings.arch_embedder.generate_embedding')
    @patch('app.rag.embeddings.arch_embedder.store_embedding')
    def test_embed_chunks(self, mock_store, mock_generate, db_session):
        mock_generate.return_value = [0.1] * 1536
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        chunk = ArchCodeChunk(
            scan_id=scan.id,
            file_path="sandbox:d-drive/Orb/app/main.py",
            chunk_type=ChunkType.FUNCTION,
            chunk_name="main",
            descriptor="def main() | Entry point",
        )
        db_session.add(chunk)
        db_session.commit()
        
        embedder = ArchitectureEmbedder(db_session, scan.id)
        stats = embedder._embed_chunks()
        
        assert stats["embedded"] == 1
    
    @patch('app.rag.embeddings.arch_embedder.generate_embedding')
    @patch('app.rag.embeddings.arch_embedder.store_embedding')
    def test_embed_all(self, mock_store, mock_generate, db_session):
        mock_generate.return_value = [0.1] * 1536
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        # Add directory
        directory = ArchDirectoryIndex(
            scan_id=scan.id,
            canonical_path="sandbox:d-drive/Orb/app",
            name="app",
            summary="App directory",
        )
        db_session.add(directory)
        
        # Add chunk
        chunk = ArchCodeChunk(
            scan_id=scan.id,
            file_path="sandbox:d-drive/Orb/app/main.py",
            chunk_type=ChunkType.FUNCTION,
            chunk_name="main",
            descriptor="def main()",
        )
        db_session.add(chunk)
        db_session.commit()
        
        embedder = ArchitectureEmbedder(db_session, scan.id)
        stats = embedder.embed_all()
        
        assert stats["directories"] == 1
        assert stats["chunks"] == 1
        assert stats["errors"] == 0
    
    @patch('app.rag.embeddings.arch_embedder.generate_embedding')
    @patch('app.rag.embeddings.arch_embedder.store_embedding')
    def test_handles_errors(self, mock_store, mock_generate, db_session):
        # Return None to simulate embedding failure
        mock_generate.return_value = None
        
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        directory = ArchDirectoryIndex(
            scan_id=scan.id,
            canonical_path="sandbox:d-drive/Orb/app",
            name="app",
            summary="Test summary",
        )
        db_session.add(directory)
        db_session.commit()
        
        embedder = ArchitectureEmbedder(db_session, scan.id)
        stats = embedder._embed_directories()
        
        assert stats["embedded"] == 0
        assert stats["errors"] == 1
