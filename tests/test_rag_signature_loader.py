# tests/test_rag_signature_loader.py
"""Tests for signature loader."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import json
import os

from sqlalchemy import create_engine, Column, Integer, text, event
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

from app.rag.chunking.signature_loader import (
    SignatureLoader,
    map_kind_to_chunk_type,
    find_latest_signatures_file,
)
from app.rag.models import ChunkType, ArchScanRun, ArchCodeChunk


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


class TestMapKind:
    def test_function(self):
        assert map_kind_to_chunk_type("function") == ChunkType.FUNCTION
    
    def test_async_function(self):
        assert map_kind_to_chunk_type("async_function") == ChunkType.ASYNC_FUNCTION
    
    def test_class(self):
        assert map_kind_to_chunk_type("class") == ChunkType.CLASS


class TestFindLatestFile:
    def test_find_latest(self, tmp_path):
        # Create multiple signature files
        (tmp_path / "SIGNATURES_20260101_120000.json").write_text("{}")
        import time
        time.sleep(0.01)
        (tmp_path / "SIGNATURES_20260102_120000.json").write_text("{}")
        time.sleep(0.01)
        (tmp_path / "SIGNATURES_20260103_120000.json").write_text("{}")
        
        latest = find_latest_signatures_file(str(tmp_path))
        assert latest is not None
        assert "20260103" in latest
    
    def test_no_files(self, tmp_path):
        latest = find_latest_signatures_file(str(tmp_path))
        assert latest is None


class TestSignatureLoader:
    def test_load_from_file(self, db_session, tmp_path):
        # Create scan run
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        # Create mock SIGNATURES.json
        signatures = {
            "scan_repo_root": "D:\\\\Orb",
            "by_file": {
                "app/main.py": [
                    {
                        "kind": "function",
                        "name": "main",
                        "line": 10,
                        "end_line": 50,
                        "signature": "()",
                        "docstring": "Main entry point.",
                    },
                    {
                        "kind": "async_function",
                        "name": "async_main",
                        "line": 60,
                        "end_line": 100,
                        "signature": "()",
                        "docstring": "Async entry point.",
                    },
                ],
            },
        }
        
        sig_file = tmp_path / "SIGNATURES_test.json"
        sig_file.write_text(json.dumps(signatures))
        
        # Load
        loader = SignatureLoader(db_session, scan.id)
        stats = loader.load_from_file(str(sig_file))
        
        # Verify
        assert stats["files_processed"] == 1
        assert stats["chunks_created"] >= 1
        
        # Check database
        chunks = db_session.query(ArchCodeChunk).filter_by(
            scan_id=scan.id
        ).all()
        assert len(chunks) >= 1
    
    def test_preserves_signature(self, db_session, tmp_path):
        scan = ArchScanRun(status="running")
        db_session.add(scan)
        db_session.commit()
        
        signatures = {
            "scan_repo_root": "D:\\\\Orb",
            "by_file": {
                "app/utils.py": [
                    {
                        "kind": "function",
                        "name": "process",
                        "line": 1,
                        "end_line": 10,
                        "signature": "(data: dict, count: int = 5) -> bool",
                        "docstring": "Process data.",
                        "parameters": ["data", "count"],
                        "returns": "bool",
                    },
                ],
            },
        }
        
        sig_file = tmp_path / "SIGNATURES_test.json"
        sig_file.write_text(json.dumps(signatures))
        
        loader = SignatureLoader(db_session, scan.id)
        loader.load_from_file(str(sig_file))
        
        chunk = db_session.query(ArchCodeChunk).filter_by(
            scan_id=scan.id,
            chunk_name="process"
        ).first()
        
        assert chunk is not None
        assert "def process" in chunk.signature
        assert "data: dict" in chunk.signature
        assert chunk.docstring == "Process data."
        assert chunk.returns == "bool"
