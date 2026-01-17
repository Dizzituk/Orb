# tests/test_rag_pipeline.py
"""Tests for RAG pipeline."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import json
from unittest.mock import patch

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

from app.rag.pipeline import RAGPipeline, run_rag_pipeline, get_latest_scan_id
from app.rag.models import ArchScanRun


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


class TestRAGPipeline:
    @patch('app.rag.pipeline.embed_architecture_scan')
    def test_pipeline_creates_scan_run(self, mock_embed, db_session, tmp_path):
        # Mock embedding to avoid needing API key
        mock_embed.return_value = {"directories": 0, "chunks": 0, "errors": 0}
        
        # Create minimal test files
        signatures = {
            "scan_repo_root": "D:\\\\Orb",
            "by_file": {},
        }
        index = {
            "scanned_files": [],
        }
        
        sig_file = tmp_path / "SIGNATURES_test.json"
        idx_file = tmp_path / "INDEX_test.json"
        
        sig_file.write_text(json.dumps(signatures))
        idx_file.write_text(json.dumps(index))
        
        # Run pipeline
        pipeline = RAGPipeline(db_session, str(tmp_path))
        stats = pipeline.run()
        
        assert stats["scan_id"] > 0
        
        # Verify scan run exists and is complete
        scan = db_session.query(ArchScanRun).filter_by(id=stats["scan_id"]).first()
        assert scan is not None
        assert scan.status == "complete"
    
    @patch('app.rag.pipeline.embed_architecture_scan')
    def test_pipeline_with_data(self, mock_embed, db_session, tmp_path):
        mock_embed.return_value = {"directories": 1, "chunks": 2, "errors": 0}
        
        # Create test files with actual data
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
                        "docstring": "Entry point.",
                    },
                ],
            },
        }
        index = {
            "scanned_files": [
                {"path": "app/main.py", "bytes": 1000, "lines": 50},
            ],
        }
        
        sig_file = tmp_path / "SIGNATURES_test.json"
        idx_file = tmp_path / "INDEX_test.json"
        
        sig_file.write_text(json.dumps(signatures))
        idx_file.write_text(json.dumps(index))
        
        pipeline = RAGPipeline(db_session, str(tmp_path))
        stats = pipeline.run()
        
        assert stats["chunks"] >= 1
    
    def test_get_latest_scan_id(self, db_session):
        # No complete scans initially
        assert get_latest_scan_id(db_session) is None
        
        # Add a complete scan
        scan = ArchScanRun(status="complete")
        db_session.add(scan)
        db_session.commit()
        
        assert get_latest_scan_id(db_session) == scan.id
    
    def test_pipeline_fails_without_signatures(self, db_session, tmp_path):
        # Empty directory - no signatures file
        pipeline = RAGPipeline(db_session, str(tmp_path))
        
        with pytest.raises(FileNotFoundError):
            pipeline.run()
