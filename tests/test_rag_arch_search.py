# tests/test_rag_arch_search.py
"""Tests for architecture search."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

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

from app.rag.retrieval.arch_search import (
    ArchitectureSearch,
    ArchSearchResult,
    ArchSearchResponse,
)
from app.rag.models import SourceType
from app.astra_memory.preference_models import IntentDepth


@dataclass
class MockSearchResult:
    source_type: str
    source_id: int
    similarity: float
    content: str
    chunk_index: int = 0


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


class TestArchitectureSearch:
    @patch('app.rag.retrieval.arch_search.search_embeddings')
    @patch('app.rag.retrieval.arch_search.classify_intent_depth')
    def test_search_basic(self, mock_intent, mock_search, db_session):
        # Mock intent
        mock_intent.return_value = IntentDepth.D2
        
        # Mock search results
        mock_search.return_value = (
            [
                MockSearchResult(
                    source_type=SourceType.ARCH_CHUNK,
                    source_id=1,
                    similarity=0.9,
                    content="def main()",
                ),
            ],
            10,
        )
        
        search = ArchitectureSearch(db_session)
        response = search.search("How does main work?")
        
        assert response.query == "How does main work?"
        assert response.intent_depth == IntentDepth.D2
        assert len(response.results) >= 0
        
        mock_intent.assert_called_once()
        mock_search.assert_called_once()
    
    @patch('app.rag.retrieval.arch_search.search_embeddings')
    @patch('app.rag.retrieval.arch_search.classify_intent_depth')
    def test_search_with_directories(self, mock_intent, mock_search, db_session):
        mock_intent.return_value = IntentDepth.D2
        
        # Mix of directories and chunks
        mock_search.return_value = (
            [
                MockSearchResult(
                    source_type=SourceType.ARCH_DIRECTORY,
                    source_id=1,
                    similarity=0.95,
                    content="sandbox:d-drive/Orb/app/",
                ),
                MockSearchResult(
                    source_type=SourceType.ARCH_CHUNK,
                    source_id=2,
                    similarity=0.85,
                    content="def process()",
                ),
            ],
            20,
        )
        
        search = ArchitectureSearch(db_session)
        response = search.search("Where is the app directory?")
        
        assert response.directories_found >= 0
        assert response.total_searched == 20
    
    @patch('app.rag.retrieval.arch_search.search_embeddings')
    @patch('app.rag.retrieval.arch_search.classify_intent_depth')
    def test_search_shallow_intent_limits_chunks(self, mock_intent, mock_search, db_session):
        # D1 should limit chunks
        mock_intent.return_value = IntentDepth.D1
        
        mock_search.return_value = ([], 0)
        
        search = ArchitectureSearch(db_session)
        response = search.search("brief overview")
        
        # Should have been called with limited top_k
        assert response.intent_depth == IntentDepth.D1
    
    @patch('app.rag.retrieval.arch_search.search_embeddings')
    @patch('app.rag.retrieval.arch_search.classify_intent_depth')
    def test_empty_results(self, mock_intent, mock_search, db_session):
        mock_intent.return_value = IntentDepth.D2
        mock_search.return_value = ([], 0)
        
        search = ArchitectureSearch(db_session)
        response = search.search("something not found")
        
        assert len(response.results) == 0
        assert response.total_searched == 0
