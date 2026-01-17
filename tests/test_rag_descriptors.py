# tests/test_rag_descriptors.py
"""Tests for descriptor generator."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from sqlalchemy import Column, Integer

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

from app.rag.descriptors.descriptor_gen import (
    generate_chunk_descriptor,
    _truncate_docstring,
    estimate_tokens,
)
from app.rag.models import ArchCodeChunk, ChunkType


class TestTruncateDocstring:
    def test_short(self):
        doc = "Short docstring."
        assert _truncate_docstring(doc) == doc
    
    def test_first_sentence(self):
        doc = "First sentence. Second sentence. Third."
        assert _truncate_docstring(doc) == "First sentence."
    
    def test_long(self):
        doc = "A" * 200
        result = _truncate_docstring(doc)
        assert len(result) < 160
    
    def test_empty(self):
        assert _truncate_docstring("") == ""
        assert _truncate_docstring(None) == ""


class TestGenerateDescriptor:
    def test_function(self):
        chunk = ArchCodeChunk(
            scan_id=1,
            file_path="sandbox:d-drive/Orb/app/main.py",
            chunk_type=ChunkType.ASYNC_FUNCTION,
            chunk_name="stream_chat",
            signature="async def stream_chat(req: Request) -> Response",
            docstring="Stream chat responses with routing.",
            decorators_json='["@router.post(\\"/chat\\")"]',
        )
        
        descriptor = generate_chunk_descriptor(chunk)
        
        assert "stream_chat" in descriptor
        assert "Stream chat" in descriptor
        assert "router.post" in descriptor
    
    def test_class(self):
        chunk = ArchCodeChunk(
            scan_id=1,
            file_path="sandbox:d-drive/Orb/app/models.py",
            chunk_type=ChunkType.CLASS,
            chunk_name="StreamRouter",
            signature="class StreamRouter",
            docstring="Main routing engine.",
            bases_json='["BaseRouter", "ABC"]',
        )
        
        descriptor = generate_chunk_descriptor(chunk)
        
        assert "StreamRouter" in descriptor
        assert "Bases:" in descriptor
    
    def test_no_signature(self):
        chunk = ArchCodeChunk(
            scan_id=1,
            file_path="sandbox:d-drive/Orb/app/utils.py",
            chunk_type=ChunkType.FUNCTION,
            chunk_name="helper",
            signature=None,
            docstring="Helper function.",
        )
        
        descriptor = generate_chunk_descriptor(chunk)
        
        assert "def helper" in descriptor
        assert "Helper function" in descriptor


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0
    
    def test_words(self):
        text = "This is a test"
        tokens = estimate_tokens(text)
        assert tokens >= 4
