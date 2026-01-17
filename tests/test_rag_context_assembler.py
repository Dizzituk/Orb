# tests/test_rag_context_assembler.py
"""Tests for context assembler."""

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

from app.rag.retrieval.context_assembler import (
    ContextAssembler,
    AssembledContext,
)
from app.rag.retrieval.arch_search import (
    ArchSearchResult,
    ArchSearchResponse,
)
from app.rag.models import SourceType
from app.astra_memory.preference_models import IntentDepth


class TestContextAssembler:
    def test_respects_token_limit(self):
        # Create many results with large signatures/docstrings
        results = [
            ArchSearchResult(
                source_type=SourceType.ARCH_CHUNK,
                source_id=i,
                score=0.9 - i * 0.01,
                content="test",
                canonical_path=f"sandbox:d-drive/file{i}.py",
                name=f"function_{i}",
                signature=f"def function_{i}(" + ", ".join([f"arg{j}: str" for j in range(10)]) + ") -> dict",
                docstring="This is a long docstring. " * 20,
            )
            for i in range(20)
        ]
        
        response = ArchSearchResponse(
            query="test",
            intent_depth=IntentDepth.D2,
            results=results,
            total_searched=20,
            directories_found=0,
            chunks_found=20,
        )
        
        assembler = ContextAssembler(max_tokens=500)
        context = assembler.assemble(response)
        
        assert context.total_tokens <= 550
        assert context.truncated is True
    
    def test_directories_first(self):
        results = [
            ArchSearchResult(
                source_type=SourceType.ARCH_CHUNK,
                source_id=1,
                score=0.9,
                content="chunk content",
                canonical_path="sandbox:d-drive/file.py",
                name="function",
            ),
            ArchSearchResult(
                source_type=SourceType.ARCH_DIRECTORY,
                source_id=1,
                score=0.8,
                content="directory content",
                canonical_path="sandbox:d-drive/app",
                name="app",
            ),
        ]
        
        response = ArchSearchResponse(
            query="test",
            intent_depth=IntentDepth.D2,
            results=results,
            total_searched=2,
            directories_found=1,
            chunks_found=1,
        )
        
        assembler = ContextAssembler()
        context = assembler.assemble(response)
        
        # Directories should come before chunks
        dir_pos = context.text.find("Directories")
        code_pos = context.text.find("Code")
        
        if dir_pos >= 0 and code_pos >= 0:
            assert dir_pos < code_pos
    
    def test_includes_line_numbers(self):
        results = [
            ArchSearchResult(
                source_type=SourceType.ARCH_CHUNK,
                source_id=1,
                score=0.9,
                content="test",
                canonical_path="sandbox:d-drive/file.py",
                name="main",
                start_line=10,
                end_line=50,
            ),
        ]
        
        response = ArchSearchResponse(
            query="test",
            intent_depth=IntentDepth.D2,
            results=results,
            total_searched=1,
            directories_found=0,
            chunks_found=1,
        )
        
        assembler = ContextAssembler()
        context = assembler.assemble(response)
        
        assert ":10" in context.text
    
    def test_empty_results(self):
        response = ArchSearchResponse(
            query="test",
            intent_depth=IntentDepth.D2,
            results=[],
            total_searched=0,
            directories_found=0,
            chunks_found=0,
        )
        
        assembler = ContextAssembler()
        context = assembler.assemble(response)
        
        assert context.text == ""
        assert context.total_tokens == 0
        assert context.truncated is False
    
    def test_includes_signature(self):
        results = [
            ArchSearchResult(
                source_type=SourceType.ARCH_CHUNK,
                source_id=1,
                score=0.9,
                content="test",
                canonical_path="sandbox:d-drive/file.py",
                name="process",
                signature="def process(data: dict) -> bool",
            ),
        ]
        
        response = ArchSearchResponse(
            query="test",
            intent_depth=IntentDepth.D2,
            results=results,
            total_searched=1,
            directories_found=0,
            chunks_found=1,
        )
        
        assembler = ContextAssembler()
        context = assembler.assemble(response)
        
        assert "def process" in context.text
        assert "```python" in context.text
