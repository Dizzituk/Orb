# tests/test_rag_directory_summary.py
"""Tests for directory summary generation."""

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

from app.rag.indexing.directory_summary import (
    generate_directory_summary,
    extension_to_language,
    format_lines,
    estimate_tokens,
)
from app.rag.models import ArchDirectoryIndex


class TestExtensionMapping:
    def test_python(self):
        assert extension_to_language(".py") == "Python"
    
    def test_typescript(self):
        assert extension_to_language(".ts") == "TypeScript"
    
    def test_unknown(self):
        result = extension_to_language(".xyz")
        assert "XYZ" in result


class TestFormatLines:
    def test_small(self):
        assert format_lines(500) == "500"
    
    def test_thousands(self):
        assert "K" in format_lines(5000)
    
    def test_millions(self):
        assert "M" in format_lines(1500000)


class TestGenerateSummary:
    def test_basic(self):
        directory = ArchDirectoryIndex(
            canonical_path="sandbox:d-drive/Orb/app/llm",
            name="llm",
            file_count=28,
            total_lines=18500,
            extensions_json='{ ".py": 25, ".md": 3 }',
        )
        
        summary = generate_directory_summary(directory)
        
        assert "sandbox:d-drive/Orb/app/llm/" in summary
        assert "28 files" in summary
        assert "Python" in summary
    
    def test_no_extensions(self):
        directory = ArchDirectoryIndex(
            canonical_path="sandbox:d-drive/Orb/empty",
            name="empty",
            file_count=0,
            total_lines=0,
            extensions_json=None,
        )
        
        summary = generate_directory_summary(directory)
        assert "sandbox:d-drive/Orb/empty/" in summary
        assert "0 files" in summary
    
    def test_zero_lines(self):
        directory = ArchDirectoryIndex(
            canonical_path="sandbox:d-drive/Orb/config",
            name="config",
            file_count=5,
            total_lines=0,
            extensions_json='{ ".json": 5 }',
        )
        
        summary = generate_directory_summary(directory)
        assert "5 files" in summary
        # Should not include lines info when zero
        assert "0 lines" not in summary


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0
    
    def test_none(self):
        assert estimate_tokens(None) == 0
    
    def test_words(self):
        text = "This is a test sentence with several words"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens > len(text.split())  # Should be more than word count
