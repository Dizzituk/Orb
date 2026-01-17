# tests/test_rag_content_capture.py
"""Tests for content capture."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import tempfile
import os

from app.rag.capture.content_capture import (
    capture_file_content,
    is_binary_content,
    is_capturable_file,
)


class TestBinaryDetection:
    def test_text_not_binary(self):
        data = b"Hello, world!\nThis is text.\n"
        assert is_binary_content(data) is False
    
    def test_null_byte_is_binary(self):
        data = b"Hello\x00World"
        assert is_binary_content(data) is True


class TestCapturableFile:
    def test_py_capturable(self):
        assert is_capturable_file("main.py") is True
    
    def test_ts_capturable(self):
        assert is_capturable_file("app.tsx") is True
    
    def test_exe_not_capturable(self):
        assert is_capturable_file("app.exe") is False
    
    def test_dockerfile_capturable(self):
        assert is_capturable_file("Dockerfile") is True


class TestCapture:
    def test_capture_py_file(self, tmp_path):
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def main():\n    pass\n")
        
        result = capture_file_content(
            str(test_file),
            "test.py",
            test_file.stat().st_size,
            "repo",
        )
        
        assert result["success"] is True
        assert "def main" in result["content"]
        assert result["line_count"] == 3
    
    def test_redaction(self, tmp_path):
        # Create file with secret
        test_file = tmp_path / "config.py"
        test_file.write_text("api_key = 'sk-abcdefghijklmnopqrstuvwxyz123456'\n")
        
        result = capture_file_content(
            str(test_file),
            "config.py",
            test_file.stat().st_size,
            "repo",
        )
        
        assert result["success"] is True
        assert "sk-abc" not in result["content"]
        assert "[REDACTED" in result["content"]
        assert result["redaction_count"] >= 1
    
    def test_size_limit(self, tmp_path):
        # Create large file
        test_file = tmp_path / "large.txt"
        test_file.write_text("x" * (200 * 1024))  # 200KB
        
        result = capture_file_content(
            str(test_file),
            "large.txt",
            test_file.stat().st_size,
            "user",  # 100KB limit
        )
        
        assert result["success"] is False
        assert "large" in result["skip_reason"].lower()
