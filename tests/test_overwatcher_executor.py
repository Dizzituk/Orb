# FILE: tests/test_overwatcher_executor.py
"""Unit tests for Overwatcher executor (Block 8)."""

import pytest
from app.overwatcher.schemas import Chunk, ChunkVerification, FileAction
from app.overwatcher.executor import (
    check_diff_boundaries,
    normalize_path,
    extract_files_from_output,
    build_implementation_prompt,
)


class TestNormalizePath:
    def test_forward_slashes(self):
        assert normalize_path("app/foo/bar.py") == "app/foo/bar.py"

    def test_backslashes(self):
        assert normalize_path("app\\foo\\bar.py") == "app/foo/bar.py"

    def test_strip_base(self):
        assert normalize_path("D:/Orb/app/foo.py", "D:/Orb") == "app/foo.py"

    def test_strip_base_with_backslashes(self):
        assert normalize_path("D:\\Orb\\app\\foo.py", "D:\\Orb") == "app/foo.py"


class TestCheckDiffBoundaries:
    def test_all_allowed(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test",
            objective="Test",
            allowed_files={
                "add": ["app/new.py"],
                "modify": ["app/existing.py"],
                "delete_candidates": [],
            },
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=["app/new.py"],
            files_modified=["app/existing.py"],
            files_deleted=[],
        )
        
        assert result.passed is True
        assert len(result.violations) == 0

    def test_unauthorized_add(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test",
            objective="Test",
            allowed_files={
                "add": ["app/allowed.py"],
                "modify": [],
                "delete_candidates": [],
            },
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=["app/unauthorized.py"],
            files_modified=[],
            files_deleted=[],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].action == "added"
        assert "unauthorized.py" in result.violations[0].file_path

    def test_unauthorized_modify(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test",
            objective="Test",
            allowed_files={
                "add": [],
                "modify": ["app/allowed.py"],
                "delete_candidates": [],
            },
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=[],
            files_modified=["app/forbidden.py"],
            files_deleted=[],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].action == "modified"

    def test_unauthorized_delete(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test",
            objective="Test",
            allowed_files={
                "add": [],
                "modify": [],
                "delete_candidates": ["app/ok_to_delete.py"],
            },
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=[],
            files_modified=[],
            files_deleted=["app/not_ok.py"],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].action == "deleted"

    def test_mixed_allowed_and_forbidden(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test",
            objective="Test",
            allowed_files={
                "add": ["app/new.py"],
                "modify": ["app/modify.py"],
                "delete_candidates": [],
            },
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=["app/new.py", "app/sneaky.py"],
            files_modified=["app/modify.py"],
            files_deleted=[],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert "sneaky.py" in result.violations[0].file_path


class TestExtractFilesFromOutput:
    def test_python_code_block(self):
        output = '''Here's the implementation:

```python:app/auth.py
def authenticate():
    pass
```

Done!'''
        
        files = extract_files_from_output(output)
        assert "app/auth.py" in files
        assert "def authenticate()" in files["app/auth.py"]

    def test_multiple_files(self):
        output = '''Files:

```python:app/foo.py
# foo
```

```python:app/bar.py
# bar
```
'''
        files = extract_files_from_output(output)
        assert len(files) == 2
        assert "app/foo.py" in files
        assert "app/bar.py" in files

    def test_file_header_pattern(self):
        output = '''# FILE: app/test.py
```python
def test():
    pass
```
'''
        files = extract_files_from_output(output)
        assert "app/test.py" in files

    def test_no_files(self):
        output = "Just some text without any code blocks."
        files = extract_files_from_output(output)
        assert len(files) == 0


class TestBuildImplementationPrompt:
    def test_prompt_contains_chunk_info(self):
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Add auth module",
            objective="Implement JWT authentication",
            allowed_files={
                "add": ["app/auth.py"],
                "modify": ["app/main.py"],
                "delete_candidates": [],
            },
            verification=ChunkVerification(
                commands=["pytest tests/test_auth.py"],
            ),
        )
        
        
        prompt = build_implementation_prompt(chunk)
        
        assert "CHUNK-001" in prompt
        assert "Add auth module" in prompt
        assert "app/auth.py" in prompt
        assert "app/main.py" in prompt
        assert "pytest tests/test_auth.py" in prompt
