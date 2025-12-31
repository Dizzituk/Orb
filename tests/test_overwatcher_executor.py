# FILE: tests/test_overwatcher_executor.py
"""
Tests for app/overwatcher/executor.py
Implementation executor with diff boundary enforcement.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Test Normalize Path
# ============================================================================

class TestNormalizePath:
    """Test path normalization."""
    
    def test_forward_slashes(self):
        """Test forward slashes are preserved."""
        from app.overwatcher.executor import normalize_path
        assert normalize_path("app/models/user.py") == "app/models/user.py"
    
    def test_backslashes(self):
        """Test backslashes are converted to forward slashes."""
        from app.overwatcher.executor import normalize_path
        assert normalize_path("app\\models\\user.py") == "app/models/user.py"
    
    def test_strip_base(self):
        """Test base directory is stripped."""
        from app.overwatcher.executor import normalize_path
        result = normalize_path("D:/Project/app/models/user.py", "D:/Project")
        assert result == "app/models/user.py"
    
    def test_strip_base_with_backslashes(self):
        """Test base stripping works with backslashes."""
        from app.overwatcher.executor import normalize_path
        result = normalize_path("D:\\Project\\app\\models\\user.py", "D:\\Project")
        assert result == "app/models/user.py"
    
    def test_empty_base(self):
        """Test empty base directory."""
        from app.overwatcher.executor import normalize_path
        assert normalize_path("app/models/user.py", "") == "app/models/user.py"
    
    def test_whitespace_stripped(self):
        """Test whitespace is stripped."""
        from app.overwatcher.executor import normalize_path
        assert normalize_path("  app/models/user.py  ") == "app/models/user.py"


# ============================================================================
# Test Check Diff Boundaries
# ============================================================================

class TestCheckDiffBoundaries:
    """Test diff boundary checking."""
    
    def _make_chunk(self, add=None, modify=None, delete=None):
        """Helper to create a mock chunk."""
        from app.overwatcher.schemas import Chunk, ChunkVerification
        
        return Chunk(
            chunk_id="chunk-1",
            title="Test Chunk",
            objective="Test objective",
            allowed_files={
                "add": add or [],
                "modify": modify or [],
                "delete_candidates": delete or [],
            },
            steps=[],
            verification=ChunkVerification(commands=[]),
        )
    
    def test_all_allowed(self):
        """Test all changes within boundaries passes."""
        from app.overwatcher.executor import check_diff_boundaries
        
        chunk = self._make_chunk(
            add=["new_file.py"],
            modify=["existing.py"],
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=["new_file.py"],
            files_modified=["existing.py"],
            files_deleted=[],
        )
        
        assert result.passed is True
        assert len(result.violations) == 0
    
    def test_unauthorized_add(self):
        """Test unauthorized file addition is caught."""
        from app.overwatcher.executor import check_diff_boundaries
        
        chunk = self._make_chunk(add=["allowed.py"])
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=["unauthorized.py"],
            files_modified=[],
            files_deleted=[],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].action == "added"
    
    def test_unauthorized_modify(self):
        """Test unauthorized file modification is caught."""
        from app.overwatcher.executor import check_diff_boundaries
        
        chunk = self._make_chunk(modify=["allowed.py"])
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=[],
            files_modified=["unauthorized.py"],
            files_deleted=[],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].action == "modified"
    
    def test_unauthorized_delete(self):
        """Test unauthorized file deletion is caught."""
        from app.overwatcher.executor import check_diff_boundaries
        
        chunk = self._make_chunk(delete=["allowed.py"])
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=[],
            files_modified=[],
            files_deleted=["unauthorized.py"],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].action == "deleted"
    
    def test_mixed_allowed_and_unauthorized(self):
        """Test mix of allowed and unauthorized changes."""
        from app.overwatcher.executor import check_diff_boundaries
        
        chunk = self._make_chunk(
            add=["allowed_new.py"],
            modify=["allowed_edit.py"],
        )
        
        result = check_diff_boundaries(
            chunk=chunk,
            files_added=["allowed_new.py", "unauthorized_new.py"],
            files_modified=["allowed_edit.py"],
            files_deleted=[],
        )
        
        assert result.passed is False
        assert len(result.violations) == 1


# ============================================================================
# Test Extract Files From Output
# ============================================================================

class TestExtractFilesFromOutput:
    """Test file extraction from LLM output."""
    
    def test_python_code_block(self):
        """Test extraction of Python code block."""
        from app.overwatcher.executor import extract_files_from_output
        
        output = """# FILE: app/models.py
```python
class User:
    pass
```"""
        
        files = extract_files_from_output(output)
        assert "app/models.py" in files
        assert "class User:" in files["app/models.py"]
    
    def test_multiple_files(self):
        """Test extraction of multiple files."""
        from app.overwatcher.executor import extract_files_from_output
        
        output = """# FILE: app/models.py
```python
class User:
    pass
```

# FILE: app/views.py
```python
def index():
    return "Hello"
```"""
        
        files = extract_files_from_output(output)
        assert len(files) == 2
        assert "app/models.py" in files
        assert "app/views.py" in files
    
    def test_file_header_with_spaces(self):
        """Test file header with extra spaces."""
        from app.overwatcher.executor import extract_files_from_output
        
        output = """#   FILE:   app/models.py  
```python
class User:
    pass
```"""
        
        files = extract_files_from_output(output)
        assert "app/models.py" in files
    
    def test_no_files(self):
        """Test output with no files returns empty dict."""
        from app.overwatcher.executor import extract_files_from_output
        
        output = "Just some text without any file markers."
        files = extract_files_from_output(output)
        assert files == {}


# ============================================================================
# Test Parse Spec Headers
# ============================================================================

class TestParseSpecHeaders:
    """Test SPEC_ID and SPEC_HASH header parsing."""
    
    def test_valid_headers(self):
        """Test parsing valid spec headers."""
        from app.overwatcher.executor import parse_spec_headers
        
        output = """SPEC_ID: spec-123
SPEC_HASH: abc123def456
Some other content here."""
        
        spec_id, spec_hash = parse_spec_headers(output)
        assert spec_id == "spec-123"
        assert spec_hash == "abc123def456"
    
    def test_missing_headers(self):
        """Test handling missing headers."""
        from app.overwatcher.executor import parse_spec_headers
        
        output = "Just some content without headers."
        spec_id, spec_hash = parse_spec_headers(output)
        assert spec_id is None
        assert spec_hash is None
    
    def test_partial_headers(self):
        """Test handling partial headers."""
        from app.overwatcher.executor import parse_spec_headers
        
        output = """SPEC_ID: spec-123
Some content but no hash."""
        
        spec_id, spec_hash = parse_spec_headers(output)
        assert spec_id == "spec-123"
        assert spec_hash is None
    
    def test_empty_output(self):
        """Test handling empty output."""
        from app.overwatcher.executor import parse_spec_headers
        
        spec_id, spec_hash = parse_spec_headers("")
        assert spec_id is None
        assert spec_hash is None
    
    def test_none_output(self):
        """Test handling None output."""
        from app.overwatcher.executor import parse_spec_headers
        
        spec_id, spec_hash = parse_spec_headers(None)
        assert spec_id is None
        assert spec_hash is None


# ============================================================================
# Test Build Implementation Prompt
# ============================================================================

class TestBuildImplementationPrompt:
    """Test implementation prompt building."""
    
    def _make_chunk(self):
        """Helper to create a test chunk."""
        from app.overwatcher.schemas import Chunk, ChunkStep, ChunkVerification, FileAction
        
        return Chunk(
            chunk_id="chunk-1",
            title="Add User Model",
            objective="Create the user model with basic fields",
            allowed_files={
                "add": ["app/models/user.py"],
                "modify": ["app/models/__init__.py"],
                "delete_candidates": [],
            },
            steps=[
                ChunkStep(
                    step_id="step-1",
                    description="Create the User model class",
                    file_path="app/models/user.py",
                    action=FileAction.ADD,
                    details="Include id, name, email fields",
                ),
            ],
            verification=ChunkVerification(
                commands=["pytest tests/test_user.py", "ruff check app/models/"],
            ),
        )
    
    def test_prompt_contains_chunk_info(self):
        """Test prompt includes chunk information."""
        from app.overwatcher.executor import build_implementation_prompt
        
        chunk = self._make_chunk()
        spec_id = "spec-abc123"
        spec_hash = "hash-def456"
        
        system, user = build_implementation_prompt(chunk, spec_id, spec_hash)
        
        # System prompt should contain spec echo instructions
        assert "SPEC_ID" in system
        assert "SPEC_HASH" in system
        assert spec_id in system
        assert spec_hash in system
        
        # User prompt should contain chunk details
        assert chunk.chunk_id in user
        assert chunk.title in user
        assert chunk.objective in user
    
    def test_prompt_contains_allowed_files(self):
        """Test prompt includes allowed files."""
        from app.overwatcher.executor import build_implementation_prompt
        
        chunk = self._make_chunk()
        _, user = build_implementation_prompt(chunk, "spec-1", "hash-1")
        
        assert "app/models/user.py" in user
        assert "app/models/__init__.py" in user
    
    def test_prompt_contains_verification_commands(self):
        """Test prompt includes verification commands."""
        from app.overwatcher.executor import build_implementation_prompt
        
        chunk = self._make_chunk()
        _, user = build_implementation_prompt(chunk, "spec-1", "hash-1")
        
        assert "pytest tests/test_user.py" in user
        assert "ruff check" in user
    
    def test_prompt_contains_steps(self):
        """Test prompt includes steps."""
        from app.overwatcher.executor import build_implementation_prompt
        
        chunk = self._make_chunk()
        _, user = build_implementation_prompt(chunk, "spec-1", "hash-1")
        
        assert "Create the User model class" in user
    
    def test_prompt_returns_tuple(self):
        """Test function returns (system, user) tuple."""
        from app.overwatcher.executor import build_implementation_prompt
        
        chunk = self._make_chunk()
        result = build_implementation_prompt(chunk, "spec-1", "hash-1")
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)  # system prompt
        assert isinstance(result[1], str)  # user prompt


# ============================================================================
# Test Configuration Constants
# ============================================================================

class TestConfiguration:
    """Test configuration constants."""
    
    def test_implementer_provider_defined(self):
        """Test implementer provider is defined."""
        from app.overwatcher.executor import IMPLEMENTER_PROVIDER
        assert IMPLEMENTER_PROVIDER is not None
    
    def test_implementer_model_defined(self):
        """Test implementer model is defined."""
        from app.overwatcher.executor import IMPLEMENTER_MODEL
        assert IMPLEMENTER_MODEL is not None
    
    def test_fallback_model_defined(self):
        """Test fallback model is defined."""
        from app.overwatcher.executor import IMPLEMENTER_FALLBACK_MODEL
        assert IMPLEMENTER_FALLBACK_MODEL is not None
    
    def test_max_output_tokens_defined(self):
        """Test max output tokens is defined."""
        from app.overwatcher.executor import IMPLEMENTER_MAX_OUTPUT_TOKENS
        assert IMPLEMENTER_MAX_OUTPUT_TOKENS > 0


# ============================================================================
# Test Rollback Functions
# ============================================================================

class TestRollbackSupport:
    """Test rollback support functions."""
    
    def test_create_backup_exists(self):
        """Test create_backup function exists."""
        from app.overwatcher.executor import create_backup
        assert callable(create_backup)
    
    def test_rollback_chunk_exists(self):
        """Test rollback_chunk function exists."""
        from app.overwatcher.executor import rollback_chunk
        assert callable(rollback_chunk)


# ============================================================================
# Test Execute Chunk (async)
# ============================================================================

class TestExecuteChunk:
    """Test chunk execution function."""
    
    def test_execute_chunk_exists(self):
        """Test execute_chunk function exists."""
        from app.overwatcher.executor import execute_chunk
        assert callable(execute_chunk)
    
    def test_execute_chunk_is_async(self):
        """Test execute_chunk is an async function."""
        import asyncio
        from app.overwatcher.executor import execute_chunk
        
        assert asyncio.iscoroutinefunction(execute_chunk)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
