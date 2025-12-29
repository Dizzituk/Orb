# FILE: tests/test_sandbox.py
"""Tests for Overwatcher sandbox integration.

Tests the sandbox bridge modules:
- SandboxClient
- SandboxVerifier
- SandboxExecutor
- EvidenceLoader

Note: Most tests mock the sandbox HTTP calls since Windows Sandbox
may not be available in all test environments.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import pytest
from unittest.mock import Mock, patch, MagicMock

# =============================================================================
# SandboxClient Tests
# =============================================================================

class TestSandboxClient:
    """Tests for SandboxClient HTTP wrapper."""
    
    def test_discover_base_url_default(self):
        """Test URL discovery falls back to default."""
        from app.overwatcher.sandbox_client import SandboxClient
        
        with patch("builtins.open", side_effect=FileNotFoundError):
            client = SandboxClient(base_url=None)
            assert "192.168" in client.base_url or "127.0.0.1" in client.base_url
    
    def test_health_parse(self):
        """Test health response parsing."""
        from app.overwatcher.sandbox_client import SandboxClient, HealthResponse
        
        mock_response = {
            "status": "ok",
            "repo_root": "C:\\Orb",
            "cache_root": "C:\\Orb\\_sandbox_cache",
            "artifact_root": "C:\\Orb\\_sandbox_cache\\jobs",
            "scratch_root": "C:\\Orb\\_sandbox_cache\\scratch",
            "version": "0.2.0",
        }
        
        client = SandboxClient(base_url="http://localhost:8765")
        
        with patch.object(client, "_request", return_value=mock_response):
            health = client.health()
            
            assert health.status == "ok"
            assert health.repo_root == "C:\\Orb"
            assert health.version == "0.2.0"
            assert client._connected is True
    
    def test_repo_tree_parse(self):
        """Test repo tree response parsing."""
        from app.overwatcher.sandbox_client import SandboxClient, FileEntry
        
        mock_response = [
            {"path": "app/main.py", "size_bytes": 1000, "sha256": "abc123"},
            {"path": "app/router.py", "size_bytes": 500},
        ]
        
        client = SandboxClient(base_url="http://localhost:8765")
        
        with patch.object(client, "_request", return_value=mock_response):
            tree = client.repo_tree(include_hashes=True)
            
            assert len(tree) == 2
            assert tree[0].path == "app/main.py"
            assert tree[0].size_bytes == 1000
            assert tree[0].sha256 == "abc123"
            assert tree[1].sha256 is None
    
    def test_shell_result_parse(self):
        """Test shell run response parsing."""
        from app.overwatcher.sandbox_client import SandboxClient, ShellResult
        
        mock_response = {
            "ok": True,
            "exit_code": 0,
            "duration_ms": 150,
            "stdout": "5 passed in 1.2s",
            "stderr": "",
        }
        
        client = SandboxClient(base_url="http://localhost:8765")
        
        with patch.object(client, "_request", return_value=mock_response):
            result = client.shell_run("pytest tests/")
            
            assert result.ok is True
            assert result.exit_code == 0
            assert result.duration_ms == 150
            assert "passed" in result.stdout


# =============================================================================
# SandboxVerifier Tests
# =============================================================================

class TestSandboxVerifier:
    """Tests for sandbox verification functions."""
    
    def test_shell_to_command_result(self):
        """Test conversion from ShellResult to CommandResult."""
        from app.overwatcher.sandbox_client import ShellResult
        from app.overwatcher.sandbox_verifier import shell_to_command_result
        
        shell_result = ShellResult(
            ok=True,
            exit_code=0,
            duration_ms=100,
            stdout="test output",
            stderr="",
        )
        
        cmd_result = shell_to_command_result(shell_result, "pytest tests/")
        
        assert cmd_result.command == "pytest tests/"
        assert cmd_result.exit_code == 0
        assert cmd_result.passed is True
        assert cmd_result.stdout == "test output"
    
    def test_parse_pytest_counts(self):
        """Test pytest output parsing."""
        from app.overwatcher.sandbox_verifier import parse_pytest_counts
        
        output = """
        ===== test session starts =====
        collected 10 items
        
        tests/test_foo.py .....F....
        
        ===== 8 passed, 2 failed in 1.5s =====
        """
        
        passed, failed = parse_pytest_counts(output)
        assert passed == 8
        assert failed == 2
    
    def test_parse_pytest_all_pass(self):
        """Test pytest output with all passing."""
        from app.overwatcher.sandbox_verifier import parse_pytest_counts
        
        output = "===== 15 passed in 0.5s ====="
        
        passed, failed = parse_pytest_counts(output)
        assert passed == 15
        assert failed == 0
    
    def test_parse_lint_error_count(self):
        """Test lint output error counting."""
        from app.overwatcher.sandbox_verifier import parse_lint_error_count
        
        output = """app/foo.py:10:5: E501 line too long
app/bar.py:20:1: F401 unused import
app/baz.py:30:10: E302 expected 2 blank lines"""
        
        count = parse_lint_error_count(output)
        assert count == 3
    
    def test_parse_type_error_count(self):
        """Test mypy output error counting."""
        from app.overwatcher.sandbox_verifier import parse_type_error_count
        
        output = """app/foo.py:10: error: Incompatible types
app/bar.py:20: error: Missing return statement
app/baz.py:30: note: See above for context
Found 2 errors in 2 files"""
        
        count = parse_type_error_count(output)
        assert count == 2


# =============================================================================
# SandboxExecutor Tests
# =============================================================================

class TestSandboxExecutor:
    """Tests for sandbox executor functions."""
    
    def test_check_sandbox_boundaries_pass(self):
        """Test boundary check passes for allowed files."""
        from app.overwatcher.sandbox_executor import check_sandbox_boundaries
        from app.overwatcher.schemas import Chunk, ChunkVerification
        from app.overwatcher.sandbox_client import FileEntry
        
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test chunk",
            objective="Test",
            allowed_files={
                "add": ["app/new_file.py"],
                "modify": ["app/existing.py"],
                "delete_candidates": [],
            },
            verification=ChunkVerification(),
        )
        
        files_to_write = {
            "app/new_file.py": "# new file",
            "app/existing.py": "# modified",
        }
        
        existing_tree = {
            "app/existing.py": FileEntry(path="app/existing.py", size_bytes=100),
        }
        
        result = check_sandbox_boundaries(chunk, files_to_write, existing_tree)
        
        assert result.passed is True
        assert len(result.violations) == 0
        assert "app/new_file.py" in result.files_added
        assert "app/existing.py" in result.files_modified
    
    def test_check_sandbox_boundaries_violation(self):
        """Test boundary check fails for disallowed files."""
        from app.overwatcher.sandbox_executor import check_sandbox_boundaries
        from app.overwatcher.schemas import Chunk, ChunkVerification
        from app.overwatcher.sandbox_client import FileEntry
        
        chunk = Chunk(
            chunk_id="CHUNK-001",
            title="Test chunk",
            objective="Test",
            allowed_files={
                "add": ["app/allowed.py"],
                "modify": [],
                "delete_candidates": [],
            },
            verification=ChunkVerification(),
        )
        
        files_to_write = {
            "app/allowed.py": "# allowed",
            "app/forbidden.py": "# not allowed!",
        }
        
        existing_tree = {}
        
        result = check_sandbox_boundaries(chunk, files_to_write, existing_tree)
        
        assert result.passed is False
        assert len(result.violations) == 1
        assert result.violations[0].file_path == "app/forbidden.py"


# =============================================================================
# EvidenceLoader Tests
# =============================================================================

class TestEvidenceLoader:
    """Tests for evidence pack loading."""
    
    def test_evidence_pack_get_symbol_by_name(self):
        """Test symbol lookup by name."""
        from app.overwatcher.evidence_loader import EvidencePack, Symbol
        
        pack = EvidencePack()
        pack.symbols = [
            Symbol(name="foo", kind="function", line=10, file_path="app/main.py"),
            Symbol(name="Bar", kind="class", line=20, file_path="app/main.py"),
        ]
        
        sym = pack.get_symbol_by_name("foo")
        assert sym is not None
        assert sym.kind == "function"
        
        missing = pack.get_symbol_by_name("nonexistent")
        assert missing is None
    
    def test_evidence_pack_get_symbols_in_file(self):
        """Test getting symbols by file."""
        from app.overwatcher.evidence_loader import EvidencePack, Symbol
        
        pack = EvidencePack()
        pack.symbols = [
            Symbol(name="a", kind="function", line=1, file_path="app/foo.py"),
            Symbol(name="b", kind="function", line=2, file_path="app/foo.py"),
            Symbol(name="c", kind="function", line=3, file_path="app/bar.py"),
        ]
        
        foo_syms = pack.get_symbols_in_file("app/foo.py")
        assert len(foo_syms) == 2
        
        bar_syms = pack.get_symbols_in_file("app/bar.py")
        assert len(bar_syms) == 1
    
    def test_evidence_pack_is_human_review_required(self):
        """Test human review path checking."""
        from app.overwatcher.evidence_loader import EvidencePack
        
        pack = EvidencePack()
        pack.human_review_paths = [
            "app/crypto/encryption.py",
            "app/auth/login.py",
        ]
        
        assert pack.is_human_review_required("app/crypto/encryption.py") is True
        assert pack.is_human_review_required("app/regular/file.py") is False
    
    def test_check_modification_safety(self):
        """Test modification safety check."""
        from app.overwatcher.evidence_loader import (
            EvidencePack, 
            TestMapping, 
            CoChangeHint,
            Invariant,
            check_modification_safety
        )
        
        pack = EvidencePack()
        pack.human_review_paths = ["app/crypto/key.py"]
        pack.test_mappings = [
            TestMapping(source_path="app/router.py", test_paths=["tests/test_router.py"]),
        ]
        pack.co_change_hints = [
            CoChangeHint(file_a="app/router.py", file_b="app/routes.py", reason="Always change together"),
        ]
        pack.invariants = [
            Invariant(name="key_length", description="Key must be 256 bits", check_paths=["app/crypto/key.py"]),
        ]
        
        # Check safe modification
        result = check_modification_safety(pack, ["app/router.py"])
        
        assert len(result["requires_human_review"]) == 0
        assert "tests/test_router.py" in result["affected_tests"]
        assert len(result["co_change_warnings"]) == 1  # Should warn about routes.py
        
        # Check modification requiring human review
        result = check_modification_safety(pack, ["app/crypto/key.py"])
        
        assert "app/crypto/key.py" in result["requires_human_review"]
        assert len(result["invariant_checks"]) == 1
    
    def test_to_architecture_context(self):
        """Test architecture context generation."""
        from app.overwatcher.evidence_loader import EvidencePack, Symbol, Route
        
        pack = EvidencePack()
        pack.generated_at = "2024-01-01T00:00:00"
        pack.repo_root = "D:\\Orb"
        pack.all_paths = ["app/main.py", "app/router.py"]
        pack.top_level_counts = {"app": 2, "tests": 1}
        pack.symbols = [
            Symbol(name="main", kind="function", line=10, file_path="app/main.py"),
        ]
        pack.routes = [
            Route(path="/api/chat", method="POST", handler="chat_handler", handler_line=20, file_path="app/router.py"),
        ]
        
        context = pack.to_architecture_context()
        
        assert "Repository Evidence Summary" in context
        assert "D:\\Orb" in context
        assert "app: 2 files" in context
        assert "/api/chat" in context


# =============================================================================
# Integration Placeholder
# =============================================================================

class TestSandboxIntegration:
    """Integration tests (require actual sandbox)."""
    
    @pytest.mark.skip(reason="Requires running Windows Sandbox")
    def test_full_sandbox_cycle(self):
        """Test full sandbox write -> verify cycle."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
