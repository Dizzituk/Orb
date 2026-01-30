# FILE: tests/test_file_discovery.py
"""Tests for file discovery system.

Tests the file_discovery module used by SpecGate to find files
for multi-file operations.

Note: Tests mock the SandboxClient since Windows Sandbox may not
be available in all test environments.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import Mock, MagicMock


# =============================================================================
# Dataclass Tests
# =============================================================================

class TestDataclasses:
    """Tests for file discovery dataclasses."""
    
    def test_line_match_creation(self):
        """Test LineMatch dataclass."""
        from app.pot_spec.grounded.file_discovery import LineMatch
        
        lm = LineMatch(line_number=42, line_content="# TODO: fix this")
        
        assert lm.line_number == 42
        assert lm.line_content == "# TODO: fix this"
    
    def test_line_match_to_dict(self):
        """Test LineMatch serialization."""
        from app.pot_spec.grounded.file_discovery import LineMatch
        
        lm = LineMatch(line_number=10, line_content="hello world")
        d = lm.to_dict()
        
        assert d == {"line_number": 10, "line_content": "hello world"}
    
    def test_file_match_creation(self):
        """Test FileMatch dataclass."""
        from app.pot_spec.grounded.file_discovery import FileMatch, LineMatch
        
        fm = FileMatch(
            path=r"D:\Orb\app\main.py",
            occurrence_count=3,
            line_matches=[
                LineMatch(line_number=10, line_content="# TODO: first"),
                LineMatch(line_number=25, line_content="# TODO: second"),
            ],
        )
        
        assert fm.path == r"D:\Orb\app\main.py"
        assert fm.occurrence_count == 3
        assert len(fm.line_matches) == 2
    
    def test_file_match_to_dict(self):
        """Test FileMatch serialization."""
        from app.pot_spec.grounded.file_discovery import FileMatch, LineMatch
        
        fm = FileMatch(
            path=r"D:\Orb\test.py",
            occurrence_count=1,
            line_matches=[LineMatch(line_number=5, line_content="test")],
        )
        d = fm.to_dict()
        
        assert d["path"] == r"D:\Orb\test.py"
        assert d["occurrence_count"] == 1
        assert len(d["line_matches"]) == 1
    
    def test_discovery_result_success(self):
        """Test DiscoveryResult for successful search."""
        from app.pot_spec.grounded.file_discovery import DiscoveryResult, FileMatch
        
        result = DiscoveryResult(
            success=True,
            search_pattern="TODO",
            total_files=5,
            total_occurrences=12,
            files=[FileMatch(path="test.py", occurrence_count=2)],
            truncated=False,
            duration_ms=150,
            roots_searched=[r"D:\Orb"],
        )
        
        assert result.success is True
        assert result.total_files == 5
        assert result.total_occurrences == 12
        assert len(result.files) == 1
    
    def test_discovery_result_error(self):
        """Test DiscoveryResult for failed search."""
        from app.pot_spec.grounded.file_discovery import DiscoveryResult
        
        result = DiscoveryResult(
            success=False,
            search_pattern="TODO",
            total_files=0,
            total_occurrences=0,
            error_message="Connection failed",
            roots_searched=[r"D:\Orb"],
        )
        
        assert result.success is False
        assert result.error_message == "Connection failed"
    
    def test_discovery_result_to_dict(self):
        """Test DiscoveryResult serialization."""
        from app.pot_spec.grounded.file_discovery import DiscoveryResult
        
        result = DiscoveryResult(
            success=True,
            search_pattern="DEBUG",
            total_files=2,
            total_occurrences=5,
            roots_searched=[r"D:\Orb"],
        )
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["search_pattern"] == "DEBUG"
        assert d["total_files"] == 2
    
    def test_discovery_result_preview(self):
        """Test human-readable preview generation."""
        from app.pot_spec.grounded.file_discovery import (
            DiscoveryResult, FileMatch, LineMatch
        )
        
        result = DiscoveryResult(
            success=True,
            search_pattern="TODO",
            total_files=2,
            total_occurrences=4,
            files=[
                FileMatch(
                    path=r"D:\Orb\app\main.py",
                    occurrence_count=2,
                    line_matches=[
                        LineMatch(line_number=10, line_content="# TODO: fix"),
                        LineMatch(line_number=20, line_content="# TODO: cleanup"),
                    ],
                ),
                FileMatch(
                    path=r"D:\Orb\app\router.py",
                    occurrence_count=2,
                    line_matches=[LineMatch(line_number=5, line_content="# TODO")],
                ),
            ],
            roots_searched=[r"D:\Orb"],
        )
        
        preview = result.get_file_preview(max_files=10)
        
        assert "TODO" in preview
        assert "2 files" in preview
        assert "4 occurrences" in preview
        assert "main.py" in preview


# =============================================================================
# discover_files() Tests
# =============================================================================

class TestDiscoverFiles:
    """Tests for discover_files() function."""
    
    def _make_mock_client(self, stdout: str, ok: bool = True, exit_code: int = 0) -> Mock:
        """Create mock SandboxClient with configured shell_run response."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.ok = ok
        mock_result.exit_code = exit_code
        mock_result.stdout = stdout
        mock_result.stderr = ""
        mock_result.duration_ms = 100
        mock_client.shell_run.return_value = mock_result
        return mock_client
    
    def test_discover_files_basic_pattern(self):
        """Test basic pattern search returns results."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        # Simulate Select-String output: path|line_number|line_content
        stdout = (
            r"D:\Orb\app\main.py|10|# TODO: fix this" + "\n"
            r"D:\Orb\app\main.py|25|# TODO: another" + "\n"
            r"D:\Orb\app\router.py|5|# TODO: refactor"
        )
        mock_client = self._make_mock_client(stdout)
        
        result = discover_files("TODO", mock_client)
        
        assert result.success is True
        assert result.total_files == 2
        assert result.total_occurrences == 3
        assert mock_client.shell_run.called
    
    def test_discover_files_empty_results(self):
        """Test handling of no matches."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        # Exit code 1 = no matches in Select-String (not an error)
        mock_client = self._make_mock_client("", ok=False, exit_code=1)
        
        result = discover_files("NONEXISTENT_PATTERN_12345", mock_client)
        
        # Should still succeed, just with 0 files
        # Note: exit code 1 with empty stdout is "no matches"
        assert result.total_files == 0
        assert result.total_occurrences == 0
    
    def test_discover_files_with_file_filter(self):
        """Test file extension filter is passed correctly."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        mock_client = self._make_mock_client(r"D:\Orb\test.py|1|match")
        
        result = discover_files("match", mock_client, file_filter="*.py")
        
        assert result.success is True
        # Verify the filter was included in the command
        call_args = mock_client.shell_run.call_args
        command = call_args[1]["command"] if "command" in call_args[1] else call_args[0][0]
        assert "*.py" in command
    
    def test_discover_files_max_results(self):
        """Test max_results truncation."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        # Generate more files than max_results
        lines = [f"D:\\Orb\\file{i}.py|1|match" for i in range(10)]
        stdout = "\n".join(lines)
        mock_client = self._make_mock_client(stdout)
        
        result = discover_files("match", mock_client, max_results=5)
        
        assert result.success is True
        assert result.total_files == 5
        assert result.truncated is True
    
    def test_discover_files_max_samples_per_file(self):
        """Test line sample limiting per file."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        # Same file with many matches
        lines = [f"D:\\Orb\\test.py|{i}|match line {i}" for i in range(10)]
        stdout = "\n".join(lines)
        mock_client = self._make_mock_client(stdout)
        
        result = discover_files("match", mock_client, max_samples_per_file=3)
        
        assert result.success is True
        assert result.total_files == 1
        assert result.total_occurrences == 10
        assert len(result.files[0].line_matches) == 3  # Limited to 3
    
    def test_discover_files_error_handling(self):
        """Test graceful error handling."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        # Simulate error (exit code 2 = actual error, not just "no matches")
        mock_client = self._make_mock_client("", ok=False, exit_code=2)
        mock_client.shell_run.return_value.stderr = "Access denied"
        
        result = discover_files("pattern", mock_client)
        
        assert result.success is False
        assert "error" in result.error_message.lower() or "Access denied" in result.error_message
    
    def test_discover_files_timeout(self):
        """Test timeout is passed to shell_run."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        mock_client = self._make_mock_client("")
        
        discover_files("pattern", mock_client, timeout_seconds=60)
        
        call_args = mock_client.shell_run.call_args
        assert call_args[1]["timeout_seconds"] == 60
    
    def test_discover_files_custom_roots(self):
        """Test custom search roots."""
        from app.pot_spec.grounded.file_discovery import discover_files
        
        mock_client = self._make_mock_client("")
        custom_roots = [r"C:\CustomPath"]
        
        result = discover_files("pattern", mock_client, roots=custom_roots)
        
        call_args = mock_client.shell_run.call_args
        command = call_args[1]["command"] if "command" in call_args[1] else call_args[0][0]
        assert "CustomPath" in command
        assert result.roots_searched == custom_roots


# =============================================================================
# discover_files_by_extension() Tests
# =============================================================================

class TestDiscoverFilesByExtension:
    """Tests for discover_files_by_extension() function."""
    
    def _make_mock_client(self, stdout: str) -> Mock:
        """Create mock SandboxClient."""
        mock_client = Mock()
        mock_result = Mock()
        mock_result.ok = True
        mock_result.exit_code = 0
        mock_result.stdout = stdout
        mock_result.stderr = ""
        mock_result.duration_ms = 50
        mock_client.shell_run.return_value = mock_result
        return mock_client
    
    def test_discover_by_extension_basic(self):
        """Test basic extension search."""
        from app.pot_spec.grounded.file_discovery import discover_files_by_extension
        
        stdout = (
            r"D:\Orb\app\main.py" + "\n"
            r"D:\Orb\app\router.py" + "\n"
            r"D:\Orb\tests\test_main.py"
        )
        mock_client = self._make_mock_client(stdout)
        
        result = discover_files_by_extension(".py", mock_client)
        
        assert result.success is True
        assert result.total_files == 3
        assert result.search_pattern == "*.py"
    
    def test_discover_by_extension_normalize(self):
        """Test extension normalization (handles various formats)."""
        from app.pot_spec.grounded.file_discovery import discover_files_by_extension
        
        mock_client = self._make_mock_client(r"D:\Orb\test.py")
        
        # All these should work the same
        for ext in [".py", "py", "*.py"]:
            result = discover_files_by_extension(ext, mock_client)
            assert result.search_pattern == "*.py"
    
    def test_discover_by_extension_empty(self):
        """Test empty results."""
        from app.pot_spec.grounded.file_discovery import discover_files_by_extension
        
        mock_client = self._make_mock_client("")
        
        result = discover_files_by_extension(".xyz", mock_client)
        
        assert result.success is True
        assert result.total_files == 0


# =============================================================================
# Parser Tests
# =============================================================================

class TestParsers:
    """Tests for output parsing functions."""
    
    def test_parse_select_string_output(self):
        """Test Select-String output parsing."""
        from app.pot_spec.grounded.file_discovery import _parse_select_string_output
        
        stdout = (
            r"D:\Orb\a.py|10|line content 1" + "\n"
            r"D:\Orb\a.py|20|line content 2" + "\n"
            r"D:\Orb\b.py|5|other content"
        )
        
        files, total, truncated = _parse_select_string_output(
            stdout, max_results=100, max_samples_per_file=5
        )
        
        assert len(files) == 2
        assert total == 3
        assert truncated is False
        
        # Check first file
        a_file = next(f for f in files if "a.py" in f.path)
        assert a_file.occurrence_count == 2
        assert len(a_file.line_matches) == 2
    
    def test_parse_select_string_malformed(self):
        """Test handling of malformed output lines."""
        from app.pot_spec.grounded.file_discovery import _parse_select_string_output
        
        stdout = (
            "malformed line without pipes\n"
            r"D:\Orb\good.py|10|valid line" + "\n"
            "another bad line"
        )
        
        files, total, truncated = _parse_select_string_output(
            stdout, max_results=100, max_samples_per_file=5
        )
        
        # Should only parse the valid line
        assert len(files) == 1
        assert total == 1
    
    def test_parse_file_list_output(self):
        """Test file list output parsing."""
        from app.pot_spec.grounded.file_discovery import _parse_file_list_output
        
        stdout = (
            r"D:\Orb\file1.py" + "\n"
            r"D:\Orb\file2.py" + "\n"
            r"D:\Orb\file3.py"
        )
        
        files, truncated = _parse_file_list_output(stdout, max_results=100)
        
        assert len(files) == 3
        assert truncated is False
        assert files[0].path == r"D:\Orb\file1.py"


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for module configuration defaults."""
    
    def test_default_roots(self):
        """Test default search roots are configured."""
        from app.pot_spec.grounded.file_discovery import DEFAULT_ROOTS
        
        assert len(DEFAULT_ROOTS) > 0
        assert any("Orb" in root for root in DEFAULT_ROOTS)
    
    def test_default_exclusions(self):
        """Test default exclusions are configured."""
        from app.pot_spec.grounded.file_discovery import DEFAULT_EXCLUSIONS
        
        assert ".git" in DEFAULT_EXCLUSIONS
        assert "node_modules" in DEFAULT_EXCLUSIONS
        assert "__pycache__" in DEFAULT_EXCLUSIONS
    
    def test_default_file_extensions(self):
        """Test default file extensions are configured."""
        from app.pot_spec.grounded.file_discovery import DEFAULT_FILE_EXTENSIONS
        
        assert ".py" in DEFAULT_FILE_EXTENSIONS
        assert ".js" in DEFAULT_FILE_EXTENSIONS


# =============================================================================
# Integration Test (requires actual sandbox)
# =============================================================================

@pytest.mark.skip(reason="Requires running Windows Sandbox - run manually")
class TestFileDiscoveryIntegration:
    """Integration tests requiring actual sandbox connection."""
    
    def test_real_discover_files(self):
        """Test actual file discovery against sandbox."""
        from app.overwatcher.sandbox_client import get_sandbox_client
        from app.pot_spec.grounded.file_discovery import discover_files
        
        client = get_sandbox_client()
        
        # Search for something that definitely exists
        result = discover_files("import", client, file_filter="*.py", max_results=10)
        
        assert result.success is True
        assert result.total_files > 0
        print(result.get_file_preview())
