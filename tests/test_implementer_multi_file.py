# FILE: tests/test_implementer_multi_file.py
"""
Tests for Implementer Multi-File Operations (v1.11 - Phase 5)

Tests:
- run_multi_file_search() - read-only search across multiple files
- run_multi_file_refactor() - batch search/replace with verification
- run_multi_file_operation() - dispatcher function
- Progress callbacks
- Error handling and consecutive error limits

Version Notes:
-------------
v1.0 (2026-01-28): Initial Phase 5 tests
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from dataclasses import dataclass


class TestMultiFileSearchBasic:
    """Tests for run_multi_file_search() basic functionality."""
    
    @pytest.mark.asyncio
    async def test_search_returns_discovery_results(self):
        """Search operation returns pre-computed discovery results."""
        from app.overwatcher.implementer import run_multi_file_search
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "search",
            "search_pattern": "TODO",
            "total_files": 15,
            "total_occurrences": 42,
            "target_files": ["app/main.py", "app/router.py"],
            "file_preview": "app/main.py:10: # TODO fix this",
        }
        
        result = await run_multi_file_search(multi_file=multi_file)
        
        assert result.success is True
        assert result.operation == "search"
        assert result.total_files == 15
        assert result.search_pattern == "TODO"
        assert result.files_modified == 0  # Read-only
    
    @pytest.mark.asyncio
    async def test_search_not_multi_file(self):
        """Returns error if not a multi-file operation."""
        from app.overwatcher.implementer import run_multi_file_search
        
        multi_file = {"is_multi_file": False}
        
        result = await run_multi_file_search(multi_file=multi_file)
        
        assert result.success is False
        assert "Not a multi-file operation" in result.error
    
    @pytest.mark.asyncio
    async def test_search_with_progress_callback(self):
        """Progress callback is called on completion."""
        from app.overwatcher.implementer import run_multi_file_search
        
        progress_calls = []
        def track_progress(data):
            progress_calls.append(data)
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "search",
            "search_pattern": "TEST",
            "total_files": 5,
            "total_occurrences": 10,
        }
        
        await run_multi_file_search(
            multi_file=multi_file,
            progress_callback=track_progress
        )
        
        assert len(progress_calls) >= 1
        assert any(p.get("type") == "complete" for p in progress_calls)


class TestMultiFileRefactorBasic:
    """Tests for run_multi_file_refactor() basic functionality."""
    
    @pytest.mark.asyncio
    async def test_refactor_requires_confirmation(self):
        """Refactor fails if not confirmed."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "requires_confirmation": True,
            "confirmed": False,
            "search_pattern": "Orb",
            "replacement_pattern": "Astra",
            "target_files": ["app/main.py"],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file)
        
        assert result.success is False
        assert "requires confirmation" in result.error
        assert result.awaiting_confirmation is True
    
    @pytest.mark.asyncio
    async def test_refactor_no_target_files(self):
        """Refactor fails with no target files."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "test",
            "replacement_pattern": "TEST",
            "target_files": [],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file)
        
        assert result.success is False
        assert "No target files" in result.error
    
    @pytest.mark.asyncio
    async def test_refactor_no_search_pattern(self):
        """Refactor fails with no search pattern."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "",
            "replacement_pattern": "TEST",
            "target_files": ["app/main.py"],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file)
        
        assert result.success is False
        assert "No search pattern" in result.error
    
    @pytest.mark.asyncio
    async def test_refactor_not_multi_file(self):
        """Returns error if not a multi-file operation."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        multi_file = {"is_multi_file": False}
        
        result = await run_multi_file_refactor(multi_file=multi_file)
        
        assert result.success is False
        assert "Not a multi-file operation" in result.error


class TestMultiFileRefactorWithSandbox:
    """Tests for run_multi_file_refactor() with mocked sandbox client."""
    
    @pytest.mark.asyncio
    async def test_refactor_processes_all_files(self):
        """Refactor processes each file in target_files."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        # Mock sandbox client
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        
        # Simulate: read returns "Hello Orb!", write succeeds, verify returns "Hello Astra!"
        mock_client.shell_run = MagicMock(side_effect=[
            # File 1: Read
            MagicMock(exit_code=0, stdout="Hello Orb!", stderr=""),
            # File 1: Write
            MagicMock(exit_code=0, stdout="", stderr=""),
            # File 1: Verify
            MagicMock(exit_code=0, stdout="Hello Astra!", stderr=""),
            # File 2: Read
            MagicMock(exit_code=0, stdout="Welcome to Orb", stderr=""),
            # File 2: Write
            MagicMock(exit_code=0, stdout="", stderr=""),
            # File 2: Verify
            MagicMock(exit_code=0, stdout="Welcome to Astra", stderr=""),
        ])
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "requires_confirmation": True,
            "confirmed": True,
            "search_pattern": "Orb",
            "replacement_pattern": "Astra",
            "target_files": ["app/main.py", "app/config.py"],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file, client=mock_client)
        
        assert result.success is True
        assert result.files_modified == 2
        assert result.total_replacements == 2
        assert result.files_failed == 0
    
    @pytest.mark.asyncio
    async def test_refactor_handles_read_error(self):
        """Refactor handles file read errors gracefully."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.shell_run = MagicMock(return_value=MagicMock(
            exit_code=1, stdout="", stderr="File not found"
        ))
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "test",
            "replacement_pattern": "TEST",
            "target_files": ["nonexistent.py"],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file, client=mock_client)
        
        assert result.files_failed == 1
        assert "Could not read file" in result.errors[0]
    
    @pytest.mark.asyncio
    async def test_refactor_stops_after_consecutive_errors(self):
        """Refactor aborts after MAX consecutive errors."""
        from app.overwatcher.implementer import run_multi_file_refactor, MULTI_FILE_MAX_ERRORS
        
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.shell_run = MagicMock(return_value=MagicMock(
            exit_code=1, stdout="", stderr="Error"
        ))
        
        # Create more files than the error limit
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "test",
            "replacement_pattern": "TEST",
            "target_files": [f"file{i}.py" for i in range(20)],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file, client=mock_client)
        
        assert result.success is False
        assert "consecutive errors" in result.error
        # Should have stopped before processing all 20 files
        assert result.files_failed <= MULTI_FILE_MAX_ERRORS
    
    @pytest.mark.asyncio
    async def test_refactor_skips_unchanged_files(self):
        """Files without matches are marked unchanged."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.shell_run = MagicMock(return_value=MagicMock(
            exit_code=0, stdout="No matches here", stderr=""
        ))
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "NOTFOUND",
            "replacement_pattern": "REPLACED",
            "target_files": ["app/main.py"],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file, client=mock_client)
        
        assert result.success is True
        assert result.files_unchanged == 1
        assert result.files_modified == 0
    
    @pytest.mark.asyncio
    async def test_refactor_calls_progress_callback(self):
        """Progress callback is called for each file."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.shell_run = MagicMock(side_effect=[
            MagicMock(exit_code=0, stdout="Hello Orb", stderr=""),
            MagicMock(exit_code=0, stdout="", stderr=""),
            MagicMock(exit_code=0, stdout="Hello Astra", stderr=""),
        ])
        
        progress_calls = []
        def progress_callback(data):
            progress_calls.append(data)
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "Orb",
            "replacement_pattern": "Astra",
            "target_files": ["app/main.py"],
        }
        
        await run_multi_file_refactor(
            multi_file=multi_file,
            client=mock_client,
            progress_callback=progress_callback
        )
        
        # Should have: processing, success, complete
        assert len(progress_calls) >= 2
        assert any(p.get("type") == "progress" for p in progress_calls)
        assert any(p.get("type") == "complete" for p in progress_calls)
    
    @pytest.mark.asyncio
    async def test_refactor_sandbox_unavailable(self):
        """Refactor fails when sandbox is not available."""
        from app.overwatcher.implementer import run_multi_file_refactor
        
        mock_client = MagicMock()
        mock_client.is_connected.return_value = False
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "test",
            "replacement_pattern": "TEST",
            "target_files": ["app/main.py"],
        }
        
        result = await run_multi_file_refactor(multi_file=multi_file, client=mock_client)
        
        assert result.success is False
        assert "Sandbox not available" in result.error


class TestMultiFileOperationDispatcher:
    """Tests for run_multi_file_operation() dispatcher."""
    
    @pytest.mark.asyncio
    async def test_routes_to_search(self):
        """Dispatcher routes search operations to search handler."""
        from app.overwatcher.implementer import run_multi_file_operation
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "search",
            "search_pattern": "TODO",
            "total_files": 5,
            "total_occurrences": 10,
        }
        
        result = await run_multi_file_operation(multi_file=multi_file)
        
        assert result.operation == "search"
    
    @pytest.mark.asyncio
    async def test_routes_to_refactor(self):
        """Dispatcher routes refactor operations to refactor handler."""
        from app.overwatcher.implementer import run_multi_file_operation
        
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        mock_client.shell_run = MagicMock(return_value=MagicMock(
            exit_code=0, stdout="", stderr=""
        ))
        
        multi_file = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "confirmed": True,
            "search_pattern": "test",
            "replacement_pattern": "TEST",
            "target_files": [],  # Empty, will return quickly
        }
        
        result = await run_multi_file_operation(multi_file=multi_file, client=mock_client)
        
        assert result.operation == "refactor"
    
    @pytest.mark.asyncio
    async def test_defaults_to_search(self):
        """Dispatcher defaults to search when operation_type is missing."""
        from app.overwatcher.implementer import run_multi_file_operation
        
        multi_file = {
            "is_multi_file": True,
            # No operation_type specified
            "search_pattern": "TEST",
            "total_files": 3,
        }
        
        result = await run_multi_file_operation(multi_file=multi_file)
        
        assert result.operation == "search"


class TestMultiFileResultDataclass:
    """Tests for MultiFileResult dataclass."""
    
    def test_to_dict(self):
        """MultiFileResult.to_dict() returns correct dictionary."""
        from app.overwatcher.implementer import MultiFileResult
        
        result = MultiFileResult(
            success=True,
            operation="refactor",
            search_pattern="old",
            replacement_pattern="new",
            total_files=10,
            files_modified=8,
            files_unchanged=1,
            files_failed=1,
            total_replacements=25,
        )
        
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["operation"] == "refactor"
        assert d["search_pattern"] == "old"
        assert d["replacement_pattern"] == "new"
        assert d["total_files"] == 10
        assert d["files_modified"] == 8
        assert d["total_replacements"] == 25
    
    def test_default_values(self):
        """MultiFileResult has sensible defaults."""
        from app.overwatcher.implementer import MultiFileResult
        
        result = MultiFileResult(success=True, operation="search")
        
        assert result.search_pattern == ""
        assert result.replacement_pattern == ""
        assert result.total_files == 0
        assert result.errors == []
        assert result.details == []
        assert result.awaiting_confirmation is False


class TestConstantsExported:
    """Verify constants are exported correctly."""
    
    def test_max_errors_constant(self):
        """MULTI_FILE_MAX_ERRORS is exported."""
        from app.overwatcher.implementer import MULTI_FILE_MAX_ERRORS
        assert MULTI_FILE_MAX_ERRORS == 10
    
    def test_verify_timeout_constant(self):
        """MULTI_FILE_VERIFY_TIMEOUT is exported."""
        from app.overwatcher.implementer import MULTI_FILE_VERIFY_TIMEOUT
        assert MULTI_FILE_VERIFY_TIMEOUT == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
