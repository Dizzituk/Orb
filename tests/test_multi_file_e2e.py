# FILE: tests/test_multi_file_e2e.py
"""
End-to-End Tests for Multi-File Operations (Level 3 - Phase 6)

Tests the complete flow from user intent through to execution:
1. Intent detection (tier0_rules)
2. Spec generation (spec_generation)
3. Implementation (implementer)
4. Progress streaming (batch_ops)

These tests use mock sandbox clients to avoid requiring
an actual sandbox environment.

Version Notes:
-------------
v1.0 (2026-01-28): Initial Phase 6 E2E tests
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio


class TestMultiFileSearchE2E:
    """End-to-end tests for multi-file search operations."""
    
    @pytest.mark.asyncio
    async def test_find_all_todo_flow(self):
        """
        Complete flow: "Find all TODO comments in the codebase"
        
        1. tier0 detects MULTI_FILE_SEARCH intent
        2. SpecGate runs discovery, populates MultiFileOperation
        3. Implementer returns search results
        """
        from app.translation.tier0_rules import check_multi_file_trigger
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        from app.overwatcher.implementer import run_multi_file_operation
        
        # Step 1: Intent detection
        result = check_multi_file_trigger("Find all TODO comments in the codebase")
        
        assert result.matched is True
        assert result.intent.value == "MULTI_FILE_SEARCH"
        
        # Step 2: Build MultiFileOperation (simulating SpecGate)
        multi_file = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="TODO",
            total_files=15,
            total_occurrences=42,
            target_files=["app/main.py", "app/router.py"],
            file_preview="app/main.py:10: # TODO: fix this\napp/router.py:25: # TODO: refactor",
        )
        
        assert multi_file.is_multi_file is True
        assert multi_file.requires_confirmation is False  # Search is read-only
        
        # Step 3: Implementer execution
        result = await run_multi_file_operation(multi_file=multi_file.to_dict())
        
        assert result.success is True
        assert result.operation == "search"
        assert result.total_files == 15
        assert result.files_modified == 0  # Read-only
    
    def test_search_intent_detection_variants(self):
        """Test various search intent patterns are detected."""
        from app.translation.tier0_rules import check_multi_file_trigger
        
        test_cases = [
            ("Find all TODO comments in the codebase", True),
            ("List all files containing error handling", True),
            ("Search codebase for deprecated functions", True),
            ("Count all occurrences of logger.error", True),
            # These should NOT match (no scope indicator)
            ("Find TODO", False),
            ("Search for bugs", False),
        ]
        
        for query, should_match in test_cases:
            result = check_multi_file_trigger(query)
            assert result.matched == should_match, f"Query '{query}' should {'match' if should_match else 'not match'}"


class TestMultiFileRefactorE2E:
    """End-to-end tests for multi-file refactor operations."""
    
    @pytest.mark.asyncio
    async def test_replace_everywhere_flow(self):
        """
        Complete flow: "Replace Orb with Astra everywhere"
        
        1. tier0 detects MULTI_FILE_REFACTOR intent
        2. SpecGate runs discovery, returns confirmation question
        3. User confirms
        4. Implementer processes all files
        5. Progress streamed
        """
        from app.translation.tier0_rules import check_multi_file_trigger
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        from app.overwatcher.implementer import run_multi_file_refactor
        
        # Step 1: Intent detection
        result = check_multi_file_trigger("Replace Orb with Astra everywhere")
        
        assert result.matched is True
        assert result.intent.value == "MULTI_FILE_REFACTOR"
        
        # Step 2: Build MultiFileOperation
        multi_file = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="Orb",
            replacement_pattern="Astra",
            total_files=3,
            total_occurrences=10,
            target_files=["app/main.py", "app/config.py", "README.md"],
            requires_confirmation=True,
            confirmed=False,
        )
        
        assert multi_file.requires_confirmation is True
        
        # Step 3: Confirmation (simulated - user says yes)
        multi_file.confirmed = True
        
        # Step 4: Implementer execution with mock sandbox
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        
        # Simulate: read returns "Orb", write succeeds, verify returns "Astra"
        mock_client.shell_run = MagicMock(side_effect=[
            # File 1
            MagicMock(exit_code=0, stdout="Welcome to Orb!", stderr=""),
            MagicMock(exit_code=0, stdout="", stderr=""),
            MagicMock(exit_code=0, stdout="Welcome to Astra!", stderr=""),
            # File 2
            MagicMock(exit_code=0, stdout="Orb config", stderr=""),
            MagicMock(exit_code=0, stdout="", stderr=""),
            MagicMock(exit_code=0, stdout="Astra config", stderr=""),
            # File 3
            MagicMock(exit_code=0, stdout="# Orb README", stderr=""),
            MagicMock(exit_code=0, stdout="", stderr=""),
            MagicMock(exit_code=0, stdout="# Astra README", stderr=""),
        ])
        
        # Step 5: Track progress
        progress_events = []
        def track_progress(data):
            progress_events.append(data)
        
        result = await run_multi_file_refactor(
            multi_file=multi_file.to_dict(),
            client=mock_client,
            progress_callback=track_progress
        )
        
        # Verify results
        assert result.success is True
        assert result.files_modified == 3
        assert result.total_replacements == 3
        
        # Verify progress was streamed
        assert len(progress_events) > 0
        assert any(e.get("type") == "complete" for e in progress_events)
    
    @pytest.mark.asyncio
    async def test_refactor_unconfirmed_fails(self):
        """Refactor without confirmation should fail."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        from app.overwatcher.implementer import run_multi_file_refactor
        
        multi_file = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="test",
            replacement_pattern="TEST",
            target_files=["file.py"],
            requires_confirmation=True,
            confirmed=False,  # NOT confirmed
        )
        
        result = await run_multi_file_refactor(multi_file=multi_file.to_dict())
        
        assert result.success is False
        assert "confirmation" in result.error
    
    def test_refactor_intent_detection_variants(self):
        """Test various refactor intent patterns are detected."""
        from app.translation.tier0_rules import check_multi_file_trigger
        
        test_cases = [
            ("Replace Orb with Astra everywhere", True),
            ("Change all foo to bar in the codebase", True),
            ("Rename old_function to new_function everywhere", True),
            ("Update all print to logger everywhere", True),
            # These should NOT match (no scope indicator)
            ("Replace foo with bar", False),
            ("Change this to that", False),
        ]
        
        for query, should_match in test_cases:
            result = check_multi_file_trigger(query)
            assert result.matched == should_match, f"Query '{query}' should {'match' if should_match else 'not match'}"


class TestProgressStreaming:
    """Tests for SSE progress streaming."""
    
    @pytest.mark.asyncio
    async def test_batch_progress_stream(self):
        """Test BatchProgressStream generates correct SSE events."""
        from app.llm.local_tools.zobie.streams.batch_ops import (
            BatchProgressStream,
        )
        
        stream = BatchProgressStream(operation="refactor", total_files=3)
        callback = stream.get_progress_callback()
        
        # Simulate progress updates in background
        async def simulate_progress():
            await asyncio.sleep(0.1)
            callback({"type": "progress", "current": 1, "total": 3, "file": "a.py", "status": "success"})
            await asyncio.sleep(0.1)
            callback({"type": "progress", "current": 2, "total": 3, "file": "b.py", "status": "success"})
            await asyncio.sleep(0.1)
            callback({"type": "complete", "files_modified": 2, "success": True})
        
        asyncio.create_task(simulate_progress())
        
        # Collect events
        events = []
        async for event in stream.generate(timeout=5.0):
            events.append(event)
            if "complete" in event:
                break
        
        # Verify events
        assert len(events) >= 3  # start + 2 progress + complete
        assert any("start" in e for e in events)
        assert any("complete" in e for e in events)
    
    def test_format_completion_summary_search(self):
        """Test completion summary formatting for search."""
        from app.llm.local_tools.zobie.streams.batch_ops import format_completion_summary
        
        result = {
            "operation": "search",
            "search_pattern": "TODO",
            "total_files": 15,
            "total_occurrences": 42,
        }
        
        summary = format_completion_summary(result)
        
        assert "Search Complete" in summary
        assert "TODO" in summary
        assert "15" in summary
        assert "42" in summary
    
    def test_format_completion_summary_refactor_success(self):
        """Test completion summary formatting for successful refactor."""
        from app.llm.local_tools.zobie.streams.batch_ops import format_completion_summary
        
        result = {
            "operation": "refactor",
            "success": True,
            "search_pattern": "Orb",
            "replacement_pattern": "Astra",
            "files_modified": 10,
            "files_unchanged": 2,
            "files_failed": 0,
            "total_replacements": 25,
        }
        
        summary = format_completion_summary(result)
        
        assert "✅" in summary
        assert "Refactor Complete" in summary
        assert "Orb" in summary
        assert "Astra" in summary
        assert "10" in summary
    
    def test_format_completion_summary_refactor_failed(self):
        """Test completion summary formatting for failed refactor."""
        from app.llm.local_tools.zobie.streams.batch_ops import format_completion_summary
        
        result = {
            "operation": "refactor",
            "success": False,
            "search_pattern": "test",
            "replacement_pattern": "TEST",
            "files_modified": 0,
            "files_unchanged": 0,
            "files_failed": 5,
            "errors": ["file1.py: Write failed", "file2.py: Read error"],
        }
        
        summary = format_completion_summary(result)
        
        assert "❌" in summary
        assert "Refactor Failed" in summary
        assert "Errors:" in summary
    
    def test_format_progress_line_variants(self):
        """Test progress line formatting for different statuses."""
        from app.llm.local_tools.zobie.streams.batch_ops import format_progress_line
        
        test_cases = [
            ({"current": 1, "total": 10, "file": "app/main.py", "status": "processing"}, "⏳"),
            ({"current": 2, "total": 10, "file": "app/router.py", "status": "success", "replacements": 5}, "✅"),
            ({"current": 3, "total": 10, "file": "app/config.py", "status": "unchanged"}, "➖"),
            ({"current": 4, "total": 10, "file": "app/error.py", "status": "error"}, "❌"),
        ]
        
        for data, expected_emoji in test_cases:
            line = format_progress_line(data)
            assert expected_emoji in line


class TestRegressionSingleFile:
    """Ensure single-file operations still work."""
    
    @pytest.mark.asyncio
    async def test_single_file_not_affected(self):
        """Regular single-file specs should work unchanged."""
        from app.overwatcher.implementer import run_multi_file_operation, MultiFileResult
        
        # Non-multi-file spec should fail gracefully
        spec = {
            "is_multi_file": False,
        }
        
        result = await run_multi_file_operation(multi_file=spec)
        
        # Should return error for non-multi-file spec
        assert result.success is False
        assert "Not a multi-file operation" in result.error
    
    def test_tier0_single_file_queries_not_matched(self):
        """Single-file queries should NOT match multi-file patterns."""
        from app.translation.tier0_rules import check_multi_file_trigger
        
        single_file_queries = [
            "Fix bug in app/main.py",
            "Read the config file",
            "Update the README",
            "What does this function do?",
            "Help me understand the code",
        ]
        
        for query in single_file_queries:
            result = check_multi_file_trigger(query)
            assert result.matched is False, f"Single-file query '{query}' should NOT match multi-file patterns"


class TestMultiFileOperationDataclass:
    """Tests for MultiFileOperation dataclass integration."""
    
    def test_multi_file_operation_to_dict(self):
        """MultiFileOperation.to_dict() produces valid dict for Implementer."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="old",
            replacement_pattern="new",
            target_files=["a.py", "b.py"],
            total_files=2,
            total_occurrences=10,
            requires_confirmation=True,
            confirmed=True,
        )
        
        d = mf.to_dict()
        
        assert d["is_multi_file"] is True
        assert d["operation_type"] == "refactor"
        assert d["search_pattern"] == "old"
        assert d["replacement_pattern"] == "new"
        assert d["target_files"] == ["a.py", "b.py"]
        assert d["confirmed"] is True
    
    def test_multi_file_operation_from_dict(self):
        """MultiFileOperation.from_dict() reconstructs correctly."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        data = {
            "is_multi_file": True,
            "operation_type": "search",
            "search_pattern": "TODO",
            "total_files": 15,
            "total_occurrences": 42,
        }
        
        mf = MultiFileOperation.from_dict(data)
        
        assert mf.is_multi_file is True
        assert mf.operation_type == "search"
        assert mf.search_pattern == "TODO"
        assert mf.total_files == 15


class TestStreamExports:
    """Verify stream exports are correct."""
    
    def test_batch_ops_exports_available(self):
        """All batch_ops exports are available from streams module."""
        from app.llm.local_tools.zobie.streams import (
            BatchProgressMessage,
            BatchProgressStream,
            create_progress_callback,
            create_sync_callback,
            format_completion_summary,
            format_progress_line,
            generate_batch_operation_stream,
        )
        
        # Just verify imports work
        assert BatchProgressMessage is not None
        assert BatchProgressStream is not None
        assert format_completion_summary is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
