# FILE: tests/test_multi_file_operation.py
"""Tests for MultiFileOperation dataclass and GroundedPOTSpec multi-file support.

Tests the Level 3 multi-file operation models added in spec_models.py v1.20.
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest


# =============================================================================
# MultiFileOperation Tests
# =============================================================================

class TestMultiFileOperation:
    """Tests for MultiFileOperation dataclass."""
    
    def test_import(self):
        """Verify MultiFileOperation can be imported."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        assert MultiFileOperation is not None
    
    def test_default_values(self):
        """Test default values are correct."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        mf = MultiFileOperation()
        
        assert mf.is_multi_file is False
        assert mf.operation_type == ""
        assert mf.search_pattern == ""
        assert mf.replacement_pattern == ""
        assert mf.target_files == []
        assert mf.total_files == 0
        assert mf.total_occurrences == 0
        assert mf.file_filter is None
        assert mf.file_preview == ""
        assert mf.discovery_truncated is False
        assert mf.discovery_duration_ms == 0
        assert mf.roots_searched == []
        assert mf.requires_confirmation is False
        assert mf.confirmed is False
        assert mf.error_message is None
    
    def test_search_operation(self):
        """Test search operation configuration."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="TODO",
            target_files=["file1.py", "file2.py"],
            total_files=2,
            total_occurrences=5,
            roots_searched=["D:\\Orb"],
        )
        
        assert mf.is_multi_file is True
        assert mf.operation_type == "search"
        assert mf.requires_confirmation is False  # Search doesn't need confirmation
    
    def test_refactor_operation(self):
        """Test refactor operation configuration."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="Orb",
            replacement_pattern="Astra",
            target_files=["file1.py", "file2.py", "file3.py"],
            total_files=3,
            total_occurrences=10,
            requires_confirmation=True,  # Refactor needs confirmation
        )
        
        assert mf.is_multi_file is True
        assert mf.operation_type == "refactor"
        assert mf.replacement_pattern == "Astra"
        assert mf.requires_confirmation is True
        assert mf.confirmed is False
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="DEBUG",
            target_files=["a.py", "b.py"],
            total_files=2,
            total_occurrences=4,
            roots_searched=["D:\\Orb"],
            discovery_duration_ms=150,
        )
        
        d = mf.to_dict()
        
        assert isinstance(d, dict)
        assert d["is_multi_file"] is True
        assert d["operation_type"] == "search"
        assert d["search_pattern"] == "DEBUG"
        assert d["target_files"] == ["a.py", "b.py"]
        assert d["total_files"] == 2
        assert d["total_occurrences"] == 4
        assert d["roots_searched"] == ["D:\\Orb"]
        assert d["discovery_duration_ms"] == 150
        assert d["error_message"] is None
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        data = {
            "is_multi_file": True,
            "operation_type": "refactor",
            "search_pattern": "old_name",
            "replacement_pattern": "new_name",
            "target_files": ["x.py"],
            "total_files": 1,
            "total_occurrences": 3,
            "requires_confirmation": True,
            "confirmed": True,
        }
        
        mf = MultiFileOperation.from_dict(data)
        
        assert mf.is_multi_file is True
        assert mf.operation_type == "refactor"
        assert mf.search_pattern == "old_name"
        assert mf.replacement_pattern == "new_name"
        assert mf.requires_confirmation is True
        assert mf.confirmed is True
    
    def test_from_dict_with_defaults(self):
        """Test from_dict handles missing keys gracefully."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        # Minimal data
        data = {"is_multi_file": True}
        
        mf = MultiFileOperation.from_dict(data)
        
        assert mf.is_multi_file is True
        assert mf.operation_type == ""
        assert mf.target_files == []
        assert mf.error_message is None
    
    def test_roundtrip_serialization(self):
        """Test to_dict/from_dict roundtrip."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        original = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="foo",
            replacement_pattern="bar",
            target_files=["a.py", "b.py", "c.py"],
            total_files=3,
            total_occurrences=15,
            file_filter="*.py",
            discovery_truncated=True,
            roots_searched=["D:\\Orb", "D:\\orb-desktop"],
            requires_confirmation=True,
            error_message=None,
        )
        
        d = original.to_dict()
        restored = MultiFileOperation.from_dict(d)
        
        assert restored.is_multi_file == original.is_multi_file
        assert restored.operation_type == original.operation_type
        assert restored.search_pattern == original.search_pattern
        assert restored.replacement_pattern == original.replacement_pattern
        assert restored.target_files == original.target_files
        assert restored.total_files == original.total_files
        assert restored.discovery_truncated == original.discovery_truncated
    
    def test_error_message(self):
        """Test error_message field."""
        from app.pot_spec.grounded.spec_models import MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="test",
            error_message="Connection failed to sandbox",
        )
        
        assert mf.error_message == "Connection failed to sandbox"
        d = mf.to_dict()
        assert d["error_message"] == "Connection failed to sandbox"


# =============================================================================
# GroundedPOTSpec with MultiFileOperation Tests
# =============================================================================

class TestGroundedPOTSpecMultiFile:
    """Tests for GroundedPOTSpec multi-file support."""
    
    def test_multi_file_field_exists(self):
        """Test multi_file field is present on GroundedPOTSpec."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        spec = GroundedPOTSpec(goal="Test goal")
        
        assert hasattr(spec, "multi_file")
        assert spec.multi_file is None  # Default is None
    
    def test_spec_with_multi_file(self):
        """Test GroundedPOTSpec accepts MultiFileOperation."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="TODO",
            total_files=5,
        )
        
        spec = GroundedPOTSpec(
            goal="Find all TODO comments",
            multi_file=mf,
        )
        
        assert spec.multi_file is not None
        assert spec.multi_file.is_multi_file is True
        assert spec.multi_file.search_pattern == "TODO"
    
    def test_get_multi_file_summary_empty(self):
        """Test get_multi_file_summary returns empty for non-multi-file specs."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec
        
        spec = GroundedPOTSpec(goal="Regular goal")
        
        summary = spec.get_multi_file_summary()
        
        assert summary == ""
    
    def test_get_multi_file_summary_not_multi_file(self):
        """Test get_multi_file_summary returns empty when is_multi_file=False."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(is_multi_file=False)  # Explicitly not multi-file
        spec = GroundedPOTSpec(goal="Test", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert summary == ""
    
    def test_get_multi_file_summary_search(self):
        """Test get_multi_file_summary for search operations."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="TODO",
            total_files=5,
            total_occurrences=12,
            roots_searched=["D:\\Orb"],
        )
        spec = GroundedPOTSpec(goal="Find TODOs", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "SEARCH" in summary
        assert "`TODO`" in summary
        assert "5 files" in summary
        assert "12 occurrences" in summary
        assert "D:\\Orb" in summary
    
    def test_get_multi_file_summary_refactor(self):
        """Test get_multi_file_summary for refactor operations."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="Orb",
            replacement_pattern="Astra",
            total_files=10,
            total_occurrences=50,
            requires_confirmation=True,
            confirmed=False,
        )
        spec = GroundedPOTSpec(goal="Rename Orb to Astra", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "REFACTOR" in summary
        assert "`Orb`" in summary
        assert "`Astra`" in summary
        assert "10 files" in summary
        assert "requires confirmation" in summary.lower()
    
    def test_get_multi_file_summary_refactor_remove(self):
        """Test get_multi_file_summary for remove operations (empty replacement)."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="console.log",
            replacement_pattern="",  # Empty = remove
            total_files=3,
            total_occurrences=8,
        )
        spec = GroundedPOTSpec(goal="Remove console.log", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "Remove all occurrences" in summary
    
    def test_get_multi_file_summary_with_preview(self):
        """Test get_multi_file_summary includes file preview."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="DEBUG",
            total_files=2,
            total_occurrences=5,
            file_preview="  1. app/main.py (3 matches)\n  2. app/config.py (2 matches)",
        )
        spec = GroundedPOTSpec(goal="Find DEBUG", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "File Preview" in summary
        assert "app/main.py" in summary
        assert "app/config.py" in summary
    
    def test_get_multi_file_summary_truncated(self):
        """Test get_multi_file_summary shows truncation warning."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="import",
            target_files=["f1.py", "f2.py", "f3.py"],  # Only 3 in list
            total_files=100,  # But 100 total
            total_occurrences=500,
            discovery_truncated=True,
        )
        spec = GroundedPOTSpec(goal="Find imports", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "truncated" in summary.lower()
        assert "3" in summary  # showing first 3
        assert "100" in summary  # of 100
    
    def test_get_multi_file_summary_confirmed(self):
        """Test get_multi_file_summary shows confirmed status."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="refactor",
            search_pattern="old",
            replacement_pattern="new",
            total_files=5,
            total_occurrences=10,
            requires_confirmation=True,
            confirmed=True,  # User confirmed
        )
        spec = GroundedPOTSpec(goal="Refactor", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "confirmed" in summary.lower()
        assert "✅" in summary
    
    def test_get_multi_file_summary_with_error(self):
        """Test get_multi_file_summary shows error message."""
        from app.pot_spec.grounded.spec_models import GroundedPOTSpec, MultiFileOperation
        
        mf = MultiFileOperation(
            is_multi_file=True,
            operation_type="search",
            search_pattern="test",
            total_files=0,
            error_message="Sandbox connection failed",
        )
        spec = GroundedPOTSpec(goal="Search test", multi_file=mf)
        
        summary = spec.get_multi_file_summary()
        
        assert "error" in summary.lower()
        assert "Sandbox connection failed" in summary
        assert "❌" in summary


# =============================================================================
# Package Export Tests
# =============================================================================

class TestPackageExports:
    """Test that MultiFileOperation is exported correctly."""
    
    def test_import_from_grounded_package(self):
        """Test importing from app.pot_spec.grounded."""
        from app.pot_spec.grounded import MultiFileOperation
        assert MultiFileOperation is not None
    
    def test_in_all_list(self):
        """Test MultiFileOperation is in __all__."""
        from app.pot_spec import grounded
        assert "MultiFileOperation" in grounded.__all__


# =============================================================================
# Phase 4 Integration Tests (v1.33)
# =============================================================================

class TestMultiFileIntentDetection:
    """Tests for _detect_multi_file_intent helper function."""
    
    def test_import_helper(self):
        """Verify _detect_multi_file_intent can be imported."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        assert _detect_multi_file_intent is not None
    
    def test_detect_search_find_all(self):
        """Test detection of 'find all X' pattern."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        result = _detect_multi_file_intent("Find all TODO comments in the codebase", None)
        
        assert result is not None
        assert result["is_multi_file"] is True
        assert result["operation_type"] == "search"
        assert "todo" in result["search_pattern"].lower()
    
    def test_detect_search_list_files(self):
        """Test detection of 'list files containing' pattern."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        result = _detect_multi_file_intent("List all files containing DEBUG", None)
        
        assert result is not None
        assert result["is_multi_file"] is True
        assert result["operation_type"] == "search"
    
    def test_detect_refactor_replace(self):
        """Test detection of 'replace X with Y everywhere' pattern."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        result = _detect_multi_file_intent("Replace Orb with Astra everywhere", None)
        
        assert result is not None
        assert result["is_multi_file"] is True
        assert result["operation_type"] == "refactor"
        assert "orb" in result["search_pattern"].lower()
        assert "astra" in result["replacement_pattern"].lower()
    
    def test_detect_refactor_rename(self):
        """Test detection of 'rename X to Y' pattern."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        result = _detect_multi_file_intent("Rename old_function to new_function across the codebase", None)
        
        assert result is not None
        assert result["is_multi_file"] is True
        assert result["operation_type"] == "refactor"
    
    def test_no_detection_without_scope(self):
        """Test that patterns without scope keywords don't trigger detection."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        # No scope keywords like 'all', 'everywhere', 'codebase'
        result = _detect_multi_file_intent("Find the bug", None)
        
        assert result is None
    
    def test_no_detection_single_file(self):
        """Test that single file operations don't trigger detection."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        result = _detect_multi_file_intent("Fix the bug in app/main.py", None)
        
        assert result is None
    
    def test_constraints_hint_override(self):
        """Test that constraints_hint multi_file_metadata takes precedence."""
        from app.pot_spec.grounded.spec_generation import _detect_multi_file_intent
        
        constraints = {
            "multi_file_metadata": {
                "is_multi_file": True,
                "operation_type": "search",
                "search_pattern": "OVERRIDE_PATTERN",
            }
        }
        
        result = _detect_multi_file_intent("Find something else", constraints)
        
        assert result is not None
        assert result["search_pattern"] == "OVERRIDE_PATTERN"


class TestBuildMultiFileOperationAsync:
    """Tests for _build_multi_file_operation async helper function."""
    
    def test_import_helper(self):
        """Verify _build_multi_file_operation can be imported."""
        from app.pot_spec.grounded.spec_generation import _build_multi_file_operation
        import asyncio
        import inspect
        
        assert _build_multi_file_operation is not None
        # Verify it's an async function
        assert inspect.iscoroutinefunction(_build_multi_file_operation)
    
    @pytest.mark.asyncio
    async def test_sandbox_unavailable_returns_error(self):
        """Test graceful handling when sandbox client is unavailable."""
        from app.pot_spec.grounded.spec_generation import _build_multi_file_operation
        from app.pot_spec.grounded import spec_generation
        
        # Temporarily disable sandbox client
        original_available = spec_generation._SANDBOX_CLIENT_AVAILABLE
        spec_generation._SANDBOX_CLIENT_AVAILABLE = False
        
        try:
            result = await _build_multi_file_operation(
                operation_type="search",
                search_pattern="test",
            )
            
            assert result is not None
            assert result.is_multi_file is True
            assert result.error_message is not None
            assert "not available" in result.error_message.lower()
        finally:
            # Restore original state
            spec_generation._SANDBOX_CLIENT_AVAILABLE = original_available
    
    @pytest.mark.asyncio
    async def test_search_operation_type(self):
        """Test search operation returns correct type."""
        from app.pot_spec.grounded.spec_generation import _build_multi_file_operation
        from app.pot_spec.grounded import spec_generation
        
        # Skip if sandbox unavailable
        if not spec_generation._SANDBOX_CLIENT_AVAILABLE:
            pytest.skip("Sandbox client not available")
        
        result = await _build_multi_file_operation(
            operation_type="search",
            search_pattern="TODO",
        )
        
        assert result.operation_type == "search"
        assert result.requires_confirmation is False  # Search doesn't need confirmation
    
    @pytest.mark.asyncio
    async def test_refactor_requires_confirmation(self):
        """Test refactor operation requires confirmation."""
        from app.pot_spec.grounded.spec_generation import _build_multi_file_operation
        from app.pot_spec.grounded import spec_generation
        
        # Skip if sandbox unavailable
        if not spec_generation._SANDBOX_CLIENT_AVAILABLE:
            pytest.skip("Sandbox client not available")
        
        result = await _build_multi_file_operation(
            operation_type="refactor",
            search_pattern="old",
            replacement_pattern="new",
        )
        
        assert result.operation_type == "refactor"
        assert result.requires_confirmation is True
        assert result.confirmed is False


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
