# FILE: tests/test_architecture_commands.py
"""
Tests for architecture commands:
- UPDATE ARCHITECTURE: Scan repo → store in .architecture/
- CREATE ARCHITECTURE MAP: Load .architecture/ → generate markdown

Tests the trigger detection, file operations, and data conversion.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import json
import tempfile
from datetime import datetime, timezone


# =============================================================================
# Trigger Detection Tests
# =============================================================================

class TestTriggerDetection:
    """Test command trigger patterns."""
    
    def test_update_arch_triggers(self):
        from app.llm.local_tools.archmap_helpers import _UPDATE_ARCH_TRIGGER_SET
        
        # Should match
        assert "update architecture" in _UPDATE_ARCH_TRIGGER_SET
        assert "update arch" in _UPDATE_ARCH_TRIGGER_SET
        assert "update your architecture" in _UPDATE_ARCH_TRIGGER_SET
        assert "/update_architecture" in _UPDATE_ARCH_TRIGGER_SET
        assert "/update_arch" in _UPDATE_ARCH_TRIGGER_SET
        
        # Case insensitive check
        assert "UPDATE ARCHITECTURE".lower() in _UPDATE_ARCH_TRIGGER_SET
    
    def test_archmap_triggers(self):
        from app.llm.local_tools.archmap_helpers import _ARCHMAP_TRIGGER_SET
        
        # Should match
        assert "create architecture map" in _ARCHMAP_TRIGGER_SET
        assert "arch map" in _ARCHMAP_TRIGGER_SET
        assert "architecture map" in _ARCHMAP_TRIGGER_SET
        assert "/arch_map" in _ARCHMAP_TRIGGER_SET
        assert "/architecture_map" in _ARCHMAP_TRIGGER_SET
    
    def test_triggers_are_distinct(self):
        from app.llm.local_tools.archmap_helpers import (
            _UPDATE_ARCH_TRIGGER_SET,
            _ARCHMAP_TRIGGER_SET,
        )
        
        # No overlap between trigger sets
        overlap = _UPDATE_ARCH_TRIGGER_SET & _ARCHMAP_TRIGGER_SET
        assert len(overlap) == 0, f"Triggers overlap: {overlap}"


# =============================================================================
# Architecture Directory Tests
# =============================================================================

class TestArchitectureDirectory:
    """Test .architecture/ directory operations."""
    
    def test_get_architecture_dir_creates(self):
        from app.llm.local_tools.archmap_helpers import get_architecture_dir
        import app.llm.local_tools.archmap_helpers as helpers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Override the directory
            original = helpers.ARCHITECTURE_DIR
            helpers.ARCHITECTURE_DIR = tmpdir + "/.architecture"
            
            try:
                arch_dir = get_architecture_dir()
                assert arch_dir.exists()
                assert arch_dir.is_dir()
            finally:
                helpers.ARCHITECTURE_DIR = original
    
    def test_architecture_exists_false_when_empty(self):
        from app.llm.local_tools.archmap_helpers import architecture_exists
        import app.llm.local_tools.archmap_helpers as helpers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original = helpers.ARCHITECTURE_DIR
            helpers.ARCHITECTURE_DIR = tmpdir + "/.architecture"
            
            try:
                assert architecture_exists() is False
            finally:
                helpers.ARCHITECTURE_DIR = original
    
    def test_architecture_exists_true_when_files_present(self):
        from app.llm.local_tools.archmap_helpers import (
            architecture_exists,
            get_architecture_dir,
        )
        import app.llm.local_tools.archmap_helpers as helpers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original = helpers.ARCHITECTURE_DIR
            helpers.ARCHITECTURE_DIR = tmpdir + "/.architecture"
            
            try:
                arch_dir = get_architecture_dir()
                
                # Create required files
                (arch_dir / "manifest.json").write_text('{"version": "1.0"}')
                (arch_dir / "files.json").write_text('{}')
                
                assert architecture_exists() is True
            finally:
                helpers.ARCHITECTURE_DIR = original


# =============================================================================
# Data Loading Tests
# =============================================================================

class TestDataLoading:
    """Test loading architecture data files."""
    
    def test_load_manifest(self):
        from app.llm.local_tools.archmap_helpers import (
            load_architecture_manifest,
            get_architecture_dir,
        )
        import app.llm.local_tools.archmap_helpers as helpers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original = helpers.ARCHITECTURE_DIR
            helpers.ARCHITECTURE_DIR = tmpdir + "/.architecture"
            
            try:
                arch_dir = get_architecture_dir()
                
                manifest_data = {
                    "version": "1.0",
                    "last_scan": "2025-12-31T10:00:00Z",
                    "files_count": 42,
                }
                (arch_dir / "manifest.json").write_text(json.dumps(manifest_data))
                
                loaded = load_architecture_manifest()
                assert loaded["version"] == "1.0"
                assert loaded["files_count"] == 42
            finally:
                helpers.ARCHITECTURE_DIR = original
    
    def test_load_files(self):
        from app.llm.local_tools.archmap_helpers import (
            load_architecture_files,
            get_architecture_dir,
        )
        import app.llm.local_tools.archmap_helpers as helpers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original = helpers.ARCHITECTURE_DIR
            helpers.ARCHITECTURE_DIR = tmpdir + "/.architecture"
            
            try:
                arch_dir = get_architecture_dir()
                
                files_data = {
                    "app/main.py": {
                        "path": "app/main.py",
                        "language": "python",
                        "functions": [{"name": "main", "line": 10}],
                    }
                }
                (arch_dir / "files.json").write_text(json.dumps(files_data))
                
                loaded = load_architecture_files()
                assert "app/main.py" in loaded
                assert loaded["app/main.py"]["language"] == "python"
            finally:
                helpers.ARCHITECTURE_DIR = original
    
    def test_load_missing_returns_empty(self):
        from app.llm.local_tools.archmap_helpers import (
            load_architecture_manifest,
            load_architecture_files,
            load_architecture_enums,
        )
        import app.llm.local_tools.archmap_helpers as helpers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original = helpers.ARCHITECTURE_DIR
            helpers.ARCHITECTURE_DIR = tmpdir + "/.nonexistent"
            
            try:
                assert load_architecture_manifest() == {}
                assert load_architecture_files() == {}
                assert load_architecture_enums() == {}
            finally:
                helpers.ARCHITECTURE_DIR = original


# =============================================================================
# Data Conversion Tests
# =============================================================================

class TestDataConversion:
    """Test converting zobie output to architecture format."""
    
    def test_convert_symbols_to_functions(self):
        from app.llm.local_tools.zobie_tools import _convert_to_architecture_format
        
        index_data = {
            "scanned_files": [
                {
                    "path": "app/main.py",
                    "language": "python",
                    "bytes": 1000,
                    "symbols": [
                        {"kind": "function", "name": "main", "line": 10},
                        {"kind": "function", "name": "setup", "line": 20},
                    ],
                    "imports": ["os", "sys"],
                }
            ]
        }
        
        files = _convert_to_architecture_format(index_data, "/tmp")
        
        assert "app/main.py" in files
        assert len(files["app/main.py"]["functions"]) == 2
        assert files["app/main.py"]["functions"][0]["name"] == "main"
    
    def test_convert_symbols_to_classes(self):
        from app.llm.local_tools.zobie_tools import _convert_to_architecture_format
        
        index_data = {
            "scanned_files": [
                {
                    "path": "app/models.py",
                    "language": "python",
                    "bytes": 500,
                    "symbols": [
                        {"kind": "class", "name": "User", "line": 5},
                        {"kind": "class", "name": "Project", "line": 25},
                    ],
                }
            ]
        }
        
        files = _convert_to_architecture_format(index_data, "/tmp")
        
        assert "app/models.py" in files
        assert len(files["app/models.py"]["classes"]) == 2
        assert files["app/models.py"]["classes"][0]["name"] == "User"
    
    def test_convert_enums(self):
        from app.llm.local_tools.zobie_tools import _convert_to_architecture_format
        
        index_data = {
            "scanned_files": [
                {
                    "path": "app/schemas.py",
                    "language": "python",
                    "bytes": 300,
                    "symbols": [],
                    "enums": [
                        {
                            "name": "Status",
                            "line": 10,
                            "base": "Enum",
                            "members": ["PENDING", "ACTIVE", "DONE"],
                        }
                    ],
                }
            ]
        }
        
        files = _convert_to_architecture_format(index_data, "/tmp")
        
        assert "app/schemas.py" in files
        # Enums are stored as classes
        assert len(files["app/schemas.py"]["classes"]) == 1
        assert files["app/schemas.py"]["classes"][0]["name"] == "Status"
        assert "PENDING" in files["app/schemas.py"]["classes"][0]["members"]
    
    def test_extract_enums_index(self):
        from app.llm.local_tools.zobie_tools import _extract_enums_index
        
        index_data = {
            "scanned_files": [
                {
                    "path": "app/schemas.py",
                    "enums": [
                        {
                            "name": "JobType",
                            "line": 15,
                            "base": "StrEnum",
                            "members": ["BUILD", "TEST", "DEPLOY"],
                            "member_count": 3,
                        }
                    ],
                }
            ]
        }
        
        enums = _extract_enums_index(index_data)
        
        assert "app/schemas.py::JobType" in enums
        assert enums["app/schemas.py::JobType"]["member_count"] == 3
    
    def test_extract_routes_index(self):
        from app.llm.local_tools.zobie_tools import _extract_routes_index
        
        index_data = {
            "scanned_files": [
                {
                    "path": "app/router.py",
                    "routes": [
                        {
                            "method": "GET",
                            "path": "/api/users",
                            "line": 20,
                            "handler": "list_users",
                            "decorator_target": "router",
                        },
                        {
                            "method": "POST",
                            "path": "/api/users",
                            "line": 35,
                            "handler": "create_user",
                            "decorator_target": "router",
                        },
                    ],
                }
            ]
        }
        
        routes = _extract_routes_index(index_data)
        
        assert "GET /api/users" in routes
        assert "POST /api/users" in routes
        assert routes["GET /api/users"]["function"] == "list_users"


# =============================================================================
# Prompt Building Tests
# =============================================================================

class TestPromptBuilding:
    """Test architecture map prompt generation."""
    
    def test_build_prompt_includes_manifest(self):
        from app.llm.local_tools.archmap_helpers import build_archmap_prompt
        
        manifest = {"version": "1.0", "files_count": 10}
        files = {}
        enums = {}
        routes = {}
        imports = {}
        
        prompt = build_archmap_prompt(manifest, files, enums, routes, imports)
        
        assert "1.0" in prompt
        assert "files_count" in prompt
    
    def test_build_prompt_includes_files(self):
        from app.llm.local_tools.archmap_helpers import build_archmap_prompt
        
        manifest = {}
        files = {
            "app/main.py": {
                "classes": [{"name": "App", "line": 5, "docstring": "Main app"}],
                "functions": [{"name": "run", "line": 10, "signature": "()"}],
            }
        }
        enums = {}
        routes = {}
        imports = {}
        
        prompt = build_archmap_prompt(manifest, files, enums, routes, imports)
        
        assert "app/main.py" in prompt
        assert "App" in prompt
        assert "run" in prompt
    
    def test_build_prompt_includes_enums(self):
        from app.llm.local_tools.archmap_helpers import build_archmap_prompt
        
        manifest = {}
        files = {}
        enums = {
            "app/schemas.py::Status": {
                "members": [{"name": "ACTIVE"}, {"name": "INACTIVE"}],
            }
        }
        routes = {}
        imports = {}
        
        prompt = build_archmap_prompt(manifest, files, enums, routes, imports)
        
        assert "Status" in prompt
        assert "ACTIVE" in prompt


# =============================================================================
# Security Tests
# =============================================================================

class TestSecurity:
    """Test security deny patterns."""
    
    def test_deny_env_file(self):
        from app.llm.local_tools.archmap_helpers import _is_denied_repo_path
        
        assert _is_denied_repo_path(".env") is True
        assert _is_denied_repo_path("config/.env") is True
        assert _is_denied_repo_path("/.env") is True
    
    def test_allow_env_example(self):
        from app.llm.local_tools.archmap_helpers import _is_denied_repo_path
        
        # .env.example should NOT be denied (only exact .env is denied)
        assert _is_denied_repo_path(".env.example") is False
        assert _is_denied_repo_path(".env.sample") is False
        assert _is_denied_repo_path(".env.template") is False
    
    def test_deny_key_files(self):
        from app.llm.local_tools.archmap_helpers import _is_denied_repo_path
        
        assert _is_denied_repo_path("server.key") is True
        assert _is_denied_repo_path("cert.pem") is True
        assert _is_denied_repo_path("app.pfx") is True
    
    def test_deny_git(self):
        from app.llm.local_tools.archmap_helpers import _is_denied_repo_path
        
        assert _is_denied_repo_path(".git") is True
        assert _is_denied_repo_path(".git/config") is True
    
    def test_allow_normal_files(self):
        from app.llm.local_tools.archmap_helpers import _is_denied_repo_path
        
        assert _is_denied_repo_path("app/main.py") is False
        assert _is_denied_repo_path("package.json") is False
        assert _is_denied_repo_path("README.md") is False


# =============================================================================
# Model Configuration Tests
# =============================================================================

class TestModelConfig:
    """Test model configuration values."""
    
    def test_archmap_uses_opus(self):
        from app.llm.local_tools.archmap_helpers import ARCHMAP_PROVIDER, ARCHMAP_MODEL
        
        assert ARCHMAP_PROVIDER == "anthropic"
        assert "opus" in ARCHMAP_MODEL.lower() or "claude" in ARCHMAP_MODEL.lower()
    
    def test_has_fallback_config(self):
        from app.llm.local_tools.archmap_helpers import (
            ARCHMAP_FALLBACK_PROVIDER,
            ARCHMAP_FALLBACK_MODEL,
        )
        
        assert ARCHMAP_FALLBACK_PROVIDER is not None
        assert ARCHMAP_FALLBACK_MODEL is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
