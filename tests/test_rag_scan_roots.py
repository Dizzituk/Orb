# tests/test_rag_scan_roots.py
"""Tests for scan root configuration."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest

from app.rag.config.scan_roots import (
    is_system_path,
    is_allowed_user_subdir,
    should_skip_directory,
    should_skip_file,
    is_scannable_path,
    get_root_alias,
    get_scan_targets,
    get_zobie_output_dir,
    get_latest_zobie_file,
    ZOBIE_OUTPUT_DIR,
)


class TestSystemPath:
    def test_windows_excluded(self):
        assert is_system_path(r"C:\Windows\System32") is True
    
    def test_program_files_excluded(self):
        assert is_system_path(r"C:\Program Files\App") is True
    
    def test_program_files_x86_excluded(self):
        assert is_system_path(r"C:\Program Files (x86)\App") is True
    
    def test_programdata_excluded(self):
        assert is_system_path(r"C:\ProgramData\App") is True
    
    def test_d_drive_allowed(self):
        assert is_system_path(r"D:\Projects") is False
    
    def test_case_insensitive(self):
        assert is_system_path(r"c:\windows\system32") is True


class TestUserHomeFilter:
    def test_desktop_allowed(self):
        path = str(Path.home() / "Desktop" / "file.txt")
        assert is_allowed_user_subdir(path) is True
    
    def test_documents_allowed(self):
        path = str(Path.home() / "Documents")
        assert is_allowed_user_subdir(path) is True
    
    def test_projects_allowed(self):
        path = str(Path.home() / "Projects" / "myproject")
        assert is_allowed_user_subdir(path) is True
    
    def test_downloads_allowed(self):
        path = str(Path.home() / "Downloads")
        assert is_allowed_user_subdir(path) is True
    
    def test_d_drive_bypasses(self):
        assert is_allowed_user_subdir(r"D:\anything") is True
    
    def test_user_home_root_allowed(self):
        path = str(Path.home())
        assert is_allowed_user_subdir(path) is True


class TestDirectorySkip:
    def test_node_modules(self):
        assert should_skip_directory("node_modules") is True
    
    def test_git(self):
        assert should_skip_directory(".git") is True
    
    def test_pycache(self):
        assert should_skip_directory("__pycache__") is True
    
    def test_venv(self):
        assert should_skip_directory("venv") is True
        assert should_skip_directory(".venv") is True
    
    def test_app_allowed(self):
        assert should_skip_directory("app") is False
    
    def test_src_allowed(self):
        assert should_skip_directory("src") is False
    
    def test_case_insensitive(self):
        assert should_skip_directory("Node_Modules") is True
        assert should_skip_directory(".GIT") is True


class TestFileSkip:
    def test_exe_skipped(self):
        skip, reason = should_skip_file("setup.exe")
        assert skip is True
        assert "Extension" in reason
    
    def test_dll_skipped(self):
        skip, _ = should_skip_file("library.dll")
        assert skip is True
    
    def test_zip_skipped(self):
        skip, _ = should_skip_file("archive.zip")
        assert skip is True
    
    def test_py_allowed(self):
        skip, reason = should_skip_file("main.py")
        assert skip is False
        assert reason is None
    
    def test_js_allowed(self):
        skip, _ = should_skip_file("index.js")
        assert skip is False
    
    def test_large_file(self):
        skip, reason = should_skip_file("data.csv", 100 * 1024 * 1024)
        assert skip is True
        assert "Size" in reason
    
    def test_small_file_allowed(self):
        skip, _ = should_skip_file("data.csv", 1024)
        assert skip is False


class TestScannable:
    def test_d_drive_scannable(self):
        ok, reason = is_scannable_path(r"D:\Orb\app\main.py")
        assert ok is True
        assert reason is None
    
    def test_system_not_scannable(self):
        ok, reason = is_scannable_path(r"C:\Windows\cmd.exe")
        assert ok is False
        assert "System path" in reason
    
    def test_node_modules_not_scannable(self):
        ok, reason = is_scannable_path(r"D:\proj\node_modules\x")
        assert ok is False
        assert "Skip dir" in reason
    
    def test_git_not_scannable(self):
        ok, reason = is_scannable_path(r"D:\Orb\.git\config")
        assert ok is False
        assert "Skip dir" in reason


class TestRootAlias:
    def test_d_drive(self):
        result = get_root_alias(r"D:\Orb\app")
        assert result is not None
        alias, kind, zone = result
        assert alias == "d-drive"
        assert kind == "sandbox"
    
    def test_d_drive_root(self):
        result = get_root_alias("D:\\")
        assert result is not None
        alias, kind, zone = result
        assert alias == "d-drive"
    
    def test_user_home(self):
        result = get_root_alias(str(Path.home() / "Documents"))
        assert result is not None
        alias, kind, zone = result
        assert alias == "user-home"
        assert kind == "user"


class TestScanTargets:
    def test_returns_existing_roots(self):
        targets = get_scan_targets()
        assert isinstance(targets, list)
        # D:\ should exist on the target system
        assert any("D:" in t or "d:" in t.lower() for t in targets)


class TestZobieConfig:
    def test_zobie_output_dir_set(self):
        output_dir = get_zobie_output_dir()
        assert output_dir is not None
        assert "Tools" in output_dir or "tools" in output_dir.lower()
    
    def test_zobie_output_dir_constant(self):
        assert ZOBIE_OUTPUT_DIR is not None
        assert isinstance(ZOBIE_OUTPUT_DIR, str)
    
    def test_get_latest_signatures(self):
        # This may return None if no files exist, which is acceptable
        result = get_latest_zobie_file("SIGNATURES_")
        if result is not None:
            assert "SIGNATURES_" in result
            assert result.endswith(".json")
    
    def test_get_latest_index(self):
        result = get_latest_zobie_file("INDEX_")
        if result is not None:
            assert "INDEX_" in result
            assert result.endswith(".json")
