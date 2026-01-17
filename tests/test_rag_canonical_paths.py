# tests/test_rag_canonical_paths.py
"""Tests for canonical path utilities."""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import os

from app.rag.utils.canonical_paths import (
    canonicalize_path,
    parse_canonical_path,
    canonical_to_absolute,
    get_canonical_directory,
    is_under_canonical_prefix,
    get_path_depth,
)


class TestCanonicalize:
    def test_d_drive_file(self):
        canonical, alias, kind, zone = canonicalize_path(r"D:\Orb\app\main.py")
        assert canonical == "sandbox:d-drive/Orb/app/main.py"
        assert alias == "d-drive"
        assert kind == "sandbox"
    
    def test_d_drive_root(self):
        canonical, _, _, _ = canonicalize_path("D:\\")
        assert canonical == "sandbox:d-drive"
    
    def test_forward_slashes(self):
        canonical, _, _, _ = canonicalize_path(r"D:\Orb\app\llm")
        assert "\\" not in canonical
    
    def test_system_path_raises(self):
        with pytest.raises(ValueError):
            canonicalize_path(r"C:\Windows\System32")
    
    def test_node_modules_raises(self):
        with pytest.raises(ValueError):
            canonicalize_path(r"D:\proj\node_modules\pkg")


class TestParse:
    def test_full_path(self):
        result = parse_canonical_path("sandbox:d-drive/Orb/app/main.py")
        assert result["root_kind"] == "sandbox"
        assert result["alias"] == "d-drive"
        assert result["relative_path"] == "Orb/app/main.py"
    
    def test_root_only(self):
        result = parse_canonical_path("sandbox:d-drive")
        assert result["relative_path"] == ""
    
    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_canonical_path("no-colon-here")


class TestRoundTrip:
    def test_preserves_path(self):
        original = r"D:\Orb\app\llm\router.py"
        canonical, _, _, _ = canonicalize_path(original)
        recovered = canonical_to_absolute(canonical)
        assert os.path.normpath(recovered).lower() == os.path.normpath(original).lower()
    
    def test_preserves_deep_path(self):
        original = r"D:\Orb\app\rag\config\scan_roots.py"
        canonical, _, _, _ = canonicalize_path(original)
        recovered = canonical_to_absolute(canonical)
        assert os.path.normpath(recovered).lower() == os.path.normpath(original).lower()
    
    def test_preserves_root(self):
        original = "D:\\"
        canonical, _, _, _ = canonicalize_path(original)
        recovered = canonical_to_absolute(canonical)
        assert os.path.normpath(recovered).lower() == os.path.normpath(original).lower()


class TestCanonicalToAbsolute:
    def test_full_path(self):
        result = canonical_to_absolute("sandbox:d-drive/Orb/app/main.py")
        assert "Orb" in result
        assert "main.py" in result
    
    def test_root_only(self):
        result = canonical_to_absolute("sandbox:d-drive")
        assert result.rstrip("\\") in ["D:", "D:\\"]
    
    def test_unknown_alias_raises(self):
        with pytest.raises(ValueError):
            canonical_to_absolute("sandbox:unknown-alias/file.py")


class TestDirectory:
    def test_nested(self):
        result = get_canonical_directory("sandbox:d-drive/Orb/app/llm/router.py")
        assert result == "sandbox:d-drive/Orb/app/llm"
    
    def test_top_level(self):
        result = get_canonical_directory("sandbox:d-drive/Orb/main.py")
        assert result == "sandbox:d-drive/Orb"
    
    def test_single_level(self):
        result = get_canonical_directory("sandbox:d-drive/file.py")
        assert result == "sandbox:d-drive"


class TestDepth:
    def test_root_zero(self):
        assert get_path_depth("sandbox:d-drive") == 0
    
    def test_one_level(self):
        assert get_path_depth("sandbox:d-drive/Orb") == 1
    
    def test_nested(self):
        assert get_path_depth("sandbox:d-drive/Orb/app/llm") == 3
    
    def test_deep(self):
        assert get_path_depth("sandbox:d-drive/Orb/app/rag/config/scan_roots.py") == 5


class TestPrefix:
    def test_under(self):
        assert is_under_canonical_prefix(
            "sandbox:d-drive/Orb/app/main.py",
            "sandbox:d-drive/Orb"
        ) is True
    
    def test_not_under(self):
        assert is_under_canonical_prefix(
            "sandbox:d-drive/tools/x.py",
            "sandbox:d-drive/Orb"
        ) is False
    
    def test_exact_match(self):
        assert is_under_canonical_prefix(
            "sandbox:d-drive/Orb",
            "sandbox:d-drive/Orb"
        ) is True
    
    def test_case_insensitive(self):
        assert is_under_canonical_prefix(
            "SANDBOX:D-DRIVE/Orb/app",
            "sandbox:d-drive/Orb"
        ) is True
    
    def test_trailing_slash_ignored(self):
        assert is_under_canonical_prefix(
            "sandbox:d-drive/Orb/app/",
            "sandbox:d-drive/Orb/"
        ) is True
