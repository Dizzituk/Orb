# FILE: tests/test_scan_security_v121.py
"""
v1.21 SCAN_ONLY Security Tests

These tests verify that SCAN_ONLY operations are constrained to the sandbox
workspace only and NEVER allow scanning the host PC filesystem.

CRITICAL SECURITY REQUIREMENT:
- Scans can ONLY target D:\Orb and D:\orb-desktop (SAFE_DEFAULT_SCAN_ROOTS)
- Bare drive letters (D:\, C:\) MUST be rejected
- "Entire D drive" MUST be interpreted as "entire allowed workspace"
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pot_spec.spec_gate_grounded import (
    _extract_scan_params,
    _is_path_within_allowed_roots,
    validate_scan_roots,
    SAFE_DEFAULT_SCAN_ROOTS,
    FORBIDDEN_SCAN_ROOTS,
)


class TestScanSecurity:
    """Tests for v1.21 SCAN_ONLY security hardening."""
    
    # =========================================================================
    # TEST A: "Scan the entire D drive and find Orb/ORB/orb"
    # Expected: scan_roots == SAFE_DEFAULT_SCAN_ROOTS, NOT "D:\"
    # =========================================================================
    
    def test_entire_d_drive_maps_to_safe_defaults(self):
        """
        Test A: 'Entire D drive' request should use SAFE_DEFAULT_SCAN_ROOTS,
        NOT bare drive letter 'D:\'.
        """
        text = "Scan the entire D drive and find Orb/ORB/orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None, "Should return scan params"
        assert result["scan_roots"] == SAFE_DEFAULT_SCAN_ROOTS, \
            f"Expected {SAFE_DEFAULT_SCAN_ROOTS}, got {result['scan_roots']}"
        assert "D:\\" not in result["scan_roots"], \
            "Bare drive root 'D:\\' must NOT be in scan_roots"
        assert "Orb" in result["scan_terms"], "Should extract 'Orb' term"
        assert "ORB" in result["scan_terms"], "Should extract 'ORB' term"
        assert "orb" in result["scan_terms"], "Should extract 'orb' term"
    
    # =========================================================================
    # TEST B: "Scan D:\Orb and D:\orb-desktop for orb"
    # Expected: scan_roots contains valid paths, no bare drive letter
    # =========================================================================
    
    def test_explicit_allowed_paths_accepted(self):
        """
        Test B: Explicit paths within allowed roots should be accepted.
        """
        text = "Scan D:\\Orb and D:\\orb-desktop for orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None, "Should return scan params"
        # Both paths should be accepted (they're within allowed roots)
        assert "D:\\Orb" in result["scan_roots"] or "D:\\orb-desktop" in result["scan_roots"], \
            f"Should accept valid paths, got {result['scan_roots']}"
        assert "D:\\" not in result["scan_roots"], \
            "Bare drive root 'D:\\' must NOT be in scan_roots"
    
    # =========================================================================
    # TEST C: "Scan D:\ for orb"
    # Expected: scan_roots == SAFE_DEFAULT_SCAN_ROOTS (bare drive rejected)
    # =========================================================================
    
    def test_bare_drive_rejected_fallback_to_defaults(self):
        """
        Test C: Bare drive letter 'D:\' should be rejected, 
        fallback to SAFE_DEFAULT_SCAN_ROOTS.
        """
        text = "Scan D:\\ for orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None, "Should return scan params"
        assert "D:\\" not in result["scan_roots"], \
            "Bare drive root 'D:\\' must be rejected"
        # Should fallback to safe defaults
        assert result["scan_roots"] == SAFE_DEFAULT_SCAN_ROOTS or \
               all(root in SAFE_DEFAULT_SCAN_ROOTS or 
                   any(root.lower().startswith(allowed.lower()) for allowed in SAFE_DEFAULT_SCAN_ROOTS) 
                   for root in result["scan_roots"]), \
            f"Should use safe defaults, got {result['scan_roots']}"
    
    # =========================================================================
    # TEST D: "Scan C:\Users\... for orb"
    # Expected: scan_roots == SAFE_DEFAULT_SCAN_ROOTS (path outside allowed)
    # =========================================================================
    
    def test_path_outside_allowed_roots_rejected(self):
        """
        Test D: Path outside allowed roots should be rejected,
        fallback to SAFE_DEFAULT_SCAN_ROOTS.
        """
        text = "Scan C:\\Users\\SomeUser\\Documents for orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None, "Should return scan params"
        assert "C:\\Users\\SomeUser\\Documents" not in result["scan_roots"], \
            "Path outside allowed roots must be rejected"
        assert "C:\\" not in result["scan_roots"], \
            "C:\\ drive must be rejected"
        # Should fallback to safe defaults
        assert result["scan_roots"] == SAFE_DEFAULT_SCAN_ROOTS, \
            f"Should use safe defaults, got {result['scan_roots']}"


class TestPathValidation:
    """Tests for path validation security functions."""
    
    def test_is_path_within_allowed_roots_accepts_orb(self):
        """D:\Orb should be accepted."""
        assert _is_path_within_allowed_roots("D:\\Orb") is True
    
    def test_is_path_within_allowed_roots_accepts_orb_desktop(self):
        """D:\orb-desktop should be accepted."""
        assert _is_path_within_allowed_roots("D:\\orb-desktop") is True
    
    def test_is_path_within_allowed_roots_accepts_subdirs(self):
        """Subdirectories within allowed roots should be accepted."""
        assert _is_path_within_allowed_roots("D:\\Orb\\app\\pot_spec") is True
        assert _is_path_within_allowed_roots("D:\\orb-desktop\\src") is True
    
    def test_is_path_within_allowed_roots_rejects_bare_drive(self):
        """Bare drive letters should be rejected."""
        assert _is_path_within_allowed_roots("D:\\") is False
        assert _is_path_within_allowed_roots("C:\\") is False
        assert _is_path_within_allowed_roots("D:") is False
        assert _is_path_within_allowed_roots("C:") is False
    
    def test_is_path_within_allowed_roots_rejects_outside_paths(self):
        """Paths outside allowed roots should be rejected."""
        assert _is_path_within_allowed_roots("C:\\Users\\SomeUser") is False
        assert _is_path_within_allowed_roots("D:\\OtherFolder") is False
        assert _is_path_within_allowed_roots("E:\\SomeDir") is False


class TestValidateScanRoots:
    """Tests for the validate_scan_roots security gate."""
    
    def test_validate_removes_bare_drives(self):
        """validate_scan_roots should remove bare drive letters."""
        valid, rejected = validate_scan_roots(["D:\\", "D:\\Orb"])
        
        assert "D:\\" not in valid, "Bare drive should be rejected"
        assert "D:\\" in rejected or "D:" in rejected, "Bare drive should be in rejected list"
        assert "D:\\Orb" in valid, "Valid path should be accepted"
    
    def test_validate_removes_outside_paths(self):
        """validate_scan_roots should remove paths outside allowed roots."""
        valid, rejected = validate_scan_roots(["C:\\Users", "D:\\Orb"])
        
        assert "C:\\Users" not in valid, "Outside path should be rejected"
        assert "C:\\Users" in rejected, "Outside path should be in rejected list"
        assert "D:\\Orb" in valid, "Valid path should be accepted"
    
    def test_validate_returns_defaults_when_all_rejected(self):
        """When all paths are rejected, should return SAFE_DEFAULT_SCAN_ROOTS."""
        valid, rejected = validate_scan_roots(["C:\\", "E:\\", "F:\\SomeDir"])
        
        assert valid == SAFE_DEFAULT_SCAN_ROOTS, \
            f"Should fallback to safe defaults, got {valid}"
        assert len(rejected) == 3, "All 3 paths should be rejected"


class TestScanTermsExtraction:
    """Tests for scan terms extraction (should still work correctly)."""
    
    def test_extracts_slash_separated_terms(self):
        """Should extract Orb/ORB/orb as separate terms."""
        text = "Scan D:\\Orb for Orb/ORB/orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None
        # Should have all three case variants
        terms_lower = [t.lower() for t in result["scan_terms"]]
        assert "orb" in terms_lower, "Should extract 'orb' term"
    
    def test_extracts_quoted_terms(self):
        """Should extract quoted terms."""
        text = 'Find files containing "Orb", "ORB", or "orb" in D:\\Orb'
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None
        assert len(result["scan_terms"]) >= 1, "Should extract at least one term"
    
    def test_case_sensitive_when_variants_present(self):
        """Should set case_sensitive when multiple case variants present."""
        text = "Scan D:\\Orb for Orb/ORB/orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None
        # When multiple case variants are provided, should be case_sensitive
        if len(result["scan_terms"]) > 1:
            assert result["scan_case_mode"] == "case_sensitive", \
                "Multiple case variants should trigger case_sensitive mode"


class TestWeaverFormatInputs:
    """Tests for Weaver-format inputs (summarized user intent)."""
    
    def test_weaver_d_colon_format(self):
        """Weaver format 'on D:' should use safe defaults."""
        text = "Find all files/folders on D: with names or contents containing Orb"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None
        assert "D:\\" not in result["scan_roots"], \
            "Bare drive root must be rejected even in Weaver format"
        assert result["scan_roots"] == SAFE_DEFAULT_SCAN_ROOTS, \
            f"Should use safe defaults, got {result['scan_roots']}"
    
    def test_weaver_d_drive_title_format(self):
        """Weaver format 'D: Orb reference scanner' should use safe defaults."""
        text = "D: Orb reference scanner - find all Orb/ORB/orb references"
        intent = {"raw_text": text}
        
        result = _extract_scan_params(text, intent)
        
        assert result is not None
        assert "D:\\" not in result["scan_roots"], \
            "Bare drive root must be rejected"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
