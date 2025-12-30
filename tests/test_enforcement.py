# FILE: tests/test_enforcement.py
"""Tests for Overwatcher enforcement and incident reports.

Tests the safety layer:
- Code detection
- Token limits
- Structure validation
- Incident report generation
"""

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import json
import pytest
from datetime import datetime, timezone


# =============================================================================
# Enforcement Tests
# =============================================================================

class TestCodeDetection:
    """Tests for code detection in Overwatcher output."""
    
    def test_detect_code_fence(self):
        """Should detect code fences."""
        from app.overwatcher.enforcement import detect_code_violations, ViolationType
        
        text = """
        The fix is:
        ```python
        def foo():
            return 42
        ```
        """
        
        violations = detect_code_violations(text)
        assert len(violations) >= 1
        assert any(v.type == ViolationType.CODE_FENCE for v in violations)
    
    def test_detect_function_definition(self):
        """Should detect function definitions."""
        from app.overwatcher.enforcement import detect_code_violations, ViolationType
        
        text = """
        Create this function:
        def calculate_total(items):
            return sum(items)
        """
        
        violations = detect_code_violations(text)
        assert any(v.type == ViolationType.FUNCTION_DEF for v in violations)
    
    def test_detect_shell_commands(self):
        """Should detect shell commands."""
        from app.overwatcher.enforcement import detect_code_violations, ViolationType
        
        text = """
        Run these commands:
        $ pip install requests
        $ pytest tests/
        """
        
        violations = detect_code_violations(text)
        assert any(v.type == ViolationType.SHELL_COMMAND for v in violations)
    
    def test_detect_powershell_commands(self):
        """Should detect PowerShell commands."""
        from app.overwatcher.enforcement import detect_code_violations, ViolationType
        
        text = """
        Execute:
        PS> Get-ChildItem -Path C:\\
        Copy-Item foo.txt bar.txt
        """
        
        violations = detect_code_violations(text)
        assert any(v.type == ViolationType.SHELL_COMMAND for v in violations)
    
    def test_detect_diff_patch(self):
        """Should detect diff/patch content."""
        from app.overwatcher.enforcement import detect_code_violations, ViolationType
        
        text = """
        Apply this patch:
        --- a/file.py
        +++ b/file.py
        @@ -1,3 +1,4 @@
        """
        
        violations = detect_code_violations(text)
        assert any(v.type == ViolationType.DIFF_PATCH for v in violations)
    
    def test_detect_import_statements(self):
        """Should detect import statements."""
        from app.overwatcher.enforcement import detect_code_violations, ViolationType
        
        text = """
        Add these imports:
        import os
        from pathlib import Path
        """
        
        violations = detect_code_violations(text)
        assert any(v.type == ViolationType.IMPORT_STATEMENT for v in violations)
    
    def test_clean_output_no_violations(self):
        """Clean diagnostic text should have no violations."""
        from app.overwatcher.enforcement import detect_code_violations
        
        text = """
        DECISION: FAIL
        
        DIAGNOSIS: The test is failing because the function returns None
        instead of the expected value. The root cause is a missing return
        statement in the calculate_total function.
        
        FIX_ACTIONS:
        1. Add a return statement to calculate_total
        2. Ensure the function handles empty lists
        
        VERIFICATION: Run pytest tests/test_calc.py
        """
        
        violations = detect_code_violations(text)
        # Should have no code violations (or minimal)
        code_violations = [v for v in violations if v.type.value != "shell_command"]
        # "Run pytest" might trigger shell detection, which is fine as a warning
        assert len(code_violations) == 0


class TestTokenLimits:
    """Tests for token limit enforcement."""
    
    def test_estimate_tokens(self):
        """Token estimation should be reasonable."""
        from app.overwatcher.enforcement import estimate_tokens
        
        text = "Hello world " * 100  # ~1200 chars
        tokens = estimate_tokens(text)
        
        # Should be roughly 300 tokens (1200 / 4)
        assert 250 <= tokens <= 400
    
    def test_within_default_limit(self):
        """Output within default limit should pass."""
        from app.overwatcher.enforcement import enforce_overwatcher_output
        
        text = "DECISION: PASS\nDIAGNOSIS: All tests passing." * 10
        result = enforce_overwatcher_output(text)
        
        # May have some violations but token should be OK
        token_violations = [v for v in result.violations 
                          if v.type.value == "token_limit"]
        assert len(token_violations) == 0
    
    def test_exceeds_max_limit(self):
        """Output exceeding max limit should fail."""
        from app.overwatcher.enforcement import enforce_overwatcher_output, ViolationType
        
        text = "x" * 10000  # ~2500 tokens
        result = enforce_overwatcher_output(text, max_tokens=2000)
        
        token_violations = [v for v in result.violations 
                          if v.type == ViolationType.TOKEN_LIMIT]
        assert len(token_violations) == 1
    
    def test_break_glass_allows_more(self):
        """Break glass should allow more tokens."""
        from app.overwatcher.enforcement import enforce_overwatcher_output
        
        text = "x" * 10000  # ~2500 tokens
        
        # Without break glass - should fail
        result1 = enforce_overwatcher_output(text, max_tokens=2000, break_glass=False)
        assert not result1.valid or any(v.type.value == "token_limit" for v in result1.violations)
        
        # With break glass - should pass token check
        result2 = enforce_overwatcher_output(text, max_tokens=2000, break_glass=True)
        token_errors = [v for v in result2.violations 
                       if v.type.value == "token_limit" and v.severity == "error"]
        assert len(token_errors) == 0


class TestStructureValidation:
    """Tests for output structure validation."""
    
    def test_valid_structure(self):
        """Valid structure should pass."""
        from app.overwatcher.enforcement import validate_structure
        
        output = {
            "decision": "FAIL",
            "diagnosis": "Test failed due to assertion error",
            "fix_actions": [
                {"description": "Fix the return value"}
            ],
        }
        
        violations = validate_structure(output)
        assert len(violations) == 0
    
    def test_missing_decision(self):
        """Missing decision should fail."""
        from app.overwatcher.enforcement import validate_structure, ViolationType
        
        output = {
            "diagnosis": "Something is wrong",
        }
        
        violations = validate_structure(output)
        assert any(v.type == ViolationType.MISSING_FIELD for v in violations)
    
    def test_missing_diagnosis(self):
        """Missing diagnosis should fail."""
        from app.overwatcher.enforcement import validate_structure, ViolationType
        
        output = {
            "decision": "FAIL",
        }
        
        violations = validate_structure(output)
        assert any(v.type == ViolationType.MISSING_FIELD for v in violations)
    
    def test_invalid_decision_value(self):
        """Invalid decision value should fail."""
        from app.overwatcher.enforcement import validate_structure, ViolationType
        
        output = {
            "decision": "MAYBE",  # Invalid
            "diagnosis": "Not sure",
        }
        
        violations = validate_structure(output)
        assert any(v.type == ViolationType.INVALID_STRUCTURE for v in violations)


class TestEnforceAndReject:
    """Tests for the main enforcement function."""
    
    def test_valid_output_accepted(self):
        """Valid output should be accepted."""
        from app.overwatcher.enforcement import enforce_and_reject
        
        raw = "DECISION: PASS\nDIAGNOSIS: All tests pass"
        parsed = {"decision": "PASS", "diagnosis": "All tests pass"}
        
        valid, output, result = enforce_and_reject(raw, parsed)
        
        # May have warnings but should be valid
        assert output == parsed
    
    def test_code_output_rejected(self):
        """Output with code should be rejected."""
        from app.overwatcher.enforcement import enforce_and_reject
        
        raw = """
        DECISION: FAIL
        DIAGNOSIS: Fix needed
        ```python
        def fix():
            pass
        ```
        """
        parsed = {"decision": "FAIL", "diagnosis": "Fix needed"}
        
        valid, output, result = enforce_and_reject(raw, parsed)
        
        assert valid is False
        assert output is None
        assert len(result.violations) > 0


# =============================================================================
# Incident Report Tests
# =============================================================================

class TestIncidentReport:
    """Tests for incident report generation."""
    
    def test_build_incident_report(self):
        """Should build a complete incident report."""
        from app.overwatcher.incident_report import (
            build_incident_report,
            StrikeRecord,
            IncidentSeverity,
        )
        
        strikes = [
            StrikeRecord(
                strike_number=1,
                timestamp="2024-01-01T00:00:00Z",
                error_signature_hash="abc123",
                exception_type="AssertionError",
                failing_component="test_foo",
                diagnosis="Assertion failed",
                fix_actions_attempted=["Fix return value"],
                outcome="same_error",
                error_output_excerpt="AssertionError: expected 1, got None",
            ),
            StrikeRecord(
                strike_number=2,
                timestamp="2024-01-01T00:01:00Z",
                error_signature_hash="abc123",
                exception_type="AssertionError",
                failing_component="test_foo",
                diagnosis="Still failing",
                fix_actions_attempted=["Add return statement"],
                outcome="same_error",
                error_output_excerpt="AssertionError: expected 1, got None",
                deep_research_used=True,
                deep_research_findings="Common issue with None returns",
            ),
            StrikeRecord(
                strike_number=3,
                timestamp="2024-01-01T00:02:00Z",
                error_signature_hash="abc123",
                exception_type="AssertionError",
                failing_component="test_foo",
                diagnosis="Cannot fix automatically",
                fix_actions_attempted=["Manual review needed"],
                outcome="same_error",
                error_output_excerpt="AssertionError: expected 1, got None",
            ),
        ]
        
        report = build_incident_report(
            job_id="job-123",
            chunk_id="CHUNK-001",
            stage="verification",
            error_signature_hash="abc123",
            exception_type="AssertionError",
            strikes=strikes,
            root_cause_hypothesis="Missing return statement in function",
            affected_files=["app/calc.py"],
            blockers=["Unknown root cause"],
            suggested_actions=["Manual code review", "Check function logic"],
        )
        
        assert report.job_id == "job-123"
        assert report.chunk_id == "CHUNK-001"
        assert len(report.strikes) == 3
        assert report.error_signature_hash == "abc123"
        assert not report.finalized
    
    def test_finalize_report(self):
        """Finalizing should compute hash and lock report."""
        from app.overwatcher.incident_report import build_incident_report, StrikeRecord
        
        strikes = [
            StrikeRecord(
                strike_number=1,
                timestamp="2024-01-01T00:00:00Z",
                error_signature_hash="abc123",
                exception_type="AssertionError",
                failing_component="test_foo",
                diagnosis="Assertion failed",
                fix_actions_attempted=[],
                outcome="same_error",
                error_output_excerpt="Error",
            ),
        ]
        
        report = build_incident_report(
            job_id="job-123",
            chunk_id="CHUNK-001",
            stage="verification",
            error_signature_hash="abc123",
            exception_type="AssertionError",
            strikes=strikes,
            root_cause_hypothesis="Unknown",
            affected_files=[],
            blockers=[],
            suggested_actions=[],
        )
        
        assert not report.finalized
        assert report.report_hash == ""
        
        report.finalize()
        
        assert report.finalized
        assert report.report_hash != ""
        assert len(report.report_hash) == 64  # SHA256
    
    def test_finalize_twice_raises(self):
        """Finalizing twice should raise error."""
        from app.overwatcher.incident_report import build_incident_report, StrikeRecord
        
        strikes = [
            StrikeRecord(
                strike_number=1,
                timestamp="2024-01-01T00:00:00Z",
                error_signature_hash="abc123",
                exception_type="AssertionError",
                failing_component="test_foo",
                diagnosis="Assertion failed",
                fix_actions_attempted=[],
                outcome="same_error",
                error_output_excerpt="Error",
            ),
        ]
        
        report = build_incident_report(
            job_id="job-123",
            chunk_id="CHUNK-001",
            stage="verification",
            error_signature_hash="abc123",
            exception_type="AssertionError",
            strikes=strikes,
            root_cause_hypothesis="Unknown",
            affected_files=[],
            blockers=[],
            suggested_actions=[],
        )
        
        report.finalize()
        
        with pytest.raises(ValueError, match="already finalized"):
            report.finalize()
    
    def test_to_markdown(self):
        """Should generate readable markdown."""
        from app.overwatcher.incident_report import build_incident_report, StrikeRecord
        
        strikes = [
            StrikeRecord(
                strike_number=1,
                timestamp="2024-01-01T00:00:00Z",
                error_signature_hash="abc123",
                exception_type="AssertionError",
                failing_component="test_foo",
                diagnosis="Assertion failed",
                fix_actions_attempted=["Fix return"],
                outcome="same_error",
                error_output_excerpt="Error",
            ),
        ]
        
        report = build_incident_report(
            job_id="job-123",
            chunk_id="CHUNK-001",
            stage="verification",
            error_signature_hash="abc123",
            exception_type="AssertionError",
            strikes=strikes,
            root_cause_hypothesis="Missing return",
            affected_files=["app/calc.py"],
            blockers=["Unknown cause"],
            suggested_actions=["Review code"],
        )
        
        md = report.to_markdown()
        
        assert "# Incident Report" in md
        assert "job-123" in md
        assert "CHUNK-001" in md
        assert "Strike 1" in md
        assert "AssertionError" in md
        assert "Missing return" in md
    
    def test_classify_error_category(self):
        """Should correctly classify errors."""
        from app.overwatcher.incident_report import classify_error_category, IncidentCategory
        
        # Test failures
        cat = classify_error_category("AssertionError", "pytest collected 5 items")
        assert cat == IncidentCategory.TEST_FAILURE
        
        # Type errors
        cat = classify_error_category("TypeError", "mypy found 3 errors")
        assert cat == IncidentCategory.TYPE_ERROR
        
        # Import errors
        cat = classify_error_category("ModuleNotFoundError", "No module named foo")
        assert cat == IncidentCategory.DEPENDENCY_ERROR
    
    def test_classify_severity(self):
        """Should correctly classify severity."""
        from app.overwatcher.incident_report import (
            classify_severity, 
            IncidentCategory,
            IncidentSeverity,
        )
        
        # Critical for auth files
        sev = classify_severity(IncidentCategory.CODE_ERROR, ["app/auth/login.py"])
        assert sev == IncidentSeverity.CRITICAL
        
        # Critical for boundary violations
        sev = classify_severity(IncidentCategory.BOUNDARY_VIOLATION, ["app/foo.py"])
        assert sev == IncidentSeverity.CRITICAL
        
        # High for test failures
        sev = classify_severity(IncidentCategory.TEST_FAILURE, ["app/foo.py"])
        assert sev == IncidentSeverity.HIGH


class TestImplementerValidation:
    """Tests for Implementer output validation."""
    
    def test_normal_code_allowed(self):
        """Implementer should be allowed to produce code."""
        from app.overwatcher.enforcement import is_implementer_output_valid
        
        output = """
        ```python
        def calculate_total(items):
            return sum(items)
        ```
        """
        
        assert is_implementer_output_valid(output) is True
    
    def test_dangerous_patterns_blocked(self):
        """Dangerous patterns should be blocked even for Implementer."""
        from app.overwatcher.enforcement import is_implementer_output_valid
        
        # rm -rf /
        assert is_implementer_output_valid("rm -rf /") is False
        
        # Fork bomb
        assert is_implementer_output_valid(":(){:|:&};:") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
