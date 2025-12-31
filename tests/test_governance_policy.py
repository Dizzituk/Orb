"""
Test suite for ASTRA Global Governance Policy
Spec Version: 1.0
Date: 2025-12-31

Run with: pytest test_governance_policy.py -v
"""

import pytest
import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional
from enum import StrEnum


# =============================================================================
# ENUMS (to be moved to Orb/app/overwatcher/ once approved)
# =============================================================================

class SignatureType(StrEnum):
    ERROR = "error"
    SPEC_HOLE = "spec_hole"


class Decision(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NEEDS_INFO = "needs_info"
    HARD_STOP = "hard_stop"
    CONTINUE = "continue"


class HoleType(StrEnum):
    MISSING_INFO = "missing_info"
    CONTRADICTION = "contradiction"
    AMBIGUITY = "ambiguity"
    SAFETY_GAP = "safety_gap"


class OverrideType(StrEnum):
    RESUME_AFTER_STRIKE3 = "resume_after_strike3"
    CLEAR_SIGNATURE = "clear_signature"
    FORCE_CONTINUE = "force_continue"


# =============================================================================
# VIOLATION PATTERNS (Section 5.4)
# =============================================================================

VIOLATION_PATTERNS = {
    "CODE_FENCE": r"```[\s\S]*?```",
    "FUNCTION_DEF": r"^\s*(def |async def |function |const .* = |class )",
    "IMPORT_STATEMENT": r"^\s*(import |from .* import |require\(|#include)",
    "SHELL_COMMAND": r"^\s*(\$|>|PS>|C:\\>|#)\s+\w+",
    "DIFF_PATCH": r"^(\+\+\+|---|@@|\+|\-)\s",
    "FILE_WRITE": r"(open\(|write\(|fs\.write|echo .* >)",
    "INDENTED_CODE": r"^    +(def |if |for |while |return |import |class )",
    "DATA_STRUCTURE": r"^\s*[\[\{][\s\S]{50,}[\]\}]\s*$",
}


# =============================================================================
# NORMALIZATION (Section 2.2.3)
# =============================================================================

def normalize_path_to_relative(match: re.Match) -> str:
    """Convert absolute path to repo-relative."""
    path = match.group(0)
    # Find common repo markers
    markers = ["Orb\\", "Orb/", "orb-desktop\\", "orb-desktop/", "sandbox_controller\\", "sandbox_controller/"]
    for marker in markers:
        if marker in path:
            idx = path.find(marker)
            return path[idx:]
    return path


def compute_error_signature(raw_error: str, stage: str) -> str:
    """
    Compute ErrorSignature per spec Section 2.
    
    Args:
        raw_error: Raw error text (may include timestamps, paths, etc.)
        stage: Pipeline stage (BUILD_TRIAGE, SPEC_GATE, etc.)
    
    Returns:
        64-character hex string (sha256)
    """
    text = raw_error
    
    # 1. Strip timestamps (ISO and common formats)
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)  # Time only
    
    # 2. Normalize Windows paths to repo-relative
    text = re.sub(r'[A-Z]:\\[^\s:]+', normalize_path_to_relative, text)
    
    # 3. Normalize Unix paths
    text = re.sub(r'/home/[^/]+/', '', text)
    text = re.sub(r'/Users/[^/]+/', '', text)
    
    # 4. Strip memory addresses
    text = re.sub(r'0x[0-9a-fA-F]+', '', text)
    
    # 5. Strip UUIDs
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '', text, flags=re.IGNORECASE)
    
    # 6. Strip line numbers that might shift
    # Keep file:line format but normalize volatile line numbers
    # text = re.sub(r':(\d+):', ':LINE:', text)
    
    # 7. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8. Lowercase for comparison
    text = text.lower()
    
    # 9. Truncate to prevent huge signatures
    text = text[:500]
    
    # 10. Combine with stage
    normalized = f"{stage}::{text}"
    
    return hashlib.sha256(normalized.encode()).hexdigest()


def compute_spec_hole_signature(holes: list[tuple[str, str]]) -> str:
    """
    Compute SpecHoleSignature per spec Section 3.
    
    Args:
        holes: List of (hole_type, canonical_field_name) tuples
    
    Returns:
        64-character hex string (sha256)
    """
    # Sort for determinism
    sorted_holes = sorted(set(holes))
    
    # Create canonical string
    canonical = "|".join(f"{h[0]}:{h[1]}" for h in sorted_holes)
    
    return hashlib.sha256(canonical.encode()).hexdigest()


# =============================================================================
# STRIKE STATE (Section 4)
# =============================================================================

@dataclass
class StrikeEvent:
    signature: str
    signature_type: SignatureType
    strike_number: int
    timestamp: str
    diagnosis_summary: str = ""


@dataclass
class StrikeResult:
    decision: Decision
    strike_count: int
    requires_incident_report: bool = False
    incident_report: Optional[dict] = None


@dataclass
class StrikeState:
    job_id: str
    strikes_by_error_sig: dict[str, int] = field(default_factory=dict)
    strikes_by_spec_hole_sig: dict[str, int] = field(default_factory=dict)
    history: list[StrikeEvent] = field(default_factory=list)
    
    def get_strike_count(self, signature: str) -> int:
        """Get current strike count for a signature."""
        return self.strikes_by_error_sig.get(signature, 0)
    
    def record_strike(self, signature: str, sig_type: SignatureType = SignatureType.ERROR) -> StrikeResult:
        """
        Record a strike for a signature.
        
        Returns StrikeResult with decision and whether incident report needed.
        """
        if sig_type == SignatureType.ERROR:
            current = self.strikes_by_error_sig.get(signature, 0)
            new_count = current + 1
            self.strikes_by_error_sig[signature] = new_count
        else:
            current = self.strikes_by_spec_hole_sig.get(signature, 0)
            new_count = current + 1
            self.strikes_by_spec_hole_sig[signature] = new_count
        
        self.history.append(StrikeEvent(
            signature=signature,
            signature_type=sig_type,
            strike_number=new_count,
            timestamp="2025-12-31T00:00:00Z"
        ))
        
        if new_count >= 3:
            return StrikeResult(
                decision=Decision.HARD_STOP,
                strike_count=new_count,
                requires_incident_report=True,
                incident_report={
                    "job_id": self.job_id,
                    "signature_value": signature,
                    "signature_type": sig_type.value,
                    "strike_history": [
                        {"strike": e.strike_number, "timestamp": e.timestamp}
                        for e in self.history if e.signature == signature
                    ]
                }
            )
        
        return StrikeResult(
            decision=Decision.CONTINUE,
            strike_count=new_count,
            requires_incident_report=False
        )
    
    def clear_signature(self, signature: str):
        """Clear strike count for a signature (human override)."""
        self.strikes_by_error_sig.pop(signature, None)
        self.strikes_by_spec_hole_sig.pop(signature, None)


# =============================================================================
# OUTPUT VALIDATION (Section 5)
# =============================================================================

def contains_violation(text: str) -> tuple[bool, Optional[str]]:
    """
    Check if text contains any policy violations.
    
    Returns:
        (has_violation, violation_type or None)
    """
    for violation_type, pattern in VIOLATION_PATTERNS.items():
        if re.search(pattern, text, re.MULTILINE):
            return True, violation_type
    return False, None


def validate_overwatcher_output(output: dict) -> tuple[bool, list[str]]:
    """
    Validate Overwatcher output against schema.
    
    Returns:
        (is_valid, list of errors)
    """
    errors = []
    
    # Required fields
    required = ["stage_name", "signature_type", "signature_value", "strike_count", "decision", "diagnosis", "constraints"]
    for field in required:
        if field not in output:
            errors.append(f"Missing required field: {field}")
    
    # Decision must be valid enum
    if "decision" in output:
        valid_decisions = {"CONTINUE", "HARD_STOP"}
        if output["decision"] not in valid_decisions:
            errors.append(f"Invalid decision: {output['decision']}")
    
    # If not HARD_STOP, must have fix_actions and verification
    if output.get("decision") != "HARD_STOP":
        if "fix_actions" not in output:
            errors.append("Missing fix_actions (required when decision != HARD_STOP)")
        if "verification" not in output:
            errors.append("Missing verification (required when decision != HARD_STOP)")
    
    # Check for violations in text fields
    text_fields = ["diagnosis", "constraints"]
    for field in text_fields:
        if field in output:
            content = output[field]
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            has_violation, violation_type = contains_violation(str(content))
            if has_violation:
                errors.append(f"Policy violation in {field}: {violation_type}")
    
    # Diagnosis max 8 items
    if "diagnosis" in output and isinstance(output["diagnosis"], list):
        if len(output["diagnosis"]) > 8:
            errors.append(f"Diagnosis has {len(output['diagnosis'])} items (max 8)")
    
    return len(errors) == 0, errors


# =============================================================================
# TESTS: ErrorSignature (Section 9.1)
# =============================================================================

class TestErrorSignature:
    """INV-ERR-*: ErrorSignature invariant tests"""
    
    def test_signature_determinism(self):
        """INV-ERR-001: Same error produces same signature"""
        error1 = "ValueError: expected str, got int at app/foo.py:42"
        error2 = "ValueError: expected str, got int at app/foo.py:42"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 == sig2, "Identical errors must produce identical signatures"
    
    def test_signature_ignores_timestamp(self):
        """INV-ERR-002: Different timestamps produce same signature"""
        error1 = "2025-12-31T10:00:00 ValueError: expected str, got int"
        error2 = "2025-12-31T11:30:45 ValueError: expected str, got int"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 == sig2, "Timestamps must be normalized out"
    
    def test_signature_normalizes_windows_paths(self):
        """INV-ERR-003: Different absolute paths produce same signature"""
        error1 = "Error at C:\\Users\\taz\\Orb\\app\\foo.py"
        error2 = "Error at D:\\Orb\\app\\foo.py"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 == sig2, "Absolute paths must be normalized to repo-relative"
    
    def test_signature_normalizes_unix_paths(self):
        """INV-ERR-003b: Unix home paths normalized"""
        error1 = "Error at /home/taz/Orb/app/foo.py"
        error2 = "Error at /home/other/Orb/app/foo.py"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 == sig2, "Unix home paths must be normalized"
    
    def test_signature_distinguishes_errors(self):
        """INV-ERR-004: Different errors produce different signatures"""
        error1 = "ValueError: x"
        error2 = "KeyError: x"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 != sig2, "Different error types must produce different signatures"
    
    def test_signature_strips_uuids(self):
        """INV-ERR-005: UUIDs are stripped"""
        error1 = "Job abc12345-1234-5678-9abc-def012345678 failed: ValueError"
        error2 = "Job 00000000-0000-0000-0000-000000000000 failed: ValueError"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 == sig2, "UUIDs must be stripped"
    
    def test_signature_strips_memory_addresses(self):
        """INV-ERR-006: Memory addresses are stripped"""
        error1 = "Object at 0x7fff5fbff8c0 is invalid"
        error2 = "Object at 0x1234567890ab is invalid"
        sig1 = compute_error_signature(error1, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error2, "BUILD_TRIAGE")
        assert sig1 == sig2, "Memory addresses must be stripped"
    
    def test_signature_includes_stage(self):
        """INV-ERR-007: Different stages produce different signatures"""
        error = "ValueError: x"
        sig1 = compute_error_signature(error, "BUILD_TRIAGE")
        sig2 = compute_error_signature(error, "PROMOTION_GATE")
        assert sig1 != sig2, "Stage context must be included in signature"
    
    def test_signature_is_64_chars(self):
        """INV-ERR-008: Signature is valid sha256 hex"""
        error = "ValueError: test"
        sig = compute_error_signature(error, "BUILD_TRIAGE")
        assert len(sig) == 64, "Signature must be 64 characters"
        assert all(c in "0123456789abcdef" for c in sig), "Signature must be hex"


# =============================================================================
# TESTS: Strike Counter (Section 9.2)
# =============================================================================

class TestStrikeCounter:
    """INV-STRIKE-*: Strike counter invariant tests"""
    
    def test_strike_increment(self):
        """INV-STRIKE-001: Strike increments on same signature"""
        state = StrikeState(job_id="test")
        sig = "a" * 64
        
        result1 = state.record_strike(sig)
        assert state.get_strike_count(sig) == 1
        assert result1.strike_count == 1
        
        result2 = state.record_strike(sig)
        assert state.get_strike_count(sig) == 2
        assert result2.strike_count == 2
    
    def test_strike_reset_on_new_sig(self):
        """INV-STRIKE-002: New signature starts at strike 1"""
        state = StrikeState(job_id="test")
        sig_a = "a" * 64
        sig_b = "b" * 64
        
        state.record_strike(sig_a)
        state.record_strike(sig_a)  # Strike 2 for sig_a
        
        result = state.record_strike(sig_b)  # New signature
        assert result.strike_count == 1, "New signature must start at strike 1"
        assert state.get_strike_count(sig_a) == 2, "Old signature count preserved"
    
    def test_strike_3_hard_stop(self):
        """INV-STRIKE-003: Strike 3 triggers HARD_STOP"""
        state = StrikeState(job_id="test")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        
        assert result.decision == Decision.HARD_STOP
        assert result.requires_incident_report is True
        assert result.incident_report is not None
    
    def test_strike_3_incident_has_signature(self):
        """INV-STRIKE-004: Incident report contains signature"""
        state = StrikeState(job_id="test-job")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        
        assert result.incident_report["signature_value"] == sig
        assert result.incident_report["job_id"] == "test-job"
    
    def test_human_override_clears_signature(self):
        """INV-STRIKE-005: Human override clears signature"""
        state = StrikeState(job_id="test")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        state.clear_signature(sig)
        
        assert state.get_strike_count(sig) == 0
    
    def test_strike_history_preserved(self):
        """INV-STRIKE-006: Strike history is preserved"""
        state = StrikeState(job_id="test")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        
        assert len(state.history) == 2
        assert all(e.signature == sig for e in state.history)


# =============================================================================
# TESTS: Output Contract (Section 9.3)
# =============================================================================

class TestOutputContract:
    """INV-CONTRACT-*: Output contract invariant tests"""
    
    def test_no_code_fence(self):
        """INV-CONTRACT-001a: Code fences detected"""
        text = "Here is the fix:\n```python\ndef foo(): pass\n```"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "CODE_FENCE"
    
    def test_no_function_def(self):
        """INV-CONTRACT-001b: Function definitions detected"""
        text = "def process_data(x):\n    return x"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "FUNCTION_DEF"
    
    def test_no_shell_command(self):
        """INV-CONTRACT-001c: Shell commands detected"""
        text = "Run this:\n$ pip install requests"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "SHELL_COMMAND"
    
    def test_no_powershell_command(self):
        """INV-CONTRACT-001d: PowerShell commands detected"""
        text = "Execute:\nPS> Get-Process"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "SHELL_COMMAND"
    
    def test_no_import_statement(self):
        """INV-CONTRACT-001e: Import statements detected"""
        text = "import os\nfrom pathlib import Path"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "IMPORT_STATEMENT"
    
    def test_valid_output_passes(self):
        """INV-CONTRACT-002: Valid schema passes validation"""
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 1,
            "decision": "CONTINUE",
            "diagnosis": ["Error in module X", "Caused by missing dependency"],
            "fix_actions": [{"action_id": "FA001", "description": "Install missing package"}],
            "constraints": ["Do not modify production config"],
            "verification": {"required_evidence": [{"evidence_id": "EV001", "description": "pip freeze output"}]}
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is True, f"Should be valid, got errors: {errors}"
    
    def test_missing_required_field(self):
        """INV-CONTRACT-003: Missing required field fails"""
        output = {
            "stage_name": "BUILD_TRIAGE",
            # Missing other required fields
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is False
        assert any("Missing required field" in e for e in errors)
    
    def test_hard_stop_no_fix_actions_ok(self):
        """INV-CONTRACT-004: HARD_STOP doesn't require fix_actions"""
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 3,
            "decision": "HARD_STOP",
            "diagnosis": ["Unrecoverable error"],
            "constraints": ["No further attempts"]
            # No fix_actions or verification - OK for HARD_STOP
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is True, f"HARD_STOP should not require fix_actions: {errors}"
    
    def test_diagnosis_max_8_items(self):
        """INV-CONTRACT-005: Diagnosis limited to 8 items"""
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 1,
            "decision": "CONTINUE",
            "diagnosis": [f"Item {i}" for i in range(10)],  # 10 items
            "fix_actions": [],
            "constraints": [],
            "verification": {}
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is False
        assert any("max 8" in e for e in errors)
    
    def test_code_in_diagnosis_fails(self):
        """INV-CONTRACT-006: Code in diagnosis fails validation"""
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 1,
            "decision": "CONTINUE",
            "diagnosis": ["def fix_it(): pass"],  # Contains code!
            "fix_actions": [],
            "constraints": [],
            "verification": {}
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is False
        assert any("Policy violation" in e for e in errors)


# =============================================================================
# TESTS: SpecHoleSignature (Section 9.4)
# =============================================================================

class TestSpecHoleSignature:
    """INV-SPEC-*: Spec Gate invariant tests"""
    
    def test_same_holes_same_signature(self):
        """INV-SPEC-001: Same holes produce same signature"""
        holes1 = [("MISSING_INFO", "rollback_strategy"), ("AMBIGUITY", "success_criteria")]
        holes2 = [("AMBIGUITY", "success_criteria"), ("MISSING_INFO", "rollback_strategy")]  # Different order
        
        sig1 = compute_spec_hole_signature(holes1)
        sig2 = compute_spec_hole_signature(holes2)
        
        assert sig1 == sig2, "Order should not affect signature"
    
    def test_different_holes_different_signature(self):
        """INV-SPEC-002: Different holes produce different signatures"""
        holes1 = [("MISSING_INFO", "rollback_strategy")]
        holes2 = [("MISSING_INFO", "budget_tokens")]
        
        sig1 = compute_spec_hole_signature(holes1)
        sig2 = compute_spec_hole_signature(holes2)
        
        assert sig1 != sig2
    
    def test_duplicate_holes_deduped(self):
        """INV-SPEC-003: Duplicate holes are deduped"""
        holes1 = [("MISSING_INFO", "rollback_strategy")]
        holes2 = [("MISSING_INFO", "rollback_strategy"), ("MISSING_INFO", "rollback_strategy")]
        
        sig1 = compute_spec_hole_signature(holes1)
        sig2 = compute_spec_hole_signature(holes2)
        
        assert sig1 == sig2, "Duplicates should be removed"
    
    def test_spec_hole_strike_3_hard_stop(self):
        """INV-SPEC-004: Spec hole strike 3 triggers HARD_STOP"""
        state = StrikeState(job_id="test")
        sig = compute_spec_hole_signature([("MISSING_INFO", "rollback_strategy")])
        
        state.record_strike(sig, SignatureType.SPEC_HOLE)
        state.record_strike(sig, SignatureType.SPEC_HOLE)
        result = state.record_strike(sig, SignatureType.SPEC_HOLE)
        
        assert result.decision == Decision.HARD_STOP


# =============================================================================
# TESTS: Incident Report (Section 9.5)
# =============================================================================

class TestIncidentReport:
    """INV-INCIDENT-*: Incident report invariant tests"""
    
    def test_incident_created_on_strike_3(self):
        """INV-INCIDENT-001: Incident created on Strike 3"""
        state = StrikeState(job_id="test")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        
        assert result.incident_report is not None
        assert result.incident_report["signature_value"] == sig
    
    def test_incident_has_strike_history(self):
        """INV-INCIDENT-002: Incident contains strike history"""
        state = StrikeState(job_id="test")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        
        history = result.incident_report["strike_history"]
        assert len(history) == 3
        assert history[0]["strike"] == 1
        assert history[1]["strike"] == 2
        assert history[2]["strike"] == 3
    
    def test_incident_has_job_id(self):
        """INV-INCIDENT-003: Incident contains job_id"""
        state = StrikeState(job_id="job-12345")
        sig = "a" * 64
        
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        
        assert result.incident_report["job_id"] == "job-12345"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
