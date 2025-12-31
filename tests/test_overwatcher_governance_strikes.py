# FILE: tests/test_overwatcher_governance_strikes.py
"""
Tests for app/overwatcher governance and strike logic
Global Governance Policy - ErrorSignature, SpecHoleSignature, Strike State
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
import re
import json
import hashlib
import tempfile
from dataclasses import dataclass, field
from typing import Optional
from enum import StrEnum
from datetime import datetime, timezone


# =============================================================================
# Enums (mirrors app/overwatcher/schemas.py)
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
# Violation Patterns
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
# Normalization
# =============================================================================

def normalize_path_to_relative(match: re.Match) -> str:
    path = match.group(0)
    markers = ["Orb\\", "Orb/", "orb-desktop\\", "orb-desktop/", "sandbox_controller\\", "sandbox_controller/"]
    for marker in markers:
        if marker in path:
            idx = path.find(marker)
            return path[idx:]
    return path


def compute_error_signature(raw_error: str, stage: str) -> str:
    text = raw_error
    text = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*Z?', '', text)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    text = re.sub(r'[A-Z]:\\[^\s:]+', normalize_path_to_relative, text)
    text = re.sub(r'/home/[^/]+/', '', text)
    text = re.sub(r'/Users/[^/]+/', '', text)
    text = re.sub(r'0x[0-9a-fA-F]+', '', text)
    text = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = text[:500]
    normalized = f"{stage}::{text}"
    return hashlib.sha256(normalized.encode()).hexdigest()


def compute_spec_hole_signature(holes: list[tuple[str, str]]) -> str:
    sorted_holes = sorted(set(holes))
    canonical = "|".join(f"{h[0]}:{h[1]}" for h in sorted_holes)
    return hashlib.sha256(canonical.encode()).hexdigest()


# =============================================================================
# Strike State
# =============================================================================

@dataclass
class StrikeEvent:
    signature: str
    signature_type: SignatureType
    strike_number: int
    timestamp: str
    diagnosis_summary: str = ""
    
    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "signature_type": self.signature_type.value,
            "strike_number": self.strike_number,
            "timestamp": self.timestamp,
            "diagnosis_summary": self.diagnosis_summary
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "StrikeEvent":
        return cls(
            signature=d["signature"],
            signature_type=SignatureType(d["signature_type"]),
            strike_number=d["strike_number"],
            timestamp=d["timestamp"],
            diagnosis_summary=d.get("diagnosis_summary", "")
        )


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
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def get_strike_count(self, signature: str) -> int:
        return self.strikes_by_error_sig.get(signature, 0)
    
    def record_strike(self, signature: str, sig_type: SignatureType = SignatureType.ERROR) -> StrikeResult:
        if sig_type == SignatureType.ERROR:
            current = self.strikes_by_error_sig.get(signature, 0)
            new_count = current + 1
            self.strikes_by_error_sig[signature] = new_count
        else:
            current = self.strikes_by_spec_hole_sig.get(signature, 0)
            new_count = current + 1
            self.strikes_by_spec_hole_sig[signature] = new_count
        
        now = datetime.now(timezone.utc).isoformat()
        self.updated_at = now
        
        self.history.append(StrikeEvent(
            signature=signature,
            signature_type=sig_type,
            strike_number=new_count,
            timestamp=now
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
        self.strikes_by_error_sig.pop(signature, None)
        self.strikes_by_spec_hole_sig.pop(signature, None)
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "strikes_by_error_sig": self.strikes_by_error_sig,
            "strikes_by_spec_hole_sig": self.strikes_by_spec_hole_sig,
            "history": [e.to_dict() for e in self.history]
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "StrikeState":
        return cls(
            job_id=d["job_id"],
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            strikes_by_error_sig=d.get("strikes_by_error_sig", {}),
            strikes_by_spec_hole_sig=d.get("strikes_by_spec_hole_sig", {}),
            history=[StrikeEvent.from_dict(e) for e in d.get("history", [])]
        )
    
    def save(self, jobs_dir: Path) -> Path:
        governance_dir = jobs_dir / self.job_id / "governance"
        governance_dir.mkdir(parents=True, exist_ok=True)
        state_file = governance_dir / "strike_state.json"
        with open(state_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return state_file
    
    @classmethod
    def load(cls, jobs_dir: Path, job_id: str) -> Optional["StrikeState"]:
        state_file = jobs_dir / job_id / "governance" / "strike_state.json"
        if not state_file.exists():
            return None
        with open(state_file) as f:
            data = json.load(f)
        return cls.from_dict(data)


# =============================================================================
# Output Validation
# =============================================================================

def contains_violation(text: str) -> tuple[bool, Optional[str]]:
    for violation_type, pattern in VIOLATION_PATTERNS.items():
        if re.search(pattern, text, re.MULTILINE):
            return True, violation_type
    return False, None


def validate_overwatcher_output(output: dict) -> tuple[bool, list[str]]:
    errors = []
    required = ["stage_name", "signature_type", "signature_value", "strike_count", "decision", "diagnosis", "constraints"]
    for fld in required:
        if fld not in output:
            errors.append(f"Missing required field: {fld}")
    
    if "decision" in output:
        valid_decisions = {"CONTINUE", "HARD_STOP"}
        if output["decision"] not in valid_decisions:
            errors.append(f"Invalid decision: {output['decision']}")
    
    if output.get("decision") != "HARD_STOP":
        if "fix_actions" not in output:
            errors.append("Missing fix_actions (required when decision != HARD_STOP)")
        if "verification" not in output:
            errors.append("Missing verification (required when decision != HARD_STOP)")
    
    text_fields = ["diagnosis", "constraints"]
    for fld in text_fields:
        if fld in output:
            content = output[fld]
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            has_violation, violation_type = contains_violation(str(content))
            if has_violation:
                errors.append(f"Policy violation in {fld}: {violation_type}")
    
    if "diagnosis" in output and isinstance(output["diagnosis"], list):
        if len(output["diagnosis"]) > 8:
            errors.append(f"Diagnosis has {len(output['diagnosis'])} items (max 8)")
    
    return len(errors) == 0, errors


# =============================================================================
# Tests: ErrorSignature
# =============================================================================

class TestErrorSignature:
    def test_signature_determinism(self):
        error1 = "ValueError: expected str, got int at app/foo.py:42"
        error2 = "ValueError: expected str, got int at app/foo.py:42"
        assert compute_error_signature(error1, "BUILD_TRIAGE") == compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_ignores_timestamp(self):
        error1 = "2025-12-31T10:00:00 ValueError: expected str, got int"
        error2 = "2025-12-31T11:30:45 ValueError: expected str, got int"
        assert compute_error_signature(error1, "BUILD_TRIAGE") == compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_normalizes_windows_paths(self):
        error1 = "Error at C:\\Users\\taz\\Orb\\app\\foo.py"
        error2 = "Error at D:\\Orb\\app\\foo.py"
        assert compute_error_signature(error1, "BUILD_TRIAGE") == compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_normalizes_unix_paths(self):
        error1 = "Error at /home/taz/Orb/app/foo.py"
        error2 = "Error at /home/other/Orb/app/foo.py"
        assert compute_error_signature(error1, "BUILD_TRIAGE") == compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_distinguishes_errors(self):
        error1 = "ValueError: x"
        error2 = "KeyError: x"
        assert compute_error_signature(error1, "BUILD_TRIAGE") != compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_strips_uuids(self):
        error1 = "Job abc12345-1234-5678-9abc-def012345678 failed: ValueError"
        error2 = "Job 00000000-0000-0000-0000-000000000000 failed: ValueError"
        assert compute_error_signature(error1, "BUILD_TRIAGE") == compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_strips_memory_addresses(self):
        error1 = "Object at 0x7fff5fbff8c0 is invalid"
        error2 = "Object at 0x1234567890ab is invalid"
        assert compute_error_signature(error1, "BUILD_TRIAGE") == compute_error_signature(error2, "BUILD_TRIAGE")
    
    def test_signature_includes_stage(self):
        error = "ValueError: x"
        assert compute_error_signature(error, "BUILD_TRIAGE") != compute_error_signature(error, "PROMOTION_GATE")
    
    def test_signature_is_64_chars(self):
        sig = compute_error_signature("ValueError: test", "BUILD_TRIAGE")
        assert len(sig) == 64
        assert all(c in "0123456789abcdef" for c in sig)


# =============================================================================
# Tests: Strike Counter
# =============================================================================

class TestStrikeCounter:
    def test_strike_increment(self):
        state = StrikeState(job_id="test")
        sig = "a" * 64
        result1 = state.record_strike(sig)
        assert state.get_strike_count(sig) == 1
        assert result1.strike_count == 1
        result2 = state.record_strike(sig)
        assert state.get_strike_count(sig) == 2
        assert result2.strike_count == 2
    
    def test_strike_reset_on_new_sig(self):
        state = StrikeState(job_id="test")
        state.record_strike("a" * 64)
        state.record_strike("a" * 64)
        result = state.record_strike("b" * 64)
        assert result.strike_count == 1
        assert state.get_strike_count("a" * 64) == 2
    
    def test_strike_3_hard_stop(self):
        state = StrikeState(job_id="test")
        sig = "a" * 64
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        assert result.decision == Decision.HARD_STOP
        assert result.requires_incident_report is True
        assert result.incident_report is not None
    
    def test_strike_3_incident_has_signature(self):
        state = StrikeState(job_id="test-job")
        sig = "a" * 64
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        assert result.incident_report["signature_value"] == sig
        assert result.incident_report["job_id"] == "test-job"
    
    def test_human_override_clears_signature(self):
        state = StrikeState(job_id="test")
        sig = "a" * 64
        state.record_strike(sig)
        state.record_strike(sig)
        state.clear_signature(sig)
        assert state.get_strike_count(sig) == 0
    
    def test_strike_history_preserved(self):
        state = StrikeState(job_id="test")
        sig = "a" * 64
        state.record_strike(sig)
        state.record_strike(sig)
        assert len(state.history) == 2
        assert all(e.signature == sig for e in state.history)


# =============================================================================
# Tests: Output Contract
# =============================================================================

class TestOutputContract:
    def test_no_code_fence(self):
        text = "Here is the fix:\n```python\ndef foo(): pass\n```"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "CODE_FENCE"
    
    def test_no_function_def(self):
        text = "def process_data(x):\n    return x"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "FUNCTION_DEF"
    
    def test_no_shell_command(self):
        text = "Run this:\n$ pip install requests"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "SHELL_COMMAND"
    
    def test_no_powershell_command(self):
        text = "Execute:\nPS> Get-Process"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "SHELL_COMMAND"
    
    def test_no_import_statement(self):
        text = "import os\nfrom pathlib import Path"
        has_violation, vtype = contains_violation(text)
        assert has_violation is True
        assert vtype == "IMPORT_STATEMENT"
    
    def test_valid_output_passes(self):
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
        output = {"stage_name": "BUILD_TRIAGE"}
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is False
        assert any("Missing required field" in e for e in errors)
    
    def test_hard_stop_no_fix_actions_ok(self):
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 3,
            "decision": "HARD_STOP",
            "diagnosis": ["Unrecoverable error"],
            "constraints": ["No further attempts"]
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is True, f"HARD_STOP should not require fix_actions: {errors}"
    
    def test_diagnosis_max_8_items(self):
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 1,
            "decision": "CONTINUE",
            "diagnosis": [f"Item {i}" for i in range(10)],
            "fix_actions": [],
            "constraints": [],
            "verification": {}
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is False
        assert any("max 8" in e for e in errors)
    
    def test_code_in_diagnosis_fails(self):
        output = {
            "stage_name": "BUILD_TRIAGE",
            "signature_type": "ERROR",
            "signature_value": "a" * 64,
            "strike_count": 1,
            "decision": "CONTINUE",
            "diagnosis": ["def fix_it(): pass"],
            "fix_actions": [],
            "constraints": [],
            "verification": {}
        }
        is_valid, errors = validate_overwatcher_output(output)
        assert is_valid is False
        assert any("Policy violation" in e for e in errors)


# =============================================================================
# Tests: SpecHoleSignature
# =============================================================================

class TestSpecHoleSignature:
    def test_same_holes_same_signature(self):
        holes1 = [("MISSING_INFO", "rollback_strategy"), ("AMBIGUITY", "success_criteria")]
        holes2 = [("AMBIGUITY", "success_criteria"), ("MISSING_INFO", "rollback_strategy")]
        assert compute_spec_hole_signature(holes1) == compute_spec_hole_signature(holes2)
    
    def test_different_holes_different_signature(self):
        holes1 = [("MISSING_INFO", "rollback_strategy")]
        holes2 = [("MISSING_INFO", "budget_tokens")]
        assert compute_spec_hole_signature(holes1) != compute_spec_hole_signature(holes2)
    
    def test_duplicate_holes_deduped(self):
        holes1 = [("MISSING_INFO", "rollback_strategy")]
        holes2 = [("MISSING_INFO", "rollback_strategy"), ("MISSING_INFO", "rollback_strategy")]
        assert compute_spec_hole_signature(holes1) == compute_spec_hole_signature(holes2)
    
    def test_spec_hole_strike_3_hard_stop(self):
        state = StrikeState(job_id="test")
        sig = compute_spec_hole_signature([("MISSING_INFO", "rollback_strategy")])
        state.record_strike(sig, SignatureType.SPEC_HOLE)
        state.record_strike(sig, SignatureType.SPEC_HOLE)
        result = state.record_strike(sig, SignatureType.SPEC_HOLE)
        assert result.decision == Decision.HARD_STOP


# =============================================================================
# Tests: Incident Report
# =============================================================================

class TestIncidentReport:
    def test_incident_created_on_strike_3(self):
        state = StrikeState(job_id="test")
        sig = "a" * 64
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        assert result.incident_report is not None
        assert result.incident_report["signature_value"] == sig
    
    def test_incident_has_strike_history(self):
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
        state = StrikeState(job_id="job-12345")
        sig = "a" * 64
        state.record_strike(sig)
        state.record_strike(sig)
        result = state.record_strike(sig)
        assert result.incident_report["job_id"] == "job-12345"


# =============================================================================
# Tests: File Persistence
# =============================================================================

class TestStrikeStatePersistence:
    def test_save_creates_governance_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state = StrikeState(job_id="test-job-001")
            saved_path = state.save(jobs_dir)
            assert saved_path.exists()
            assert saved_path.parent.name == "governance"
            assert saved_path.parent.parent.name == "test-job-001"
    
    def test_save_creates_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state = StrikeState(job_id="test-job-001")
            state.record_strike("sig123")
            saved_path = state.save(jobs_dir)
            with open(saved_path) as f:
                data = json.load(f)
            assert data["job_id"] == "test-job-001"
            assert "sig123" in data["strikes_by_error_sig"]
    
    def test_load_restores_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state1 = StrikeState(job_id="test-job-001")
            state1.record_strike("sig_a")
            state1.record_strike("sig_a")
            state1.save(jobs_dir)
            state2 = StrikeState.load(jobs_dir, "test-job-001")
            assert state2 is not None
            assert state2.job_id == "test-job-001"
            assert state2.get_strike_count("sig_a") == 2
            assert len(state2.history) == 2
    
    def test_load_nonexistent_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state = StrikeState.load(jobs_dir, "nonexistent-job")
            assert state is None
    
    def test_save_preserves_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state1 = StrikeState(job_id="test-job-001")
            state1.record_strike("sig_a", SignatureType.ERROR)
            state1.record_strike("sig_b", SignatureType.SPEC_HOLE)
            state1.save(jobs_dir)
            state2 = StrikeState.load(jobs_dir, "test-job-001")
            assert len(state2.history) == 2
            assert state2.history[0].signature == "sig_a"
            assert state2.history[0].signature_type == SignatureType.ERROR
            assert state2.history[1].signature == "sig_b"
            assert state2.history[1].signature_type == SignatureType.SPEC_HOLE
    
    def test_save_updates_updated_at(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state = StrikeState(job_id="test-job-001")
            initial_updated = state.updated_at
            state.record_strike("sig_a")
            state.save(jobs_dir)
            assert state.updated_at != initial_updated
    
    def test_correct_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state = StrikeState(job_id="abc-123")
            saved_path = state.save(jobs_dir)
            expected = jobs_dir / "abc-123" / "governance" / "strike_state.json"
            assert saved_path == expected
    
    def test_multiple_saves_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            state = StrikeState(job_id="test-job-001")
            state.record_strike("sig_a")
            state.save(jobs_dir)
            state.record_strike("sig_a")
            state.save(jobs_dir)
            loaded = StrikeState.load(jobs_dir, "test-job-001")
            assert loaded.get_strike_count("sig_a") == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
