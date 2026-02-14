# FILE: app/orchestrator/phase_checkout_models.py
"""
Phase Checkout — Data Models.

Defines result types for each verification check performed during
Phase Checkout (Stage 9). Used by phase_checkout.py and consumed
by segment_loop.py for routing decisions.

v1.0 (2026-02-14): Initial implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# =============================================================================
# INDIVIDUAL CHECK RESULTS
# =============================================================================

@dataclass
class BootTestResult:
    """Result of the application boot test."""
    status: str  # "pass", "fail", "error", "skipped"
    stdout: str = ""
    stderr: str = ""
    error_summary: str = ""
    traceback_file: Optional[str] = None  # file that caused the failure
    traceback_segment: Optional[str] = None  # segment that produced it
    duration_ms: int = 0


@dataclass
class SizeViolation:
    """A single file that exceeds size constraints."""
    file_path: str
    line_count: int
    kb_size: float
    max_function_lines: int = 0
    max_function_name: str = ""
    produced_by_segment: str = ""
    violation_type: str = ""  # "file_too_large", "function_too_large"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_count": self.line_count,
            "kb_size": self.kb_size,
            "max_function_lines": self.max_function_lines,
            "max_function_name": self.max_function_name,
            "produced_by_segment": self.produced_by_segment,
            "violation_type": self.violation_type,
        }


@dataclass
class SizeValidationResult:
    """Result of output file size validation."""
    status: str  # "pass", "fail"
    files_checked: int = 0
    violations: List[SizeViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "files_checked": self.files_checked,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass
class ContractViolation:
    """A skeleton contract breach."""
    segment_id: str
    violation_type: str  # "missing_export", "scope_violation", "phantom_file"
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "violation_type": self.violation_type,
            "detail": self.detail,
        }


@dataclass
class ContractCheckResult:
    """Result of skeleton contract verification."""
    status: str  # "pass", "fail", "skipped"
    violations: List[ContractViolation] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "violations": [v.to_dict() for v in self.violations],
        }


# =============================================================================
# FAILURE ROUTING
# =============================================================================

@dataclass
class FailureRouting:
    """Decision on where to route a phase checkout failure for retry."""
    target_stage: str  # "stage_4_specgate", "stage_5_critical", "stage_8_overwatcher"
    target_segment: Optional[str] = None  # specific segment, or None for all
    target_file: Optional[str] = None  # specific file that failed
    reason: str = ""
    rollback_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_stage": self.target_stage,
            "target_segment": self.target_segment,
            "target_file": self.target_file,
            "reason": self.reason,
            "rollback_required": self.rollback_required,
        }


# =============================================================================
# AGGREGATE RESULT
# =============================================================================

@dataclass
class PhaseCheckoutResult:
    """Complete result of Phase Checkout (Stage 9)."""
    job_id: str
    status: str = "pending"  # "pass", "fail", "error"

    # Individual checks
    boot_test: Optional[BootTestResult] = None
    size_validation: Optional[SizeValidationResult] = None
    contract_check: Optional[ContractCheckResult] = None

    # Routing (if failed)
    routing: Optional[FailureRouting] = None

    # Retry tracking
    attempt: int = 1
    max_attempts: int = 3

    # Metadata
    timestamp: str = ""
    duration_ms: int = 0
    checks_run: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "boot_test": self.boot_test.__dict__ if self.boot_test else None,
            "size_validation": self.size_validation.to_dict() if self.size_validation else None,
            "contract_check": self.contract_check.to_dict() if self.contract_check else None,
            "routing": self.routing.to_dict() if self.routing else None,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "checks_run": self.checks_run,
        }

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    @property
    def needs_retry(self) -> bool:
        return self.status == "fail" and self.attempt < self.max_attempts

    @property
    def exhausted(self) -> bool:
        return self.status == "fail" and self.attempt >= self.max_attempts
