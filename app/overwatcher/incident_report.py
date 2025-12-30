# FILE: app/overwatcher/incident_report.py
"""Incident Report: Strike 3 Hard Stop Artifact.

When a chunk/stage exhausts all 3 strikes on the same ErrorSignature,
the system MUST halt and produce a formal incident report.

This artifact:
1. Documents the failure chain (all 3 attempts)
2. Captures full diagnostic context
3. Requires human review before any retry
4. Is immutable once written

NO automatic retry after Strike 3. Human must:
- Review the incident report
- Manually approve retry with new job ID
- Or mark as "won't fix" / "needs redesign"
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class IncidentSeverity(str, Enum):
    """Severity classification for incidents."""
    CRITICAL = "critical"      # System integrity at risk
    HIGH = "high"              # Feature blocked, no workaround
    MEDIUM = "medium"          # Feature blocked, workaround exists
    LOW = "low"                # Minor issue, can proceed


class IncidentCategory(str, Enum):
    """Category of failure."""
    CODE_ERROR = "code_error"           # Implementation bug
    TEST_FAILURE = "test_failure"       # Tests don't pass
    LINT_ERROR = "lint_error"           # Code quality gate
    TYPE_ERROR = "type_error"           # Type checking failure
    BOUNDARY_VIOLATION = "boundary"     # Chunk touched forbidden files
    DEPENDENCY_ERROR = "dependency"     # Missing/broken dependency
    ENVIRONMENT_ERROR = "environment"   # System/env issue
    SPEC_INCOMPLETE = "spec_incomplete" # Spec Gate couldn't close holes
    UNKNOWN = "unknown"


class ResolutionStatus(str, Enum):
    """Status of incident resolution."""
    OPEN = "open"                       # Awaiting human review
    INVESTIGATING = "investigating"     # Human is reviewing
    BLOCKED = "blocked"                 # Needs external input
    WONT_FIX = "wont_fix"              # Decided not to fix
    REDESIGN_REQUIRED = "redesign"      # Needs architecture change
    RESOLVED = "resolved"               # Fixed in subsequent job


# =============================================================================
# Strike Record
# =============================================================================

@dataclass
class StrikeRecord:
    """Record of a single strike attempt."""
    strike_number: int
    timestamp: str
    error_signature_hash: str
    exception_type: str
    failing_component: str  # test name, file path, or stage
    
    # What was tried
    diagnosis: str
    fix_actions_attempted: List[str]
    
    # What happened
    outcome: str  # "same_error", "different_error", "partial_success"
    error_output_excerpt: str  # Truncated to 500 chars
    
    # Strike 2 only
    deep_research_used: bool = False
    deep_research_findings: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strike_number": self.strike_number,
            "timestamp": self.timestamp,
            "error_signature_hash": self.error_signature_hash,
            "exception_type": self.exception_type,
            "failing_component": self.failing_component,
            "diagnosis": self.diagnosis,
            "fix_actions_attempted": self.fix_actions_attempted,
            "outcome": self.outcome,
            "error_output_excerpt": self.error_output_excerpt,
            "deep_research_used": self.deep_research_used,
            "deep_research_findings": self.deep_research_findings,
        }


# =============================================================================
# Incident Report
# =============================================================================

@dataclass
class IncidentReport:
    """Formal incident report for Strike 3 hard stop.
    
    This is the ONLY artifact produced on Strike 3.
    It is immutable and requires human review.
    """
    
    # Identity
    incident_id: str  # SHA256 of job_id + chunk_id + timestamp
    job_id: str
    chunk_id: str
    stage: str  # "verification", "spec_gate", "implementation", etc.
    
    # Classification
    severity: IncidentSeverity
    category: IncidentCategory
    
    # Timeline
    created_at: str
    first_strike_at: str
    final_strike_at: str
    
    # The error that couldn't be fixed
    error_signature_hash: str
    exception_type: str
    root_cause_hypothesis: str  # Overwatcher's best guess
    
    # Full strike history
    strikes: List[StrikeRecord] = field(default_factory=list)
    
    # Context
    spec_id: str = ""
    spec_hash: str = ""
    architecture_hash: str = ""
    affected_files: List[str] = field(default_factory=list)
    
    # What would be needed to fix
    blockers: List[str] = field(default_factory=list)
    suggested_human_actions: List[str] = field(default_factory=list)
    
    # Resolution tracking
    resolution_status: ResolutionStatus = ResolutionStatus.OPEN
    resolution_notes: str = ""
    resolved_by: str = ""
    resolved_at: str = ""
    follow_up_job_id: str = ""
    
    # Immutability
    report_hash: str = ""  # SHA256 of report contents (set on finalize)
    finalized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "job_id": self.job_id,
            "chunk_id": self.chunk_id,
            "stage": self.stage,
            "severity": self.severity.value,
            "category": self.category.value,
            "created_at": self.created_at,
            "first_strike_at": self.first_strike_at,
            "final_strike_at": self.final_strike_at,
            "error_signature_hash": self.error_signature_hash,
            "exception_type": self.exception_type,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "strikes": [s.to_dict() for s in self.strikes],
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "architecture_hash": self.architecture_hash,
            "affected_files": self.affected_files,
            "blockers": self.blockers,
            "suggested_human_actions": self.suggested_human_actions,
            "resolution_status": self.resolution_status.value,
            "resolution_notes": self.resolution_notes,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "follow_up_job_id": self.follow_up_job_id,
            "report_hash": self.report_hash,
            "finalized": self.finalized,
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_markdown(self) -> str:
        """Generate human-readable markdown report."""
        lines = []
        lines.append(f"# Incident Report: {self.incident_id[:16]}")
        lines.append("")
        lines.append(f"**Status:** {self.resolution_status.value.upper()}")
        lines.append(f"**Severity:** {self.severity.value.upper()}")
        lines.append(f"**Category:** {self.category.value}")
        lines.append("")
        
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Job ID:** `{self.job_id}`")
        lines.append(f"- **Chunk ID:** `{self.chunk_id}`")
        lines.append(f"- **Stage:** {self.stage}")
        lines.append(f"- **Created:** {self.created_at}")
        lines.append("")
        
        lines.append("## Error Details")
        lines.append("")
        lines.append(f"- **Exception:** `{self.exception_type}`")
        lines.append(f"- **Signature:** `{self.error_signature_hash}`")
        lines.append("")
        lines.append("### Root Cause Hypothesis")
        lines.append("")
        lines.append(self.root_cause_hypothesis or "_No hypothesis provided_")
        lines.append("")
        
        lines.append("## Strike History")
        lines.append("")
        for strike in self.strikes:
            lines.append(f"### Strike {strike.strike_number}")
            lines.append(f"- **Time:** {strike.timestamp}")
            lines.append(f"- **Component:** {strike.failing_component}")
            lines.append(f"- **Diagnosis:** {strike.diagnosis}")
            lines.append(f"- **Actions Tried:** {', '.join(strike.fix_actions_attempted) or 'None'}")
            lines.append(f"- **Outcome:** {strike.outcome}")
            if strike.deep_research_used:
                lines.append(f"- **Deep Research:** {strike.deep_research_findings[:200]}...")
            lines.append("")
        
        if self.affected_files:
            lines.append("## Affected Files")
            lines.append("")
            for f in self.affected_files:
                lines.append(f"- `{f}`")
            lines.append("")
        
        if self.blockers:
            lines.append("## Blockers")
            lines.append("")
            for b in self.blockers:
                lines.append(f"- {b}")
            lines.append("")
        
        if self.suggested_human_actions:
            lines.append("## Suggested Actions for Human Review")
            lines.append("")
            for i, action in enumerate(self.suggested_human_actions, 1):
                lines.append(f"{i}. {action}")
            lines.append("")
        
        lines.append("## Verification")
        lines.append("")
        lines.append(f"- **Spec ID:** `{self.spec_id}`")
        lines.append(f"- **Spec Hash:** `{self.spec_hash[:16]}...`" if self.spec_hash else "- **Spec Hash:** _None_")
        lines.append(f"- **Report Hash:** `{self.report_hash[:16]}...`" if self.report_hash else "- **Report Hash:** _Not finalized_")
        lines.append(f"- **Finalized:** {self.finalized}")
        lines.append("")
        
        if self.resolution_status != ResolutionStatus.OPEN:
            lines.append("## Resolution")
            lines.append("")
            lines.append(f"- **Status:** {self.resolution_status.value}")
            lines.append(f"- **Notes:** {self.resolution_notes or '_None_'}")
            lines.append(f"- **Resolved By:** {self.resolved_by or '_N/A_'}")
            lines.append(f"- **Resolved At:** {self.resolved_at or '_N/A_'}")
            if self.follow_up_job_id:
                lines.append(f"- **Follow-up Job:** `{self.follow_up_job_id}`")
            lines.append("")
        
        return "\n".join(lines)
    
    def finalize(self) -> None:
        """Finalize the report, making it immutable.
        
        Once finalized:
        - report_hash is computed
        - No further modifications allowed
        - Must be stored to artifact storage
        """
        if self.finalized:
            raise ValueError("Report already finalized")
        
        # Compute hash of report contents (excluding hash fields)
        content = {
            "incident_id": self.incident_id,
            "job_id": self.job_id,
            "chunk_id": self.chunk_id,
            "stage": self.stage,
            "severity": self.severity.value,
            "category": self.category.value,
            "error_signature_hash": self.error_signature_hash,
            "exception_type": self.exception_type,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "strikes": [s.to_dict() for s in self.strikes],
            "blockers": self.blockers,
            "suggested_human_actions": self.suggested_human_actions,
        }
        
        content_str = json.dumps(content, sort_keys=True)
        self.report_hash = hashlib.sha256(content_str.encode()).hexdigest()
        self.finalized = True


# =============================================================================
# Report Builder
# =============================================================================

def generate_incident_id(job_id: str, chunk_id: str, timestamp: str) -> str:
    """Generate unique incident ID."""
    content = f"{job_id}:{chunk_id}:{timestamp}"
    return hashlib.sha256(content.encode()).hexdigest()


def classify_error_category(
    exception_type: str,
    error_output: str,
) -> IncidentCategory:
    """Classify error into category based on exception and output."""
    exception_lower = exception_type.lower()
    output_lower = error_output.lower()
    
    # Test failures
    if "assert" in exception_lower or "pytest" in output_lower:
        return IncidentCategory.TEST_FAILURE
    
    # Type errors
    if "type" in exception_lower or "mypy" in output_lower:
        return IncidentCategory.TYPE_ERROR
    
    # Lint errors
    if "ruff" in output_lower or "lint" in output_lower:
        return IncidentCategory.LINT_ERROR
    
    # Dependency errors
    if "modulenotfound" in exception_lower or "import" in exception_lower:
        return IncidentCategory.DEPENDENCY_ERROR
    
    # Environment errors
    if "permission" in exception_lower or "filenotfound" in exception_lower:
        return IncidentCategory.ENVIRONMENT_ERROR
    
    # Boundary violations (check output)
    if "boundary" in output_lower or "not in allowed" in output_lower:
        return IncidentCategory.BOUNDARY_VIOLATION
    
    # Default to code error
    return IncidentCategory.CODE_ERROR


def classify_severity(
    category: IncidentCategory,
    affected_files: List[str],
) -> IncidentSeverity:
    """Classify severity based on category and scope."""
    # Critical paths
    critical_patterns = ["auth", "crypto", "security", "migration"]
    for f in affected_files:
        if any(p in f.lower() for p in critical_patterns):
            return IncidentSeverity.CRITICAL
    
    # Category-based
    if category == IncidentCategory.BOUNDARY_VIOLATION:
        return IncidentSeverity.CRITICAL  # Safety violation
    
    if category in (IncidentCategory.TEST_FAILURE, IncidentCategory.CODE_ERROR):
        return IncidentSeverity.HIGH
    
    if category in (IncidentCategory.TYPE_ERROR, IncidentCategory.LINT_ERROR):
        return IncidentSeverity.MEDIUM
    
    return IncidentSeverity.MEDIUM


def build_incident_report(
    job_id: str,
    chunk_id: str,
    stage: str,
    error_signature_hash: str,
    exception_type: str,
    strikes: List[StrikeRecord],
    root_cause_hypothesis: str,
    affected_files: List[str],
    blockers: List[str],
    suggested_actions: List[str],
    spec_id: str = "",
    spec_hash: str = "",
    architecture_hash: str = "",
) -> IncidentReport:
    """Build a complete incident report from strike data.
    
    Args:
        job_id: Job UUID
        chunk_id: Chunk identifier
        stage: Stage where failure occurred
        error_signature_hash: Hash of the error that couldn't be fixed
        exception_type: Type of exception
        strikes: List of StrikeRecord objects (should be 3)
        root_cause_hypothesis: Overwatcher's best guess at root cause
        affected_files: Files involved in the failure
        blockers: Known blockers preventing fix
        suggested_actions: Actions for human to take
        spec_id: Spec identifier
        spec_hash: Spec hash for verification
        architecture_hash: Architecture hash
    
    Returns:
        IncidentReport (not yet finalized)
    """
    now = datetime.now(timezone.utc).isoformat()
    
    # Get timestamps from strikes
    first_strike_at = strikes[0].timestamp if strikes else now
    final_strike_at = strikes[-1].timestamp if strikes else now
    
    # Generate incident ID
    incident_id = generate_incident_id(job_id, chunk_id, now)
    
    # Classify
    category = classify_error_category(
        exception_type,
        strikes[-1].error_output_excerpt if strikes else ""
    )
    severity = classify_severity(category, affected_files)
    
    return IncidentReport(
        incident_id=incident_id,
        job_id=job_id,
        chunk_id=chunk_id,
        stage=stage,
        severity=severity,
        category=category,
        created_at=now,
        first_strike_at=first_strike_at,
        final_strike_at=final_strike_at,
        error_signature_hash=error_signature_hash,
        exception_type=exception_type,
        root_cause_hypothesis=root_cause_hypothesis,
        strikes=strikes,
        spec_id=spec_id,
        spec_hash=spec_hash,
        architecture_hash=architecture_hash,
        affected_files=affected_files,
        blockers=blockers,
        suggested_human_actions=suggested_actions,
    )


# =============================================================================
# Storage
# =============================================================================

def store_incident_report(
    report: IncidentReport,
    artifact_root: str,
) -> str:
    """Store incident report to artifact storage.
    
    Creates:
    - incident_<id>.json (machine-readable)
    - incident_<id>.md (human-readable)
    
    Returns:
        Path to JSON report
    """
    if not report.finalized:
        report.finalize()
    
    # Create incidents directory
    incidents_dir = Path(artifact_root) / "incidents"
    incidents_dir.mkdir(parents=True, exist_ok=True)
    
    short_id = report.incident_id[:16]
    
    # Write JSON
    json_path = incidents_dir / f"incident_{short_id}.json"
    json_path.write_text(report.to_json(), encoding="utf-8")
    
    # Write Markdown
    md_path = incidents_dir / f"incident_{short_id}.md"
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    
    logger.info(f"[incident_report] Stored incident {short_id} to {incidents_dir}")
    
    return str(json_path)


def load_incident_report(path: str) -> IncidentReport:
    """Load incident report from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    strikes = [
        StrikeRecord(**s) for s in data.get("strikes", [])
    ]
    
    return IncidentReport(
        incident_id=data["incident_id"],
        job_id=data["job_id"],
        chunk_id=data["chunk_id"],
        stage=data["stage"],
        severity=IncidentSeverity(data["severity"]),
        category=IncidentCategory(data["category"]),
        created_at=data["created_at"],
        first_strike_at=data["first_strike_at"],
        final_strike_at=data["final_strike_at"],
        error_signature_hash=data["error_signature_hash"],
        exception_type=data["exception_type"],
        root_cause_hypothesis=data["root_cause_hypothesis"],
        strikes=strikes,
        spec_id=data.get("spec_id", ""),
        spec_hash=data.get("spec_hash", ""),
        architecture_hash=data.get("architecture_hash", ""),
        affected_files=data.get("affected_files", []),
        blockers=data.get("blockers", []),
        suggested_human_actions=data.get("suggested_human_actions", []),
        resolution_status=ResolutionStatus(data.get("resolution_status", "open")),
        resolution_notes=data.get("resolution_notes", ""),
        resolved_by=data.get("resolved_by", ""),
        resolved_at=data.get("resolved_at", ""),
        follow_up_job_id=data.get("follow_up_job_id", ""),
        report_hash=data.get("report_hash", ""),
        finalized=data.get("finalized", False),
    )


__all__ = [
    # Enums
    "IncidentSeverity",
    "IncidentCategory",
    "ResolutionStatus",
    # Models
    "StrikeRecord",
    "IncidentReport",
    # Builders
    "generate_incident_id",
    "classify_error_category",
    "classify_severity",
    "build_incident_report",
    # Storage
    "store_incident_report",
    "load_incident_report",
]
