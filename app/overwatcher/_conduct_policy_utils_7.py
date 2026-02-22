from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ConductRule(str, Enum):
    """
    The seven core conduct rules that govern all Overwatcher operations.
    
    These rules are non-negotiable and apply universally.
    """
    SPEC_FIDELITY = "RULE_1_SPEC_FIDELITY"
    DISCOVERY_BEFORE_ACTION = "RULE_2_DISCOVERY_BEFORE_ACTION"
    EVIDENCE_BASED_EXECUTION = "RULE_3_EVIDENCE_BASED_EXECUTION"
    NO_SILENT_SUBSTITUTION = "RULE_4_NO_SILENT_SUBSTITUTION"
    PREFER_UNCERTAINTY = "RULE_5_PREFER_UNCERTAINTY"
    POLICY_VIOLATION_DETECTION = "RULE_6_POLICY_VIOLATION_DETECTION"
    GRACEFUL_FAILURE = "RULE_7_GRACEFUL_FAILURE"

class ConductViolationType(str, Enum):
    """
    Types of conduct policy violations.
    
    These are distinct from code enforcement violations (ViolationType).
    These represent behavioral/procedural violations at the governance level.
    """
    # Rule 1: Spec Fidelity
    CREATED_EXISTING_RESOURCE = "created_resource_spec_said_exists"
    WORKAROUND_INSTEAD_OF_FAIL = "workaround_instead_of_fail"
    PRECONDITION_MISSING_CONTINUED = "continued_with_missing_precondition"
    
    # Rule 2: Discovery Before Action
    ACTED_WITHOUT_VERIFICATION = "acted_without_verifying_target"
    ASSUMED_PATH_EXISTS = "assumed_path_without_check"
    SKIPPED_ENUMERATION = "skipped_location_enumeration"
    
    # Rule 3: Evidence-Based Execution
    NO_EVIDENCE_PRODUCED = "no_evidence_produced"
    INCOMPLETE_EVIDENCE_TRAIL = "incomplete_evidence_trail"
    MISSING_INSPECTION_LOG = "missing_inspection_log"
    MISSING_FINDING_LOG = "missing_finding_log"
    
    # Rule 4: No Silent Substitution
    ASSUMED_FILE_PATH = "assumed_file_path"
    INVENTED_RESOURCE = "invented_missing_resource"
    CREATED_SUBSTITUTE_RESOURCE = "created_substitute_for_missing"
    REDIRECTED_EXECUTION = "redirected_to_different_location"
    SPEC_REINTERPRETED = "spec_meaning_modified"
    
    # Rule 5: Prefer Uncertainty
    GUESSED_INSTEAD_OF_ASKING = "guessed_instead_of_clarification"
    UNCLEAR_CONDITIONS_PROCEEDED = "proceeded_with_unclear_conditions"
    
    # Rule 6: Policy Violations (Meta)
    SYSTEMATIC_SAFETY_FAILURE = "systematic_safety_failure"
    EXECUTION_WITHOUT_EVIDENCE = "execution_without_evidence_trail"
    
    # Rule 7: Graceful Failure
    IMPROPER_FAILURE_HANDLING = "improper_failure_handling"
    UNAUTHORIZED_RECOVERY_ATTEMPT = "unauthorized_recovery_attempt"
    UNCLEAR_FAILURE_REASON = "unclear_failure_reason"

class ViolationSeverity(str, Enum):
    """Severity levels for conduct violations."""
    CRITICAL = "critical"    # Immediate job termination required
    ERROR = "error"          # Job should fail, violation logged
    WARNING = "warning"      # Job may continue, violation logged
    INFO = "info"            # For audit trail, no action required

@dataclass
class ConductViolation:
    """
    A single conduct policy violation.
    
    Captures the full context of what rule was violated, how, and evidence.
    """
    rule: ConductRule
    violation_type: ConductViolationType
    message: str
    severity: ViolationSeverity = ViolationSeverity.ERROR
    evidence: Dict[str, Any] = field(default_factory=dict)
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    remediation_hint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for logging/storage."""
        return {
            "rule": self.rule.value,
            "violation_type": self.violation_type.value,
            "message": self.message,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "timestamp": self.timestamp.isoformat(),
            "remediation_hint": self.remediation_hint,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConductViolation":
        """Deserialize from dictionary."""
        return cls(
            rule=ConductRule(data["rule"]),
            violation_type=ConductViolationType(data["violation_type"]),
            message=data["message"],
            severity=ViolationSeverity(data.get("severity", "error")),
            evidence=data.get("evidence", {}),
            spec_id=data.get("spec_id"),
            spec_hash=data.get("spec_hash"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.utcnow(),
            remediation_hint=data.get("remediation_hint"),
        )

@dataclass
class EvidenceRecord:
    """
    Evidence of an inspection or action taken during execution.
    
    Every operation must produce evidence showing what was inspected,
    what was found, and why the chosen action was valid.
    """
    action: str
    target: str
    result: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    inspected_path: Optional[str] = None
    found_state: Optional[Dict[str, Any]] = None
    decision_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "target": self.target,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "inspected_path": self.inspected_path,
            "found_state": self.found_state,
            "decision_reason": self.decision_reason,
        }

@dataclass
class ConductComplianceResult:
    """
    Result of conduct policy compliance evaluation.
    
    This is the Overwatcher's verdict on whether execution followed
    the global conduct rules.
    """
    compliant: bool
    violations: List[ConductViolation] = field(default_factory=list)
    evidence_trail: List[EvidenceRecord] = field(default_factory=list)
    spec_id: Optional[str] = None
    spec_hash: Optional[str] = None
    evaluation_timestamp: datetime = field(default_factory=datetime.utcnow)
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "compliant": self.compliant,
            "violations": [v.to_dict() for v in self.violations],
            "evidence_trail": [e.to_dict() for e in self.evidence_trail],
            "spec_id": self.spec_id,
            "spec_hash": self.spec_hash,
            "evaluation_timestamp": self.evaluation_timestamp.isoformat(),
            "summary": self.summary,
        }
    
    @property
    def critical_violations(self) -> List[ConductViolation]:
        """Return only critical violations."""
        return [v for v in self.violations if v.severity == ViolationSeverity.CRITICAL]
    
    @property
    def has_critical(self) -> bool:
        """Check if any critical violations exist."""
        return len(self.critical_violations) > 0

@dataclass
class ResourceExistenceSpec:
    """
    Specification of a resource's expected existence state.
    
    Used to validate Rule 1 (Spec Fidelity) - if spec says resource exists,
    we must not create it.
    """
    path: str
    must_exist: bool
    resource_type: str  # "file", "folder", "record", "service"
    description: Optional[str] = None

@dataclass
class DiscoveryResult:
    """
    Result of a discovery operation (Rule 2: Discovery Before Action).
    
    Captures what was looked for, what was found, and the discovery method.
    """
    target: str
    exists: bool
    discovery_method: str  # "stat", "enumerate", "query", "verify"
    locations_checked: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "exists": self.exists,
            "discovery_method": self.discovery_method,
            "locations_checked": self.locations_checked,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
