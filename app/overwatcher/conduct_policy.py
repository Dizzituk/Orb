# FILE: app/overwatcher/conduct_policy.py
"""
Global Overwatcher Conduct Policy v1.0

This module defines the universal behavioral contract for all LLMs and agents
operating within the ASTRA system. These rules are binding and apply to every
job, every pipeline, every time.

The Overwatcher is the safety authority. Its role is to ensure execution:
- Follows the spec literally
- Never fabricates missing conditions
- Produces evidence of actions taken
- Fails cleanly when requirements are unmet

SCOPE: All pipelines, all tasks, including:
- File operations
- Coding tasks
- System interaction
- Repository changes
- Architecture synthesis
- Test execution
- Sandbox behavior

VERSION HISTORY:
- v1.0 (2026-01): Initial formalization of global conduct rules
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# POLICY RULE IDENTIFIERS
# =============================================================================

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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

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


# =============================================================================
# POLICY RULES - FORMAL DEFINITIONS
# =============================================================================

# The seven rules in machine-readable format
GLOBAL_CONDUCT_RULES: Dict[ConductRule, Dict[str, Any]] = {
    ConductRule.SPEC_FIDELITY: {
        "id": "RULE_1",
        "name": "Spec Fidelity Is Mandatory",
        "description": (
            "If a spec states that a resource already exists, you must not create it. "
            "If a required precondition is missing, the correct result is FAIL, not workaround. "
            "This applies system-wide."
        ),
        "violations": [
            ConductViolationType.CREATED_EXISTING_RESOURCE,
            ConductViolationType.WORKAROUND_INSTEAD_OF_FAIL,
            ConductViolationType.PRECONDITION_MISSING_CONTINUED,
        ],
        "severity_default": ViolationSeverity.CRITICAL,
    },
    ConductRule.DISCOVERY_BEFORE_ACTION: {
        "id": "RULE_2",
        "name": "Discovery Before Action",
        "description": (
            "Before performing any operation, you must verify that the target resource "
            "actually exists. Examples: If modifying a file → confirm the file exists. "
            "If locating a folder → enumerate locations and confirm it exists. "
            "If editing a record → retrieve and verify it. "
            "If the resource does not exist → FAIL gracefully."
        ),
        "violations": [
            ConductViolationType.ACTED_WITHOUT_VERIFICATION,
            ConductViolationType.ASSUMED_PATH_EXISTS,
            ConductViolationType.SKIPPED_ENUMERATION,
        ],
        "severity_default": ViolationSeverity.ERROR,
    },
    ConductRule.EVIDENCE_BASED_EXECUTION: {
        "id": "RULE_3",
        "name": "Evidence-Based Execution",
        "description": (
            "Every job must produce evidence logs showing: "
            "What was inspected, What was found, Why the chosen action was valid. "
            "If evidence cannot be produced → treat execution as invalid."
        ),
        "violations": [
            ConductViolationType.NO_EVIDENCE_PRODUCED,
            ConductViolationType.INCOMPLETE_EVIDENCE_TRAIL,
            ConductViolationType.MISSING_INSPECTION_LOG,
            ConductViolationType.MISSING_FINDING_LOG,
        ],
        "severity_default": ViolationSeverity.ERROR,
    },
    ConductRule.NO_SILENT_SUBSTITUTION: {
        "id": "RULE_4",
        "name": "No Silent Substitution",
        "description": (
            "You must NOT: assume file paths, invent missing resources, "
            "create resources that are supposed to pre-exist, "
            "redirect execution to a different location, "
            "weaken or reinterpret the spec to make execution easier. "
            "If execution cannot proceed as written → report FAIL, do not improvise."
        ),
        "violations": [
            ConductViolationType.ASSUMED_FILE_PATH,
            ConductViolationType.INVENTED_RESOURCE,
            ConductViolationType.CREATED_SUBSTITUTE_RESOURCE,
            ConductViolationType.REDIRECTED_EXECUTION,
            ConductViolationType.SPEC_REINTERPRETED,
        ],
        "severity_default": ViolationSeverity.CRITICAL,
    },
    ConductRule.PREFER_UNCERTAINTY: {
        "id": "RULE_5",
        "name": "Prefer Uncertainty Over Invention",
        "description": (
            "If conditions are unclear: Ask for clarification rather than guessing. "
            "Guessing is prohibited."
        ),
        "violations": [
            ConductViolationType.GUESSED_INSTEAD_OF_ASKING,
            ConductViolationType.UNCLEAR_CONDITIONS_PROCEEDED,
        ],
        "severity_default": ViolationSeverity.WARNING,
    },
    ConductRule.POLICY_VIOLATION_DETECTION: {
        "id": "RULE_6",
        "name": "Policy Violations",
        "description": (
            "The following automatically trigger policy violation state: "
            "Creating a resource the spec says must already exist, "
            "Acting on a resource without verifying it, "
            "Modifying the meaning of the spec, "
            "Completing execution without evidence trail. "
            "These are systemic safety failures."
        ),
        "violations": [
            ConductViolationType.SYSTEMATIC_SAFETY_FAILURE,
            ConductViolationType.EXECUTION_WITHOUT_EVIDENCE,
        ],
        "severity_default": ViolationSeverity.CRITICAL,
    },
    ConductRule.GRACEFUL_FAILURE: {
        "id": "RULE_7",
        "name": "Fail Gracefully",
        "description": (
            "If execution cannot proceed: Explain precisely why, "
            "Do not attempt recovery unless explicitly permitted, "
            "Suggest next steps only if appropriate, Exit cleanly."
        ),
        "violations": [
            ConductViolationType.IMPROPER_FAILURE_HANDLING,
            ConductViolationType.UNAUTHORIZED_RECOVERY_ATTEMPT,
            ConductViolationType.UNCLEAR_FAILURE_REASON,
        ],
        "severity_default": ViolationSeverity.ERROR,
    },
}


# =============================================================================
# EDGE CASES AND AMBIGUITY RULES
# =============================================================================

EDGE_CASE_RULES: Dict[str, Dict[str, Any]] = {
    "partial_existence": {
        "description": (
            "Resource exists but is incomplete or corrupted. "
            "Treat as NON-EXISTENT for spec fidelity purposes. "
            "Report the partial state and FAIL."
        ),
        "ruling": "FAIL_WITH_EVIDENCE",
        "applies_to": [ConductRule.SPEC_FIDELITY, ConductRule.DISCOVERY_BEFORE_ACTION],
    },
    "ambiguous_path_spec": {
        "description": (
            "Spec references a path that could resolve to multiple locations. "
            "Do NOT guess. Request clarification or enumerate all possibilities and ask."
        ),
        "ruling": "REQUEST_CLARIFICATION",
        "applies_to": [ConductRule.PREFER_UNCERTAINTY, ConductRule.NO_SILENT_SUBSTITUTION],
    },
    "transient_resource": {
        "description": (
            "Resource existed during discovery but disappeared before action. "
            "FAIL - do not attempt retry without explicit permission."
        ),
        "ruling": "FAIL_NO_RETRY",
        "applies_to": [ConductRule.DISCOVERY_BEFORE_ACTION, ConductRule.GRACEFUL_FAILURE],
    },
    "read_only_resource": {
        "description": (
            "Resource exists but is read-only when write is required. "
            "Report the constraint and FAIL. Do not attempt workarounds."
        ),
        "ruling": "FAIL_WITH_CONSTRAINT_REPORT",
        "applies_to": [ConductRule.SPEC_FIDELITY, ConductRule.NO_SILENT_SUBSTITUTION],
    },
    "permission_denied": {
        "description": (
            "Resource exists but access is denied. "
            "Report the permission issue and FAIL. Do not escalate without authorization."
        ),
        "ruling": "FAIL_WITH_PERMISSION_REPORT",
        "applies_to": [ConductRule.DISCOVERY_BEFORE_ACTION, ConductRule.GRACEFUL_FAILURE],
    },
    "network_resource_timeout": {
        "description": (
            "Network resource did not respond within timeout. "
            "Treat as discovery failure. Report and FAIL. "
            "Do not assume resource doesn't exist - state is unknown."
        ),
        "ruling": "FAIL_UNKNOWN_STATE",
        "applies_to": [ConductRule.DISCOVERY_BEFORE_ACTION, ConductRule.PREFER_UNCERTAINTY],
    },
    "symlink_resolution": {
        "description": (
            "Path is a symlink. Must verify actual target exists. "
            "Broken symlinks = resource does not exist."
        ),
        "ruling": "VERIFY_RESOLVED_TARGET",
        "applies_to": [ConductRule.DISCOVERY_BEFORE_ACTION],
    },
    "case_sensitivity_mismatch": {
        "description": (
            "Spec path differs from filesystem path by case only. "
            "On case-insensitive systems, this may match. "
            "On case-sensitive systems, treat as different. "
            "Always report the exact match status."
        ),
        "ruling": "REPORT_EXACT_MATCH_STATUS",
        "applies_to": [ConductRule.DISCOVERY_BEFORE_ACTION, ConductRule.EVIDENCE_BASED_EXECUTION],
    },
    "empty_resource": {
        "description": (
            "File/folder/record exists but is empty. "
            "Empty is still EXISTS unless spec explicitly requires content."
        ),
        "ruling": "EXISTS_UNLESS_CONTENT_REQUIRED",
        "applies_to": [ConductRule.SPEC_FIDELITY],
    },
    "spec_silent_on_existence": {
        "description": (
            "Spec doesn't explicitly state whether resource should exist. "
            "Default to DISCOVERY_REQUIRED - verify before acting."
        ),
        "ruling": "DISCOVERY_REQUIRED",
        "applies_to": [ConductRule.DISCOVERY_BEFORE_ACTION, ConductRule.PREFER_UNCERTAINTY],
    },
}


# =============================================================================
# COMPLIANCE EVALUATION
# =============================================================================

class ConductPolicyEvaluator:
    """
    Evaluates execution against the Global Overwatcher Conduct Policy.
    
    This is the compliance engine that determines whether a job followed
    all conduct rules correctly.
    """
    
    def __init__(self, spec_id: Optional[str] = None, spec_hash: Optional[str] = None):
        """
        Initialize evaluator with optional spec context.
        
        Args:
            spec_id: The spec identifier (job_id typically)
            spec_hash: The spec hash for anchoring
        """
        self.spec_id = spec_id
        self.spec_hash = spec_hash
        self.violations: List[ConductViolation] = []
        self.evidence_trail: List[EvidenceRecord] = []
    
    def add_evidence(
        self,
        action: str,
        target: str,
        result: str,
        inspected_path: Optional[str] = None,
        found_state: Optional[Dict[str, Any]] = None,
        decision_reason: Optional[str] = None,
    ) -> None:
        """
        Add evidence of an action to the trail.
        
        Every operation should call this to document what was done.
        """
        record = EvidenceRecord(
            action=action,
            target=target,
            result=result,
            inspected_path=inspected_path,
            found_state=found_state,
            decision_reason=decision_reason,
        )
        self.evidence_trail.append(record)
        logger.debug(f"[CONDUCT] Evidence recorded: {action} -> {target}: {result}")
    
    def record_violation(
        self,
        rule: ConductRule,
        violation_type: ConductViolationType,
        message: str,
        severity: Optional[ViolationSeverity] = None,
        evidence: Optional[Dict[str, Any]] = None,
        remediation_hint: Optional[str] = None,
    ) -> ConductViolation:
        """
        Record a conduct violation.
        
        Args:
            rule: The conduct rule that was violated
            violation_type: Specific type of violation
            message: Human-readable description
            severity: Override severity (uses rule default if None)
            evidence: Supporting evidence for the violation
            remediation_hint: Suggestion for how to fix
            
        Returns:
            The created violation record
        """
        if severity is None:
            rule_def = GLOBAL_CONDUCT_RULES.get(rule, {})
            severity = rule_def.get("severity_default", ViolationSeverity.ERROR)
        
        violation = ConductViolation(
            rule=rule,
            violation_type=violation_type,
            message=message,
            severity=severity,
            evidence=evidence or {},
            spec_id=self.spec_id,
            spec_hash=self.spec_hash,
            remediation_hint=remediation_hint,
        )
        self.violations.append(violation)
        logger.warning(f"[CONDUCT] Violation: {rule.value} - {message}")
        return violation
    
    def check_spec_fidelity(
        self,
        resource_specs: List[ResourceExistenceSpec],
        actual_states: Dict[str, bool],
        created_resources: List[str],
    ) -> List[ConductViolation]:
        """
        Check Rule 1: Spec Fidelity.
        
        Verifies that:
        - Resources spec says must exist were not created
        - Resources spec says must not exist were not used as if existing
        
        Args:
            resource_specs: List of resource existence specifications
            actual_states: Map of resource paths to their actual existence state
            created_resources: List of resources that were created during execution
            
        Returns:
            List of violations found
        """
        violations = []
        
        for spec in resource_specs:
            if spec.must_exist and spec.path in created_resources:
                v = self.record_violation(
                    rule=ConductRule.SPEC_FIDELITY,
                    violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
                    message=f"Spec states '{spec.path}' must exist, but it was created. This violates spec fidelity.",
                    evidence={
                        "spec_path": spec.path,
                        "spec_must_exist": spec.must_exist,
                        "spec_type": spec.resource_type,
                        "action_taken": "created",
                    },
                    remediation_hint="Verify resource exists before proceeding. If missing, FAIL instead of creating.",
                )
                violations.append(v)
        
        return violations
    
    def check_discovery(
        self,
        required_targets: List[str],
        discovery_results: Dict[str, DiscoveryResult],
    ) -> List[ConductViolation]:
        """
        Check Rule 2: Discovery Before Action.
        
        Verifies that all required targets were discovered before being acted upon.
        
        Args:
            required_targets: List of targets that needed discovery
            discovery_results: Map of targets to their discovery results
            
        Returns:
            List of violations found
        """
        violations = []
        
        for target in required_targets:
            if target not in discovery_results:
                v = self.record_violation(
                    rule=ConductRule.DISCOVERY_BEFORE_ACTION,
                    violation_type=ConductViolationType.ACTED_WITHOUT_VERIFICATION,
                    message=f"Target '{target}' was acted upon without prior discovery/verification.",
                    evidence={
                        "target": target,
                        "discovery_performed": False,
                    },
                    remediation_hint="Always verify target existence before performing operations.",
                )
                violations.append(v)
        
        return violations
    
    def check_evidence_completeness(
        self,
        required_evidence_types: List[str],
    ) -> List[ConductViolation]:
        """
        Check Rule 3: Evidence-Based Execution.
        
        Verifies that the evidence trail is complete and covers all required types.
        
        Args:
            required_evidence_types: List of evidence types that must be present
                                    (e.g., ["inspection", "finding", "decision"])
            
        Returns:
            List of violations found
        """
        violations = []
        
        if not self.evidence_trail:
            v = self.record_violation(
                rule=ConductRule.EVIDENCE_BASED_EXECUTION,
                violation_type=ConductViolationType.NO_EVIDENCE_PRODUCED,
                message="No evidence was recorded during execution.",
                remediation_hint="All operations must produce evidence logs.",
            )
            violations.append(v)
            return violations
        
        evidence_actions = {e.action for e in self.evidence_trail}
        
        for required_type in required_evidence_types:
            if required_type not in evidence_actions:
                v = self.record_violation(
                    rule=ConductRule.EVIDENCE_BASED_EXECUTION,
                    violation_type=ConductViolationType.INCOMPLETE_EVIDENCE_TRAIL,
                    message=f"Required evidence type '{required_type}' is missing from trail.",
                    evidence={
                        "required_type": required_type,
                        "present_types": list(evidence_actions),
                    },
                    remediation_hint=f"Ensure '{required_type}' is logged during execution.",
                )
                violations.append(v)
        
        return violations
    
    def evaluate(
        self,
        resource_specs: Optional[List[ResourceExistenceSpec]] = None,
        actual_states: Optional[Dict[str, bool]] = None,
        created_resources: Optional[List[str]] = None,
        required_targets: Optional[List[str]] = None,
        discovery_results: Optional[Dict[str, DiscoveryResult]] = None,
        required_evidence_types: Optional[List[str]] = None,
    ) -> ConductComplianceResult:
        """
        Perform full compliance evaluation.
        
        This is the main entry point for checking conduct policy compliance.
        
        Returns:
            ConductComplianceResult with compliance status and any violations
        """
        # Check Rule 1: Spec Fidelity
        if resource_specs and actual_states and created_resources:
            self.check_spec_fidelity(resource_specs, actual_states, created_resources)
        
        # Check Rule 2: Discovery Before Action
        if required_targets and discovery_results is not None:
            self.check_discovery(required_targets, discovery_results)
        
        # Check Rule 3: Evidence-Based Execution
        if required_evidence_types:
            self.check_evidence_completeness(required_evidence_types)
        
        # Determine overall compliance
        compliant = len(self.violations) == 0
        
        # Generate summary
        if compliant:
            summary = "All conduct rules satisfied. Execution is compliant."
        else:
            critical_count = len([v for v in self.violations if v.severity == ViolationSeverity.CRITICAL])
            error_count = len([v for v in self.violations if v.severity == ViolationSeverity.ERROR])
            summary = f"Conduct violations detected: {critical_count} critical, {error_count} errors."
        
        return ConductComplianceResult(
            compliant=compliant,
            violations=self.violations.copy(),
            evidence_trail=self.evidence_trail.copy(),
            spec_id=self.spec_id,
            spec_hash=self.spec_hash,
            summary=summary,
        )
    
    def reset(self) -> None:
        """Reset evaluator state for a new evaluation."""
        self.violations = []
        self.evidence_trail = []


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_resource_spec(
    path: str,
    must_exist: bool,
    resource_type: str = "file",
    description: Optional[str] = None,
) -> ResourceExistenceSpec:
    """
    Create a resource existence specification.
    
    Args:
        path: Path to the resource
        must_exist: True if spec says resource must already exist
        resource_type: Type of resource ("file", "folder", "record", "service")
        description: Optional description
        
    Returns:
        ResourceExistenceSpec instance
    """
    return ResourceExistenceSpec(
        path=path,
        must_exist=must_exist,
        resource_type=resource_type,
        description=description,
    )


def create_discovery_result(
    target: str,
    exists: bool,
    method: str,
    locations_checked: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> DiscoveryResult:
    """
    Create a discovery result record.
    
    Args:
        target: What was being discovered
        exists: Whether it was found
        method: How discovery was performed
        locations_checked: List of locations that were checked
        metadata: Additional metadata about the discovery
        
    Returns:
        DiscoveryResult instance
    """
    return DiscoveryResult(
        target=target,
        exists=exists,
        discovery_method=method,
        locations_checked=locations_checked or [],
        metadata=metadata or {},
    )


def format_compliance_report(result: ConductComplianceResult) -> str:
    """
    Format a compliance result as a human-readable report.
    
    Args:
        result: The compliance result to format
        
    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "CONDUCT POLICY COMPLIANCE REPORT",
        "=" * 60,
        f"Spec ID: {result.spec_id or 'N/A'}",
        f"Spec Hash: {result.spec_hash or 'N/A'}",
        f"Evaluated: {result.evaluation_timestamp.isoformat()}",
        f"Status: {'COMPLIANT' if result.compliant else 'NON-COMPLIANT'}",
        "",
        result.summary,
        "",
    ]
    
    if result.violations:
        lines.append("-" * 60)
        lines.append("VIOLATIONS:")
        lines.append("-" * 60)
        for i, v in enumerate(result.violations, 1):
            lines.append(f"\n{i}. [{v.severity.value.upper()}] {v.rule.value}")
            lines.append(f"   Type: {v.violation_type.value}")
            lines.append(f"   Message: {v.message}")
            if v.remediation_hint:
                lines.append(f"   Remediation: {v.remediation_hint}")
    
    if result.evidence_trail:
        lines.append("")
        lines.append("-" * 60)
        lines.append("EVIDENCE TRAIL:")
        lines.append("-" * 60)
        for i, e in enumerate(result.evidence_trail, 1):
            lines.append(f"\n{i}. {e.action} -> {e.target}")
            lines.append(f"   Result: {e.result}")
            if e.decision_reason:
                lines.append(f"   Reason: {e.decision_reason}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def get_rule_description(rule: ConductRule) -> str:
    """Get the full description of a conduct rule."""
    rule_def = GLOBAL_CONDUCT_RULES.get(rule, {})
    return rule_def.get("description", "No description available.")


def get_edge_case_ruling(edge_case_id: str) -> Optional[Dict[str, Any]]:
    """Get the ruling for a specific edge case."""
    return EDGE_CASE_RULES.get(edge_case_id)


def compute_spec_hash(spec_content: str) -> str:
    """
    Compute a hash of spec content for anchoring.
    
    Args:
        spec_content: The spec content to hash
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(spec_content.encode()).hexdigest()[:16]


# =============================================================================
# SCENARIO EXAMPLES (For Documentation)
# =============================================================================

SCENARIO_EXAMPLES: Dict[str, Dict[str, Any]] = {
    "correct_discovery_then_modify": {
        "title": "Correct: Discovery then Modify",
        "description": "Agent verifies file exists before modifying",
        "is_violation": False,
        "steps": [
            "1. Spec says: 'Modify config.yaml'",
            "2. Agent calls: stat('config.yaml') -> EXISTS",
            "3. Agent records evidence: inspected='config.yaml', found='exists'",
            "4. Agent modifies file",
            "5. Agent records evidence: action='modify', target='config.yaml'",
        ],
        "outcome": "COMPLIANT - Discovery performed before action",
    },
    "violation_create_existing": {
        "title": "Violation: Creating Resource Spec Says Exists",
        "description": "Agent creates a file the spec says must already exist",
        "is_violation": True,
        "violated_rule": ConductRule.SPEC_FIDELITY,
        "steps": [
            "1. Spec says: 'Edit existing config.yaml'",
            "2. Agent calls: stat('config.yaml') -> NOT EXISTS",
            "3. Agent creates config.yaml (VIOLATION)",
        ],
        "outcome": "NON-COMPLIANT - Should have FAILed when file not found",
    },
    "violation_no_discovery": {
        "title": "Violation: Acting Without Discovery",
        "description": "Agent modifies file without verifying it exists",
        "is_violation": True,
        "violated_rule": ConductRule.DISCOVERY_BEFORE_ACTION,
        "steps": [
            "1. Spec says: 'Update settings.json'",
            "2. Agent directly calls: write('settings.json', ...) (VIOLATION)",
            "3. No prior existence check performed",
        ],
        "outcome": "NON-COMPLIANT - Must verify before acting",
    },
    "violation_silent_substitution": {
        "title": "Violation: Silent Path Substitution",
        "description": "Agent assumes a different path when specified path not found",
        "is_violation": True,
        "violated_rule": ConductRule.NO_SILENT_SUBSTITUTION,
        "steps": [
            "1. Spec says: 'Read /data/input.csv'",
            "2. Agent finds /data/input.csv missing",
            "3. Agent silently reads /backup/input.csv instead (VIOLATION)",
        ],
        "outcome": "NON-COMPLIANT - Should have FAILed, not substituted",
    },
    "correct_graceful_failure": {
        "title": "Correct: Graceful Failure",
        "description": "Agent fails cleanly when precondition missing",
        "is_violation": False,
        "steps": [
            "1. Spec says: 'Process existing report.pdf'",
            "2. Agent calls: stat('report.pdf') -> NOT EXISTS",
            "3. Agent records: discovery failed, target missing",
            "4. Agent returns: FAIL with reason 'Required file report.pdf not found'",
        ],
        "outcome": "COMPLIANT - Failed gracefully with clear reason",
    },
    "violation_guessing": {
        "title": "Violation: Guessing Instead of Asking",
        "description": "Agent guesses intent instead of requesting clarification",
        "is_violation": True,
        "violated_rule": ConductRule.PREFER_UNCERTAINTY,
        "steps": [
            "1. Spec says: 'Update the config' (ambiguous - which config?)",
            "2. Agent guesses: 'probably means config.yaml' (VIOLATION)",
            "3. Agent modifies config.yaml",
        ],
        "outcome": "NON-COMPLIANT - Should have requested clarification",
    },
    "correct_request_clarification": {
        "title": "Correct: Requesting Clarification",
        "description": "Agent asks for clarity when spec is ambiguous",
        "is_violation": False,
        "steps": [
            "1. Spec says: 'Update the config' (ambiguous)",
            "2. Agent finds multiple config files: config.yaml, config.json, settings.ini",
            "3. Agent returns: NEEDS_CLARIFICATION with question 'Which config file?'",
        ],
        "outcome": "COMPLIANT - Preferred uncertainty over invention",
    },
    "violation_unauthorized_recovery": {
        "title": "Violation: Unauthorized Recovery Attempt",
        "description": "Agent attempts recovery without explicit permission",
        "is_violation": True,
        "violated_rule": ConductRule.GRACEFUL_FAILURE,
        "steps": [
            "1. Spec says: 'Read data.json'",
            "2. Agent finds file corrupted",
            "3. Agent attempts to restore from backup (VIOLATION)",
        ],
        "outcome": "NON-COMPLIANT - Should have failed and reported, not recovered",
    },
}


def get_scenario_example(scenario_id: str) -> Optional[Dict[str, Any]]:
    """Get a scenario example by ID."""
    return SCENARIO_EXAMPLES.get(scenario_id)


def list_scenario_examples() -> List[str]:
    """List all available scenario example IDs."""
    return list(SCENARIO_EXAMPLES.keys())
