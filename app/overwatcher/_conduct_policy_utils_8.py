from __future__ import annotations
import logging
from app.overwatcher._conduct_policy_utils_7 import ConductComplianceResult, ConductRule, ConductViolation, ConductViolationType, DiscoveryResult, EvidenceRecord, ResourceExistenceSpec, ViolationSeverity
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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
