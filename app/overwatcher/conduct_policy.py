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
from app.overwatcher._conduct_policy_utils_6 import compute_spec_hash, create_discovery_result, create_resource_spec, format_compliance_report, get_edge_case_ruling, get_rule_description, get_scenario_example, list_scenario_examples
from app.overwatcher._conduct_policy_utils_7 import ConductComplianceResult, ConductRule, ConductViolation, ConductViolationType, DiscoveryResult, EvidenceRecord, ResourceExistenceSpec, ViolationSeverity
from app.overwatcher._conduct_policy_utils_8 import ConductPolicyEvaluator

logger = logging.getLogger(__name__)


# =============================================================================
# POLICY RULE IDENTIFIERS
# =============================================================================


# =============================================================================
# DATA STRUCTURES
# =============================================================================


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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


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
