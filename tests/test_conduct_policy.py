# FILE: tests/test_conduct_policy.py
"""
Tests for Global Overwatcher Conduct Policy.

Comprehensive test coverage for:
- All 7 conduct rules
- All violation types
- Edge case handling
- Compliance evaluation
- Evidence trail management
- Scenario examples
"""
import pytest
from datetime import datetime
from typing import Dict, List, Any

# Import the conduct policy module
from app.overwatcher.conduct_policy import (
    # Enums
    ConductRule,
    ConductViolationType,
    ViolationSeverity,
    # Data structures
    ConductViolation,
    EvidenceRecord,
    ConductComplianceResult,
    ResourceExistenceSpec,
    DiscoveryResult,
    # Evaluator
    ConductPolicyEvaluator,
    # Helper functions
    create_resource_spec,
    create_discovery_result,
    format_compliance_report,
    get_rule_description,
    get_edge_case_ruling,
    compute_spec_hash,
    get_scenario_example,
    list_scenario_examples,
    # Constants
    GLOBAL_CONDUCT_RULES,
    EDGE_CASE_RULES,
    SCENARIO_EXAMPLES,
)


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestConductRuleEnum:
    """Test ConductRule enum definition."""
    
    def test_all_seven_rules_defined(self):
        """Verify all 7 conduct rules are defined."""
        expected_rules = [
            "SPEC_FIDELITY",
            "DISCOVERY_BEFORE_ACTION",
            "EVIDENCE_BASED_EXECUTION",
            "NO_SILENT_SUBSTITUTION",
            "PREFER_UNCERTAINTY",
            "POLICY_VIOLATION_DETECTION",
            "GRACEFUL_FAILURE",
        ]
        actual_rules = [r.name for r in ConductRule]
        assert len(actual_rules) == 7, f"Expected 7 rules, got {len(actual_rules)}"
        for expected in expected_rules:
            assert expected in actual_rules, f"Missing rule: {expected}"
    
    def test_rule_values_are_prefixed(self):
        """Verify rule values follow RULE_N_ pattern."""
        for rule in ConductRule:
            assert rule.value.startswith("RULE_"), f"Rule {rule.name} value should start with RULE_"
    
    def test_rules_are_strings(self):
        """Verify rules inherit from str."""
        for rule in ConductRule:
            assert isinstance(rule.value, str)


class TestConductViolationTypeEnum:
    """Test ConductViolationType enum definition."""
    
    def test_violation_types_exist(self):
        """Verify violation types are defined."""
        assert len(list(ConductViolationType)) > 0
    
    def test_each_rule_has_violations(self):
        """Verify each rule has associated violation types in GLOBAL_CONDUCT_RULES."""
        for rule in ConductRule:
            rule_def = GLOBAL_CONDUCT_RULES.get(rule)
            assert rule_def is not None, f"Rule {rule.name} missing from GLOBAL_CONDUCT_RULES"
            violations = rule_def.get("violations", [])
            assert len(violations) > 0, f"Rule {rule.name} has no associated violations"
    
    def test_spec_fidelity_violations(self):
        """Test Rule 1 has correct violation types."""
        expected = [
            ConductViolationType.CREATED_EXISTING_RESOURCE,
            ConductViolationType.WORKAROUND_INSTEAD_OF_FAIL,
            ConductViolationType.PRECONDITION_MISSING_CONTINUED,
        ]
        rule_def = GLOBAL_CONDUCT_RULES[ConductRule.SPEC_FIDELITY]
        assert rule_def["violations"] == expected
    
    def test_discovery_violations(self):
        """Test Rule 2 has correct violation types."""
        expected = [
            ConductViolationType.ACTED_WITHOUT_VERIFICATION,
            ConductViolationType.ASSUMED_PATH_EXISTS,
            ConductViolationType.SKIPPED_ENUMERATION,
        ]
        rule_def = GLOBAL_CONDUCT_RULES[ConductRule.DISCOVERY_BEFORE_ACTION]
        assert rule_def["violations"] == expected


class TestViolationSeverityEnum:
    """Test ViolationSeverity enum definition."""
    
    def test_severity_levels_defined(self):
        """Verify all severity levels exist."""
        expected = ["CRITICAL", "ERROR", "WARNING", "INFO"]
        actual = [s.name for s in ViolationSeverity]
        assert actual == expected
    
    def test_critical_is_most_severe(self):
        """Test severity ordering concept."""
        # CRITICAL should be first in definition order
        severities = list(ViolationSeverity)
        assert severities[0] == ViolationSeverity.CRITICAL


# =============================================================================
# DATA STRUCTURE TESTS
# =============================================================================

class TestConductViolation:
    """Test ConductViolation dataclass."""
    
    def test_create_violation(self):
        """Test basic violation creation."""
        v = ConductViolation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Test violation",
        )
        assert v.rule == ConductRule.SPEC_FIDELITY
        assert v.violation_type == ConductViolationType.CREATED_EXISTING_RESOURCE
        assert v.message == "Test violation"
        assert v.severity == ViolationSeverity.ERROR  # Default
    
    def test_violation_to_dict(self):
        """Test serialization to dict."""
        v = ConductViolation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Test",
            severity=ViolationSeverity.CRITICAL,
            evidence={"key": "value"},
            spec_id="job-123",
            spec_hash="abc123",
            remediation_hint="Fix it",
        )
        d = v.to_dict()
        assert d["rule"] == "RULE_1_SPEC_FIDELITY"
        assert d["violation_type"] == "created_resource_spec_said_exists"
        assert d["message"] == "Test"
        assert d["severity"] == "critical"
        assert d["evidence"] == {"key": "value"}
        assert d["spec_id"] == "job-123"
        assert d["remediation_hint"] == "Fix it"
    
    def test_violation_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "rule": "RULE_1_SPEC_FIDELITY",
            "violation_type": "created_resource_spec_said_exists",
            "message": "Test",
            "severity": "critical",
            "evidence": {"key": "value"},
            "spec_id": "job-123",
            "spec_hash": "abc123",
            "timestamp": "2026-01-02T12:00:00",
            "remediation_hint": "Fix it",
        }
        v = ConductViolation.from_dict(data)
        assert v.rule == ConductRule.SPEC_FIDELITY
        assert v.violation_type == ConductViolationType.CREATED_EXISTING_RESOURCE
        assert v.severity == ViolationSeverity.CRITICAL
    
    def test_violation_timestamp_auto_set(self):
        """Test timestamp is automatically set."""
        v = ConductViolation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Test",
        )
        assert isinstance(v.timestamp, datetime)


class TestEvidenceRecord:
    """Test EvidenceRecord dataclass."""
    
    def test_create_evidence(self):
        """Test basic evidence creation."""
        e = EvidenceRecord(
            action="inspect",
            target="/path/to/file",
            result="exists",
        )
        assert e.action == "inspect"
        assert e.target == "/path/to/file"
        assert e.result == "exists"
    
    def test_evidence_to_dict(self):
        """Test serialization."""
        e = EvidenceRecord(
            action="inspect",
            target="/path/to/file",
            result="exists",
            inspected_path="/actual/path",
            found_state={"size": 1024},
            decision_reason="File found, proceeding",
        )
        d = e.to_dict()
        assert d["action"] == "inspect"
        assert d["target"] == "/path/to/file"
        assert d["found_state"] == {"size": 1024}
        assert "timestamp" in d


class TestConductComplianceResult:
    """Test ConductComplianceResult dataclass."""
    
    def test_compliant_result(self):
        """Test creating a compliant result."""
        r = ConductComplianceResult(
            compliant=True,
            summary="All good",
        )
        assert r.compliant is True
        assert len(r.violations) == 0
        assert r.has_critical is False
    
    def test_non_compliant_result(self):
        """Test creating a non-compliant result."""
        v = ConductViolation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Bad",
            severity=ViolationSeverity.CRITICAL,
        )
        r = ConductComplianceResult(
            compliant=False,
            violations=[v],
            summary="Violations found",
        )
        assert r.compliant is False
        assert len(r.violations) == 1
        assert r.has_critical is True
    
    def test_critical_violations_property(self):
        """Test filtering critical violations."""
        violations = [
            ConductViolation(
                rule=ConductRule.SPEC_FIDELITY,
                violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
                message="Critical",
                severity=ViolationSeverity.CRITICAL,
            ),
            ConductViolation(
                rule=ConductRule.DISCOVERY_BEFORE_ACTION,
                violation_type=ConductViolationType.ACTED_WITHOUT_VERIFICATION,
                message="Error",
                severity=ViolationSeverity.ERROR,
            ),
        ]
        r = ConductComplianceResult(compliant=False, violations=violations)
        assert len(r.critical_violations) == 1
        assert r.critical_violations[0].severity == ViolationSeverity.CRITICAL
    
    def test_result_to_dict(self):
        """Test serialization."""
        r = ConductComplianceResult(
            compliant=True,
            spec_id="job-123",
            spec_hash="abc",
            summary="OK",
        )
        d = r.to_dict()
        assert d["compliant"] is True
        assert d["spec_id"] == "job-123"
        assert d["summary"] == "OK"


class TestResourceExistenceSpec:
    """Test ResourceExistenceSpec dataclass."""
    
    def test_create_spec(self):
        """Test basic spec creation."""
        s = ResourceExistenceSpec(
            path="/config/app.yaml",
            must_exist=True,
            resource_type="file",
            description="Main config",
        )
        assert s.path == "/config/app.yaml"
        assert s.must_exist is True
        assert s.resource_type == "file"


class TestDiscoveryResult:
    """Test DiscoveryResult dataclass."""
    
    def test_create_discovery(self):
        """Test basic discovery result creation."""
        d = DiscoveryResult(
            target="/path/to/file",
            exists=True,
            discovery_method="stat",
            locations_checked=["/path/to/file"],
        )
        assert d.target == "/path/to/file"
        assert d.exists is True
        assert d.discovery_method == "stat"
    
    def test_discovery_to_dict(self):
        """Test serialization."""
        d = DiscoveryResult(
            target="file.txt",
            exists=False,
            discovery_method="enumerate",
            locations_checked=["/a", "/b", "/c"],
            metadata={"checked_count": 3},
        )
        result = d.to_dict()
        assert result["exists"] is False
        assert result["locations_checked"] == ["/a", "/b", "/c"]


# =============================================================================
# EVALUATOR TESTS
# =============================================================================

class TestConductPolicyEvaluator:
    """Test ConductPolicyEvaluator class."""
    
    def test_create_evaluator(self):
        """Test basic evaluator creation."""
        e = ConductPolicyEvaluator(spec_id="job-1", spec_hash="hash123")
        assert e.spec_id == "job-1"
        assert e.spec_hash == "hash123"
        assert len(e.violations) == 0
        assert len(e.evidence_trail) == 0
    
    def test_add_evidence(self):
        """Test adding evidence to trail."""
        e = ConductPolicyEvaluator()
        e.add_evidence(
            action="inspect",
            target="/file.txt",
            result="exists",
            decision_reason="Proceeding with modification",
        )
        assert len(e.evidence_trail) == 1
        assert e.evidence_trail[0].action == "inspect"
    
    def test_record_violation(self):
        """Test recording a violation."""
        e = ConductPolicyEvaluator(spec_id="job-1")
        v = e.record_violation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Created file that should exist",
        )
        assert len(e.violations) == 1
        assert v.spec_id == "job-1"
        assert v.severity == ViolationSeverity.CRITICAL  # Default for Rule 1
    
    def test_record_violation_custom_severity(self):
        """Test recording violation with custom severity."""
        e = ConductPolicyEvaluator()
        v = e.record_violation(
            rule=ConductRule.PREFER_UNCERTAINTY,
            violation_type=ConductViolationType.GUESSED_INSTEAD_OF_ASKING,
            message="Guessed",
            severity=ViolationSeverity.ERROR,  # Override default
        )
        assert v.severity == ViolationSeverity.ERROR
    
    def test_reset(self):
        """Test evaluator reset."""
        e = ConductPolicyEvaluator()
        e.add_evidence("test", "target", "result")
        e.record_violation(
            ConductRule.SPEC_FIDELITY,
            ConductViolationType.CREATED_EXISTING_RESOURCE,
            "Test",
        )
        assert len(e.violations) == 1
        assert len(e.evidence_trail) == 1
        
        e.reset()
        assert len(e.violations) == 0
        assert len(e.evidence_trail) == 0


class TestSpecFidelityCheck:
    """Test Rule 1: Spec Fidelity checking."""
    
    def test_compliant_no_creation_of_existing(self):
        """Test compliant case - didn't create what should exist."""
        e = ConductPolicyEvaluator()
        specs = [
            create_resource_spec("/config.yaml", must_exist=True, resource_type="file"),
        ]
        actual_states = {"/config.yaml": True}
        created = []  # Nothing created
        
        violations = e.check_spec_fidelity(specs, actual_states, created)
        assert len(violations) == 0
    
    def test_violation_created_existing_resource(self):
        """Test violation - created resource that spec says must exist."""
        e = ConductPolicyEvaluator()
        specs = [
            create_resource_spec("/config.yaml", must_exist=True, resource_type="file"),
        ]
        actual_states = {"/config.yaml": True}  # Now exists
        created = ["/config.yaml"]  # But we created it!
        
        violations = e.check_spec_fidelity(specs, actual_states, created)
        assert len(violations) == 1
        assert violations[0].violation_type == ConductViolationType.CREATED_EXISTING_RESOURCE
        assert violations[0].severity == ViolationSeverity.CRITICAL


class TestDiscoveryCheck:
    """Test Rule 2: Discovery Before Action checking."""
    
    def test_compliant_discovery_performed(self):
        """Test compliant case - discovery was performed."""
        e = ConductPolicyEvaluator()
        required = ["/file.txt"]
        discoveries = {
            "/file.txt": create_discovery_result("/file.txt", True, "stat", ["/file.txt"]),
        }
        
        violations = e.check_discovery(required, discoveries)
        assert len(violations) == 0
    
    def test_violation_no_discovery(self):
        """Test violation - acted without discovery."""
        e = ConductPolicyEvaluator()
        required = ["/file.txt"]
        discoveries = {}  # No discovery performed
        
        violations = e.check_discovery(required, discoveries)
        assert len(violations) == 1
        assert violations[0].violation_type == ConductViolationType.ACTED_WITHOUT_VERIFICATION
    
    def test_partial_discovery(self):
        """Test violation - discovery for some but not all targets."""
        e = ConductPolicyEvaluator()
        required = ["/file1.txt", "/file2.txt", "/file3.txt"]
        discoveries = {
            "/file1.txt": create_discovery_result("/file1.txt", True, "stat"),
            # file2 and file3 not discovered
        }
        
        violations = e.check_discovery(required, discoveries)
        assert len(violations) == 2  # Two targets not discovered


class TestEvidenceCompletenessCheck:
    """Test Rule 3: Evidence-Based Execution checking."""
    
    def test_compliant_all_evidence_present(self):
        """Test compliant case - all required evidence present."""
        e = ConductPolicyEvaluator()
        e.add_evidence("inspection", "/file", "checked")
        e.add_evidence("finding", "/file", "exists")
        e.add_evidence("decision", "/file", "proceed")
        
        violations = e.check_evidence_completeness(["inspection", "finding", "decision"])
        assert len(violations) == 0
    
    def test_violation_no_evidence(self):
        """Test violation - no evidence at all."""
        e = ConductPolicyEvaluator()
        # No evidence added
        
        violations = e.check_evidence_completeness(["inspection"])
        assert len(violations) == 1
        assert violations[0].violation_type == ConductViolationType.NO_EVIDENCE_PRODUCED
    
    def test_violation_missing_evidence_type(self):
        """Test violation - some evidence types missing."""
        e = ConductPolicyEvaluator()
        e.add_evidence("inspection", "/file", "checked")
        # Missing "finding" and "decision"
        
        violations = e.check_evidence_completeness(["inspection", "finding", "decision"])
        assert len(violations) == 2


class TestFullEvaluation:
    """Test full compliance evaluation."""
    
    def test_fully_compliant_evaluation(self):
        """Test a fully compliant execution."""
        e = ConductPolicyEvaluator(spec_id="job-1", spec_hash="hash123")
        
        # Add evidence
        e.add_evidence("inspection", "/config.yaml", "found")
        e.add_evidence("finding", "/config.yaml", "exists, size=1024")
        e.add_evidence("decision", "/config.yaml", "proceed with modification")
        
        # Evaluate
        result = e.evaluate(
            resource_specs=[create_resource_spec("/config.yaml", True, "file")],
            actual_states={"/config.yaml": True},
            created_resources=[],
            required_targets=["/config.yaml"],
            discovery_results={
                "/config.yaml": create_discovery_result("/config.yaml", True, "stat"),
            },
            required_evidence_types=["inspection", "finding", "decision"],
        )
        
        assert result.compliant is True
        assert len(result.violations) == 0
        assert len(result.evidence_trail) == 3
        assert result.spec_id == "job-1"
    
    def test_non_compliant_multiple_violations(self):
        """Test evaluation with multiple violations."""
        e = ConductPolicyEvaluator()
        
        # No evidence added
        
        result = e.evaluate(
            resource_specs=[create_resource_spec("/config.yaml", True, "file")],
            actual_states={"/config.yaml": True},
            created_resources=["/config.yaml"],  # Violation: created existing
            required_targets=["/config.yaml"],
            discovery_results={},  # Violation: no discovery
            required_evidence_types=["inspection"],  # Violation: no evidence
        )
        
        assert result.compliant is False
        assert len(result.violations) >= 3


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_resource_spec(self):
        """Test resource spec factory."""
        s = create_resource_spec("/path", True, "folder", "Test folder")
        assert s.path == "/path"
        assert s.must_exist is True
        assert s.resource_type == "folder"
        assert s.description == "Test folder"
    
    def test_create_discovery_result(self):
        """Test discovery result factory."""
        d = create_discovery_result(
            "file.txt",
            True,
            "stat",
            ["/a", "/b"],
            {"mtime": 12345},
        )
        assert d.target == "file.txt"
        assert d.exists is True
        assert d.metadata["mtime"] == 12345
    
    def test_format_compliance_report_compliant(self):
        """Test report formatting for compliant result."""
        result = ConductComplianceResult(
            compliant=True,
            spec_id="job-1",
            summary="All good",
        )
        report = format_compliance_report(result)
        assert "COMPLIANT" in report
        assert "job-1" in report
    
    def test_format_compliance_report_violations(self):
        """Test report formatting with violations."""
        v = ConductViolation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Bad action",
            severity=ViolationSeverity.CRITICAL,
            remediation_hint="Don't do that",
        )
        result = ConductComplianceResult(
            compliant=False,
            violations=[v],
            summary="Violations found",
        )
        report = format_compliance_report(result)
        assert "NON-COMPLIANT" in report
        assert "VIOLATIONS" in report
        assert "CRITICAL" in report
        assert "Bad action" in report
        assert "Don't do that" in report
    
    def test_get_rule_description(self):
        """Test getting rule descriptions."""
        desc = get_rule_description(ConductRule.SPEC_FIDELITY)
        assert "spec" in desc.lower()
        assert "exist" in desc.lower()
    
    def test_get_edge_case_ruling(self):
        """Test getting edge case rulings."""
        ruling = get_edge_case_ruling("partial_existence")
        assert ruling is not None
        assert ruling["ruling"] == "FAIL_WITH_EVIDENCE"
    
    def test_get_edge_case_ruling_not_found(self):
        """Test getting non-existent edge case."""
        ruling = get_edge_case_ruling("nonexistent_case")
        assert ruling is None
    
    def test_compute_spec_hash(self):
        """Test spec hash computation."""
        content = "This is a spec"
        h1 = compute_spec_hash(content)
        h2 = compute_spec_hash(content)
        assert h1 == h2
        assert len(h1) == 16
        
        # Different content -> different hash
        h3 = compute_spec_hash("Different spec")
        assert h3 != h1


# =============================================================================
# SCENARIO EXAMPLE TESTS
# =============================================================================

class TestScenarioExamples:
    """Test scenario examples are properly defined."""
    
    def test_scenarios_exist(self):
        """Test that scenarios are defined."""
        scenarios = list_scenario_examples()
        assert len(scenarios) > 0
    
    def test_correct_scenarios_marked_not_violation(self):
        """Test that correct scenarios are not violations."""
        correct_scenarios = [
            "correct_discovery_then_modify",
            "correct_graceful_failure",
            "correct_request_clarification",
        ]
        for scenario_id in correct_scenarios:
            scenario = get_scenario_example(scenario_id)
            if scenario:  # Only test if exists
                assert scenario["is_violation"] is False, f"{scenario_id} should not be a violation"
    
    def test_violation_scenarios_marked_as_violations(self):
        """Test that violation scenarios are properly marked."""
        violation_scenarios = [
            "violation_create_existing",
            "violation_no_discovery",
            "violation_silent_substitution",
            "violation_guessing",
            "violation_unauthorized_recovery",
        ]
        for scenario_id in violation_scenarios:
            scenario = get_scenario_example(scenario_id)
            if scenario:  # Only test if exists
                assert scenario["is_violation"] is True, f"{scenario_id} should be a violation"
                assert "violated_rule" in scenario, f"{scenario_id} should specify violated rule"
    
    def test_scenario_has_required_fields(self):
        """Test each scenario has required fields."""
        required_fields = ["title", "description", "is_violation", "steps", "outcome"]
        for scenario_id, scenario in SCENARIO_EXAMPLES.items():
            for field in required_fields:
                assert field in scenario, f"Scenario {scenario_id} missing field: {field}"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge case rule definitions."""
    
    def test_all_edge_cases_have_ruling(self):
        """Test all edge cases have a ruling."""
        for case_id, case_def in EDGE_CASE_RULES.items():
            assert "ruling" in case_def, f"Edge case {case_id} missing ruling"
            assert "description" in case_def, f"Edge case {case_id} missing description"
            assert "applies_to" in case_def, f"Edge case {case_id} missing applies_to"
    
    def test_edge_case_applies_to_valid_rules(self):
        """Test edge cases reference valid conduct rules."""
        for case_id, case_def in EDGE_CASE_RULES.items():
            for rule in case_def["applies_to"]:
                assert isinstance(rule, ConductRule), f"Edge case {case_id} references invalid rule"
    
    def test_partial_existence_case(self):
        """Test partial existence edge case."""
        case = get_edge_case_ruling("partial_existence")
        assert case["ruling"] == "FAIL_WITH_EVIDENCE"
        assert ConductRule.SPEC_FIDELITY in case["applies_to"]
    
    def test_ambiguous_path_case(self):
        """Test ambiguous path edge case."""
        case = get_edge_case_ruling("ambiguous_path_spec")
        assert case["ruling"] == "REQUEST_CLARIFICATION"
        assert ConductRule.PREFER_UNCERTAINTY in case["applies_to"]
    
    def test_transient_resource_case(self):
        """Test transient resource edge case."""
        case = get_edge_case_ruling("transient_resource")
        assert case["ruling"] == "FAIL_NO_RETRY"
    
    def test_permission_denied_case(self):
        """Test permission denied edge case."""
        case = get_edge_case_ruling("permission_denied")
        assert "FAIL" in case["ruling"]
        assert ConductRule.GRACEFUL_FAILURE in case["applies_to"]


# =============================================================================
# GLOBAL CONDUCT RULES STRUCTURE TESTS
# =============================================================================

class TestGlobalConductRulesStructure:
    """Test the GLOBAL_CONDUCT_RULES constant structure."""
    
    def test_all_rules_have_definition(self):
        """Test all ConductRule values have definitions."""
        for rule in ConductRule:
            assert rule in GLOBAL_CONDUCT_RULES, f"Rule {rule.name} missing from GLOBAL_CONDUCT_RULES"
    
    def test_rule_definitions_have_required_fields(self):
        """Test each rule definition has required fields."""
        required_fields = ["id", "name", "description", "violations", "severity_default"]
        for rule, definition in GLOBAL_CONDUCT_RULES.items():
            for field in required_fields:
                assert field in definition, f"Rule {rule.name} missing field: {field}"
    
    def test_rule_ids_are_numbered(self):
        """Test rule IDs follow RULE_N pattern."""
        for rule, definition in GLOBAL_CONDUCT_RULES.items():
            assert definition["id"].startswith("RULE_"), f"Rule {rule.name} ID should start with RULE_"
    
    def test_rule_violations_are_valid_types(self):
        """Test rule violations reference valid violation types."""
        for rule, definition in GLOBAL_CONDUCT_RULES.items():
            for violation_type in definition["violations"]:
                assert isinstance(violation_type, ConductViolationType), \
                    f"Rule {rule.name} has invalid violation type"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests simulating real scenarios."""
    
    def test_scenario_file_modification_compliant(self):
        """Test compliant file modification workflow."""
        e = ConductPolicyEvaluator(spec_id="job-modify-config")
        
        # Step 1: Discovery
        e.add_evidence("inspection", "/config.yaml", "checking existence")
        discovery = create_discovery_result("/config.yaml", True, "stat", ["/config.yaml"])
        e.add_evidence("finding", "/config.yaml", f"exists={discovery.exists}")
        
        # Step 2: Decision
        e.add_evidence("decision", "/config.yaml", "proceeding with modification")
        
        # Evaluate
        result = e.evaluate(
            resource_specs=[create_resource_spec("/config.yaml", True, "file")],
            actual_states={"/config.yaml": True},
            created_resources=[],
            required_targets=["/config.yaml"],
            discovery_results={"/config.yaml": discovery},
            required_evidence_types=["inspection", "finding", "decision"],
        )
        
        assert result.compliant is True
    
    def test_scenario_missing_file_graceful_fail(self):
        """Test graceful failure when required file missing."""
        e = ConductPolicyEvaluator(spec_id="job-process-report")
        
        # Step 1: Discovery - file not found
        e.add_evidence("inspection", "/report.pdf", "checking existence")
        discovery = create_discovery_result("/report.pdf", False, "stat", ["/report.pdf"])
        e.add_evidence("finding", "/report.pdf", f"exists={discovery.exists}")
        
        # Step 2: Graceful failure
        e.add_evidence("failure", "/report.pdf", "required file not found, failing gracefully")
        
        # Evaluate - should be compliant because we handled missing file correctly
        result = e.evaluate(
            required_targets=["/report.pdf"],
            discovery_results={"/report.pdf": discovery},
            required_evidence_types=["inspection", "finding", "failure"],
        )
        
        # This is compliant because we discovered, found missing, and recorded failure
        assert result.compliant is True
    
    def test_scenario_multiple_files_partial_discovery(self):
        """Test partial discovery across multiple files."""
        e = ConductPolicyEvaluator(spec_id="job-batch")
        
        # Only discover first file
        e.add_evidence("inspection", "/file1.txt", "checked")
        
        result = e.evaluate(
            required_targets=["/file1.txt", "/file2.txt", "/file3.txt"],
            discovery_results={
                "/file1.txt": create_discovery_result("/file1.txt", True, "stat"),
            },
        )
        
        assert result.compliant is False
        # Should have violations for file2 and file3
        violation_targets = [v.evidence.get("target") for v in result.violations]
        assert "/file2.txt" in violation_targets
        assert "/file3.txt" in violation_targets


# =============================================================================
# SERIALIZATION ROUND-TRIP TESTS
# =============================================================================

class TestSerializationRoundTrip:
    """Test serialization and deserialization round trips."""
    
    def test_violation_round_trip(self):
        """Test ConductViolation survives serialization."""
        original = ConductViolation(
            rule=ConductRule.NO_SILENT_SUBSTITUTION,
            violation_type=ConductViolationType.ASSUMED_FILE_PATH,
            message="Assumed path without verification",
            severity=ViolationSeverity.CRITICAL,
            evidence={"assumed": "/path/a", "actual": "/path/b"},
            spec_id="job-x",
            spec_hash="hash123",
            remediation_hint="Verify paths before use",
        )
        
        serialized = original.to_dict()
        restored = ConductViolation.from_dict(serialized)
        
        assert restored.rule == original.rule
        assert restored.violation_type == original.violation_type
        assert restored.message == original.message
        assert restored.severity == original.severity
        assert restored.evidence == original.evidence
        assert restored.spec_id == original.spec_id
        assert restored.remediation_hint == original.remediation_hint
    
    def test_compliance_result_to_dict(self):
        """Test ConductComplianceResult serialization."""
        v = ConductViolation(
            rule=ConductRule.SPEC_FIDELITY,
            violation_type=ConductViolationType.CREATED_EXISTING_RESOURCE,
            message="Test",
        )
        e = EvidenceRecord(
            action="test",
            target="target",
            result="result",
        )
        result = ConductComplianceResult(
            compliant=False,
            violations=[v],
            evidence_trail=[e],
            spec_id="job-1",
            spec_hash="hash",
            summary="Test summary",
        )
        
        d = result.to_dict()
        assert d["compliant"] is False
        assert len(d["violations"]) == 1
        assert len(d["evidence_trail"]) == 1
        assert d["spec_id"] == "job-1"
