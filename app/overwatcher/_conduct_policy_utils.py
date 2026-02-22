from __future__ import annotations
import hashlib
from typing import Any, Dict, List, Optional


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
    from .conduct_policy import ResourceExistenceSpec
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
    from .conduct_policy import DiscoveryResult
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
    from .conduct_policy import ConductComplianceResult
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
    from .conduct_policy import ConductRule, GLOBAL_CONDUCT_RULES
    rule_def = GLOBAL_CONDUCT_RULES.get(rule, {})
    return rule_def.get("description", "No description available.")

def get_edge_case_ruling(edge_case_id: str) -> Optional[Dict[str, Any]]:
    """Get the ruling for a specific edge case."""
    from .conduct_policy import EDGE_CASE_RULES
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

def get_scenario_example(scenario_id: str) -> Optional[Dict[str, Any]]:
    """Get a scenario example by ID."""
    from .conduct_policy import SCENARIO_EXAMPLES
    return SCENARIO_EXAMPLES.get(scenario_id)

def list_scenario_examples() -> List[str]:
    """List all available scenario example IDs."""
    from .conduct_policy import SCENARIO_EXAMPLES
    return list(SCENARIO_EXAMPLES.keys())


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "ConductComplianceResult": "_conduct_policy_utils_4",
    "ConductRule": "_conduct_policy_utils_4",
    "ConductViolation": "_conduct_policy_utils_4",
    "ConductViolationType": "_conduct_policy_utils_4",
    "DiscoveryResult": "_conduct_policy_utils_4",
    "EvidenceRecord": "_conduct_policy_utils_4",
    "ResourceExistenceSpec": "_conduct_policy_utils_4",
    "ViolationSeverity": "_conduct_policy_utils_4",
    "ConductPolicyEvaluator": "_conduct_policy_utils_5",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.overwatcher.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
