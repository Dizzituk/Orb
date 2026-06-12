# Purpose: schema utils
# Called-by: app.specs._schema_utils_1
# Depends-on: app.specs._schema_utils_1, app.specs.schema
# Last-renovated: 2026-06-11
from __future__ import annotations
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from app.specs.schema import Spec

# JobKind used at runtime (value comparison) — lazy import
def _get_job_kind():
    from app.specs._schema_utils_1 import JobKind
    return JobKind


class SpecStatus(str, Enum):
    """Spec lifecycle status."""
    DRAFT = "draft"              # Created by Weaver, not yet validated
    PENDING_VALIDATION = "pending_validation"  # Sent to Spec Gate
    VALIDATED = "validated"      # Approved by Spec Gate
    REJECTED = "rejected"        # Rejected by Spec Gate
    SUPERSEDED = "superseded"    # Replaced by newer spec

@dataclass
class SpecOutput:
    """Output definition for a spec."""
    name: str
    type: str
    example: Optional[str] = None
    acceptance_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SpecStep:
    """Step in the spec execution plan."""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)  # other step ids
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class SpecRequirements:
    """Functional and non-functional requirements."""
    functional: List[str] = field(default_factory=list)
    non_functional: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SpecSafety:
    """Risk coverage and mitigations."""
    risks: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    runtime_guards: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SpecMetadata:
    """Additional metadata."""
    priority: str = "medium"  # low, medium, high
    owner: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class SpecValidationResult:
    """Result of validating a spec against the schema."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

def spec_to_markdown(spec: Spec) -> str:
    """
    Convert spec to human-readable markdown.
    
    This is the secondary format for human review.
    JSON remains canonical.
    """
    lines = []
    
    lines.append(f"# {spec.title or 'Untitled Spec'}")
    lines.append("")
    lines.append(f"**Spec ID:** `{spec.spec_id}`")
    lines.append(f"**Version:** {spec.spec_version}")
    lines.append(f"**Status:** Draft")
    
    # v1.1: Show job classification
    if spec.job_kind and spec.job_kind != _get_job_kind().UNKNOWN.value:
        lines.append(f"**Job Kind:** `{spec.job_kind}` (confidence: {spec.job_kind_confidence:.2f})")
    
    lines.append("")
    
    if spec.summary:
        lines.append("## Summary")
        lines.append(spec.summary)
        lines.append("")
    
    lines.append("## Objective")
    lines.append(spec.objective or "_Not specified_")
    lines.append("")
    
    # v1.1: Show sandbox grounding if present
    if spec.sandbox_discovery_used and spec.sandbox_input_path:
        lines.append("## Sandbox Grounding")
        lines.append("")
        lines.append(f"- **Input:** `{spec.sandbox_input_path}`")
        if spec.sandbox_output_path:
            lines.append(f"- **Output:** `{spec.sandbox_output_path}`")
        if spec.sandbox_selected_type:
            lines.append(f"- **Content Type:** {spec.sandbox_selected_type}")
        lines.append("")
    
    # Requirements
    lines.append("## Requirements")
    lines.append("")
    lines.append("### Functional")
    if spec.requirements.functional:
        for req in spec.requirements.functional:
            lines.append(f"- {req}")
    else:
        lines.append("_None specified_")
    lines.append("")
    
    lines.append("### Non-Functional")
    if spec.requirements.non_functional:
        for req in spec.requirements.non_functional:
            lines.append(f"- {req}")
    else:
        lines.append("_None specified_")
    lines.append("")
    
    # Constraints
    lines.append("## Constraints")
    lines.append("")
    if spec.constraints.budget:
        lines.append(f"- **Budget:** {spec.constraints.budget}")
    if spec.constraints.latency:
        lines.append(f"- **Latency:** {spec.constraints.latency}")
    if spec.constraints.platform:
        lines.append(f"- **Platform:** {spec.constraints.platform}")
    if spec.constraints.integrations:
        lines.append(f"- **Integrations:** {', '.join(spec.constraints.integrations)}")
    if spec.constraints.compliance:
        lines.append(f"- **Compliance:** {', '.join(spec.constraints.compliance)}")
    if not any([spec.constraints.budget, spec.constraints.latency, spec.constraints.platform,
                spec.constraints.integrations, spec.constraints.compliance]):
        lines.append("_None specified_")
    lines.append("")
    
    # Safety
    lines.append("## Safety")
    lines.append("")
    lines.append("### Risks")
    if spec.safety.risks:
        for risk in spec.safety.risks:
            lines.append(f"- {risk}")
    else:
        lines.append("_None identified_")
    lines.append("")
    
    lines.append("### Mitigations")
    if spec.safety.mitigations:
        for mit in spec.safety.mitigations:
            lines.append(f"- {mit}")
    else:
        lines.append("_None specified_")
    lines.append("")
    
    # Acceptance Criteria
    lines.append("## Acceptance Criteria")
    if spec.acceptance_criteria:
        for i, criterion in enumerate(spec.acceptance_criteria, 1):
            lines.append(f"{i}. {criterion}")
    else:
        lines.append("_None specified_")
    lines.append("")
    
    # Steps
    if spec.steps:
        lines.append("## Execution Steps")
        lines.append("")
        for step in spec.steps:
            deps = f" (depends on: {', '.join(step.dependencies)})" if step.dependencies else ""
            lines.append(f"### Step {step.id}{deps}")
            lines.append(step.description)
            if step.notes:
                lines.append(f"_Note: {step.notes}_")
            lines.append("")
    
    # Non-goals
    if spec.non_goals:
        lines.append("## Non-Goals (Out of Scope)")
        for ng in spec.non_goals:
            lines.append(f"- {ng}")
        lines.append("")
    
    # Provenance
    lines.append("---")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- **Generator:** {spec.provenance.generator_model}")
    lines.append(f"- **Created:** {spec.provenance.created_at}")
    if spec.provenance.commit_hash:
        lines.append(f"- **Commit:** `{spec.provenance.commit_hash[:12]}`")
    lines.append(f"- **Source Messages:** {len(spec.provenance.source_message_ids)} messages")
    lines.append(f"- **Token Count:** {spec.provenance.token_count}")
    
    return "\n".join(lines)


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "JobKind": "_schema_utils_1",
    "SPEC_SCHEMA_VERSION": "_schema_utils_1",
    "SpecConstraints": "_schema_utils_1",
    "SpecInput": "_schema_utils_1",
    "SpecProvenance": "_schema_utils_1",
    "validate_spec": "_schema_utils_1",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.specs.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
