# FILE: app/specs/schema.py
"""
Canonical JSON Schema for ASTRA Specs.

This defines the versioned schema that Spec Gate validates against.
All specs must conform to this schema to be accepted by the Critical Pipeline.

INVARIANT: JSON Schema is canonical truth. Markdown is secondary.
INVARIANT: Invalid JSON = rejected spec.
"""
from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

# Current schema version
SPEC_SCHEMA_VERSION = "1.0"


class SpecStatus(str, Enum):
    """Spec lifecycle status."""
    DRAFT = "draft"              # Created by Weaver, not yet validated
    PENDING_VALIDATION = "pending_validation"  # Sent to Spec Gate
    VALIDATED = "validated"      # Approved by Spec Gate
    REJECTED = "rejected"        # Rejected by Spec Gate
    SUPERSEDED = "superseded"    # Replaced by newer spec


@dataclass
class SpecInput:
    """Input definition for a spec."""
    name: str
    type: str
    required: bool = True
    example: Optional[str] = None
    source: Optional[str] = None  # file path, API, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


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
class SpecConstraints:
    """Budget, latency, platform, and compliance constraints."""
    budget: Optional[str] = None
    latency: Optional[str] = None
    platform: Optional[str] = None
    integrations: List[str] = field(default_factory=list)
    compliance: List[str] = field(default_factory=list)
    
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
class SpecProvenance:
    """Provenance tracking for reproducibility."""
    job_id: Optional[str] = None
    conversation_id: Optional[str] = None
    source_message_ids: List[int] = field(default_factory=list)
    summary_ids: List[int] = field(default_factory=list)  # if summarization used
    commit_hash: Optional[str] = None
    generator_model: str = "weaver-v1"
    token_count: int = 0
    timestamp_start: Optional[str] = None  # ISO-8601
    timestamp_end: Optional[str] = None    # ISO-8601
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Spec:
    """
    Canonical ASTRA Spec structure.
    
    This is the complete spec that Weaver generates and Spec Gate validates.
    """
    # Required identifiers
    spec_version: str = SPEC_SCHEMA_VERSION
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core content
    title: str = ""
    summary: str = ""
    objective: str = ""
    
    # Structured sections
    requirements: SpecRequirements = field(default_factory=SpecRequirements)
    constraints: SpecConstraints = field(default_factory=SpecConstraints)
    safety: SpecSafety = field(default_factory=SpecSafety)
    
    # Acceptance criteria (mandatory, can be empty)
    acceptance_criteria: List[str] = field(default_factory=list)
    
    # I/O definitions
    inputs: List[SpecInput] = field(default_factory=list)
    outputs: List[SpecOutput] = field(default_factory=list)
    
    # Execution steps
    steps: List[SpecStep] = field(default_factory=list)
    
    # Dependencies and scope
    dependencies: List[str] = field(default_factory=list)
    non_goals: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: SpecMetadata = field(default_factory=SpecMetadata)
    provenance: SpecProvenance = field(default_factory=SpecProvenance)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "spec_version": self.spec_version,
            "spec_id": self.spec_id,
            "title": self.title,
            "summary": self.summary,
            "objective": self.objective,
            "requirements": self.requirements.to_dict(),
            "constraints": self.constraints.to_dict(),
            "safety": self.safety.to_dict(),
            "acceptance_criteria": self.acceptance_criteria,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "steps": [s.to_dict() for s in self.steps],
            "dependencies": self.dependencies,
            "non_goals": self.non_goals,
            "metadata": self.metadata.to_dict(),
            "provenance": self.provenance.to_dict(),
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of spec content.
        
        Used for deduplication and change detection.
        Excludes provenance.created_at to allow same-content comparison.
        """
        # Create hashable dict excluding volatile fields
        hash_dict = self.to_dict()
        hash_dict["provenance"].pop("created_at", None)
        
        content = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Spec":
        """Create Spec from dictionary."""
        spec = cls()
        spec.spec_version = data.get("spec_version", SPEC_SCHEMA_VERSION)
        spec.spec_id = data.get("spec_id", str(uuid.uuid4()))
        spec.title = data.get("title", "")
        spec.summary = data.get("summary", "")
        spec.objective = data.get("objective", "")
        
        # Requirements
        req_data = data.get("requirements", {})
        spec.requirements = SpecRequirements(
            functional=req_data.get("functional", []),
            non_functional=req_data.get("non_functional", []),
        )
        
        # Constraints
        con_data = data.get("constraints", {})
        spec.constraints = SpecConstraints(
            budget=con_data.get("budget"),
            latency=con_data.get("latency"),
            platform=con_data.get("platform"),
            integrations=con_data.get("integrations", []),
            compliance=con_data.get("compliance", []),
        )
        
        # Safety
        safety_data = data.get("safety", {})
        spec.safety = SpecSafety(
            risks=safety_data.get("risks", []),
            mitigations=safety_data.get("mitigations", []),
            runtime_guards=safety_data.get("runtime_guards", []),
        )
        
        spec.acceptance_criteria = data.get("acceptance_criteria", [])
        
        # Inputs
        spec.inputs = [
            SpecInput(**inp) for inp in data.get("inputs", [])
        ]
        
        # Outputs
        spec.outputs = [
            SpecOutput(**out) for out in data.get("outputs", [])
        ]
        
        # Steps
        spec.steps = [
            SpecStep(**step) for step in data.get("steps", [])
        ]
        
        spec.dependencies = data.get("dependencies", [])
        spec.non_goals = data.get("non_goals", [])
        
        # Metadata
        meta_data = data.get("metadata", {})
        spec.metadata = SpecMetadata(
            priority=meta_data.get("priority", "medium"),
            owner=meta_data.get("owner"),
            tags=meta_data.get("tags", []),
        )
        
        # Provenance
        prov_data = data.get("provenance", {})
        spec.provenance = SpecProvenance(
            job_id=prov_data.get("job_id"),
            conversation_id=prov_data.get("conversation_id"),
            source_message_ids=prov_data.get("source_message_ids", []),
            summary_ids=prov_data.get("summary_ids", []),
            commit_hash=prov_data.get("commit_hash"),
            generator_model=prov_data.get("generator_model", "weaver-v1"),
            token_count=prov_data.get("token_count", 0),
            timestamp_start=prov_data.get("timestamp_start"),
            timestamp_end=prov_data.get("timestamp_end"),
            created_at=prov_data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
        
        return spec
    
    @classmethod
    def from_json(cls, json_str: str) -> "Spec":
        """Create Spec from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SpecValidationResult:
    """Result of validating a spec against the schema."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def validate_spec(spec: Spec) -> SpecValidationResult:
    """
    Validate a spec against the canonical schema.
    
    Returns validation result with errors and warnings.
    Spec Gate uses this to reject invalid specs.
    """
    errors = []
    warnings = []
    
    # Required fields
    if not spec.spec_version:
        errors.append("spec_version is required")
    if not spec.spec_id:
        errors.append("spec_id is required")
    if not spec.objective:
        errors.append("objective is required")
    
    # Version check
    if spec.spec_version != SPEC_SCHEMA_VERSION:
        warnings.append(f"spec_version '{spec.spec_version}' differs from current '{SPEC_SCHEMA_VERSION}'")
    
    # Provenance checks
    if not spec.provenance.source_message_ids:
        warnings.append("provenance.source_message_ids is empty - spec may not be reproducible")
    if not spec.provenance.commit_hash:
        warnings.append("provenance.commit_hash is missing - spec not anchored to codebase state")
    
    # Content checks
    if not spec.title:
        warnings.append("title is empty")
    if not spec.summary:
        warnings.append("summary is empty")
    if not spec.acceptance_criteria:
        warnings.append("acceptance_criteria is empty - how will you know when it's done?")
    
    # Safety checks
    if not spec.safety.risks and not spec.safety.mitigations:
        warnings.append("safety section is empty - consider adding risk coverage")
    
    return SpecValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


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
    lines.append("")
    
    if spec.summary:
        lines.append("## Summary")
        lines.append(spec.summary)
        lines.append("")
    
    lines.append("## Objective")
    lines.append(spec.objective or "_Not specified_")
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
