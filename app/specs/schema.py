# FILE: app/specs/schema.py
"""
Canonical JSON Schema for ASTRA Specs.

This defines the versioned schema that Spec Gate validates against.
All specs must conform to this schema to be accepted by the Critical Pipeline.

INVARIANT: JSON Schema is canonical truth. Markdown is secondary.
INVARIANT: Invalid JSON = rejected spec.

v1.1 (2026-01-22): Added job classification and sandbox grounding fields
- job_kind, job_kind_confidence, job_kind_reason for deterministic routing
- sandbox_* fields for micro-execution job support
- These fields enable Critical Pipeline to route correctly without guessing
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
# v1.2: Added multi-target read fields (is_multi_target_read, multi_target_files, grounding_data)
SPEC_SCHEMA_VERSION = "1.2"


# v1.2 (2026-01-31): Added multi-target read fields for Critical Pipeline
# - is_multi_target_read: Flag for multi-file read operations
# - multi_target_files: Array of file targets with content
# - grounding_data: Dict containing all grounding evidence
# CRITICAL: These fields MUST be persisted for micro_quickcheck() to pass on multi-target jobs


class SpecStatus(str, Enum):
    """Spec lifecycle status."""
    DRAFT = "draft"              # Created by Weaver, not yet validated
    PENDING_VALIDATION = "pending_validation"  # Sent to Spec Gate
    VALIDATED = "validated"      # Approved by Spec Gate
    REJECTED = "rejected"        # Rejected by Spec Gate
    SUPERSEDED = "superseded"    # Replaced by newer spec


class JobKind(str, Enum):
    """
    Job classification for pipeline routing (v1.1).
    
    Determined by SpecGate using deterministic rules (NO LLM).
    Critical Pipeline MUST obey this classification.
    """
    MICRO_EXECUTION = "micro_execution"  # Simple read/write/answer tasks (seconds)
    REPO_CHANGE = "repo_change"          # Code edits, file changes (minutes)
    ARCHITECTURE = "architecture"         # Design/build/refactor (2-5 minutes)
    UNKNOWN = "unknown"                   # Ambiguous - escalate to architecture


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
    
    v1.1: Added job classification and sandbox grounding fields for
    deterministic Critical Pipeline routing.
    """
    # Required identifiers
    spec_version: str = SPEC_SCHEMA_VERSION
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Core content
    title: str = ""
    summary: str = ""
    objective: str = ""
    
    # ==========================================================================
    # JOB CLASSIFICATION (v1.1) - For Critical Pipeline routing
    # ==========================================================================
    # These fields are set by SpecGate using DETERMINISTIC rules (no LLM).
    # Critical Pipeline MUST obey job_kind - it should NOT reclassify.
    
    job_kind: str = JobKind.UNKNOWN.value  # micro_execution | repo_change | architecture | unknown
    job_kind_confidence: float = 0.0       # 0.0 - 1.0
    job_kind_reason: str = ""              # Human-readable explanation
    
    # ==========================================================================
    # SANDBOX GROUNDING (v1.1) - For micro-execution jobs
    # ==========================================================================
    # These fields are populated by SpecGate's sandbox discovery.
    # They enable micro-execution jobs to skip architecture generation.
    
    sandbox_input_path: Optional[str] = None       # e.g., C:\Users\...\Desktop\test\test.txt
    sandbox_output_path: Optional[str] = None      # e.g., C:\Users\...\Desktop\test\reply.txt
    sandbox_generated_reply: Optional[str] = None  # Pre-generated reply content (if applicable)
    sandbox_discovery_used: bool = False           # True if sandbox inspector ran
    sandbox_selected_type: Optional[str] = None    # Content type: message, code, etc.
    sandbox_folder_path: Optional[str] = None      # Parent folder path
    sandbox_input_excerpt: Optional[str] = None    # First 500 chars of input file
    sandbox_discovery_status: Optional[str] = None # success, failed, not_attempted
    sandbox_output_mode: Optional[str] = None      # v1.2: append_in_place, separate_reply_file, chat_only
    sandbox_insertion_format: Optional[str] = None # v1.2: e.g., '\n\nAnswer:\n{reply}\n'
    
    # ==========================================================================
    # MULTI-TARGET READ FIELDS (v1.2) - For multi-file micro-execution jobs
    # ==========================================================================
    # These fields are CRITICAL for micro_quickcheck() to pass on multi-target jobs.
    # They are populated by SpecGate's multi-target discovery and must survive
    # the persistence/retrieval chain to reach Critical Pipeline.
    
    is_multi_target_read: bool = False                         # True if this is a multi-file read operation
    multi_target_files: List[Dict[str, Any]] = field(default_factory=list)  # List of file targets with content
    grounding_data: Dict[str, Any] = field(default_factory=dict)  # All grounding evidence from SpecGate
    
    # Grounding metadata (what exists in the environment)
    goal: str = ""
    what_exists: List[str] = field(default_factory=list)
    what_missing: List[str] = field(default_factory=list)
    constraints_from_repo: List[str] = field(default_factory=list)
    constraints_from_intent: List[str] = field(default_factory=list)
    proposed_steps: List[str] = field(default_factory=list)
    acceptance_tests: List[str] = field(default_factory=list)
    
    # ==========================================================================
    # STANDARD SPEC FIELDS
    # ==========================================================================
    
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
        result = {
            "spec_version": self.spec_version,
            "spec_id": self.spec_id,
            "title": self.title,
            "summary": self.summary,
            "objective": self.objective,
            
            # v1.1: Job classification fields
            "job_kind": self.job_kind,
            "job_kind_confidence": self.job_kind_confidence,
            "job_kind_reason": self.job_kind_reason,
            
            # v1.1: Sandbox grounding fields
            "sandbox_input_path": self.sandbox_input_path,
            "sandbox_output_path": self.sandbox_output_path,
            "sandbox_generated_reply": self.sandbox_generated_reply,
            "sandbox_discovery_used": self.sandbox_discovery_used,
            "sandbox_selected_type": self.sandbox_selected_type,
            "sandbox_folder_path": self.sandbox_folder_path,
            "sandbox_input_excerpt": self.sandbox_input_excerpt,
            "sandbox_discovery_status": self.sandbox_discovery_status,
            "sandbox_output_mode": self.sandbox_output_mode,
            "sandbox_insertion_format": self.sandbox_insertion_format,
            
            # v1.2: Multi-target read fields (CRITICAL for micro_quickcheck)
            "is_multi_target_read": self.is_multi_target_read,
            "multi_target_files": self.multi_target_files,
            "grounding_data": self.grounding_data,
            
            # v1.1: Grounding metadata
            "goal": self.goal,
            "what_exists": self.what_exists,
            "what_missing": self.what_missing,
            "constraints_from_repo": self.constraints_from_repo,
            "constraints_from_intent": self.constraints_from_intent,
            "proposed_steps": self.proposed_steps,
            "acceptance_tests": self.acceptance_tests,
            
            # Standard fields
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
        return result
    
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
        
        # v1.1: Job classification fields
        spec.job_kind = data.get("job_kind", JobKind.UNKNOWN.value)
        spec.job_kind_confidence = float(data.get("job_kind_confidence", 0.0))
        spec.job_kind_reason = data.get("job_kind_reason", "")
        
        # v1.1: Sandbox grounding fields
        spec.sandbox_input_path = data.get("sandbox_input_path")
        spec.sandbox_output_path = data.get("sandbox_output_path")
        spec.sandbox_generated_reply = data.get("sandbox_generated_reply")
        spec.sandbox_discovery_used = bool(data.get("sandbox_discovery_used", False))
        spec.sandbox_selected_type = data.get("sandbox_selected_type")
        spec.sandbox_folder_path = data.get("sandbox_folder_path")
        spec.sandbox_input_excerpt = data.get("sandbox_input_excerpt")
        spec.sandbox_discovery_status = data.get("sandbox_discovery_status")
        spec.sandbox_output_mode = data.get("sandbox_output_mode")
        spec.sandbox_insertion_format = data.get("sandbox_insertion_format")
        
        # v1.2: Multi-target read fields (CRITICAL for micro_quickcheck)
        spec.is_multi_target_read = bool(data.get("is_multi_target_read", False))
        spec.multi_target_files = data.get("multi_target_files", [])
        spec.grounding_data = data.get("grounding_data", {})
        
        # v1.1: Grounding metadata
        spec.goal = data.get("goal", "")
        spec.what_exists = data.get("what_exists", [])
        spec.what_missing = data.get("what_missing", [])
        spec.constraints_from_repo = data.get("constraints_from_repo", [])
        spec.constraints_from_intent = data.get("constraints_from_intent", [])
        spec.proposed_steps = data.get("proposed_steps", [])
        spec.acceptance_tests = data.get("acceptance_tests", [])
        
        # Requirements
        req_data = data.get("requirements", {})
        if isinstance(req_data, dict):
            spec.requirements = SpecRequirements(
                functional=req_data.get("functional", []),
                non_functional=req_data.get("non_functional", []),
            )
        
        # Constraints
        con_data = data.get("constraints", {})
        if isinstance(con_data, dict):
            spec.constraints = SpecConstraints(
                budget=con_data.get("budget"),
                latency=con_data.get("latency"),
                platform=con_data.get("platform"),
                integrations=con_data.get("integrations", []),
                compliance=con_data.get("compliance", []),
            )
        
        # Safety
        safety_data = data.get("safety", {})
        if isinstance(safety_data, dict):
            spec.safety = SpecSafety(
                risks=safety_data.get("risks", []),
                mitigations=safety_data.get("mitigations", []),
                runtime_guards=safety_data.get("runtime_guards", []),
            )
        
        spec.acceptance_criteria = data.get("acceptance_criteria", [])
        
        # Inputs
        inputs_data = data.get("inputs", [])
        if isinstance(inputs_data, list):
            spec.inputs = []
            for inp in inputs_data:
                if isinstance(inp, dict):
                    try:
                        spec.inputs.append(SpecInput(**inp))
                    except TypeError:
                        # Handle unexpected fields
                        spec.inputs.append(SpecInput(
                            name=inp.get("name", ""),
                            type=inp.get("type", "unknown"),
                            required=inp.get("required", True),
                            example=inp.get("example"),
                            source=inp.get("source"),
                        ))
        
        # Outputs
        outputs_data = data.get("outputs", [])
        if isinstance(outputs_data, list):
            spec.outputs = []
            for out in outputs_data:
                if isinstance(out, dict):
                    try:
                        spec.outputs.append(SpecOutput(**out))
                    except TypeError:
                        spec.outputs.append(SpecOutput(
                            name=out.get("name", ""),
                            type=out.get("type", "unknown"),
                            example=out.get("example"),
                            acceptance_criteria=out.get("acceptance_criteria", []),
                        ))
        
        # Steps
        steps_data = data.get("steps", [])
        if isinstance(steps_data, list):
            spec.steps = []
            for step in steps_data:
                if isinstance(step, dict):
                    try:
                        spec.steps.append(SpecStep(**step))
                    except TypeError:
                        spec.steps.append(SpecStep(
                            id=step.get("id", ""),
                            description=step.get("description", ""),
                            dependencies=step.get("dependencies", []),
                            notes=step.get("notes"),
                        ))
        
        spec.dependencies = data.get("dependencies", [])
        spec.non_goals = data.get("non_goals", [])
        
        # Metadata
        meta_data = data.get("metadata", {})
        if isinstance(meta_data, dict):
            spec.metadata = SpecMetadata(
                priority=meta_data.get("priority", "medium"),
                owner=meta_data.get("owner"),
                tags=meta_data.get("tags", []),
            )
        
        # Provenance
        prov_data = data.get("provenance", {})
        if isinstance(prov_data, dict):
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
    if spec.spec_version not in [SPEC_SCHEMA_VERSION, "1.0", "1.1"]:
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
    
    # v1.1: Job classification check
    if spec.job_kind == JobKind.UNKNOWN.value and spec.job_kind_confidence < 0.5:
        warnings.append("job_kind is unknown with low confidence - routing may be suboptimal")
    
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
    
    # v1.1: Show job classification
    if spec.job_kind and spec.job_kind != JobKind.UNKNOWN.value:
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
