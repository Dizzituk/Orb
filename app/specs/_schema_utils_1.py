# Purpose: schema utils 1
# Called-by: app.specs._schema_utils
# Depends-on: app.specs._schema_utils, app.specs.schema
# Last-renovated: 2026-06-11
from __future__ import annotations
from app.specs._schema_utils import SpecValidationResult
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from app.specs.schema import Spec


SPEC_SCHEMA_VERSION = "1.2"

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
