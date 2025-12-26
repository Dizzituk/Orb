# FILE: app/pot_spec/schemas.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class Requirements(BaseModel):
    must: list[str] = Field(default_factory=list)
    should: list[str] = Field(default_factory=list)
    can: list[str] = Field(default_factory=list)



class PoTSpecDraft(BaseModel):
    """LLM-facing draft output for Spec Gate (no IDs/hashes).

    This is the only shape the Spec Gate model should emit.
    The service layer will add job_id/spec_id/spec_version/spec_hash, etc.
    """
    goal: str = Field(default="")
    requirements: Requirements = Field(default_factory=Requirements)
    constraints: dict[str, Any] = Field(default_factory=dict)
    acceptance_tests: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    repo_snapshot: Optional[dict[str, Any]] = None

class PoTSpecPayload(BaseModel):
    """Canonical hash material written to spec_vN.json (NO spec_hash field).

    Rationale: spec_hash is derived from this payload's canonical bytes; embedding it
    would be self-referential and would prevent a clean "hash of exact file bytes".
    """
    # Identity / lineage
    job_id: str
    spec_id: str
    spec_version: int = 1
    parent_spec_id: Optional[str] = None

    # Metadata
    created_at: datetime
    created_by_model: str

    # Content
    goal: str
    requirements: Requirements = Field(default_factory=Requirements)
    constraints: dict[str, Any] = Field(default_factory=dict)
    acceptance_tests: list[str] = Field(default_factory=list)

    # Workflow control
    open_questions: list[str] = Field(default_factory=list)

    # Model suggestions (must be clearly marked as non-binding)
    recommendations: list[str] = Field(default_factory=list)

    # Optional provenance
    repo_snapshot: Optional[dict[str, Any]] = None


class PoTSpec(PoTSpecPayload):
    """Full spec object used by the pipeline (payload + derived spec_hash)."""
    spec_hash: str
