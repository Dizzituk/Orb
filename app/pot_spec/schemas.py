# FILE: app/pot_spec/schemas.py
"""
PoT Spec Schemas - Point-of-Truth Specification Models

v1.1 (2026-01-22): Added implementation_stack field
- Captures user's tech stack choice (language, framework, library)
- Prevents architecture drift by anchoring to user-discussed stack
- Includes stack_locked flag to indicate explicit user confirmation
- See CRITICAL_PIPELINE_FAILURE_REPORT.md for context

v1.0 (2026-01): Initial schema
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


class Requirements(BaseModel):
    must: list[str] = Field(default_factory=list)
    should: list[str] = Field(default_factory=list)
    can: list[str] = Field(default_factory=list)


class ImplementationStack(BaseModel):
    """User's chosen technology stack for implementation.
    
    v1.1 (2026-01-22): Added to prevent architecture drift.
    When the user discusses or confirms a tech stack (e.g., Python + Pygame),
    this captures that choice so architecture must honor it.
    
    Attributes:
        language: Primary programming language (e.g., "Python", "TypeScript")
        framework: Framework or library (e.g., "Pygame", "React", "FastAPI")
        runtime: Runtime environment if applicable (e.g., "Node.js", "Electron")
        stack_locked: bool = True if user explicitly confirmed this stack choice
        source: Where this was captured ("user_message", "assistant_proposal_confirmed", "spec_requirement")
        notes: Additional context about the stack choice
    """
    language: Optional[str] = Field(default=None, description="Primary programming language")
    framework: Optional[str] = Field(default=None, description="Framework or library")
    runtime: Optional[str] = Field(default=None, description="Runtime environment if applicable")
    stack_locked: bool = Field(default=False, description="True if user explicitly confirmed this stack")
    source: Optional[str] = Field(default=None, description="Where this choice was captured")
    notes: Optional[str] = Field(default=None, description="Additional context")


class PoTSpecDraft(BaseModel):
    """LLM-facing draft output for Spec Gate (no IDs/hashes).

    This is the only shape the Spec Gate model should emit.
    The service layer will add job_id/spec_id/spec_version/spec_hash, etc.
    
    v1.1 (2026-01-22): Added implementation_stack field for tech stack anchoring.
    """
    goal: str = Field(default="")
    requirements: Requirements = Field(default_factory=Requirements)
    constraints: dict[str, Any] = Field(default_factory=dict)
    acceptance_tests: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    repo_snapshot: Optional[dict[str, Any]] = None
    
    # v1.1: Tech stack anchoring
    implementation_stack: Optional[ImplementationStack] = Field(
        default=None,
        description="User's chosen tech stack - architecture MUST honor this if stack_locked=True"
    )

class PoTSpecPayload(BaseModel):
    """Canonical hash material written to spec_vN.json (NO spec_hash field).

    Rationale: spec_hash is derived from this payload's canonical bytes; embedding it
    would be self-referential and would prevent a clean "hash of exact file bytes".
    
    v1.1 (2026-01-22): Added implementation_stack field for tech stack anchoring.
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
    
    # v1.1: Tech stack anchoring
    implementation_stack: Optional[ImplementationStack] = Field(
        default=None,
        description="User's chosen tech stack - architecture MUST honor this if stack_locked=True"
    )


class PoTSpec(PoTSpecPayload):
    """Full spec object used by the pipeline (payload + derived spec_hash)."""
    spec_hash: str
