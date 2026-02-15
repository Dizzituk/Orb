# FILE: app/experience/models.py
"""
SQLAlchemy models for the Experience Database.

Tables:
- experience_patterns: Confidence-scored lessons learned from pipeline runs

Design: Every indexed field exists because a retrieval query filters on it.
Schema matches Section 3.2 of the Unified Memory System v3.0 spec.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text,
    Boolean, Index,
)
from app.db import Base


class ExperiencePattern(Base):
    """
    A structured, confidence-scored lesson learned from pipeline execution.

    Patterns are distilled from Build Journal entries after job completion.
    They represent reusable knowledge: error→fix pairs, architectural decisions,
    implementation strategies.

    Retrieval is two-stage:
    1. Index filter: job_type, stage, language, error_signature, category
    2. Semantic ranking: embedding similarity on description + root_cause + resolution
    """
    __tablename__ = "experience_patterns"

    id = Column(Integer, primary_key=True)

    # --- Classification (Stage 1 filter keys) ---
    category = Column(String(50), nullable=False, index=True)
    # Values: "error_fix", "architecture_decision", "implementation_strategy",
    #         "performance_pattern", "false_positive", "boot_resolution"

    stage = Column(String(50), nullable=False, index=True)
    # Pipeline stage that produced this: "critical_pipeline", "implementer",
    # "phase_checkout", "critique", "final_checkout", etc.

    job_type = Column(String(50), nullable=True, index=True)
    # "refactor", "app_build", "feature_add", "bug_fix"
    # NULL = universal (applies to all job types)

    language = Column(String(30), nullable=True, index=True)
    # "python", "typescript", NULL = language-agnostic

    error_signature = Column(String(64), nullable=True, index=True)
    # Deterministic hash for exact-match lookup on boot errors

    file_scope = Column(String(500), nullable=True)
    # File path pattern this applies to (e.g. "app/overwatcher/*.py")

    # --- Content ---
    description = Column(Text, nullable=False)
    # Human-readable: "When refactoring large files, always use edit-mode"

    root_cause = Column(Text, nullable=True)
    # Why the pattern exists: "Full-file LLM generation truncates files >300 lines"

    resolution = Column(Text, nullable=True)
    # What to do: "Use MODIFY with edit pairs instead of CREATE for large files"

    # --- Confidence (Section 3.3) ---
    confidence = Column(Float, nullable=False, default=0.5, index=True)
    # 0.0–1.0. Controls injection format:
    #   ≥0.8 = CONSTRAINT, 0.5–0.8 = HINT, 0.3–0.5 = BACKGROUND, <0.3 = SKIP

    times_validated = Column(Integer, default=0, nullable=False)
    # How many times this pattern was confirmed by subsequent jobs

    times_contradicted = Column(Integer, default=0, nullable=False)
    # How many times this pattern was contradicted

    times_injected = Column(Integer, default=0, nullable=False)
    # How many times this was injected into a pipeline stage prompt

    times_useful = Column(Integer, default=0, nullable=False)
    # How many times injection led to a successful outcome

    # --- Source tracking ---
    source_job_id = Column(String(100), nullable=True, index=True)
    # Job that first produced this pattern

    source_journal_index = Column(Integer, nullable=True)
    # Line index in the source journal (for traceability)

    # --- Embedding ---
    embedded = Column(Boolean, default=False, nullable=False, index=True)
    embedding_model = Column(String(100), nullable=True)
    embedded_at = Column(DateTime, nullable=True)

    # --- Timestamps ---
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    last_validated_at = Column(DateTime, nullable=True)
    last_injected_at = Column(DateTime, nullable=True)

    # --- Composite indexes for common retrieval queries ---
    __table_args__ = (
        Index("ix_exp_stage_jobtype", "stage", "job_type"),
        Index("ix_exp_stage_category", "stage", "category"),
        Index("ix_exp_confidence", "confidence"),
        Index("ix_exp_error_sig", "error_signature"),
        Index("ix_exp_stage_confidence", "stage", "confidence"),
    )

    def __repr__(self):
        return (
            f"<ExperiencePattern(id={self.id}, category={self.category}, "
            f"stage={self.stage}, confidence={self.confidence:.2f})>"
        )

    @property
    def injection_tier(self) -> str:
        """Determine injection format based on confidence."""
        if self.confidence >= 0.8:
            return "CONSTRAINT"
        if self.confidence >= 0.5:
            return "HINT"
        if self.confidence >= 0.3:
            return "BACKGROUND"
        return "SKIP"

    @property
    def is_injectable(self) -> bool:
        """Whether this pattern should be injected at all."""
        return self.confidence >= 0.3

    @property
    def embedding_text(self) -> str:
        """Text to embed for semantic search (description + root_cause + resolution)."""
        parts = [self.description or ""]
        if self.root_cause:
            parts.append(f"Root cause: {self.root_cause}")
        if self.resolution:
            parts.append(f"Resolution: {self.resolution}")
        return "\n".join(parts)
