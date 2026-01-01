# FILE: app/astra_memory/preference_models.py
"""
ASTRA Memory System - Preference & Evidence Models

This module defines:
1. PreferenceRecord - Stores user preferences with confidence tracking
2. PreferenceEvidence - Append-only ledger of evidence events
3. MemoryRecordMixin - Confidence fields for any memory record
4. HotIndex / ColdStorage - Hot/cold layer structures
5. SummaryPyramid - Multi-level summaries for large artifacts
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, String, Text, Float, Boolean,
    ForeignKey, DateTime, JSON, Index, Enum as SQLEnum,
    event,
)
from sqlalchemy.orm import relationship

from app.db import Base


# =============================================================================
# ENUMS
# =============================================================================

class ConfidenceType(str, Enum):
    """
    Two distinct confidence concepts - DO NOT MIX.
    
    RECORD: How sure we are the record is correct (truth/accuracy).
            Applies to: extracted facts, repo summaries, entity resolution, atlas nodes.
            Drivers: source reliability, verification, contradictions.
    
    PREFERENCE: How sure we are the user prefers X over Y.
                Applies to: defaults, workflows, tool choice, safety rules, routing.
                Drivers: explicit instruction, approvals, repeated behavior, contradictions.
    """
    RECORD = "record"
    PREFERENCE = "preference"


class PreferenceStrength(str, Enum):
    """
    Preference enforcement levels.
    
    SOFT: Suggestion only, can be overridden easily
    DEFAULT: Applied as default but can be changed
    HARD_RULE: Immutable via implicit behavior, requires explicit override
    """
    SOFT = "soft"
    DEFAULT = "default"
    HARD_RULE = "hard_rule"


class RecordStatus(str, Enum):
    """Status of a memory record."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DISPUTED = "disputed"
    EXPIRED = "expired"


class SignalType(str, Enum):
    """Types of evidence signals."""
    EXPLICIT = "explicit"      # Direct user instruction
    IMPLICIT = "implicit"      # Observed behavior
    APPROVAL = "approval"      # User approved a suggestion
    ONE_OFF = "one_off"        # Single instance choice
    CONTRADICTION = "contradiction"  # Conflicts with existing preference


class RetrievalCost(str, Enum):
    """Estimated cost to retrieve a record."""
    TINY = "tiny"      # Hot layer only
    MEDIUM = "medium"  # Some cold expansion
    LARGE = "large"    # Full artifact load


class IntentDepth(str, Enum):
    """
    Retrieval intent depth levels.
    
    D0: Chat - no memory or tiny
    D1: Brief - small, fast; short summaries + top facts
    D2: Normal - targeted; relevant decisions + pointers + atlas snippets
    D3: Deep - heavy; full spec/arch artifacts, multi-doc stitching
    D4: Forensic - slowest; evidence bundles, event ledger, diff history
    """
    D0 = "D0"  # Chat
    D1 = "D1"  # Brief
    D2 = "D2"  # Normal
    D3 = "D3"  # Deep
    D4 = "D4"  # Forensic


# =============================================================================
# PREFERENCE RECORD
# =============================================================================

class PreferenceRecord(Base):
    """
    User preference with confidence tracking.
    
    Stores both the preference value and all metadata needed for
    confidence scoring and enforcement decisions.
    
    Invariant: Preferences must never be treated as factual truth.
    """
    __tablename__ = "astra_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Preference identity
    preference_key = Column(String(128), unique=True, nullable=False, index=True)
    # e.g., "no_destructive_repo_commands", "default_test_framework", "tone_formal"
    
    # Preference value (JSON for flexibility)
    preference_value = Column(JSON, nullable=False)
    # e.g., true, "pytest", {"level": "formal", "emoji": false}
    
    # Strength/enforcement level
    strength = Column(
        SQLEnum(PreferenceStrength),
        default=PreferenceStrength.DEFAULT,
        nullable=False,
    )
    
    # Confidence scoring fields
    confidence = Column(Float, default=0.0, nullable=False, index=True)
    confidence_type = Column(
        SQLEnum(ConfidenceType),
        default=ConfidenceType.PREFERENCE,
        nullable=False,
    )
    evidence_count = Column(Integer, default=0, nullable=False)
    evidence_weight = Column(Float, default=0.0, nullable=False)  # Sum of weights
    source_reliability = Column(Float, default=1.0, nullable=False)
    contradiction_count = Column(Integer, default=0, nullable=False)
    
    # Status tracking
    status = Column(
        SQLEnum(RecordStatus),
        default=RecordStatus.ACTIVE,
        nullable=False,
        index=True,
    )
    
    # Temporal tracking
    last_reinforced_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Namespace for isolation
    namespace = Column(String(64), default="user_personal", nullable=False, index=True)
    
    # Component scope
    applies_to = Column(String(128), nullable=True)
    # e.g., "overwatcher", "spec_gate", "all", "llm_router"
    
    # Source tracking
    source = Column(String(64), default="user_declared", nullable=True)
    # "user_declared", "learned", "system_default", "imported"
    
    # Relationships
    evidence = relationship(
        "PreferenceEvidence",
        back_populates="preference",
        cascade="all, delete-orphan",
        order_by="PreferenceEvidence.timestamp.desc()",
    )
    
    __table_args__ = (
        Index("ix_pref_namespace_status", "namespace", "status"),
        Index("ix_pref_applies_confidence", "applies_to", "confidence"),
    )


# =============================================================================
# PREFERENCE EVIDENCE LEDGER (Append-Only)
# =============================================================================

class PreferenceEvidence(Base):
    """
    Append-only evidence ledger for preference learning.
    
    Every signal that affects preference confidence is recorded here.
    This provides full auditability and supports confidence recalculation.
    
    CRITICAL: This table is append-only. Never update or delete rows.
    """
    __tablename__ = "astra_preference_evidence"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Link to preference
    preference_key = Column(
        String(128),
        ForeignKey("astra_preferences.preference_key"),
        nullable=False,
        index=True,
    )
    
    # Signal classification
    signal_type = Column(
        SQLEnum(SignalType),
        nullable=False,
        index=True,
    )
    
    # Weight of this evidence (positive or negative)
    weight = Column(Float, nullable=False)
    
    # Timestamp (for decay calculation)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Context pointer (for audit trail)
    context_pointer = Column(String(512), nullable=True)
    # Format: "message:{id}", "job:{id}", "commit:{hash}", "artifact:{path}"
    
    # Optional details
    details = Column(JSON, nullable=True)
    # e.g., {"original_value": "x", "new_value": "y", "trigger": "user_said_always"}
    
    # Relationships
    preference = relationship("PreferenceRecord", back_populates="evidence")
    
    __table_args__ = (
        Index("ix_evidence_key_ts", "preference_key", "timestamp"),
    )


# Enforce append-only behavior
@event.listens_for(PreferenceEvidence, "before_update")
def prevent_evidence_update(mapper, connection, target):
    """Prevent updates to evidence records - append only."""
    raise ValueError("PreferenceEvidence is append-only. Cannot update existing records.")


@event.listens_for(PreferenceEvidence, "before_delete")
def prevent_evidence_delete(mapper, connection, target):
    """Prevent deletion of evidence records - append only."""
    raise ValueError("PreferenceEvidence is append-only. Cannot delete records.")


# =============================================================================
# HOT INDEX (Fast retrieval layer)
# =============================================================================

class HotIndex(Base):
    """
    Hot layer index for fast D0/D1 retrieval.
    
    Stores lightweight summaries and pointers to cold storage.
    D0/D1 queries MUST NOT fetch cold artifacts.
    """
    __tablename__ = "astra_hot_index"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Record identity
    record_type = Column(String(32), nullable=False, index=True)
    # "preference", "job", "spec", "arch", "fact", "pattern"
    
    record_id = Column(String(64), nullable=False, index=True)
    # The ID in the source table (preference_key, job_id, spec_id, etc.)
    
    # Hot layer content (for D0-D1)
    title = Column(String(256), nullable=False)
    one_liner = Column(String(512), nullable=True)  # L0 summary
    bullets_5 = Column(JSON, nullable=True)  # L1 summary: list of 5 strings
    
    # Indexing
    tags = Column(JSON, nullable=True)  # List of tag strings
    entities = Column(JSON, nullable=True)  # List of entity references
    
    # Pointers to cold storage
    cold_storage_path = Column(String(512), nullable=True)
    cold_storage_hash = Column(String(64), nullable=True)
    
    # Retrieval metadata
    retrieval_priority = Column(Float, default=0.5, nullable=False, index=True)
    retrieval_cost = Column(
        SQLEnum(RetrievalCost),
        default=RetrievalCost.TINY,
        nullable=False,
    )
    
    # Freshness
    content_hash = Column(String(64), nullable=True)
    version = Column(Integer, default=1, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_hot_type_id", "record_type", "record_id", unique=True),
        Index("ix_hot_priority", "retrieval_priority"),
    )


# =============================================================================
# SUMMARY PYRAMID (Multi-level summaries)
# =============================================================================

class SummaryPyramid(Base):
    """
    Multi-level summaries for large artifacts.
    
    Levels:
    - L0: 1 sentence
    - L1: 5 bullets
    - L2: 1-2 paragraphs
    - L3: full text (pointer to cold storage)
    
    Selection rules:
    - D1 → L0/L1
    - D2 → L1/L2 + relevant sections
    - D3 → L2/L3
    """
    __tablename__ = "astra_summary_pyramids"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Artifact identity
    artifact_type = Column(String(32), nullable=False, index=True)
    # "spec", "arch", "doc", "transcript", "report"
    
    artifact_id = Column(String(64), nullable=False, index=True)
    
    # Summary levels
    l0_sentence = Column(String(512), nullable=True)  # 1 sentence
    l1_bullets = Column(JSON, nullable=True)  # List of ~5 bullet strings
    l2_paragraphs = Column(Text, nullable=True)  # 1-2 paragraphs
    
    # L3 is always in cold storage
    l3_cold_path = Column(String(512), nullable=True)
    l3_cold_hash = Column(String(64), nullable=True)
    l3_token_estimate = Column(Integer, nullable=True)
    
    # Sectioned content for partial retrieval
    sections = Column(JSON, nullable=True)
    # [{"title": "...", "summary": "...", "cold_offset": 0, "cold_length": 1000}, ...]
    
    # Freshness
    source_hash = Column(String(64), nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index("ix_pyramid_artifact", "artifact_type", "artifact_id", unique=True),
    )


# =============================================================================
# MEMORY RECORD CONFIDENCE EXTENSION
# =============================================================================

class MemoryRecordConfidence(Base):
    """
    Confidence metadata for any memory record.
    
    This extends existing memory records (from app.memory) with
    confidence fields without modifying those tables directly.
    
    Links to records via source_type + source_id.
    """
    __tablename__ = "astra_memory_confidence"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Link to source record
    source_type = Column(String(32), nullable=False, index=True)
    # "note", "message", "file", "document", "job", "chunk"
    
    source_id = Column(Integer, nullable=False, index=True)
    
    # Confidence fields (spec section 3.1)
    confidence = Column(Float, default=0.5, nullable=False, index=True)
    confidence_type = Column(
        SQLEnum(ConfidenceType),
        default=ConfidenceType.RECORD,
        nullable=False,
    )
    evidence_count = Column(Integer, default=1, nullable=False)
    evidence_weight = Column(Float, default=1.0, nullable=False)
    source_reliability = Column(Float, default=1.0, nullable=False)
    contradiction_count = Column(Integer, default=0, nullable=False)
    
    # Status
    status = Column(
        SQLEnum(RecordStatus),
        default=RecordStatus.ACTIVE,
        nullable=False,
        index=True,
    )
    
    # Temporal
    last_reinforced_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Namespace isolation
    namespace = Column(String(64), default="general", nullable=False, index=True)
    
    __table_args__ = (
        Index("ix_memconf_source", "source_type", "source_id", unique=True),
        Index("ix_memconf_ns_conf", "namespace", "confidence"),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ConfidenceType",
    "PreferenceStrength",
    "RecordStatus",
    "SignalType",
    "RetrievalCost",
    "IntentDepth",
    # Models
    "PreferenceRecord",
    "PreferenceEvidence",
    "HotIndex",
    "SummaryPyramid",
    "MemoryRecordConfidence",
]
