# FILE: app/specs/models.py
"""
SQLAlchemy ORM models for ASTRA Spec storage.

Specs table stores canonical JSON specs created by Weaver and validated by Spec Gate.

INVARIANT: content_json is canonical truth. content_markdown is derived.
INVARIANT: Every spec must have provenance (source_message_ids, commit_hash).
"""
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from app.db import Base

# Import encrypted column types for sensitive content
from app.crypto import EncryptedText, EncryptedJSON


def utcnow():
    """Timezone-aware UTC now."""
    return datetime.now(timezone.utc)


class Spec(Base):
    """
    Canonical spec storage.
    
    Stores specs created by Weaver and validated by Spec Gate.
    JSON content is canonical - markdown is for human review only.
    """
    __tablename__ = "specs"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True)
    
    # Unique identifiers
    spec_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    spec_hash = Column(String(64), nullable=False, index=True)  # SHA-256 of content
    
    # Foreign keys
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    arch_id = Column(Integer, nullable=True)  # Optional FK to architecture (if exists)
    
    # Status
    status = Column(String(30), nullable=False, default="draft", index=True)
    # Values: draft, pending_validation, validated, rejected, superseded
    
    # Version tracking
    spec_version = Column(String(10), nullable=False, default="1.0")
    
    # Content (ENCRYPTED)
    # Canonical JSON spec - this is the source of truth
    content_json = Column(EncryptedText, nullable=False)
    
    # Human-readable markdown (derived from JSON)
    content_markdown = Column(EncryptedText, nullable=True)
    
    # Title and summary for quick lookup (not encrypted for search)
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    
    # Provenance - REQUIRED for reproducibility
    source_message_ids = Column(JSON, nullable=False, default=list)  # List of message IDs
    summary_ids = Column(JSON, nullable=True)  # List of summary IDs if summarization used
    
    # Token and timing
    token_count = Column(Integer, nullable=False, default=0)
    timestamp_start = Column(DateTime(timezone=True), nullable=True)  # First source message
    timestamp_end = Column(DateTime(timezone=True), nullable=True)    # Last source message
    
    # Git provenance
    commit_hash = Column(String(40), nullable=True)  # Full SHA
    
    # Generator info
    generator_model = Column(String(100), nullable=False, default="weaver-v1")
    
    # Validation results (filled by Spec Gate)
    validation_result = Column(JSON, nullable=True)
    # Structure: { "valid": bool, "errors": [], "warnings": [], "validated_at": str, "validator_model": str }
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)
    
    # Relationships
    project = relationship("Project", backref="specs")


class SpecQuestion(Base):
    """
    Questions raised by Spec Gate during validation.
    
    When Spec Gate finds gaps or ambiguities, it creates questions.
    These go through the Mediator to the user.
    """
    __tablename__ = "spec_questions"
    
    id = Column(Integer, primary_key=True, index=True)
    spec_id = Column(Integer, ForeignKey("specs.id"), nullable=False, index=True)
    
    # Question details
    question_id = Column(String(36), unique=True, nullable=False)  # UUID
    category = Column(String(50), nullable=False)
    # Categories: missing_requirement, contradiction, ambiguity, safety_gap, other
    
    question_text = Column(Text, nullable=False)
    context = Column(Text, nullable=True)  # Why this question matters
    
    # Response tracking
    status = Column(String(20), nullable=False, default="pending")
    # Values: pending, answered, skipped, resolved
    
    answer_text = Column(Text, nullable=True)
    answered_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    
    # Relationships
    spec = relationship("Spec", backref="questions")


class SpecHistory(Base):
    """
    History of spec changes for audit trail.
    
    Tracks status changes and re-validations.
    """
    __tablename__ = "spec_history"
    
    id = Column(Integer, primary_key=True, index=True)
    spec_id = Column(Integer, ForeignKey("specs.id"), nullable=False, index=True)
    
    # Change details
    action = Column(String(50), nullable=False)
    # Actions: created, submitted, validated, rejected, superseded, updated
    
    old_status = Column(String(30), nullable=True)
    new_status = Column(String(30), nullable=True)
    
    details = Column(JSON, nullable=True)  # Additional context
    
    # Who/what triggered the change
    triggered_by = Column(String(100), nullable=True)  # user, weaver, spec_gate, etc.
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    
    # Relationships
    spec = relationship("Spec", backref="history")
