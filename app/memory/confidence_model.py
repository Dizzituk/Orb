# FILE: app/memory/confidence_model.py
"""
SQLAlchemy model for the confidence_scores table.

Stores phrase-to-intent confidence mappings for the translation layer.
This is distinct from preference confidence — it tracks how sure ASTRA
is that a given phrase maps to a given intent.

Used by: ConfidenceStore (app/memory/domains/confidence.py — Job 5)
"""

from datetime import datetime

from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Index
from app.db import Base


class ConfidenceScore(Base):
    """
    Phrase-to-intent confidence score.

    One row per (phrase_pattern, intent) pair. Confidence is calculated
    from confirmations, corrections, and recency.

    Formula: confidence = (confirmations / (confirmations + corrections + 2)) * recency_factor
    Recency: 1.0 if used within 30 days, then 0.95^(days_since_30) decay.
    """
    __tablename__ = "confidence_scores"

    id = Column(Integer, primary_key=True, index=True)

    # The phrase pattern (normalised user input fragment)
    phrase_pattern = Column(String(500), nullable=False)

    # The intent it maps to (e.g. 'refactor', 'scan', 'chat', 'pipeline')
    intent = Column(String(100), nullable=False)

    # Current confidence score (0.0 to 1.0)
    confidence = Column(Float, nullable=False, default=0.5, index=True)

    # Learning counters
    confirmations = Column(Integer, nullable=False, default=0)
    corrections = Column(Integer, nullable=False, default=0)

    # Last time this mapping was used (for decay calculation)
    last_used = Column(DateTime, nullable=True)

    # Optional context tags (JSON array of strings)
    # e.g. ["build_mode", "refactor_active"]
    context_tags = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_confidence_phrase_intent", "phrase_pattern", "intent", unique=True),
        Index("ix_confidence_last_used", "last_used"),
        Index("ix_confidence_score", "confidence"),
    )

    def __repr__(self):
        return (
            f"<ConfidenceScore(phrase='{self.phrase_pattern[:30]}', "
            f"intent='{self.intent}', conf={self.confidence:.2f})>"
        )
