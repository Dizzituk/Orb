# FILE: app/memory/conversation_models.py
# Purpose: SQLAlchemy ORM models for the Conversational Memory Layer.
# Called-by: app.db, app.memory.conversation_service, app.memory.knowledge_extractor, app.memory.session_lifecycle (+4 more)
# Depends-on: app.crypto, app.db
# Last-renovated: 2026-06-11
"""
SQLAlchemy ORM models for the Conversational Memory Layer.

Introduces two new concepts:
  1. ConversationSession — groups messages into logical sessions with
     lifecycle states (active → idle → archived).
  2. ConversationSummary — rolling summaries of each session, generated
     by a cheap/fast model and used for cross-model context injection.

Security Level 4: summary_json uses EncryptedJSON since summaries
contain distilled conversation content.

Part of CONV-MEMORY-001: Phase 1 (Session Model + Storage).
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text,
)
from sqlalchemy.orm import relationship
from app.db import Base
from app.crypto import EncryptedJSON


# =========================================================================
# ConversationSession
# =========================================================================

class ConversationSession(Base):
    """
    Groups a contiguous block of messages into a logical session.

    Lifecycle:
        active   — messages within the last 2 hours
        idle     — no messages for 2+ hours (Layer 2 persists)
        archived — no messages for 48+ hours (Layer 3 extraction runs)

    A project can have at most one 'active' session at a time.
    New messages arriving after an archived session create a new session.
    """
    __tablename__ = "conversation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Lifecycle state: active | idle | archived
    status = Column(String(20), default="active", nullable=False, index=True)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Stats
    message_count = Column(Integer, default=0, nullable=False)

    # Comma-separated list of "provider:model" pairs seen in this session.
    # e.g. "openai:gpt-4o-mini,anthropic:claude-sonnet-4-20250514"
    # Stored as plain text — not sensitive (just model identifiers).
    models_used = Column(Text, default="", nullable=False)

    # Relationships
    summaries = relationship(
        "ConversationSummary",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ConversationSummary.version.desc()",
    )


# =========================================================================
# ConversationSummary
# =========================================================================

class ConversationSummary(Base):
    """
    Rolling summary of a conversation session.

    Each summary is versioned. When new messages arrive, the existing
    summary is loaded, merged with the new messages, and a new version
    is written. This keeps the summary up-to-date without reprocessing
    the entire conversation.

    summary_json schema (Phase 2 will define formally):
    {
        "current_topic": str,
        "key_decisions": [str],
        "open_items": [str],
        "technical_context": str,
        "attachments_described": [str],
        "user_preferences_expressed": [str],
        "models_involved": [str],
        "message_range": {"from_id": int, "to_id": int},
    }
    """
    __tablename__ = "conversation_summaries"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer,
        ForeignKey("conversation_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Monotonically increasing version within a session.
    # v1 = first summary, v2 = updated after more messages, etc.
    version = Column(Integer, default=1, nullable=False)

    # The message ID through which this summary covers.
    # Phase 2 uses this to know which messages are "unsummarised".
    summarised_through_message_id = Column(Integer, nullable=True)

    # ENCRYPTED: Contains distilled conversation content.
    summary_json = Column(EncryptedJSON, nullable=True)

    # Approximate token count of the summary (for budget calculations).
    token_estimate = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    session = relationship(
        "ConversationSession",
        back_populates="summaries",
    )


# =========================================================================
# ConversationSpine
# =========================================================================

class ConversationSpine(Base):
    """
    Lightweight per-message index for within-conversation recall.

    One row per stored Message. Holds only routing/topic metadata — NO message
    text (recall fetches the real encrypted Message by id), so the spine never
    duplicates Security-Level-4 content. Tags/entities are plaintext (like the
    astra_hot_index) so retrieval is a cheap project-scoped LIKE.

    Written synchronously by app.memory.spine_service.record_message_spine at
    the create_message chokepoint; the tagger it uses is pure keyword matching
    (~1ms), so this adds no meaningful latency to the chat path.

    Part of CONV-MEMORY-002: within-conversation spine (Build Request B).
    """
    __tablename__ = "conversation_spine"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(
        Integer,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id = Column(
        Integer,
        ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    message_id = Column(
        Integer,
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Not encrypted (metadata): "user" | "assistant"
    role = Column(String(20), nullable=False)

    # Plaintext JSON arrays — topic tags + named entities from topic_tagger.
    # Mirrors astra_hot_index.tags so a project-scoped LIKE is the whole search.
    tags = Column(Text, nullable=True)
    entities = Column(Text, nullable=True)

    created_at = Column(
        DateTime, default=datetime.utcnow, nullable=False, index=True,
    )
