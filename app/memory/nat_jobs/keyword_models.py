# FILE: app/memory/nat_jobs/keyword_models.py
# Purpose: message_keywords table — a cheap, non-decaying per-message topic index
#          (Job 1, keyword recall trail). One row per message, an INDEX not content.
# Called-by: app.db.init_db (registration), app.memory.nat_jobs.keyword_trail (CRUD)
# Depends-on: app.db.Base
# Last-renovated: 2026-06-19
"""
The keyword trail fixes a structural recall bug: the rolling summary
(app/memory/summary_generator.py) is lossy by design (<800 words, lists capped
at 10, oldest dropped), so the START of a long conversation gets compressed out
and "summarise our chat" can't recover it.

This table stores 3-8 short topic keywords per message — it does NOT compete for
a word budget, so it never forgets the conversation's start. Recall reads the
trail (a cheap DB read), not the summary.

Keywords are short topic LABELS derived from the message, not the content
itself; stored plaintext (like the conversation spine's tags) so coverage reads
are fast. The encrypted message content stays in Message.content.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text

from app.db import Base


class MessageKeywords(Base):
    """One row per message: its extracted topic keywords (JSON array as text)."""

    __tablename__ = "message_keywords"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(
        Integer, ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    session_id = Column(
        Integer, ForeignKey("conversation_sessions.id", ondelete="SET NULL"),
        nullable=True, index=True,
    )
    message_id = Column(
        Integer, ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False, unique=True, index=True,
    )
    # JSON array of short strings, e.g. ["calorie tracking","pasta","weight loss"]
    keywords = Column(Text, nullable=True)
    # "nat" if Nat extracted them, "fallback" if the deterministic tagger did.
    source = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
