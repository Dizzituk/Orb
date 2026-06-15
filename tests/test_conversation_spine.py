# FILE: tests/test_conversation_spine.py
# Purpose: Unit tests for the within-conversation memory spine (CONV-MEMORY-002).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.db, app.memory.conversation_models, app.memory.models, app.memory.service, app.memory.spine_service
# Last-renovated: 2026-06-13
"""
Unit tests for the within-conversation memory spine (Build Request B).

Covers:
  - record_message_spine writes a row carrying topic tags/entities
  - the write is idempotent (no duplicate row per message)
  - search_spine matches by entity and excludes the live window
  - build_conversation_recall pulls scrolled-out messages back as a block
  - recall stays silent for topic-free questions
  - the create_message chokepoint indexes the spine automatically

Run with: pytest tests/test_conversation_spine.py -v
"""

import pytest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
# Import models so Base.metadata registers them before create_all.
from app.memory.models import Project, Message  # noqa: F401
from app.memory.conversation_models import ConversationSpine  # noqa: F401


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def db():
    """In-memory SQLite DB with crypto stubbed (same pattern as the
    conversation-memory suite)."""
    engine = create_engine("sqlite:///:memory:")

    with patch("app.crypto.types.is_encryption_ready", return_value=True), \
         patch("app.crypto.types.encrypt_string", side_effect=lambda x: f"ENC:{x}"), \
         patch("app.crypto.types.decrypt_string",
               side_effect=lambda x: x[4:] if x.startswith("ENC:") else x):

        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        session.add(Project(
            id=1, name="test-project", description="Test",
            project_key="test-project", type="development", status="active",
        ))
        session.commit()

        yield session
        session.close()


def _make_message(db, role, content, project_id=1):
    """Persist a Message directly (bypassing the embedding path)."""
    m = Message(project_id=project_id, role=role, content=content, provider="local")
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


# =========================================================================
# Write path
# =========================================================================

class TestRecordSpine:
    def test_record_creates_row_with_topic(self, db):
        from app.memory.spine_service import record_message_spine
        from app.memory.conversation_models import ConversationSpine

        m = _make_message(db, "user", "Tell me about Anthropic and my TAO position")
        record_message_spine(db, m)

        row = (
            db.query(ConversationSpine)
            .filter(ConversationSpine.message_id == m.id)
            .one()
        )
        assert row.project_id == 1
        assert row.role == "user"
        # Entity vocab catches Anthropic + TAO; investments tag catches TAO.
        assert "Anthropic" in row.entities
        assert "TAO" in row.entities
        assert "investments" in row.tags

    def test_record_is_idempotent(self, db):
        from app.memory.spine_service import record_message_spine
        from app.memory.conversation_models import ConversationSpine

        m = _make_message(db, "user", "A note about Bittensor subnets")
        record_message_spine(db, m)
        record_message_spine(db, m)

        count = (
            db.query(ConversationSpine)
            .filter(ConversationSpine.message_id == m.id)
            .count()
        )
        assert count == 1


# =========================================================================
# Search
# =========================================================================

class TestSearchSpine:
    def test_search_by_entity_excludes_window(self, db):
        from app.memory.spine_service import record_message_spine, search_spine

        older = _make_message(db, "user", "Earlier: my view on Anthropic vs OpenAI")
        newer = _make_message(db, "user", "Now I'm asking again about Anthropic")
        record_message_spine(db, older)
        record_message_spine(db, newer)

        # newer is "in the window" — exclude it; only the older row should match.
        rows = search_spine(
            db, project_id=1,
            query_tags=[], query_entities=["Anthropic"],
            exclude_message_ids=[newer.id],
        )
        ids = [r.message_id for r in rows]
        assert older.id in ids
        assert newer.id not in ids

    def test_search_empty_without_terms(self, db):
        from app.memory.spine_service import search_spine
        assert search_spine(db, 1, [], []) == []


# =========================================================================
# Recall block
# =========================================================================

class TestRecall:
    def test_recall_pulls_older_message(self, db):
        from app.memory.spine_service import (
            record_message_spine, build_conversation_recall,
        )

        older = _make_message(
            db, "user",
            "I was telling you that Anthropic's Claude models are my favourite",
        )
        record_message_spine(db, older)

        # A fresh window that does NOT contain the older message id.
        block = build_conversation_recall(
            db, project_id=1,
            user_message="what was I telling you about Anthropic?",
            window_message_ids=[999],
        )
        assert "[CONVERSATION_RECALL]" in block
        assert "Claude models are my favourite" in block

    def test_recall_silent_without_topic(self, db):
        from app.memory.spine_service import build_conversation_recall
        block = build_conversation_recall(
            db, 1, "hello, how are you today?", window_message_ids=[],
        )
        assert block == ""

    def test_recall_silent_when_only_match_is_in_window(self, db):
        from app.memory.spine_service import (
            record_message_spine, build_conversation_recall,
        )
        m = _make_message(db, "user", "A point about Anthropic")
        record_message_spine(db, m)

        # The only matching message is already in the window → nothing to recall.
        block = build_conversation_recall(
            db, 1, "more on Anthropic please", window_message_ids=[m.id],
        )
        assert block == ""


# =========================================================================
# create_message chokepoint
# =========================================================================

class TestChokepoint:
    def test_create_message_indexes_spine(self, db):
        from app.memory import service as memory_service, schemas
        from app.memory.conversation_models import ConversationSpine

        with patch("app.embeddings.auto_index_enabled", return_value=False):
            msg = memory_service.create_message(db, schemas.MessageCreate(
                project_id=1, role="user",
                content="Logging a thought about Leigh Day and worker status",
                provider="local",
            ))

        row = (
            db.query(ConversationSpine)
            .filter(ConversationSpine.message_id == msg.id)
            .one()
        )
        assert row.role == "user"
        assert "Leigh Day" in row.entities
        assert "legal" in row.tags
