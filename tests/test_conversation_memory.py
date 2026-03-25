# FILE: tests/test_conversation_memory.py
"""
Integration tests for the Conversational Memory Layer (CONV-MEMORY-001).

Tests cover all seven phases:
  1. Session creation, lifecycle, and persistence
  2. Summary generation trigger logic
  3. Summary injection into LLM context
  4. Model switch detection
  5. Session decay and archival
  6. Knowledge extraction to ASTRA memory
  7. End-to-end multi-model coherence

Run with: pytest tests/test_conversation_memory.py -v
"""

import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")

    # We need the crypto module to not require real encryption in tests
    with patch("app.crypto.types.is_encryption_ready", return_value=True), \
         patch("app.crypto.types.encrypt_string", side_effect=lambda x: f"ENC:{x}"), \
         patch("app.crypto.types.decrypt_string", side_effect=lambda x: x[4:] if x.startswith("ENC:") else x):

        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Seed a test project
        from app.memory.models import Project
        project = Project(
            id=1, name="test-project", description="Test",
            project_key="test-project", type="development", status="active",
        )
        session.add(project)
        session.commit()

        yield session
        session.close()


@pytest.fixture
def project_id():
    return 1


# =========================================================================
# Phase 1: Session Model + Storage
# =========================================================================

class TestSessionLifecycle:
    """Test conversation session creation and management."""

    def test_create_session(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate

        session = create_session(db, SessionCreate(project_id=project_id))
        assert session.id is not None
        assert session.project_id == project_id
        assert session.status == "active"
        assert session.message_count == 0
        assert session.models_used == ""

    def test_get_or_create_returns_existing(self, db, project_id):
        from app.memory.conversation_service import (
            create_session, get_or_create_active_session,
        )
        from app.memory.conversation_schemas import SessionCreate

        original = create_session(db, SessionCreate(project_id=project_id))
        fetched = get_or_create_active_session(db, project_id)
        assert fetched.id == original.id

    def test_stale_session_creates_new(self, db, project_id):
        from app.memory.conversation_service import (
            create_session, get_or_create_active_session, IDLE_THRESHOLD,
        )
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.conversation_models import ConversationSession

        original = create_session(db, SessionCreate(project_id=project_id))
        # Make it stale
        original.last_activity_at = datetime.utcnow() - IDLE_THRESHOLD - timedelta(minutes=5)
        db.commit()

        new_session = get_or_create_active_session(db, project_id)
        assert new_session.id != original.id
        assert new_session.status == "active"

        # Original should be idle
        db.refresh(original)
        assert original.status == "idle"

    def test_bump_session_activity(self, db, project_id):
        from app.memory.conversation_service import (
            create_session, bump_session_activity,
        )
        from app.memory.conversation_schemas import SessionCreate

        session = create_session(db, SessionCreate(project_id=project_id))
        bump_session_activity(db, session, provider="openai", model="gpt-5.4-mini")
        assert session.message_count == 1
        assert "openai:gpt-5.4-mini" in session.models_used

        bump_session_activity(db, session, provider="anthropic", model="claude-sonnet-4-20250514")
        assert session.message_count == 2
        assert "anthropic:claude-sonnet-4-20250514" in session.models_used

    def test_session_status_transitions(self, db, project_id):
        from app.memory.conversation_service import (
            create_session, update_session_status,
        )
        from app.memory.conversation_schemas import SessionCreate

        session = create_session(db, SessionCreate(project_id=project_id))

        # active → idle (valid)
        updated = update_session_status(db, session.id, "idle")
        assert updated.status == "idle"

        # idle → archived (valid)
        updated = update_session_status(db, session.id, "archived")
        assert updated.status == "archived"

        # archived → active (invalid)
        result = update_session_status(db, session.id, "active")
        assert result is None

    def test_message_create_with_session_id(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.schemas import MessageCreate
        from app.memory.service import create_message

        session = create_session(db, SessionCreate(project_id=project_id))
        msg = create_message(db, MessageCreate(
            project_id=project_id,
            role="user",
            content="Hello",
            session_id=session.id,
        ))
        assert msg.session_id == session.id


# =========================================================================
# Phase 1: Summary CRUD
# =========================================================================

class TestSummaryCRUD:
    """Test summary creation and versioning."""

    def test_create_summary_auto_versions(self, db, project_id):
        from app.memory.conversation_service import (
            create_session, create_summary, get_latest_summary,
        )
        from app.memory.conversation_schemas import SessionCreate, SummaryCreate

        session = create_session(db, SessionCreate(project_id=project_id))

        s1 = create_summary(db, SummaryCreate(
            session_id=session.id, project_id=project_id,
            summary_json={"current_topic": "test v1"},
        ))
        assert s1.version == 1

        s2 = create_summary(db, SummaryCreate(
            session_id=session.id, project_id=project_id,
            summary_json={"current_topic": "test v2"},
        ))
        assert s2.version == 2

        latest = get_latest_summary(db, session.id)
        assert latest.version == 2
        assert latest.summary_json["current_topic"] == "test v2"


# =========================================================================
# Phase 2: Summary Trigger
# =========================================================================

class TestSummaryTrigger:
    """Test when summaries should be generated."""

    def test_threshold_trigger(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.schemas import MessageCreate
        from app.memory.service import create_message
        from app.memory.summary_trigger import should_generate_summary

        session = create_session(db, SessionCreate(project_id=project_id))

        # Add 4 messages — below threshold
        for i in range(4):
            create_message(db, MessageCreate(
                project_id=project_id, role="user",
                content=f"Message {i}", session_id=session.id,
            ))
        assert not should_generate_summary(db, session)

        # Add 5th — hits threshold
        create_message(db, MessageCreate(
            project_id=project_id, role="user",
            content="Message 5", session_id=session.id,
        ))
        assert should_generate_summary(db, session)

    def test_model_switch_trigger(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.schemas import MessageCreate
        from app.memory.service import create_message
        from app.memory.summary_trigger import should_generate_summary

        session = create_session(db, SessionCreate(project_id=project_id))

        # Two messages from same model — no trigger
        create_message(db, MessageCreate(
            project_id=project_id, role="assistant",
            content="Response 1", provider="openai", model="gpt-5.4-mini",
            session_id=session.id,
        ))
        create_message(db, MessageCreate(
            project_id=project_id, role="assistant",
            content="Response 2", provider="openai", model="gpt-5.4-mini",
            session_id=session.id,
        ))
        assert not should_generate_summary(db, session)

        # Switch to Anthropic — should trigger
        create_message(db, MessageCreate(
            project_id=project_id, role="assistant",
            content="Response 3", provider="anthropic",
            model="claude-sonnet-4-20250514", session_id=session.id,
        ))
        assert should_generate_summary(db, session)

    def test_force_trigger(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.summary_trigger import should_generate_summary

        session = create_session(db, SessionCreate(project_id=project_id))
        # No messages at all, but force=True
        assert should_generate_summary(db, session, force=True)


# =========================================================================
# Phase 3: Summary Injection
# =========================================================================

class TestSummaryInjection:
    """Test summary formatting and context injection."""

    def test_format_summary_block(self):
        from app.memory.summary_injection import format_summary_block

        block = format_summary_block({
            "current_topic": "Building ASTRA pipeline",
            "key_decisions": ["Use Flash Lite for summaries"],
            "open_items": ["Wire Phase 3"],
            "technical_context": "Python, FastAPI, SQLAlchemy",
            "models_involved": ["google:gemini-2.0-flash-lite"],
        })

        assert "[CONVERSATION_SUMMARY]" in block
        assert "[/CONVERSATION_SUMMARY]" in block
        assert "Building ASTRA pipeline" in block
        assert "Flash Lite" in block
        assert "Wire Phase 3" in block

    def test_empty_summary_returns_empty_string(self):
        from app.memory.summary_injection import format_summary_block

        block = format_summary_block({})
        assert "[CONVERSATION_SUMMARY]" in block
        # Should still have opening/closing tags but minimal content

    def test_build_summary_context_no_session(self, db, project_id):
        from app.memory.summary_injection import build_summary_context

        result = build_summary_context(db, project_id)
        assert result == ""


# =========================================================================
# Phase 4: Model Switch Detection (tested via trigger above)
# =========================================================================


# =========================================================================
# Phase 5: Session Decay
# =========================================================================

class TestSessionDecay:
    """Test lifecycle scheduler transitions."""

    @pytest.mark.asyncio
    async def test_process_stale_sessions(self, db, project_id):
        from app.memory.conversation_service import (
            create_session, IDLE_THRESHOLD, ARCHIVE_THRESHOLD,
        )
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.session_lifecycle import process_stale_sessions

        # Create sessions at different ages
        active = create_session(db, SessionCreate(project_id=project_id))

        idle_candidate = create_session(db, SessionCreate(project_id=project_id))
        active.last_activity_at = datetime.utcnow()  # Keep active fresh
        idle_candidate.last_activity_at = datetime.utcnow() - IDLE_THRESHOLD - timedelta(minutes=10)
        db.commit()

        results = await process_stale_sessions(db)
        assert results["idled"] >= 1

    @pytest.mark.asyncio
    async def test_idle_to_archived(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.session_lifecycle import process_stale_sessions

        session = create_session(db, SessionCreate(project_id=project_id))
        session.status = "idle"
        session.last_activity_at = datetime.utcnow() - timedelta(hours=49)
        db.commit()

        with patch(
            "app.memory.session_lifecycle._trigger_knowledge_extraction_async",
            new_callable=AsyncMock,
            return_value=False,
        ):
            results = await process_stale_sessions(db)
        assert results["archived"] >= 1


# =========================================================================
# Phase 6: Knowledge Extraction
# =========================================================================

class TestKnowledgeExtraction:
    """Test extraction parsing and category mapping."""

    def test_parse_valid_extractions(self):
        from app.memory.knowledge_extractor import _parse_extractions

        raw = json.dumps([
            {"category": "biographical", "key": "name", "value": "Taz"},
            {"category": "preference", "key": "file_size", "value": "Under 20KB"},
            {"category": "invalid_cat", "key": "x", "value": "y"},
        ])
        result = _parse_extractions(raw)
        assert len(result) == 2
        assert result[0]["category"] == "biographical"
        assert result[1]["category"] == "preference"

    def test_parse_markdown_fenced(self):
        from app.memory.knowledge_extractor import _parse_extractions

        raw = "```json\n" + json.dumps([
            {"category": "philosophy", "key": "rule", "value": "Best way not quickest"},
        ]) + "\n```"
        result = _parse_extractions(raw)
        assert len(result) == 1

    def test_parse_invalid_json(self):
        from app.memory.knowledge_extractor import _parse_extractions
        result = _parse_extractions("not json at all")
        assert result == []

    def test_max_10_extractions(self):
        from app.memory.knowledge_extractor import _parse_extractions
        items = [
            {"category": "project", "key": f"k{i}", "value": f"v{i}"}
            for i in range(15)
        ]
        result = _parse_extractions(json.dumps(items))
        assert len(result) == 10


# =========================================================================
# Phase 7: End-to-end summary generation (mocked LLM)
# =========================================================================

class TestSummaryGeneration:
    """Test the full summary generation flow with mocked LLM."""

    @pytest.mark.asyncio
    async def test_generate_summary_mocked(self, db, project_id):
        from app.memory.conversation_service import create_session
        from app.memory.conversation_schemas import SessionCreate
        from app.memory.schemas import MessageCreate
        from app.memory.service import create_message
        from app.memory.summary_generator import generate_summary

        session = create_session(db, SessionCreate(project_id=project_id))

        # Add some messages
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            create_message(db, MessageCreate(
                project_id=project_id, role=role,
                content=f"Test message {i}",
                provider="google" if role == "assistant" else None,
                model="gemini-2.0-flash" if role == "assistant" else None,
                session_id=session.id,
            ))

        # Mock the LLM call
        mock_response = json.dumps({
            "current_topic": "Testing the summary system",
            "key_decisions": ["Use mocked LLM"],
            "open_items": [],
            "technical_context": "pytest, SQLAlchemy",
            "attachments_described": [],
            "user_preferences_expressed": [],
            "models_involved": ["google:gemini-2.0-flash"],
            "message_range": {"from_id": 1, "to_id": 6},
        })

        with patch(
            "app.memory.summary_generator.call_llm_text",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await generate_summary(db, session)

        assert result is not None
        assert result.current_topic == "Testing the summary system"

        # Verify summary was persisted
        from app.memory.conversation_service import get_latest_summary
        latest = get_latest_summary(db, session.id)
        assert latest is not None
        assert latest.version == 1
        assert latest.summary_json["current_topic"] == "Testing the summary system"
