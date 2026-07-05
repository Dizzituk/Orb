# FILE: tests/test_summary_tiering.py
# Purpose: Lane B (2026-07-01) — summariser model resolves from .env ONLY
#          (no literals), summary caps are env-tunable, and the tiered
#          conversation_arc compresses instead of shedding (opener pinned).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.memory.model_env, app.memory.summary_generator, app.memory.summary_injection
"""
Lane B tests — Task 3 (tiered summary) + Task 4 (summary model env fix).

Run with: pytest tests/test_summary_tiering.py -v
"""

import json

import pytest
from unittest.mock import AsyncMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.memory.models import Project, Message  # noqa: F401
from app.memory.conversation_models import (  # noqa: F401
    ConversationSession,
    ConversationSummary,
    ConversationSpine,
)

_ALL_MODEL_VARS = (
    "SUMMARIZER_PROVIDER", "SUMMARIZER_MODEL",
    "SUMMARY_PROVIDER", "SUMMARY_MODEL",
    "DEFAULT_PROVIDER", "DEFAULT_MODEL",
)


def _clear_model_env(monkeypatch):
    for name in _ALL_MODEL_VARS:
        monkeypatch.delenv(name, raising=False)


# =========================================================================
# Fixtures (same pattern as the conversation-memory suite)
# =========================================================================

@pytest.fixture
def db():
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


def _seed_session_with_messages(db, n=4, session=None):
    """Active session + n messages attached to it. Returns the session."""
    if session is None:
        session = ConversationSession(project_id=1, status="active")
        db.add(session)
        db.commit()
        db.refresh(session)
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        db.add(Message(
            project_id=1, role=role, content=f"Test message {i}",
            provider="local", session_id=session.id,
        ))
    db.commit()
    return session


# =========================================================================
# Task 4 — model resolution is .env-only
# =========================================================================

class TestModelEnv:
    def test_summary_vars_resolve(self, monkeypatch):
        from app.memory.model_env import summary_model_from_env
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("SUMMARY_PROVIDER", "prov-a")
        monkeypatch.setenv("SUMMARY_MODEL", "model-a")
        assert summary_model_from_env() == ("prov-a", "model-a")

    def test_summarizer_alias_takes_precedence(self, monkeypatch):
        """If Lane A renames the vars to the SUMMARIZER_* convention, they win
        without a code change here."""
        from app.memory.model_env import summary_model_from_env
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("SUMMARY_PROVIDER", "prov-a")
        monkeypatch.setenv("SUMMARY_MODEL", "model-a")
        monkeypatch.setenv("SUMMARIZER_PROVIDER", "prov-b")
        monkeypatch.setenv("SUMMARIZER_MODEL", "model-b")
        assert summary_model_from_env() == ("prov-b", "model-b")

    def test_default_fallback(self, monkeypatch):
        from app.memory.model_env import summary_model_from_env
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("DEFAULT_PROVIDER", "prov-d")
        monkeypatch.setenv("DEFAULT_MODEL", "model-d")
        assert summary_model_from_env() == ("prov-d", "model-d")

    def test_unresolved_returns_empty_not_literal(self, monkeypatch):
        """Principle 1: with nothing configured there is NO baked-in model."""
        from app.memory.model_env import summary_model_from_env
        _clear_model_env(monkeypatch)
        assert summary_model_from_env() == ("", "")

    def test_generator_delegates(self, monkeypatch):
        from app.memory.summary_generator import _get_summary_model
        _clear_model_env(monkeypatch)
        monkeypatch.setenv("SUMMARY_PROVIDER", "prov-a")
        monkeypatch.setenv("SUMMARY_MODEL", "model-a")
        assert _get_summary_model() == ("prov-a", "model-a")


# =========================================================================
# Task 3 — env-tunable caps + arc rules in the prompt
# =========================================================================

class TestPromptCaps:
    def test_env_caps_flow_into_prompt(self, monkeypatch):
        from app.memory.summary_generator import _build_system_prompt
        monkeypatch.setenv("ASTRA_SUMMARY_MAX_LIST_ITEMS", "7")
        monkeypatch.setenv("ASTRA_SUMMARY_MAX_WORDS", "500")
        monkeypatch.setenv("ASTRA_SUMMARY_ARC_MAX_ITEMS", "12")
        prompt = _build_system_prompt()
        assert "under 500 words" in prompt
        assert "at most 7 items" in prompt
        assert "exceeds 12 entries" in prompt

    def test_arc_rules_present(self, monkeypatch):
        from app.memory.summary_generator import _build_system_prompt
        prompt = _build_system_prompt()
        assert '"conversation_arc"' in prompt
        assert "NEVER drop an arc entry" in prompt
        assert "OPENED" in prompt

    def test_schema_accepts_arc(self):
        from app.memory.conversation_schemas import SummaryContent
        s = SummaryContent(conversation_arc=["[09:02] opener — work log"])
        assert s.conversation_arc == ["[09:02] opener — work log"]
        assert SummaryContent().conversation_arc == []


class TestArcRender:
    def test_format_renders_arc(self):
        from app.memory.summary_injection import format_summary_block
        block = format_summary_block({
            "current_topic": "evening wrap-up",
            "conversation_arc": [
                "[09:02] good-morning work log",
                "[11:00] DeepMind model research",
                "[16:00] memory architecture",
            ],
        })
        assert "Conversation arc (chronological, whole session):" in block
        assert "good-morning work log" in block
        # Chronological: morning before afternoon in the rendered line.
        assert block.index("09:02") < block.index("16:00")

    def test_no_arc_no_line(self):
        from app.memory.summary_injection import format_summary_block
        block = format_summary_block({"current_topic": "x"})
        assert "Conversation arc" not in block
        assert "[CONVERSATION_SUMMARY]" in block


# =========================================================================
# Arc guard + unconfigured skip (generate_summary, mocked LLM)
# =========================================================================

def _mock_summary_json(**overrides):
    data = {
        "current_topic": "Testing the tiered summary",
        "key_decisions": [],
        "open_items": [],
        "technical_context": "",
        "attachments_described": [],
        "user_preferences_expressed": [],
        "models_involved": [],
        "message_range": {"from_id": 1, "to_id": 4},
    }
    data.update(overrides)
    return json.dumps(data)


class TestGenerateSummaryArc:
    @pytest.mark.asyncio
    async def test_arc_restored_when_model_drops_it(self, db, monkeypatch):
        """A model reply that loses the arc wholesale gets the previous arc
        back — the opener can never vanish from the summary channel."""
        from app.memory.conversation_service import create_summary
        from app.memory.conversation_schemas import SummaryCreate
        from app.memory.summary_generator import generate_summary

        monkeypatch.setenv("SUMMARY_PROVIDER", "prov-a")
        monkeypatch.setenv("SUMMARY_MODEL", "model-a")

        session = _seed_session_with_messages(db, n=4)
        last_msg = (
            db.query(Message).order_by(Message.id.desc()).first()
        )
        old_arc = ["[09:02] good-morning work log", "[11:00] DeepMind research"]
        create_summary(db, SummaryCreate(
            session_id=session.id, project_id=1,
            summary_json={"current_topic": "morning", "conversation_arc": old_arc},
            summarised_through_message_id=last_msg.id,
        ))
        # New unsummarised messages arrive after the v1 summary.
        _seed_session_with_messages(db, n=2, session=session)

        with patch(
            "app.memory.summary_generator.call_llm_text",
            new_callable=AsyncMock,
            return_value=_mock_summary_json(),  # no conversation_arc at all
        ):
            result = await generate_summary(db, session)

        assert result is not None
        assert result.conversation_arc == old_arc

    @pytest.mark.asyncio
    async def test_arc_cap_keeps_opener_and_tail(self, db, monkeypatch):
        from app.memory.summary_generator import generate_summary

        monkeypatch.setenv("SUMMARY_PROVIDER", "prov-a")
        monkeypatch.setenv("SUMMARY_MODEL", "model-a")
        monkeypatch.setenv("ASTRA_SUMMARY_ARC_MAX_ITEMS", "10")

        session = _seed_session_with_messages(db, n=4)
        long_arc = [f"[{h:02d}:00] phase {h}" for h in range(15)]

        with patch(
            "app.memory.summary_generator.call_llm_text",
            new_callable=AsyncMock,
            return_value=_mock_summary_json(conversation_arc=long_arc),
        ):
            result = await generate_summary(db, session)

        assert result is not None
        assert len(result.conversation_arc) == 10
        assert result.conversation_arc[0] == long_arc[0]      # opener pinned
        assert result.conversation_arc[-1] == long_arc[-1]    # newest kept

    @pytest.mark.asyncio
    async def test_skips_when_model_unconfigured(self, db, monkeypatch):
        """Principle 1: unconfigured env skips generation — no literal
        fallback model is ever called."""
        from app.memory.summary_generator import generate_summary

        _clear_model_env(monkeypatch)
        session = _seed_session_with_messages(db, n=2)

        with patch(
            "app.memory.summary_generator.call_llm_text",
            new_callable=AsyncMock,
        ) as mock_llm:
            result = await generate_summary(db, session)

        assert result is None
        mock_llm.assert_not_awaited()
