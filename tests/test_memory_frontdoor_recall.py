# FILE: tests/test_memory_frontdoor_recall.py
# Purpose: Job 2 — prove recall + summary reach the prompt THROUGH the single
#          front door (build_memory_block), the path the phone/voice uses.
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.db, app.memory.conversation_models, app.memory.models, app.memory.spine_service, app.llm.routing.memory_injection
# Last-renovated: 2026-06-15 (Job 2: read the ledger)
"""
Front-door recall/summary tests (MEMORY_JOBS_SPEC.md Job 2).

Job 0 collapsed the three injectors into one gated front door
(memory_injection.build_memory_block) and routed build_full_context — the
assembler BOTH the desktop-streaming and phone/voice paths call — through it.
That means the within-conversation recall + rolling summary now reach the model
on the voice path, which is the reported "lost-middle" bug.

These tests lock that in at the front-door level (not just the spine_service
unit level the conversation-spine suite already covers):

  - conversation_only=True path (what endpoints/chat.py /chat uses): recall of a
    scrolled-out topic, window exclusion, topic-free silence, summary backstop.
  - full path (what build_full_context → phone + desktop-streaming use): the
    same recall surfaces, with the heavy semantic sources neutralised so the
    test exercises the conversation wiring in a minimal DB.
  - the D0 selectivity gate still silences memory.
  - the per-turn observability line (Job 2 task 3) is emitted.

Run with: pytest tests/test_memory_frontdoor_recall.py -v
"""

import logging

import pytest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
# Import models so Base.metadata registers them before create_all.
from app.memory.models import Project, Message  # noqa: F401
from app.memory.conversation_models import (  # noqa: F401
    ConversationSession,
    ConversationSummary,
    ConversationSpine,
)


# =========================================================================
# Fixtures / helpers
# =========================================================================

@pytest.fixture
def db():
    """In-memory SQLite with crypto stubbed (same pattern as the
    conversation-spine + conversation-memory suites)."""
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
    """Persist a Message and index it into the spine (mirrors what the
    create_message chokepoint does in production)."""
    from app.memory.spine_service import record_message_spine
    m = Message(project_id=project_id, role=role, content=content, provider="local")
    db.add(m)
    db.commit()
    db.refresh(m)
    record_message_spine(db, m)
    return m


def _seed_long_session(db):
    """Topic A (Anthropic) discussed early, then a wall of unrelated chatter —
    the reported scenario where the early topic scrolls out of the window.

    Returns (early_message, all_window_ids) where all_window_ids is the set of
    the 20 most recent ids (what the live window would carry)."""
    early = _make_message(
        db, "user",
        "I've been reading about Anthropic's Claude models all week and I love them",
    )
    for i in range(25):
        _make_message(db, "user", f"Note number {i} about the weather and my lunch")

    from app.memory import service as memory_service
    window_ids = [m.id for m in memory_service.list_messages(db, 1, limit=20)]
    return early, window_ids


def _seed_summary(db, topic="Anthropic and the TAO position"):
    """Attach an active session + a rolling summary so the backstop has
    something to inject."""
    session = ConversationSession(project_id=1, status="active")
    db.add(session)
    db.commit()
    db.refresh(session)
    db.add(ConversationSummary(
        session_id=session.id, project_id=1, version=1,
        summary_json={
            "current_topic": topic,
            "key_decisions": ["hold TAO"],
            "open_items": [],
            "technical_context": "",
            "models_involved": [],
        },
    ))
    db.commit()


# =========================================================================
# conversation_only path — the /chat door
# =========================================================================

class TestConversationOnlyDoor:
    def test_recall_surfaces_scrolled_out_topic(self, db):
        from app.llm.routing.memory_injection import build_memory_block
        early, window_ids = _seed_long_session(db)

        block = build_memory_block(
            db, user_message="did we talk about Anthropic today?",
            project_id=1, window_message_ids=window_ids, conversation_only=True,
        )
        assert "[CONVERSATION_RECALL]" in block
        assert "Claude models" in block
        # The early topical message must be OUTSIDE the live window for this to
        # be a real recall (not just echoing what the model already sees).
        assert early.id not in window_ids

    def test_silent_for_topic_free_question(self, db):
        from app.llm.routing.memory_injection import build_memory_block
        _seed_long_session(db)
        block = build_memory_block(
            db, user_message="hmm, what should I have for lunch?",
            project_id=1, conversation_only=True,
        )
        # No recognised topic/entity → recall stays silent; no summary seeded.
        assert block == ""

    def test_summary_backstop_injected(self, db):
        from app.llm.routing.memory_injection import build_memory_block
        _seed_summary(db)
        # A question with a recognised topic so the D0 gate passes; the summary
        # is the always-on backstop even when nothing scrolled-out matches.
        block = build_memory_block(
            db, user_message="give me the status on Anthropic",
            project_id=1, conversation_only=True,
        )
        assert "[CONVERSATION_SUMMARY]" in block
        assert "Anthropic and the TAO position" in block

    def test_no_project_returns_empty(self, db):
        from app.llm.routing.memory_injection import build_memory_block
        assert build_memory_block(
            db, user_message="did we talk about Anthropic?",
            project_id=None, conversation_only=True,
        ) == ""


# =========================================================================
# full path — the build_full_context door (phone + desktop streaming)
# =========================================================================

class _EmptyRetrieval:
    """Stand-in for retrieve_for_query's result in a minimal DB."""
    records: list = []
    records_expanded = 0


@pytest.fixture
def neutralise_semantic(monkeypatch):
    """Stub the heavy db-touching semantic sources so the full path exercises
    the conversation wiring without tripping on astra_memory / rag tables that
    a minimal DB doesn't have. Uses the REAL depth classifier so the D0-D4 gate
    is genuinely tested."""
    import app.llm.routing.memory_injection as mi
    from app.astra_memory import classify_intent_depth, IntentDepth

    monkeypatch.setattr(mi, "_MEMORY_API", {
        "classify_intent_depth": classify_intent_depth,
        "retrieve_for_query": lambda **kw: _EmptyRetrieval(),
        "get_applicable_preferences": lambda *a, **k: [],
        "IntentDepth": IntentDepth,
    })
    try:
        monkeypatch.setattr(
            "app.rag.retrieval.chat_injection.build_repo_context",
            lambda *a, **k: "",
        )
    except Exception:
        # Module not importable in this env → build_memory_context's own
        # try/except swallows the import and repo context is empty anyway.
        pass


class TestFullDoor:
    def test_full_path_surfaces_recall(self, db, neutralise_semantic):
        """This is the phone/voice door: build_full_context calls
        build_memory_block with no conversation_only flag."""
        from app.llm.routing.memory_injection import build_memory_block
        _seed_long_session(db)
        block = build_memory_block(
            db, user_message="did we talk about Anthropic today?", project_id=1,
        )
        assert "[CONVERSATION_RECALL]" in block
        assert "Claude models" in block

    def test_full_path_d0_greeting_is_silent(self, db, neutralise_semantic):
        from app.llm.routing.memory_injection import build_memory_block
        _seed_long_session(db)
        # "hello" classifies D0 → the whole memory block is suppressed.
        block = build_memory_block(db, user_message="hello", project_id=1)
        assert block == ""


# =========================================================================
# Observability — Job 2 task 3 (the voice-path failure signature)
# =========================================================================

class TestObservability:
    def test_block_stats_logged_with_recall(self, db, caplog):
        from app.llm.routing.memory_injection import build_memory_block
        _, window_ids = _seed_long_session(db)
        with caplog.at_level(logging.INFO, logger="app.llm.routing.memory_injection_sources"):
            build_memory_block(
                db, user_message="did we talk about Anthropic today?",
                project_id=1, window_message_ids=window_ids, conversation_only=True,
            )
        assert "[memory_frontdoor]" in caplog.text
        assert "recall=Y" in caplog.text

    def test_block_stats_logged_when_empty(self, db, caplog):
        """The failure signature: a turn that assembled nothing still logs, so
        a long voice session with recall=n summary=n block=0 is visible."""
        from app.llm.routing.memory_injection import build_memory_block
        _seed_long_session(db)
        with caplog.at_level(logging.INFO, logger="app.llm.routing.memory_injection_sources"):
            build_memory_block(
                db, user_message="what should I have for lunch?",
                project_id=1, conversation_only=True,
            )
        assert "[memory_frontdoor]" in caplog.text
        assert "recall=n" in caplog.text
        assert "block=0chars" in caplog.text
