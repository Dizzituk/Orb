# FILE: tests/test_conversation_overview.py
# Purpose: Lane B (2026-07-01) — whole-conversation meta questions are served by
#          the durable [CONVERSATION_TIMELINE] overview (pinned opener +
#          waypoints), through the same recall source every chat path uses.
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.memory.conversation_meta, app.memory.conversation_overview, app.memory.spine_service
"""
Lane B tests — meta-question detector + conversation overview + spine wiring.

The evidence being locked in (2026-07-01 chat log):
  - "give me a full lowdown of the whole day" only covered the afternoon;
  - "what were the very first messages" was answered with an INVENTED opener.
Both questions are topicless, so the tag-gated spine recall returned "" exactly
when recall mattered most. The overview serves them from the ordered message
index, which never sheds.

Run with: pytest tests/test_conversation_overview.py -v
"""

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
# Fixtures / helpers (same pattern as the conversation-spine suite)
# =========================================================================

@pytest.fixture
def db():
    """In-memory SQLite DB with crypto stubbed."""
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
    m = Message(project_id=project_id, role=role, content=content, provider="local")
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def _seed_day(db, chatter=30):
    """Opener + a wall of later messages — the whole-day scenario. Returns
    (opener, window_ids) where window_ids are the newest-20 ids (the live
    window the model can already see)."""
    opener = _make_message(
        db, "user",
        "Good morning — here's my work log to kick the day off: centralise the AI stack",
    )
    _make_message(db, "assistant", "Morning! Logged. Let's plan the centralisation.")
    for i in range(chatter):
        _make_message(db, "user", f"Later chatter number {i} about lunch and errands")

    from app.memory import service as memory_service
    window_ids = [m.id for m in memory_service.list_messages(db, 1, limit=20)]
    return opener, window_ids


# =========================================================================
# Meta-question detector
# =========================================================================

class TestMetaDetector:
    POSITIVE = [
        # The four phrasings from the 2026-07-01 evidence + the brief:
        "give me a full lowdown of the whole day",
        "what were the very first messages",
        "how did this chat start",
        "summarise everything we've discussed",
        "what did we talk about today",
        # A few nearby phrasings that must also count:
        "gimme the lowdown on our chat",
        "what was the first thing I said?",
        "recap the entire day for me",
        # Review 2026-07-01 regression phrases (previously missed):
        "how did this chat get started",
        "tell me how this chat started",
        "what were our first messages",
        "what were my first messages today",
    ]
    NEGATIVE = [
        # Genuine lookups / trivia must NOT count (no wrongly-injected context):
        "what's the latest on anthropic",
        "what was the first message ever sent on the internet",
        # Review 2026-07-01 regression phrases: verb-led "lowdown on X" is the
        # idiomatic form of a genuine TOPIC lookup — must stay silent.
        "give me the lowdown on quantum computing",
        "need the lowdown on bitcoin",
        "I want the lowdown on the game tonight",
        # Smalltalk / the existing tests' topic-free phrases stay silent:
        "what should I have for lunch?",
        "hello, how are you today?",
        "hmm, what should I have for lunch?",
    ]

    def test_positive_phrases_fire(self):
        from app.memory.conversation_meta import is_conversation_meta_question
        for phrase in self.POSITIVE:
            assert is_conversation_meta_question(phrase), phrase

    def test_negative_phrases_stay_silent(self):
        from app.memory.conversation_meta import is_conversation_meta_question
        for phrase in self.NEGATIVE:
            assert not is_conversation_meta_question(phrase), phrase

    def test_superset_of_recall_intent(self):
        """Everything the shared web-search guard treats as recall must count
        as a meta question here (the detector is a strict superset)."""
        from app.memory.conversation_meta import is_conversation_meta_question
        for phrase in (
            "what was I telling you about Anthropic?",
            "did we talk about Anthropic today?",
            "catch me up",
        ):
            assert is_conversation_meta_question(phrase), phrase

    def test_empty_and_none_safe(self):
        from app.memory.conversation_meta import is_conversation_meta_question
        assert not is_conversation_meta_question("")


# =========================================================================
# Overview builder
# =========================================================================

class TestOverviewBuilder:
    def test_pins_real_opener(self, db):
        from app.memory.conversation_overview import (
            OVERVIEW_MARKER, build_conversation_overview,
        )
        opener, window_ids = _seed_day(db)
        block = build_conversation_overview(
            db, 1, "what were the very first messages", window_ids,
        )
        assert OVERVIEW_MARKER in block
        assert "Opening messages" in block
        assert "Good morning" in block and "work log" in block
        # The opener sits above the later content — chronological order.
        assert block.index("Good morning") < block.index("Later chatter")

    def test_window_tail_not_duplicated(self, db):
        from app.memory.conversation_overview import build_conversation_overview
        _, window_ids = _seed_day(db)
        block = build_conversation_overview(
            db, 1, "give me a full lowdown of the whole day", window_ids,
        )
        # The newest message is inside the visible window → excluded from the
        # "most recent turns" section (it would be pure duplication).
        newest = (
            db.query(Message).order_by(Message.id.desc()).first()
        )
        assert newest.content not in block

    def test_silent_for_non_meta_question(self, db):
        from app.memory.conversation_overview import build_conversation_overview
        _seed_day(db)
        assert build_conversation_overview(
            db, 1, "what should I have for lunch?", [],
        ) == ""

    def test_silent_when_disabled(self, db, monkeypatch):
        from app.memory.conversation_overview import build_conversation_overview
        _seed_day(db)
        monkeypatch.setenv("ASTRA_CONV_OVERVIEW", "0")
        assert build_conversation_overview(
            db, 1, "what were the very first messages", [],
        ) == ""

    def test_silent_without_messages(self, db):
        from app.memory.conversation_overview import build_conversation_overview
        assert build_conversation_overview(
            db, 1, "what were the very first messages", [],
        ) == ""

    def test_silent_without_project(self, db):
        from app.memory.conversation_overview import build_conversation_overview
        assert build_conversation_overview(
            db, None, "what were the very first messages", [],
        ) == ""

    def test_respects_char_budget(self, db, monkeypatch):
        from app.memory.conversation_overview import build_conversation_overview
        _seed_day(db, chatter=40)
        monkeypatch.setenv("ASTRA_CONV_OVERVIEW_MAX_CHARS", "700")
        block = build_conversation_overview(
            db, 1, "summarise everything we've discussed", [],
        )
        assert block  # opener still present under a tight budget
        assert len(block) <= 700 + 200  # header/footer allowance
        assert "Good morning" in block


# =========================================================================
# Spine wiring — the fix for "recall returned '' when it mattered most"
# =========================================================================

class TestSpineIntegration:
    def test_topicless_meta_question_served(self, db):
        """The 18:56 scenario: 'what were the very first messages' is
        topicless (no tags/entities), which used to return '' — now the
        durable timeline answers it with the REAL opener."""
        from app.memory.spine_service import build_conversation_recall
        opener, window_ids = _seed_day(db)

        block = build_conversation_recall(
            db, project_id=1,
            user_message="what were the very first messages",
            window_message_ids=window_ids,
        )
        assert "[CONVERSATION_TIMELINE]" in block
        assert "Good morning" in block

    def test_topic_recall_unchanged(self, db):
        """Existing topic-tagged recall behaviour is untouched."""
        from app.memory.spine_service import (
            record_message_spine, build_conversation_recall,
        )
        older = _make_message(
            db, "user",
            "I was telling you that Anthropic's Claude models are my favourite",
        )
        record_message_spine(db, older)

        block = build_conversation_recall(
            db, project_id=1,
            user_message="what was I telling you about Anthropic?",
            window_message_ids=[999],
        )
        assert "[CONVERSATION_RECALL]" in block
        assert "Claude models are my favourite" in block

    def test_smalltalk_still_silent(self, db):
        from app.memory.spine_service import build_conversation_recall
        _seed_day(db)
        assert build_conversation_recall(
            db, 1, "hello, how are you today?", window_message_ids=[],
        ) == ""

    def test_timeline_defers_to_topic_recall_no_double_render(self, db):
        """Review 2026-07-01: a mid-conversation message that topic recall
        renders must not ALSO appear in the timeline's waypoint/tail sections
        with a different truncation (the cross-source de-dup can't collapse
        the two renderings). Topic recall keeps priority; the timeline defers."""
        from app.memory.spine_service import (
            record_message_spine, build_conversation_recall,
        )
        # Enough pre-chatter that the topical message is NOT a pinned opener.
        for i in range(6):
            _make_message(db, "user", f"Morning chatter number {i}")
        topical = _make_message(
            db, "user",
            "I was telling you that Anthropic's Claude models are my favourite — "
            + "detail " * 60,  # >280 chars: the two channels cut differently
        )
        record_message_spine(db, topical)
        for i in range(10):
            _make_message(db, "user", f"Afternoon chatter number {i}")

        block = build_conversation_recall(
            db, project_id=1,
            user_message="what was I telling you about Anthropic at the start of the day?",
            window_message_ids=[],
        )
        # Meta + topical question: both channels fire…
        assert "[CONVERSATION_TIMELINE]" in block
        assert "[CONVERSATION_RECALL]" in block
        # …but the topical message is rendered exactly ONCE (by recall).
        assert block.count("Claude models are my favourite") == 1
