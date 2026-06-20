# FILE: tests/test_nat_keyword_trail.py
# Purpose: Guard Nat Job 1 (keyword recall trail): recall-intent gating, keyword
#          cleaning, coverage block from the trail, and extraction fallback.
# Last-renovated: 2026-06-19
import asyncio
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def _scratch_session():
    """In-memory SQLite with the message_keywords table + its FK targets.

    The FK targets (projects, conversation_sessions, messages) must be in the
    shared Base.metadata or create() raises NoReferencedTableError, so we import
    the model modules and create just that closed table set. FK enforcement is
    off by default on a fresh in-memory engine, so inserts need no parent rows.
    """
    from app.db import Base
    from app.memory import models as mem_models  # Project, Message
    from app.memory import conversation_models  # ConversationSession
    from app.memory.nat_jobs.keyword_models import MessageKeywords
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[
        mem_models.Project.__table__,
        conversation_models.ConversationSession.__table__,
        mem_models.Message.__table__,
        MessageKeywords.__table__,
    ])
    db = sessionmaker(bind=engine)()
    # The global pragma listener forces foreign_keys=ON for any sqlite engine;
    # this logic test inserts keyword rows without parent rows, so turn it off.
    db.execute(text("PRAGMA foreign_keys=OFF"))
    return db, MessageKeywords


def test_is_recall_question():
    from app.memory.nat_jobs.coverage import is_recall_question
    yes = [
        "what did we talk about today?",
        "can you summarise our conversation",
        "give me a recap of everything we discussed",
        "remind me what we covered",
        "what have we been talking about",
    ]
    no = [
        "log my breakfast",
        "what's the weather",
        "how many calories in pasta",
        "summarise this article for me",  # not the conversation
    ]
    for q in yes:
        assert is_recall_question(q), q
    for q in no:
        assert not is_recall_question(q), q


def test_keyword_clean():
    from app.memory.nat_jobs.keyword_trail import _clean
    assert _clean(["a", "A", " b ", "", "x" * 50, "c"]) == ["a", "b", "c"]
    assert _clean("not a list") == []
    assert len(_clean([str(i) for i in range(20)])) == 8  # cap


def test_coverage_block_from_trail():
    db, MK = _scratch_session()
    rows = [
        MK(project_id=1, session_id=5, message_id=10, keywords=json.dumps(["calorie tracking", "pasta"])),
        MK(project_id=1, session_id=5, message_id=12, keywords=json.dumps(["weight loss", "pasta"])),  # dup pasta
        MK(project_id=1, session_id=5, message_id=14, keywords=json.dumps(["portugal move"])),
    ]
    for r in rows:
        db.add(r)
    db.commit()

    from app.memory.nat_jobs.coverage import build_coverage_block
    # recall question -> coverage with chronological, deduped topics
    block = build_coverage_block(db, 1, "what did we talk about?")
    assert block.startswith("[CONVERSATION_COVERAGE]"), block
    assert "calorie tracking" in block and "weight loss" in block and "portugal move" in block
    assert block.count("pasta") == 1  # deduped
    assert block.index("calorie tracking") < block.index("portugal move")  # order preserved
    # non-recall question -> empty (no token bloat)
    assert build_coverage_block(db, 1, "log my dinner") == ""
    # unknown project -> empty
    assert build_coverage_block(db, 999, "what did we talk about?") == ""


def test_extract_keywords_fallback_when_nat_down():
    """Nat unreachable -> deterministic extract_tags fallback, never empty-crash."""
    import os
    os.environ.update(VLLM_BASE_URL="http://127.0.0.1:1/v1", NAT_PROVIDER="vllm_local",
                      NAT_MODEL="gemma4:e4b")

    async def _run():
        from app.memory.nat_jobs.keyword_trail import extract_keywords
        kws, source = await extract_keywords("I want to talk about my fitness and nutrition goals")
        assert isinstance(kws, list)
        assert source in ("fallback", "none")  # Nat down -> not "nat"
        # the deterministic tagger should find something for this content
        assert source == "fallback" and kws, (source, kws)

    asyncio.run(_run())


if __name__ == "__main__":
    test_is_recall_question()
    test_keyword_clean()
    test_coverage_block_from_trail()
    test_extract_keywords_fallback_when_nat_down()
    print("nat keyword trail tests passed")
