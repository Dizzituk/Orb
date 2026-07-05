# FILE: tests/test_recurring_synthesis.py
# Purpose: Guard the 2026-06-24 synthesis fixes — (1) cross-conversation recurring
#          themes (a topic recurring across >=N distinct sessions surfaces), and
#          (2) consolidation_graph merging a reverse-direction link into ONE
#          strengthened edge instead of a duplicate.
# Last-renovated: 2026-06-24
import json

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def _scratch_session():
    from app.db import Base
    from app.memory import models as mem_models
    from app.memory import conversation_models
    from app.memory.nat_jobs.keyword_models import MessageKeywords
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[
        mem_models.Project.__table__,
        conversation_models.ConversationSession.__table__,
        mem_models.Message.__table__,
        MessageKeywords.__table__,
    ])
    db = sessionmaker(bind=engine)()
    db.execute(text("PRAGMA foreign_keys=OFF"))
    return db, MessageKeywords


def _seed(db, MK):
    # Each conversation is its own PROJECT (how the phone stores them), so seed the
    # trail across distinct project_ids. The recurring channel counts distinct
    # projects per topic — no messages table needed.
    #   investments -> projects {100,101,102} = 3 (recurring)
    #   portugal    -> projects {101,100}     = 2 (below the >=3 bar)
    rows = [
        (100, ["investments", "work"]),
        (101, ["investments", "portugal"]),
        (102, ["investments"]),
        (100, ["portugal"]),
        (103, ["music"]),
    ]
    for i, (pid, kws) in enumerate(rows, start=1):
        db.add(MK(project_id=pid, message_id=i, keywords=json.dumps(kws)))
    db.commit()


def _clear_cache():
    from app.memory.nat_jobs import recurring_themes as rt
    rt._FREQ_CACHE.clear()


def test_recurring_surfaces_on_recall():
    _clear_cache()
    db, MK = _scratch_session()
    _seed(db, MK)
    from app.memory.nat_jobs.recurring_themes import build_recurring_themes_block
    block = build_recurring_themes_block(db, 1, "what do I keep coming back to?")
    assert block.startswith("[RECURRING THEMES"), block
    assert "investments (3 conversations)" in block
    assert "portugal" not in block          # 2 sessions -> below the >=3 threshold
    assert "work" not in block               # 1 session


def test_recurring_silent_when_not_recall_and_no_topic_match(monkeypatch):
    _clear_cache()
    db, MK = _scratch_session()
    _seed(db, MK)
    from app.memory.nat_jobs import recurring_themes as rt
    monkeypatch.setattr(rt, "_current_topics", lambda msg: {"unrelated"})
    assert rt.build_recurring_themes_block(db, 1, "tell me about the weather") == ""


def test_recurring_nudge_on_topical_turn(monkeypatch):
    _clear_cache()
    db, MK = _scratch_session()
    _seed(db, MK)
    from app.memory.nat_jobs import recurring_themes as rt
    # Current turn is about a recurring topic -> a one-line nudge, not the overview.
    monkeypatch.setattr(rt, "_current_topics", lambda msg: {"investments"})
    block = rt.build_recurring_themes_block(db, 1, "I moved some money into the index fund")
    assert block.startswith("[RECURRING THEME]"), block
    assert "investments (3 conversations)" in block


def test_recurring_nudge_real_freeform_keyword():
    # Free-form Nat key "index fund" recurs across 3 sessions; a non-recall turn
    # that mentions it must nudge WITHOUT monkeypatching _current_topics
    # (exercises the word-boundary substring path the coarse tagger would miss).
    _clear_cache()
    db, MK = _scratch_session()
    # "index fund" across 3 distinct conversations (projects)
    for i, pid in enumerate([200, 201, 202], start=1):
        db.add(MK(project_id=pid, message_id=i, keywords=json.dumps(["index fund"])))
    db.commit()
    from app.memory.nat_jobs.recurring_themes import build_recurring_themes_block
    block = build_recurring_themes_block(db, 999, "I just moved more money into the index fund")
    assert block.startswith("[RECURRING THEME]"), block
    assert "index fund (3 conversations)" in block
    # a turn NOT about a recurring theme stays silent
    assert build_recurring_themes_block(db, 999, "what's the weather like") == ""


def test_recurring_skips_today_question():
    # A "today" question is coverage's job; recurring must NOT inject all-time
    # themes for it (that was the "it pulled old stuff" bug).
    _clear_cache()
    db, MK = _scratch_session()
    _seed(db, MK)  # investments recurs across 3 conversations
    from app.memory.nat_jobs.recurring_themes import build_recurring_themes_block
    assert build_recurring_themes_block(db, 999, "what have we talked about today") == ""
    # ...but a genuine recurrence question still surfaces them.
    blk = build_recurring_themes_block(db, 999, "what do I keep coming back to")
    assert blk.startswith("[RECURRING THEMES"), blk


def test_coverage_today_spans_conversations():
    # "today" coverage must span ALL of today's conversations (projects), not just
    # the current one — asked from a fresh chat it still sees earlier chats today.
    db, MK = _scratch_session()
    db.add(MK(project_id=100, message_id=1, keywords=json.dumps(["anthropic", "fable"])))
    db.add(MK(project_id=101, message_id=2, keywords=json.dumps(["portugal trip"])))
    db.commit()
    from app.memory.nat_jobs.coverage import build_coverage_block
    block = build_coverage_block(db, 999, "what have we talked about today")
    assert block.startswith("[CONVERSATION_COVERAGE]"), block
    assert "anthropic" in block and "portugal trip" in block  # across projects 100 + 101


def test_recurring_disabled(monkeypatch):
    _clear_cache()
    db, MK = _scratch_session()
    _seed(db, MK)
    monkeypatch.setenv("ASTRA_RECURRING_THEMES", "0")
    from app.memory.nat_jobs.recurring_themes import build_recurring_themes_block
    assert build_recurring_themes_block(db, 1, "what do I keep talking about?") == ""


# --- consolidation_graph: reverse-direction merge -------------------------

class _FakeGraph:
    """Minimal stand-in for the directed knowledge graph: edges keyed (src,tgt)."""
    def __init__(self):
        self.entities = {}
        self.edges = {}  # (src, tgt) -> Relationship

    def get_entity(self, nid):
        return self.entities.get(nid)

    def add_entity(self, ent):
        self.entities[ent.entity_id] = ent

    def get_relationships(self, node_id, direction="both"):
        out = []
        for (s, t), rel in self.edges.items():
            if direction in ("outgoing", "both") and s == node_id:
                out.append(rel)
            if direction in ("incoming", "both") and t == node_id:
                out.append(rel)
        return out

    def add_relationship(self, rel):
        self.edges[(rel.source_id, rel.target_id)] = rel


def test_reverse_direction_link_merges_not_duplicates():
    from app.memory import consolidation_graph as cg
    g = _FakeGraph()
    # Session 1: investments funds the portugal move.
    assert cg._write_link(g, cg.DomainLink("investments", "funds", "portugal move"), 1)
    # Session 2: the SAME pair drawn the reverse way must STRENGTHEN, not duplicate.
    assert cg._write_link(g, cg.DomainLink("portugal move", "depends on", "investments"), 2)

    assert len(g.edges) == 1, f"expected one merged edge, got {list(g.edges)}"
    rel = next(iter(g.edges.values()))
    assert rel.metadata["occurrences"] == 2
    assert rel.weight == 2.0
    assert set(rel.metadata["sessions"]) == {1, 2}


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
