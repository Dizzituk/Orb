# FILE: tests/test_deep_research_task.py
# Purpose: WS5 — depth budget, lead-chasing cap, findings block contract, completion flag, checkpoint.
# Called-by: pytest
# Depends-on: app.llm.research_task, app.llm.research_context, app.llm.deep_research_engine
# Last-renovated: 2026-07-01

import json
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.idle import ledger
from app.idle.models import IdleTaskRecord
from app.idle.router import TaskContext
from app.llm import deep_research_engine as engine
from app.llm import research_task
from app.llm.research_models import ResearchRun


@pytest.fixture
def session_factory():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Explicit tables only — dodges the known fresh-DB NoReferencedTableError
    # (messages.session_id FK) under cross-test import pollution.
    Base.metadata.create_all(bind=eng, tables=[ResearchRun.__table__, IdleTaskRecord.__table__])
    maker = sessionmaker(bind=eng, autocommit=False, autoflush=False)
    yield maker
    eng.dispose()


@pytest.fixture
def fake_engine(monkeypatch):
    """Deterministic engine phases; counters record what the grind did."""
    calls = {"plan": 0, "search": 0, "search_queries": [], "gaps": 0, "synth": 0}

    async def fake_plan(question, context="", round_num=1, previous_gaps="", local_only=False):
        calls["plan"] += 1
        return [f"planned-{round_num}-a", f"planned-{round_num}-b"]

    async def fake_search(queries, pages_budget):
        calls["search"] += 1
        calls["search_queries"].append(list(queries))
        n = min(5, pages_budget)
        sources = [
            {"title": f"S{calls['search']}-{i}", "url": f"http://src/{calls['search']}/{i}", "fetched": True}
            for i in range(n)
        ]
        return sources, f"evidence from search {calls['search']}"

    async def fake_gaps(question, evidence, local_only=False):
        calls["gaps"] += 1
        return "still missing plenty"

    async def fake_synth(question, evidence, source_count, local_only=False):
        calls["synth"] += 1
        return "draft synthesis text"

    monkeypatch.setattr(engine, "_plan_queries", fake_plan)
    monkeypatch.setattr(engine, "_search_and_collect", fake_search)
    monkeypatch.setattr(engine, "_analyse_gaps", fake_gaps)
    monkeypatch.setattr(engine, "_synthesise", fake_synth)

    async def fake_claims(question, evidence):
        return [{"claim": f"claim from {evidence[:20]}", "source_url": "http://src/x"}]

    async def fake_leads(question, evidence, cap):
        return [f"lead-{i}" for i in range(cap)]

    monkeypatch.setattr(research_task, "_extract_claims", fake_claims)
    monkeypatch.setattr(research_task, "_extract_leads", fake_leads)
    return calls


def _ctx(db, session_factory, run_id, should_yield=lambda: False):
    rec = ledger.enqueue(db, "deep_research", {"research_id": run_id}, dedupe=False)
    return TaskContext(
        db=db, record=rec, params={"research_id": run_id}, fingerprint=None,
        should_yield=should_yield, session_factory=session_factory,
    )


# ── depth budget ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_budget_bounds_the_grind(session_factory, fake_engine, monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_MAX_FETCHES", "6")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_MINUTES", "60")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_ROUNDS_BACKGROUND", "8")

    db = session_factory()
    run = research_task.create_research_run(db, "what is the state of X?")
    outcome = await research_task.deep_research_handler(_ctx(db, session_factory, run.id))

    assert outcome.status == "completed"
    # round 1 fetches 5 (<6), round 2 gets budget 1 -> 6 total, then stop.
    assert fake_engine["search"] == 2
    stats = json.loads(run.stats_json)
    assert stats["fetches"] == 6
    assert run.status == "completed"
    assert run.synthesis == "draft synthesis text"
    db.close()


@pytest.mark.asyncio
async def test_time_budget_bounds_the_grind(session_factory, fake_engine, monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_MAX_FETCHES", "1000")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_MINUTES", "0.000001")

    db = session_factory()
    run = research_task.create_research_run(db, "time-bounded question")
    outcome = await research_task.deep_research_handler(_ctx(db, session_factory, run.id))

    assert outcome.status == "completed"
    assert fake_engine["search"] == 1  # budget spent after the first active round
    db.close()


# ── lead-chasing ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_leads_feed_next_round_first(session_factory, fake_engine, monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_MAX_FETCHES", "10")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_MINUTES", "60")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_LEADS_PER_ROUND", "3")

    db = session_factory()
    run = research_task.create_research_run(db, "lead chasing test")
    await research_task.deep_research_handler(_ctx(db, session_factory, run.id))

    assert len(fake_engine["search_queries"]) >= 2
    round2 = fake_engine["search_queries"][1]
    # leads from round 1 lead the round-2 query list, capped by env
    assert round2[:3] == ["lead-0", "lead-1", "lead-2"]
    assert len(round2) <= engine.MAX_QUERIES_PER_ROUND
    db.close()


@pytest.mark.asyncio
async def test_real_lead_extractor_caps_output(monkeypatch):
    async def fake_llm(prompt, system, max_tokens=200, local_only=False):
        return json.dumps([f"lead {i}" for i in range(10)])

    monkeypatch.setattr(engine, "_llm_call", fake_llm)
    leads = await research_task._extract_leads("q", "evidence", cap=3)
    assert leads == ["lead 0", "lead 1", "lead 2"]


# ── findings block contract ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_findings_block_well_formed_and_capped(session_factory):
    from app.llm.research_context import MAX_RESEARCH_BLOCK_CHARS, build_research_context

    db = session_factory()
    run = ResearchRun(
        id="abc123def456",
        query="what is the state of X?",
        status="completed",
        findings_json=json.dumps([{"claim": f"important fact {i}", "source_url": f"http://s/{i}", "round": 1} for i in range(20)]),
        sources_json=json.dumps([{"title": f"Source {i}", "url": f"http://s/{i}", "credibility_label": "established"} for i in range(15)]),
        synthesis="way too long " * 800,  # ~10KB -> must be capped
        completed_at=datetime(2026, 7, 1, 12, 0),
    )
    db.add(run)
    db.commit()

    block = await build_research_context("abc123def456", db)
    assert block.startswith("[RESEARCH_FINDINGS]")
    assert block.endswith("[/RESEARCH_FINDINGS]")
    assert len(block) <= MAX_RESEARCH_BLOCK_CHARS
    assert "what is the state of X?" in block
    assert "important fact 0" in block and "http://s/0" in block
    assert "Research completed: 2026-07-01 12:00 UTC" in block
    db.close()


@pytest.mark.asyncio
async def test_findings_block_never_raises_and_empty_on_none(session_factory):
    from app.llm.research_context import build_research_context

    db = session_factory()
    assert await build_research_context("nonexistent", db) == ""

    incomplete = ResearchRun(id="notdoneyet00", query="q", status="running")
    db.add(incomplete)
    db.commit()
    assert await build_research_context("notdoneyet00", db) == ""
    db.close()

    # A hostile db object must be swallowed, not raised.
    assert await build_research_context("anything", object()) == ""


# ── completion flag + checkpoint ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_completion_flags_and_ledger_row(session_factory, fake_engine, monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_MAX_FETCHES", "5")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_MINUTES", "60")
    flags = []
    monkeypatch.setattr(research_task, "_flag_completion", lambda db, run, f, s: flags.append((run.id, f, s)))

    db = session_factory()
    run = research_task.create_research_run(db, "flag me when done")
    outcome = await research_task.deep_research_handler(_ctx(db, session_factory, run.id))

    assert outcome.status == "completed"
    assert len(flags) == 1 and flags[0][0] == run.id and flags[0][1] >= 1
    db.close()


@pytest.mark.asyncio
async def test_user_message_checkpoints_research(session_factory, fake_engine, monkeypatch):
    monkeypatch.setenv("DEEP_RESEARCH_MAX_FETCHES", "100")
    monkeypatch.setenv("DEEP_RESEARCH_MAX_MINUTES", "60")

    db = session_factory()
    run = research_task.create_research_run(db, "interrupted research")

    yields = {"n": 0}

    def should_yield():
        yields["n"] += 1
        return yields["n"] > 1  # allow round 1, checkpoint before round 2

    ctx = _ctx(db, session_factory, run.id, should_yield=should_yield)
    outcome = await research_task.deep_research_handler(ctx)
    assert outcome.status == "paused"
    assert run.status == "paused"
    assert len(json.loads(run.findings_json)) >= 1  # round-1 findings persisted

    # Next idle window: resumes and completes under a now-tight budget.
    monkeypatch.setenv("DEEP_RESEARCH_MAX_FETCHES", "5")
    ctx2 = TaskContext(
        db=db, record=ctx.record, params=ctx.params, fingerprint=None,
        should_yield=lambda: False, session_factory=session_factory,
    )
    outcome2 = await research_task.deep_research_handler(ctx2)
    assert outcome2.status == "completed"
    assert run.status == "completed"
    db.close()


@pytest.mark.asyncio
async def test_start_tool_queues_run_and_ledger_task(session_factory, monkeypatch):
    import app.db as app_db

    monkeypatch.setattr(app_db, "get_db_session", lambda: session_factory())
    out = await research_task.start_deep_research_handler({"query": "dig into X"})
    assert out["ok"] is True and out["status"] == "queued"

    db = session_factory()
    assert db.query(ResearchRun).filter(ResearchRun.id == out["research_id"]).first() is not None
    rows = db.query(IdleTaskRecord).filter(IdleTaskRecord.task_type == "deep_research").all()
    assert len(rows) == 1 and out["research_id"] in rows[0].params_json
    db.close()
