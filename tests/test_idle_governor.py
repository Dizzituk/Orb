# FILE: tests/test_idle_governor.py
# Purpose: WS2 — idle governor, persistent ledger, checkpoint yield, catch-up, fingerprint skip.
# Called-by: pytest
# Depends-on: app.idle.*
# Last-renovated: 2026-07-01

import asyncio
import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.idle import ledger, router
from app.idle.governor import IdleGovernor
from app.idle.models import IdleTaskRecord
from app.idle.router import RecurringSpec, TaskOutcome, register_task_handler


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Explicit tables only — dodges the known fresh-DB NoReferencedTableError
    # (messages.session_id FK) under cross-test import pollution.
    Base.metadata.create_all(bind=engine, tables=[IdleTaskRecord.__table__])
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    yield maker
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_router():
    """Isolate handler/recurring registrations per test: start empty (other
    modules imported in the session may have registered real tasks), restore
    the originals afterwards."""
    h = dict(router._HANDLERS)
    f = dict(router._FINGERPRINTS)
    r = list(router._RECURRING)
    router._HANDLERS.clear()
    router._FINGERPRINTS.clear()
    router._RECURRING.clear()
    yield
    router._HANDLERS.clear()
    router._HANDLERS.update(h)
    router._FINGERPRINTS.clear()
    router._FINGERPRINTS.update(f)
    router._RECURRING[:] = r


def _governor(session_factory, monkeypatch, idle_minutes="0"):
    monkeypatch.setenv("IDLE_MINUTES", idle_minutes)
    monkeypatch.setenv("ASTRA_IDLE_GOVERNOR_ENABLED", "true")
    return IdleGovernor(session_factory=session_factory)


# ── idle trigger ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_idle_trigger_drains_queued_task(session_factory, monkeypatch):
    calls = []

    async def handler(ctx):
        calls.append(ctx.params)
        return TaskOutcome.completed(summary="did the thing", coverage="unit test")

    register_task_handler("fake_task", handler)
    db = session_factory()
    ledger.enqueue(db, "fake_task", {"x": 1})
    db.close()

    gov = _governor(session_factory, monkeypatch, idle_minutes="0")
    processed = await gov.run_pending()

    assert processed == 1
    assert calls == [{"x": 1}]
    db = session_factory()
    row = db.query(IdleTaskRecord).one()
    assert row.status == "completed"
    assert row.result_summary == "did the thing"
    db.close()


@pytest.mark.asyncio
async def test_no_drain_while_user_active(session_factory, monkeypatch):
    async def handler(ctx):  # pragma: no cover — must not run
        raise AssertionError("ran while user active")

    register_task_handler("fake_task", handler)
    db = session_factory()
    ledger.enqueue(db, "fake_task", {})
    db.close()

    gov = _governor(session_factory, monkeypatch, idle_minutes="10")
    gov.record_activity()  # fresh activity — not idle
    processed = await gov.run_pending()

    assert processed == 0
    db = session_factory()
    assert db.query(IdleTaskRecord).one().status == "queued"
    db.close()


# ── checkpoint yield ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_user_message_checkpoints_within_one_unit(session_factory, monkeypatch):
    """A message mid-task pauses at the next unit boundary with progress
    persisted; the next idle window resumes from that unit."""
    unit_log = []

    async def handler(ctx):
        progress = ctx.load_progress()
        done = list(progress.get("units_done") or [])
        for unit in ("u1", "u2", "u3"):
            if unit in done:
                continue
            if ctx.should_yield():
                ctx.save_progress({"units_done": done})
                return TaskOutcome.paused(f"yielded before {unit}")
            unit_log.append(unit)
            done.append(unit)
            if unit == "u1":
                gov.record_activity()  # user message arrives mid-task
        ctx.save_progress({"units_done": done})
        return TaskOutcome.completed(summary="all units done")

    register_task_handler("resumable", handler)
    db = session_factory()
    ledger.enqueue(db, "resumable", {})
    db.close()

    gov = _governor(session_factory, monkeypatch, idle_minutes="10")
    gov._last_activity -= 3600  # simulate an hour of silence

    processed = await gov.run_pending()
    assert processed == 1
    assert unit_log == ["u1"]  # checkpointed before u2
    db = session_factory()
    row = db.query(IdleTaskRecord).one()
    assert row.status == "paused"
    assert json.loads(row.progress_json)["units_done"] == ["u1"]
    db.close()

    # Next idle window: resumes from u2, completes.
    gov._last_activity -= 3600
    processed = await gov.run_pending()
    assert processed == 1
    assert unit_log == ["u1", "u2", "u3"]
    db = session_factory()
    assert db.query(IdleTaskRecord).one().status == "completed"
    db.close()


# ── boot catch-up ───────────────────────────────────────────────────────────


def test_catch_up_enqueues_due_recurring_task(session_factory):
    async def handler(ctx):
        return TaskOutcome.completed()

    register_task_handler(
        "daily_scrape", handler,
        recurring=RecurringSpec(task_type="daily_scrape", params={"w": "land"}, cadence_hours=24),
    )
    db = session_factory()

    assert router.catch_up(db) == 1  # never ran -> due
    assert router.catch_up(db) == 0  # already queued -> deduped

    # Complete it now -> inside cadence -> not due.
    rec = ledger.next_runnable(db)
    ledger.mark_running(db, rec)
    ledger.mark_completed(db, rec, summary="ran")
    assert router.catch_up(db) == 0

    # Backdate the completion beyond cadence -> due again (the morning-after-
    # shutdown case).
    rec.completed_at = datetime.utcnow() - timedelta(hours=25)
    db.commit()
    assert router.catch_up(db) == 1
    db.close()


def test_boot_recovery_resumes_crash_stranded_running_rows(session_factory):
    db = session_factory()
    rec = ledger.enqueue(db, "anything", {})
    ledger.mark_running(db, rec)
    ledger.save_progress(db, rec, {"units_done": ["u1"]})
    db.close()

    # Simulated restart: fresh session, recover.
    db = session_factory()
    assert ledger.recover_stale_running(db) == 1
    row = db.query(IdleTaskRecord).one()
    assert row.status == "paused"
    assert json.loads(row.progress_json)["units_done"] == ["u1"]
    db.close()


# ── fingerprint skip ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unchanged_fingerprint_skips_without_running(session_factory, monkeypatch):
    calls = []
    fp = {"value": "FP1"}

    async def handler(ctx):
        calls.append(1)
        return TaskOutcome.completed(summary="scanned")

    register_task_handler("fp_task", handler, fingerprint_fn=lambda params: fp["value"])
    gov = _governor(session_factory, monkeypatch, idle_minutes="0")

    db = session_factory()
    ledger.enqueue(db, "fp_task", {})
    db.close()
    await gov.run_pending()
    assert len(calls) == 1

    # Same inputs -> skipped, handler not invoked.
    db = session_factory()
    ledger.enqueue(db, "fp_task", {})
    db.close()
    await gov.run_pending()
    assert len(calls) == 1
    db = session_factory()
    statuses = [r.status for r in db.query(IdleTaskRecord).order_by(IdleTaskRecord.id).all()]
    assert statuses == ["completed", "skipped"]
    db.close()

    # Changed inputs -> runs again.
    fp["value"] = "FP2"
    db = session_factory()
    ledger.enqueue(db, "fp_task", {})
    db.close()
    await gov.run_pending()
    assert len(calls) == 2


# ── unknown handler + chat tool ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unknown_task_type_marks_failed(session_factory, monkeypatch):
    db = session_factory()
    ledger.enqueue(db, "no_such_task", {})
    db.close()
    gov = _governor(session_factory, monkeypatch, idle_minutes="0")
    await gov.run_pending()
    db = session_factory()
    row = db.query(IdleTaskRecord).one()
    assert row.status == "failed"
    assert "no handler" in (row.error or "")
    db.close()


@pytest.mark.asyncio
async def test_background_task_log_tool_reads_ledger(session_factory, monkeypatch):
    import app.db as app_db
    from app.idle.tools_registration import get_background_task_log_handler

    db = session_factory()
    rec = ledger.enqueue(db, "repo_map", {})
    ledger.mark_running(db, rec)
    ledger.mark_completed(db, rec, fingerprint="abc", coverage="app/", summary="+3 files mapped")
    ledger.enqueue(db, "watch_observe", {"watcher": "hardware"})
    db.close()

    monkeypatch.setattr(app_db, "get_db_session", lambda: session_factory())
    out = await get_background_task_log_handler({"limit": 5})

    assert out["ok"] is True
    assert out["recent"][0]["task"] == "repo_map"
    assert out["recent"][0]["summary"] == "+3 files mapped"
    assert out["queued_or_paused"][0]["task"] == "watch_observe"
    assert "enabled" in out["governor"]
