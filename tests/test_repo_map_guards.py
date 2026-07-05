# FILE: tests/test_repo_map_guards.py
# Purpose: Guards added 2026-07-02 after the first live drain quarantined the RAG index (sandbox down).
# Called-by: pytest
# Depends-on: app.idle.tasks, app.idle.governor
# Last-renovated: 2026-07-02
"""
Incident: sandbox unreachable -> rescan walked an empty tree -> "-1403 files"
-> whole index quarantined -> cleanup hard-deleted 8558 embedding vectors.
These tests pin the guards: sandbox pre-flight, mass-deletion tripwire (fails
BEFORE the cleanup unit), no fingerprint recorded on failure (so recovery
runs are never fingerprint-skipped), and the failed-run retry cooldown.
"""

import json
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.idle import ledger, router
from app.idle import tasks as idle_tasks
from app.idle.governor import IdleGovernor
from app.idle.models import IdleTaskRecord
from app.idle.router import register_task_handler


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[IdleTaskRecord.__table__])
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    yield maker
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_router():
    """Start with empty handler/recurring registries: modules imported at
    collection (watcher instances, idle tasks) register REAL recurring specs
    globally, and run_pending's catch_up would enqueue them into the test DB."""
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


class _FakeReport:
    def __init__(self, added=0, modified=0, deleted=0, unchanged=0):
        self.added = ["x"] * added
        self.modified = ["x"] * modified
        self.deleted = ["x"] * deleted
        self.unchanged = ["x"] * unchanged
        self.chunks_added = 0
        self.chunks_removed = 0


@pytest.fixture
def unit_spies(monkeypatch):
    """Sane-by-default fakes for the three units; spies record what ran."""
    import app.rag.rescan as rescan_mod
    import app.rag.reindex as reindex_mod

    ran = {"rescan": 0, "reindex": 0, "cleanup": 0}
    report = {"value": _FakeReport(added=2, unchanged=100)}

    monkeypatch.setattr(idle_tasks, "_sandbox_reachable", lambda: True)
    monkeypatch.setattr(idle_tasks, "repo_map_fingerprint", lambda params: "host-fp-1")

    def fake_rescan(db):
        ran["rescan"] += 1
        return report["value"]

    def fake_reindex(db, project_id=0):
        ran["reindex"] += 1
        return {"descriptors_generated": 1, "embeddings_created": 1, "errors": 0}

    def fake_cleanup(db):
        ran["cleanup"] += 1
        return 0

    monkeypatch.setattr(rescan_mod, "rescan_codebase", fake_rescan)
    monkeypatch.setattr(reindex_mod, "reindex_unembedded", fake_reindex)
    monkeypatch.setattr(reindex_mod, "cleanup_orphaned_embeddings", fake_cleanup)
    return ran, report


async def _run_repo_map(session_factory, monkeypatch):
    register_task_handler(
        idle_tasks.REPO_MAP_TASK,
        idle_tasks.repo_map_handler,
        fingerprint_fn=idle_tasks.repo_map_fingerprint,
    )
    monkeypatch.setenv("IDLE_MINUTES", "0")
    db = session_factory()
    ledger.enqueue(db, idle_tasks.REPO_MAP_TASK, {}, dedupe=False)
    db.close()
    gov = IdleGovernor(session_factory=session_factory)
    await gov.run_pending(max_tasks=1)
    db = session_factory()
    row = (
        db.query(IdleTaskRecord)
        .filter(IdleTaskRecord.task_type == idle_tasks.REPO_MAP_TASK)
        .order_by(IdleTaskRecord.id.desc())
        .first()
    )
    db.expunge(row)
    db.close()
    return row


@pytest.mark.asyncio
async def test_sandbox_down_fails_cleanly_without_touching_anything(session_factory, unit_spies, monkeypatch):
    ran, _ = unit_spies
    monkeypatch.setattr(idle_tasks, "_sandbox_reachable", lambda: False)

    row = await _run_repo_map(session_factory, monkeypatch)
    assert row.status == "failed"
    assert "sandbox unreachable" in (row.error or "")
    assert row.input_fingerprint is None  # recovery run must not be fingerprint-skipped
    assert ran == {"rescan": 0, "reindex": 0, "cleanup": 0}


@pytest.mark.asyncio
async def test_mass_deletion_tripwire_stops_before_cleanup(session_factory, unit_spies, monkeypatch):
    ran, report = unit_spies
    report["value"] = _FakeReport(deleted=1403, unchanged=0)  # tonight's incident shape

    row = await _run_repo_map(session_factory, monkeypatch)
    assert row.status == "failed"
    assert "walk failure" in (row.error or "")
    assert row.input_fingerprint is None
    assert ran["rescan"] == 1
    assert ran["reindex"] == 0 and ran["cleanup"] == 0  # vectors survive


@pytest.mark.asyncio
async def test_sane_delta_passes_all_units(session_factory, unit_spies, monkeypatch):
    ran, _ = unit_spies
    row = await _run_repo_map(session_factory, monkeypatch)
    assert row.status == "completed"
    assert row.input_fingerprint == "host-fp-1"
    assert ran == {"rescan": 1, "reindex": 1, "cleanup": 1}


@pytest.mark.asyncio
async def test_poisoned_resume_progress_also_trips(session_factory, unit_spies, monkeypatch):
    """A paused task whose stored rescan stats show mass deletion must not
    resume into reindex/cleanup."""
    ran, _ = unit_spies
    register_task_handler(idle_tasks.REPO_MAP_TASK, idle_tasks.repo_map_handler)
    db = session_factory()
    rec = ledger.enqueue(db, idle_tasks.REPO_MAP_TASK, {}, dedupe=False)
    ledger.save_progress(db, rec, {
        "units_done": ["rescan"],
        "stats": {"rescan": {"added": 0, "modified": 0, "deleted": 1403, "unchanged": 0}},
    })
    rec.status = "paused"
    db.commit()
    db.close()

    monkeypatch.setenv("IDLE_MINUTES", "0")
    gov = IdleGovernor(session_factory=session_factory)
    await gov.run_pending(max_tasks=1)

    db = session_factory()
    row = db.query(IdleTaskRecord).filter(IdleTaskRecord.task_type == idle_tasks.REPO_MAP_TASK).one()
    assert row.status == "failed"
    assert ran["reindex"] == 0 and ran["cleanup"] == 0
    db.close()


def test_failed_run_imposes_retry_cooldown(session_factory, monkeypatch):
    monkeypatch.setenv("ASTRA_IDLE_RETRY_COOLDOWN_MINUTES", "60")
    db = session_factory()
    key = ledger.task_key("repo_map", {})

    rec = ledger.enqueue(db, "repo_map", {})
    ledger.mark_running(db, rec)
    ledger.mark_failed(db, rec, "sandbox unreachable")

    assert ledger.is_due(db, key, cadence_hours=24) is False  # cooling down

    rec.completed_at = datetime.utcnow() - timedelta(minutes=61)
    db.commit()
    assert ledger.is_due(db, key, cadence_hours=24) is True  # failure never satisfies cadence

    # A completed run inside cadence still wins over everything.
    rec2 = ledger.enqueue(db, "repo_map", {})
    ledger.mark_running(db, rec2)
    ledger.mark_completed(db, rec2, summary="ok")
    assert ledger.is_due(db, key, cadence_hours=24) is False
    db.close()
