# FILE: tests/test_watchers.py
# Purpose: WS3 — watcher framework: fake watcher run, ledger append/read, tool auto-registration, threshold flag.
# Called-by: pytest
# Depends-on: app.watchers.*
# Last-renovated: 2026-07-01

from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.idle.models import IdleTaskRecord
from app.idle import ledger
from app.idle.router import TaskContext
from app.watchers import framework
from app.watchers.framework import Reading, WatcherSpec
from app.watchers.models import WatcherReading
from app.watchers.price_extract import extract_prices, median


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Explicit tables only — dodges the known fresh-DB NoReferencedTableError
    # (messages.session_id FK) under cross-test import pollution.
    Base.metadata.create_all(bind=engine, tables=[WatcherReading.__table__, IdleTaskRecord.__table__])
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    yield maker
    engine.dispose()


@pytest.fixture
def fake_spec():
    async def observe(key_cfg):
        return Reading(key=key_cfg["key"], value=key_cfg.get("fake_value", 100.0), source="test", note="fake")

    spec = WatcherSpec(
        watcher_id="fake_watch",
        title="Fake watcher",
        unit="gbp",
        keys=[{"key": "alpha", "fake_value": 100.0}, {"key": "beta", "fake_value": 250.0}],
        observe_key=observe,
        tool_name="get_fake_watch",
        tool_description="test watcher",
    )
    framework._WATCHERS[spec.watcher_id] = spec
    yield spec
    framework._WATCHERS.pop(spec.watcher_id, None)


def _ctx(db, session_factory, params, should_yield=lambda: False):
    # dedupe=False: tests build several contexts for the same task_key
    rec = ledger.enqueue(db, "watcher_observe", params, dedupe=False)
    return TaskContext(
        db=db, record=rec, params=params, fingerprint=None,
        should_yield=should_yield, session_factory=session_factory,
    )


# ── price extraction ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_snippet_observation_decodes_html_entities_and_tags(monkeypatch):
    """Brave/DDG snippets carry &#163; entities and <strong> tags around
    matched terms — the observer must decode before extracting (2026-07-02,
    found on the first live hardware scrape)."""
    import app.tools.registry as tools_registry
    from app.watchers.snippet_source import observe_via_search

    async def fake_search(input_data, context=None):
        return {
            "provider": "brave",
            "results": [
                {"title": "PSU deal", "url": "http://shop.example/psu",
                 "description": "now &#163;1,999.99 was <strong>£180</strong> more"},
            ],
        }

    monkeypatch.setattr(tools_registry, "web_search_handler", fake_search)
    reading = await observe_via_search("psu", "psu price", "£", lo=1, hi=5000)
    # median of [1999.99, 180.0]
    assert reading.value == pytest.approx(1089.99, abs=0.01)
    assert "2 prices" in reading.note


def test_price_extraction_handles_real_world_formats():
    text = "Terreno à venda: €450.000 ou 1.250,50 € por m2. GPU deal £1,999.99, PSU £180, RAM 2 300 €"
    assert extract_prices(text, "€") == [450000.0, 1250.50, 2300.0]
    assert extract_prices(text, "£") == [1999.99, 180.0]
    assert extract_prices(text, "€", lo=1, hi=2000) == [1250.50]
    assert median([3.0, 1.0, 2.0]) == 2.0
    assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert median([]) is None


# ── observe -> record (fake watcher through the real idle handler) ─────────


@pytest.mark.asyncio
async def test_fake_watcher_records_one_row_per_key_per_day(session_factory, fake_spec):
    db = session_factory()
    outcome = await framework._observe_handler(_ctx(db, session_factory, {"watcher_id": "fake_watch"}))
    assert outcome.status == "completed"
    assert "2 reading(s)" in outcome.summary

    rows = db.query(WatcherReading).order_by(WatcherReading.key).all()
    assert [(r.key, r.value, r.unit) for r in rows] == [("alpha", 100.0, "gbp"), ("beta", 250.0, "gbp")]

    # Same day again: idempotent, nothing appended.
    outcome2 = await framework._observe_handler(_ctx(db, session_factory, {"watcher_id": "fake_watch"}))
    assert outcome2.status == "completed"
    assert "0 reading(s)" in outcome2.summary
    assert db.query(WatcherReading).count() == 2
    db.close()


@pytest.mark.asyncio
async def test_watcher_checkpoints_between_keys(session_factory, fake_spec):
    db = session_factory()
    yields = {"count": 0}

    def should_yield():
        yields["count"] += 1
        return yields["count"] > 1  # allow first key, yield before second

    ctx = _ctx(db, session_factory, {"watcher_id": "fake_watch"}, should_yield=should_yield)
    outcome = await framework._observe_handler(ctx)
    assert outcome.status == "paused"
    assert db.query(WatcherReading).count() == 1

    # Resume with no more activity: finishes the remaining key only.
    ctx2 = TaskContext(
        db=db, record=ctx.record, params=ctx.params, fingerprint=None,
        should_yield=lambda: False, session_factory=session_factory,
    )
    outcome2 = await framework._observe_handler(ctx2)
    assert outcome2.status == "completed"
    assert db.query(WatcherReading).count() == 2
    db.close()


# ── ledger read helpers ─────────────────────────────────────────────────────


def test_ledger_append_and_series_read(session_factory, fake_spec):
    db = session_factory()
    for offset, val in ((2, 90.0), (1, 95.0), (0, 100.0)):
        day = (datetime.utcnow() - timedelta(days=offset)).strftime("%Y-%m-%d")
        assert framework.record_reading(db, "fake_watch", "gbp", Reading(key="alpha", value=val), day=day)
    payload = framework._series_payload(db, fake_spec, days=30)
    assert payload["latest"]["alpha"]["value"] == 100.0
    assert payload["latest"]["alpha"]["change_pct_over_window"] == pytest.approx(11.1, abs=0.1)
    assert [p["value"] for p in payload["series"]["alpha"]] == [90.0, 95.0, 100.0]
    db.close()


# ── threshold alert flag ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_move_beyond_threshold_flags_alert(session_factory, fake_spec, monkeypatch):
    monkeypatch.setenv("ASTRA_WATCHER_ALERT_PCT", "10")
    emitted = []
    monkeypatch.setattr(framework, "_emit_alert", lambda db, spec, key, prev, new, pct: emitted.append((key, prev, new, round(pct, 1))))

    db = session_factory()
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    framework.record_reading(db, "fake_watch", "gbp", Reading(key="alpha", value=100.0), day=yesterday)
    framework.record_reading(db, "fake_watch", "gbp", Reading(key="beta", value=200.0), day=yesterday)

    today = datetime.utcnow().strftime("%Y-%m-%d")
    # +20% -> flagged
    assert framework.check_move_and_alert(db, fake_spec, Reading(key="alpha", value=120.0), today) == pytest.approx(20.0)
    # -5% -> below threshold, not flagged
    assert framework.check_move_and_alert(db, fake_spec, Reading(key="beta", value=190.0), today) is None
    assert emitted == [("alpha", 100.0, 120.0, 20.0)]
    db.close()


# ── tool auto-registration ──────────────────────────────────────────────────


def test_watcher_tools_auto_register(fake_spec):
    from app.tools.registry import get_tool

    framework.register_watcher_tools()
    # The fake spec's tool AND the shipped instances' tools exist.
    assert get_tool("get_fake_watch", "v1") is not None
    assert get_tool("get_portugal_land_prices", "v1") is not None
    assert get_tool("get_hardware_prices", "v1") is not None


@pytest.mark.asyncio
async def test_watcher_tool_reads_seeded_ledger(session_factory, fake_spec, monkeypatch):
    import app.db as app_db

    db = session_factory()
    framework.record_reading(db, "fake_watch", "gbp", Reading(key="alpha", value=100.0, source="test"))
    db.close()

    monkeypatch.setattr(app_db, "get_db_session", lambda: session_factory())
    handler = framework._make_tool_handler(fake_spec)
    out = await handler({"days": 7})
    assert out["ok"] is True
    assert out["latest"]["alpha"]["value"] == 100.0
    assert "trend_summary" not in out  # summarise not requested
