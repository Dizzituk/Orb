# FILE: tests/test_sentinel_baseline.py
# Purpose: Units for app/sentinel/baseline.py — observe/trust/maintenance and the
#          14-calendar-day learn-mode date logic. In-memory SQLite, no live agent.
# Called-by: pytest
# Depends-on: app.sentinel.baseline, app.sentinel.models
# Last-renovated: 2026-06-12
from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
from app.sentinel import baseline
from app.sentinel.models import (
    SentinelAlert,
    SentinelBaseline,
    SentinelConnection,
    SentinelState,
)

_SENTINEL_TABLES = [
    SentinelBaseline.__table__,
    SentinelConnection.__table__,
    SentinelAlert.__table__,
    SentinelState.__table__,
]


@pytest.fixture()
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_SENTINEL_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


# ── observe / query ──────────────────────────────────────────────────────────

def test_observe_creates_then_increments(db):
    row, created = baseline.observe(db, "app.exe", "example.com", 443, country="US")
    assert created and row.hit_count == 1
    row2, created2 = baseline.observe(db, "app.exe", "example.com", 443)
    assert not created2 and row2.id == row.id and row2.hit_count == 2


def test_process_known_and_port_country_sets(db):
    assert not baseline.process_known(db, "app.exe")
    baseline.observe(db, "app.exe", "example.com", 443, country="US")
    baseline.observe(db, "app.exe", "10.0.0.5", 8080)
    assert baseline.process_known(db, "app.exe")
    assert baseline.known_ports_for_process(db, "app.exe") == {443, 8080}
    assert baseline.known_countries_for_process(db, "app.exe") == {"US"}


def test_listen_rows_do_not_make_process_known(db):
    baseline.observe(db, "srv.exe", "", 9000, kind="listen")
    assert not baseline.process_known(db, "srv.exe")


# ── trust ────────────────────────────────────────────────────────────────────

def test_trust_process_covers_everything(db):
    baseline.trust_process(db, "chrome.exe")
    assert baseline.is_trusted(db, "chrome.exe")
    assert baseline.is_trusted(db, "chrome.exe", "anything.com", 1234)


def test_trust_pair_is_exact(db):
    baseline.observe(db, "app.exe", "example.com", 443)
    baseline.trust_pair(db, "app.exe", "example.com", 443)
    assert baseline.is_trusted(db, "app.exe", "example.com", 443)
    assert not baseline.is_trusted(db, "app.exe", "example.com", 80)
    assert not baseline.is_trusted(db, "app.exe")


# ── maintenance ──────────────────────────────────────────────────────────────

def test_maintenance_prunes_only_stale_untrusted(db):
    old = datetime.utcnow() - timedelta(days=baseline.STALE_BASELINE_DAYS + 5)
    stale, _ = baseline.observe(db, "old.exe", "gone.com", 443)
    stale.last_seen = old
    trusted_stale, _ = baseline.observe(db, "keep.exe", "keep.com", 443)
    trusted_stale.last_seen = old
    trusted_stale.trusted = True
    fresh, _ = baseline.observe(db, "new.exe", "fresh.com", 443)
    db.commit()
    removed = baseline.maintenance(db)
    assert removed == 1
    names = {r.process_name for r in db.query(SentinelBaseline).all()}
    assert names == {"keep.exe", "new.exe"}


# ── learn mode: 14 CALENDAR days, wall-clock dates ───────────────────────────

def test_learn_mode_stamps_once(db):
    first = baseline.ensure_learn_mode_started(db)
    again = baseline.ensure_learn_mode_started(db)
    assert first == again
    assert baseline.learn_mode_started_at(db) == first


def test_learn_mode_day_zero_full_window(db, monkeypatch):
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 12))
    baseline.ensure_learn_mode_started(db)
    assert baseline.learn_mode_days_remaining(db) == baseline.LEARN_MODE_DAYS
    assert baseline.learn_mode_active(db)


def test_learn_mode_last_day_still_active(db, monkeypatch):
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 12))
    baseline.ensure_learn_mode_started(db)
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 25))  # day 13
    assert baseline.learn_mode_days_remaining(db) == 1
    assert baseline.learn_mode_active(db)


def test_learn_mode_ends_on_day_14(db, monkeypatch):
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 12))
    baseline.ensure_learn_mode_started(db)
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 26))  # day 14
    assert baseline.learn_mode_days_remaining(db) == 0
    assert not baseline.learn_mode_active(db)


def test_learn_mode_calendar_not_runtime(db, monkeypatch):
    """The PC turning off doesn't pause the clock — a stamp 20 days ago is over."""
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 12))
    baseline.set_state(db, "learn_mode_started_at", "2026-05-23")
    assert not baseline.learn_mode_active(db)


def test_learn_mode_bad_stamp_restamps(db, monkeypatch):
    monkeypatch.setattr(baseline, "_today_local", lambda: date(2026, 6, 12))
    baseline.set_state(db, "learn_mode_started_at", "not-a-date")
    started = baseline.ensure_learn_mode_started(db)
    assert started == date(2026, 6, 12)


# ── state helpers ────────────────────────────────────────────────────────────

def test_state_roundtrip(db):
    assert baseline.get_state(db, "k") is None
    baseline.set_state(db, "k", "v1")
    assert baseline.get_state(db, "k") == "v1"
    baseline.set_state(db, "k", "v2")
    assert baseline.get_state(db, "k") == "v2"
