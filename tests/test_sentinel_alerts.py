# FILE: tests/test_sentinel_alerts.py
# Purpose: Units for app/sentinel/alerts.py — create/dedupe/ack lifecycle, learn-mode
#          suppression visibility, active-file mirror + injection cooldown (nudges pattern).
# Called-by: pytest
# Depends-on: app.sentinel.alerts, app.sentinel.models
# Last-renovated: 2026-06-12
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.bridge import sentinel_alerts as bridge_sentinel_alerts
from app.db import Base
from app.sentinel import alerts
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
def db(tmp_path, monkeypatch):
    # Redirect the active-alerts mirror into the test sandbox.
    monkeypatch.setattr(alerts, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(alerts, "_ACTIVE_FILE", tmp_path / "alerts_active.json")
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine, tables=_SENTINEL_TABLES)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _make(db, severity="medium", rule_key="raw_ip", process="app.exe",
          remote="1.2.3.4:443", suppressed=False, title=None):
    return alerts.create_alert(
        db,
        severity=severity,
        rule_key=rule_key,
        title=title or f"{rule_key} on {process}",
        explanation="test explanation",
        process=process,
        remote=remote,
        suppressed=suppressed,
    )


# ── create / dedupe ──────────────────────────────────────────────────────────

def test_create_and_mirror(db):
    alert = _make(db, severity="severe")
    assert alert is not None and alert.id
    active = alerts._load_active()
    assert [i["id"] for i in active["items"]] == [alert.id]


def test_dedupe_same_signature_within_window(db):
    assert _make(db) is not None
    assert _make(db) is None  # identical rule/process/remote inside 6h window
    assert db.query(SentinelAlert).count() == 1


def test_dedupe_expires_after_window(db):
    first = _make(db)
    first.created_at = datetime.utcnow() - timedelta(hours=alerts._DEDUPE_WINDOW_HOURS + 1)
    db.commit()
    assert _make(db) is not None


def test_different_remote_not_deduped(db):
    assert _make(db, remote="1.2.3.4:443") is not None
    assert _make(db, remote="5.6.7.8:443") is not None


# ── suppression (learn mode) ─────────────────────────────────────────────────

def test_suppressed_hidden_from_default_views(db):
    _make(db, suppressed=True)
    assert alerts.list_alerts(db) == []
    assert alerts.list_alerts(db, include_suppressed=True) != []
    active = alerts._load_active()
    assert active == {} or active.get("items") in ([], None)
    counts = alerts.alert_counts(db)
    assert counts["suppressed"] == 1 and counts["unacked"] == 0


# ── severity floor (P0-1: phone feed noise fix) ──────────────────────────────

def test_min_severity_none_returns_all(db):
    _make(db, severity="low", rule_key="raw_ip")
    _make(db, severity="severe", rule_key="new_process", remote="9.9.9.9:1")
    rows = alerts.list_alerts(db)
    assert {a.severity for a in rows} == {"low", "severe"}


def test_min_severity_medium_excludes_low(db):
    _make(db, severity="low", rule_key="raw_ip")
    _make(db, severity="medium", rule_key="new_listen", remote="listen:9000")
    _make(db, severity="severe", rule_key="new_process", remote="9.9.9.9:1")
    rows = alerts.list_alerts(db, min_severity="medium")
    assert {a.severity for a in rows} == {"medium", "severe"}


def test_min_severity_severe_excludes_medium_and_low(db):
    _make(db, severity="low", rule_key="raw_ip")
    _make(db, severity="medium", rule_key="new_listen", remote="listen:9000")
    _make(db, severity="severe", rule_key="new_process", remote="9.9.9.9:1")
    rows = alerts.list_alerts(db, min_severity="severe")
    assert {a.severity for a in rows} == {"severe"}


def test_phone_feed_default_floor_drops_low(db, monkeypatch):
    """GET /bridge/sentinel-alerts must never surface a `low` row (acceptance test)."""
    monkeypatch.setattr(bridge_sentinel_alerts, "_PHONE_MIN_SEVERITY", "medium")
    _make(db, severity="low", rule_key="raw_ip")
    _make(db, severity="medium", rule_key="new_listen", remote="listen:9000")
    rows = asyncio.run(bridge_sentinel_alerts.get_sentinel_alerts(db=db, _auth=True))
    severities = {r.severity for r in rows}
    assert "low" not in severities
    assert "medium" in severities


# ── ack / action lifecycle ───────────────────────────────────────────────────

def test_ack_removes_from_active_file(db):
    alert = _make(db)
    assert alerts._load_active()["items"]
    alerts.ack_alert(db, alert.id)
    assert alerts._load_active()["items"] == []
    assert alerts.get_alert(db, alert.id).acknowledged


def test_ack_unknown_id_returns_none(db):
    assert alerts.ack_alert(db, 9999) is None


def test_mark_action_for_remote(db):
    _make(db, remote="9.9.9.9:4444")
    _make(db, rule_key="new_process", remote="9.9.9.9:53")
    _make(db, rule_key="raw_ip", process="other.exe", remote="1.1.1.1:80")
    touched = alerts.mark_action_for_remote(db, "9.9.9.9", "blocked 9.9.9.9")
    assert touched == 2
    untouched = [a for a in alerts.list_alerts(db) if a.remote.startswith("1.1.1.1")]
    assert untouched and untouched[0].action_taken is None


# ── injection (nudges pattern) ───────────────────────────────────────────────

def test_injection_prefers_severe_and_stamps_cooldown(db):
    _make(db, severity="low", rule_key="unusual_port", remote="2.2.2.2:9999")
    severe = _make(db, severity="severe", rule_key="new_listen", remote="listen:81",
                   title="New listening port 81")
    text = alerts.get_alert_for_injection()
    assert text is not None and "SEVERE" in text and str(severe.id) in text
    assert "+1 more" in text
    # cooldown: immediate second read yields nothing
    assert alerts.get_alert_for_injection() is None


def test_injection_after_cooldown(db):
    _make(db, severity="medium")
    assert alerts.get_alert_for_injection() is not None
    data = alerts._load_active()
    stale = datetime.now().astimezone() - timedelta(hours=alerts._INJECTION_COOLDOWN_HOURS + 1)
    data["last_injected_at"] = stale.isoformat()
    alerts._ACTIVE_FILE.write_text(__import__("json").dumps(data), encoding="utf-8")
    assert alerts.get_alert_for_injection() is not None


def test_injection_empty_when_no_alerts(db):
    alerts.refresh_active_file(db)
    assert alerts.get_alert_for_injection() is None
