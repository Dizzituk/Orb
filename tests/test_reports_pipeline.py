# FILE: tests/test_reports_pipeline.py
# Purpose: WS4 — report render from seeded ledger, cache sweep, document-kind path safety, marker emission.
# Called-by: pytest
# Depends-on: app.reports.*, app.bridge.artifacts, app.llm.turn_surface
# Last-renovated: 2026-07-01

import os
import time
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.idle.models import IdleTaskRecord
from app.watchers.models import WatcherReading
from app.watchers import framework
from app.watchers.framework import Reading, WatcherSpec


@pytest.fixture
def session_factory():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Explicit tables only: a full create_all trips the known fresh-DB
    # NoReferencedTableError (messages.session_id FK) once unrelated model
    # modules have been imported by earlier tests in the session.
    Base.metadata.create_all(bind=engine, tables=[WatcherReading.__table__, IdleTaskRecord.__table__])
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    yield maker
    engine.dispose()


@pytest.fixture
def fake_spec():
    async def observe(key_cfg):  # pragma: no cover — not observed in these tests
        return Reading(key=key_cfg["key"], value=1.0)

    spec = WatcherSpec(
        watcher_id="fake_land",
        title="Fake land prices",
        unit="eur_per_m2",
        keys=[{"key": "central"}, {"key": "coastal"}],
        observe_key=observe,
        tool_name="get_fake_land",
        tool_description="test",
    )
    framework._WATCHERS[spec.watcher_id] = spec
    yield spec
    framework._WATCHERS.pop(spec.watcher_id, None)


@pytest.fixture
def reports_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_CACHE_DIR", str(tmp_path))
    return tmp_path


def _seed(db, days_values):
    for offset, central, coastal in days_values:
        day = (datetime.utcnow() - timedelta(days=offset)).strftime("%Y-%m-%d")
        framework.record_reading(db, "fake_land", "eur_per_m2", Reading(key="central", value=central, source="test"), day=day)
        framework.record_reading(db, "fake_land", "eur_per_m2", Reading(key="coastal", value=coastal, source="test"), day=day)


# ── render from a seeded ledger ─────────────────────────────────────────────


def test_render_watcher_report_from_seeded_ledger(session_factory, fake_spec, reports_dir):
    from app.reports.renderer import render_watcher_report

    db = session_factory()
    _seed(db, [(2, 10.0, 20.0), (1, 11.0, 21.0), (0, 12.0, 22.0)])
    report = render_watcher_report(db, "fake_land", days=30)
    db.close()

    assert report is not None
    assert os.path.isfile(report["path"])
    assert report["filename"].startswith("report-fake-land-") and report["filename"].endswith(".html")

    html_text = open(report["path"], encoding="utf-8").read()
    assert "Fake land prices" in html_text
    assert f"Data through {report['data_through']}" in html_text
    assert report["data_through"] == datetime.utcnow().strftime("%Y-%m-%d")
    # chart embedded when plotly is available, honest note otherwise
    assert ("data:image/png;base64," in html_text) or ("chart unavailable" in html_text)
    # dated table content
    assert "central" in html_text and "coastal" in html_text and "22.0" in html_text
    # self-contained: no local filesystem references
    assert "D:\\" not in html_text


def test_render_unknown_watcher_returns_none(session_factory, reports_dir):
    from app.reports.renderer import render_watcher_report

    db = session_factory()
    assert render_watcher_report(db, "nope") is None
    db.close()


# ── cache sweep ─────────────────────────────────────────────────────────────


def test_sweep_removes_only_expired_unkept_reports(reports_dir, monkeypatch):
    from app.reports.cache import sweep_expired

    monkeypatch.setenv("REPORTS_TTL_HOURS", "1")
    old_a = reports_dir / "report-old-a.html"
    old_b = reports_dir / "report-old-b.html"
    fresh = reports_dir / "report-fresh.html"
    for f in (old_a, old_b, fresh):
        f.write_text("<html></html>", encoding="utf-8")
    stale = time.time() - 7200
    os.utime(old_a, (stale, stale))
    os.utime(old_b, (stale, stale))

    removed = sweep_expired(keep_filenames={old_b.name})  # old_b is "open in a window"
    assert removed == 1
    assert not old_a.exists()
    assert old_b.exists() and fresh.exists()


# ── document-kind artifact path safety ──────────────────────────────────────


def test_document_artifact_resolution_and_traversal_safety(reports_dir):
    from app.bridge.artifacts import resolve_artifact_path

    good = reports_dir / "report-fake-20260701-101010.html"
    good.write_text("<html></html>", encoding="utf-8")

    resolved = resolve_artifact_path("document", good.name)
    assert resolved is not None and resolved.name == good.name

    assert resolve_artifact_path("document", "../evil.html") is None      # charset
    assert resolve_artifact_path("document", "..") is None                # not a file
    assert resolve_artifact_path("document", "missing.html") is None      # absent
    assert resolve_artifact_path("audio", good.name) is None              # unwired kind stays 404


# ── surface-aware delivery + marker emission ────────────────────────────────


@pytest.mark.asyncio
async def test_bridge_turn_registers_document_artifact_marker(reports_dir):
    from app.bridge.artifacts import ARTIFACT_MARKER_RE
    from app.llm.turn_surface import begin_turn, drain_turn_artifacts, format_artifact_markers
    from app.reports.surface import deliver_report

    (reports_dir / "report-x-1.html").write_text("<html></html>", encoding="utf-8")

    begin_turn("bridge")
    out = await deliver_report("report-x-1.html", "Fake land prices")
    assert out["ok"] is True and out["delivered"] == "document_artifact"

    pending = drain_turn_artifacts()
    assert pending == [("document", "report-x-1.html")]
    marker = format_artifact_markers(pending)
    m = ARTIFACT_MARKER_RE.fullmatch(marker)
    assert m and m.group("kind") == "document" and m.group("filename") == "report-x-1.html"
    # drained means drained — the next turn starts clean
    assert drain_turn_artifacts() == []


@pytest.mark.asyncio
async def test_desktop_turn_opens_reports_window(reports_dir, monkeypatch):
    import app.media.display_client as dc
    from app.llm.turn_surface import begin_turn, drain_turn_artifacts
    from app.reports.surface import deliver_report

    opened = []

    async def fake_open(url, display="main", **kwargs):
        opened.append((url, display))
        return {"ok": True, "result": {}}

    async def fake_list():
        return {"ok": True, "result": {"windows": []}}

    monkeypatch.setattr(dc, "open_on_display", fake_open)
    monkeypatch.setattr(dc, "list_open_windows", fake_list)

    begin_turn("desktop")
    out = await deliver_report("report-y-1.html", "Fake land prices")
    assert out["ok"] is True and out["delivered"] == "reports_window"
    assert out["opened_on"] == "reports"
    assert opened[0][1] == "reports"
    assert opened[0][0].endswith("/output/reports/report-y-1.html")
    assert drain_turn_artifacts() == []  # desktop path ships no artifact


@pytest.mark.asyncio
async def test_show_report_tool_end_to_end_bridge(session_factory, fake_spec, reports_dir, monkeypatch):
    import app.db as app_db
    from app.llm.turn_surface import begin_turn, drain_turn_artifacts
    from app.reports.tools_registration import show_watcher_report_handler

    db = session_factory()
    _seed(db, [(1, 10.0, 20.0), (0, 12.0, 22.0)])
    db.close()

    monkeypatch.setattr(app_db, "get_db_session", lambda: session_factory())
    begin_turn("bridge")
    out = await show_watcher_report_handler({"watcher": "fake_land", "days": 14})
    assert out["ok"] is True
    assert out["surface"] == "bridge"
    assert out["data_through"] == datetime.utcnow().strftime("%Y-%m-%d")
    kinds = [k for k, _ in drain_turn_artifacts()]
    assert kinds == ["document"]


@pytest.mark.asyncio
async def test_show_report_tool_unknown_watcher_lists_known(reports_dir, monkeypatch, session_factory):
    import app.db as app_db
    from app.reports.tools_registration import show_watcher_report_handler

    monkeypatch.setattr(app_db, "get_db_session", lambda: session_factory())
    out = await show_watcher_report_handler({"watcher": "definitely_not_real"})
    assert out["ok"] is False
    assert "known_watchers" in out
