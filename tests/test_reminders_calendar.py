# FILE: tests/test_reminders_calendar.py
# Purpose: Calendar/reminders 2026-07-03 batch — delivered-once semantics,
#          absolute-date parsing, announce fallback, migration backfill, and
#          wiring guards for the injection/capture exemptions.
# Called-by: pytest
# Depends-on: app.reminders.*, app.db, app.invocation.classifier, app.self_model.fragments.capture
# Last-renovated: 2026-07-03
from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect as sa_inspect, text
from sqlalchemy.orm import sessionmaker

from app.reminders.models import Reminder
from app.reminders import service
from app.reminders.time_parse import parse_when


# ── helpers ──────────────────────────────────────────────────────────────────

NOW = datetime(2026, 7, 3, 20, 0).astimezone()  # Fri 3 Jul 2026, 20:00 local


@pytest.fixture()
def db():
    """Reminders-only in-memory DB. Deliberately avoids Base.metadata.create_all
    (full fresh-DB bootstrap is a known broken path — NoReferencedTableError)."""
    engine = create_engine("sqlite:///:memory:")
    Reminder.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _mk(db, text_="thing", minutes=-5, fired=False, acked=False, delivered=False):
    r = service.create(db, text=text_, due_at=datetime.now(timezone.utc) + timedelta(minutes=minutes))
    if fired:
        service.mark_fired(db, r.id)
    if acked:
        service.ack(db, r.id)
    if delivered:
        service.mark_delivered(db, r.id)
    return r


# ── absolute-date parsing (the calendar phrasing bug) ────────────────────────

@pytest.mark.parametrize("when,expect", [
    ("on the 17th of December", "2026-12-17 09:00"),
    ("17th december 4:15pm", "2026-12-17 16:15"),
    ("dec 17 4.30pm", "2026-12-17 16:30"),
    ("december 17th 2027", "2027-12-17 09:00"),
    ("17/12", "2026-12-17 09:00"),
    ("17/12/2027 8am", "2027-12-17 08:00"),
    ("on the 23rd", "2026-07-23 09:00"),
    ("on the 2nd", "2026-08-02 09:00"),          # day passed → next month
    ("on the 1st of March", "2027-03-01 09:00"),  # date passed → next year
])
def test_absolute_dates(when, expect):
    got = parse_when(when, now=NOW)
    assert got is not None, when
    assert got.strftime("%Y-%m-%d %H:%M") == expect


@pytest.mark.parametrize("when,expect", [
    ("in 20 minutes", (NOW + timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M")),
    ("tomorrow 9am", "2026-07-04 09:00"),
    ("friday 8am", "2026-07-10 08:00"),
    ("3pm", "2026-07-04 15:00"),
    ("4.30pm", "2026-07-04 16:30"),  # dot clock time (new)
])
def test_existing_phrasings_unbroken(when, expect):
    got = parse_when(when, now=NOW)
    assert got is not None and got.strftime("%Y-%m-%d %H:%M") == expect


@pytest.mark.parametrize("when,expect", [
    ("in two minutes", (NOW + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")),
    ("in twenty five minutes", (NOW + timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M")),
    ("in twenty-five minutes", (NOW + timedelta(minutes=25)).strftime("%Y-%m-%d %H:%M")),
    ("in an hour", (NOW + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")),
    ("in half an hour", (NOW + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M")),
    ("in an hour and a half", (NOW + timedelta(minutes=90)).strftime("%Y-%m-%d %H:%M")),
    ("in a couple of minutes", (NOW + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")),
    ("in a few minutes", (NOW + timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M")),
    ("in thirty seconds", (NOW + timedelta(seconds=30)).strftime("%Y-%m-%d %H:%M")),
    ("at ten", "2026-07-04 10:00"),                      # 20:00 now → tomorrow
    ("tomorrow at seven", "2026-07-04 07:00"),
    ("the seventeenth of december", "2026-12-17 09:00"),
    ("december the twenty second", "2026-12-22 09:00"),
    ("on the twenty-third", "2026-07-23 09:00"),
    ("four fifteen pm tomorrow", "2026-07-04 16:15"),
])
def test_spoken_number_words(when, expect):
    """Voice transcription hands the parser WORDS — the exact live failure was
    'in two minutes' being bounced back as unparseable (2026-07-03 20:11)."""
    got = parse_when(when, now=NOW)
    assert got is not None, when
    assert got.strftime("%Y-%m-%d %H:%M") == expect


def test_create_reminder_parse_error_instructs_self_heal():
    """On a genuinely unparseable `when`, the tool error must tell the model to
    retry with due_at_iso — never to ask the user to rephrase to a clock time."""
    from app.tools.reminder_tools import create_reminder_handler
    out = asyncio.run(create_reminder_handler({"text": "x", "when": "gibberish blorp"}, None))
    assert out["ok"] is False
    assert "due_at_iso" in out["error"] and "rephrase" in out["error"]


def test_invalid_date_falls_through_to_iso_fallback():
    assert parse_when("on the 31st of February", now=NOW) is None
    got = parse_when("on the 31st of February", model_due_at_iso="2026-12-01T10:00:00", now=NOW)
    assert got is not None and got.strftime("%Y-%m-%d %H:%M") == "2026-12-01 10:00"


# ── delivered-once semantics ─────────────────────────────────────────────────

def test_mark_delivered_idempotent(db):
    r = _mk(db, fired=True)
    first = service.mark_delivered(db, r.id).delivered_at
    again = service.mark_delivered(db, r.id).delivered_at
    assert first is not None and first == again


def test_injection_candidates_exclude_delivered_and_acked(db):
    undelivered = _mk(db, text_="undelivered", fired=True)
    _mk(db, text_="delivered", fired=True, delivered=True)
    _mk(db, text_="acked", fired=True, acked=True)
    _mk(db, text_="unfired", minutes=60)
    ids = [r.id for r in service.list_due_unacked_undelivered(db)]
    assert ids == [undelivered.id]


def test_nag_set_still_includes_delivered_but_unacked(db):
    """Desktop toast + phone catch-up keep nagging until an explicit ack —
    delivery only silences the chat-injection channel."""
    r = _mk(db, fired=True, delivered=True)
    assert [x.id for x in service.list_due_unacked(db)] == [r.id]


def test_list_in_range(db):
    inside = _mk(db, text_="inside", minutes=0)
    _mk(db, text_="way later", minutes=60 * 24 * 40)
    start = datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.now(timezone.utc) + timedelta(days=1)
    got = service.list_in_range(db, start, end)
    assert [r.id for r in got] == [inside.id]


# ── feed contract (chat-injection channel) ───────────────────────────────────

def test_feed_returns_dict_and_respects_cooldown(tmp_path, monkeypatch):
    from app.reminders import feed
    monkeypatch.setattr(feed, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(feed, "_ACTIVE_FILE", tmp_path / "active.json")

    feed.write_active_reminders([{"id": 7, "text": "brush teeth", "due_at": "2026-07-03T09:00:00"}])
    got = feed.get_due_reminder_for_injection()
    assert got == {"id": 7, "text": "brush teeth", "due_at": "2026-07-03T09:00:00"}
    # cooldown: an immediate second read returns nothing
    assert feed.get_due_reminder_for_injection() is None


def test_feed_skips_noop_rewrite(tmp_path, monkeypatch):
    """Scheduler ticks every 10s now — identical content must not touch disk
    (generated_at would change on a real rewrite)."""
    from app.reminders import feed
    monkeypatch.setattr(feed, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(feed, "_ACTIVE_FILE", tmp_path / "active.json")
    feed.write_active_reminders([{"id": 1, "text": "a", "due_at": "x"}])
    first = (tmp_path / "active.json").read_text(encoding="utf-8")
    feed.write_active_reminders([{"id": 1, "text": "a", "due_at": "x"}])
    assert (tmp_path / "active.json").read_text(encoding="utf-8") == first


def test_feed_prune_on_rewrite(tmp_path, monkeypatch):
    """Scheduler writes only undelivered reminders — a rewrite without the
    delivered one must drop it from the snapshot entirely."""
    from app.reminders import feed
    monkeypatch.setattr(feed, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(feed, "_ACTIVE_FILE", tmp_path / "active.json")
    feed.write_active_reminders([{"id": 1, "text": "a", "due_at": "x"}, {"id": 2, "text": "b", "due_at": "y"}])
    feed.write_active_reminders([{"id": 2, "text": "b", "due_at": "y"}])
    assert [i["id"] for i in feed.get_due_reminders()] == [2]


# ── announce line (LLM phrasing with hard fallback) ──────────────────────────

def test_announce_line_falls_back_when_llm_dies(monkeypatch):
    import app.llm.routing.core as core
    from app.reminders.announce import announce_line

    async def boom(message, **kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr(core, "quick_chat_async", boom)
    line = asyncio.run(announce_line("you're amazing"))
    assert line == "Taz — reminder: you're amazing"


def test_announce_line_uses_llm_and_trims(monkeypatch):
    import app.llm.routing.core as core
    from app.reminders.announce import announce_line

    async def fake(message, **kwargs):
        return '  "Taz — quick reminder: you are amazing."  \nsecond line ignored'

    monkeypatch.setattr(core, "quick_chat_async", fake)
    line = asyncio.run(announce_line("you're amazing"))
    assert line == "Taz — quick reminder: you are amazing."


# ── chat-turn delivery (2026-07-03 evening): cache + speak semantics ────────

def test_announce_line_cache_roundtrip(monkeypatch):
    from app.reminders import announce
    t = {"now": 1000.0}
    monkeypatch.setattr(announce.time, "monotonic", lambda: t["now"])
    announce._LINES.clear()
    announce.remember_line(42, "Taz — ice cream time.")
    assert announce.recall_line(42) == "Taz — ice cream time."
    t["now"] += 2 * announce.FRESH_DELIVERY_WINDOW_S + 1
    assert announce.recall_line(42) is None


def test_announce_endpoint_fresh_delivery_speaks_same_line(db):
    """Scheduler delivered seconds ago (fire-time chat turn) → the watcher must
    still chime and speak, with the exact cached line that landed in the chat."""
    from app.reminders import announce
    from app.reminders.router import announce_reminder
    announce._LINES.clear()
    r = _mk(db, text_="ice cream", fired=True, delivered=True)
    announce.remember_line(r.id, "Taz — quick one: ice cream o'clock.")
    out = asyncio.run(announce_reminder(r.id, db))
    assert out.speak is True and out.already_delivered is True
    assert out.spoken_text == "Taz — quick one: ice cream o'clock."


def test_announce_endpoint_stale_delivery_is_silent(db):
    """Desktop was off at fire time; hours later it must show the toast without
    re-blaring — the phone alarm and chat history already told him."""
    from app.reminders import announce
    from app.reminders.router import announce_reminder
    announce._LINES.clear()
    r = _mk(db, text_="old news", fired=True, delivered=True)
    r.delivered_at = datetime.utcnow() - timedelta(minutes=10)
    db.commit()
    out = asyncio.run(announce_reminder(r.id, db))
    assert out.speak is False and out.already_delivered is True


def test_announce_endpoint_undelivered_generates_and_stamps(db, monkeypatch):
    import app.reminders.router as router_mod
    from app.reminders import announce
    announce._LINES.clear()

    async def fake_line(text):
        return f"Taz — reminder: {text} (fake)"

    monkeypatch.setattr(router_mod, "announce_line", fake_line)
    r = _mk(db, text_="say it", fired=True)
    out = asyncio.run(router_mod.announce_reminder(r.id, db))
    assert out.speak is True and out.already_delivered is False
    assert "(fake)" in out.spoken_text
    db.refresh(r)
    assert r.delivered_at is not None


def test_utc_offset_stamped_on_api_output(db):
    """Naive-UTC columns must emit with +00:00 — naive ISO made the desktop
    show times an hour early (BST) and broke the phone's OffsetDateTime.parse,
    so its exact alarms never armed (2026-07-03 evening fix)."""
    from app.reminders.router import _to_out
    from app.bridge.reminders_feed import _to_out as bridge_to_out
    r = _mk(db, minutes=60)
    assert _to_out(r).due_at.endswith("+00:00")
    assert bridge_to_out(r).due_at.endswith("+00:00")


def test_scheduler_delivers_chat_turn():
    import app.reminders.scheduler as sched
    tick_src = inspect.getsource(sched.ReminderScheduler._tick)
    assert "_deliver_chat_turn" in tick_src
    deliver_src = inspect.getsource(sched._deliver_chat_turn)
    for needle in ('provider="reminder"', "create_message", "mark_delivered", "remember_line"):
        assert needle in deliver_src, needle


def test_create_message_skips_memory_for_reminder_provider():
    from app.memory.service import create_message
    src = inspect.getsource(create_message)
    assert '== "reminder"' in src


# ── migration backfill ───────────────────────────────────────────────────────

def test_migration_adds_column_and_backfills(monkeypatch, tmp_path):
    import app.db as db_module

    engine = create_engine(f"sqlite:///{tmp_path / 'mig.db'}")
    with engine.connect() as conn:
        conn.execute(text(
            "CREATE TABLE reminders (id INTEGER PRIMARY KEY, text VARCHAR(500) NOT NULL, "
            "due_at DATETIME NOT NULL, created_at DATETIME NOT NULL, fired_at DATETIME, "
            "acked_at DATETIME, source_device VARCHAR(20), recurrence VARCHAR(40))"
        ))
        conn.execute(text(
            "INSERT INTO reminders (text, due_at, created_at, fired_at) "
            "VALUES ('old fired', '2026-07-03 09:33', '2026-07-03 09:28', '2026-07-03 09:33'),"
            "       ('old unfired', '2026-12-17 09:00', '2026-07-03 09:28', NULL)"
        ))
        conn.commit()

    monkeypatch.setattr(db_module, "engine", engine)
    db_module._migrate_reminders_schema()
    db_module._migrate_reminders_schema()  # idempotent second run

    cols = {c["name"] for c in sa_inspect(engine).get_columns("reminders")}
    assert "delivered_at" in cols
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT text, delivered_at FROM reminders ORDER BY id")).fetchall()
    assert rows[0][1] is not None, "historical fired row must be backfilled as delivered"
    assert rows[1][1] is None, "unfired row must stay undelivered"


# ── wiring guards (cheap source-level checks, same pattern as prior batches) ─

def test_scheduler_writes_only_undelivered():
    from app.reminders.scheduler import ReminderScheduler
    src = inspect.getsource(ReminderScheduler._tick)
    assert "list_due_unacked_undelivered" in src


def test_memory_injection_stamps_delivered():
    import app.llm.routing.memory_injection as mi
    src = inspect.getsource(mi)
    assert "get_due_reminder_for_injection" in src
    assert "mark_delivered" in src


def test_classifier_routes_calendar_phrasing():
    from app.invocation.classifier import classify
    for phrase in ("remind me in 20 minutes", "put that on my calendar", "set a reminder for 4pm"):
        assert classify(phrase).matched_rule == "remind_or_schedule", phrase


def test_fragment_capture_skips_reminder_turns():
    from app.self_model.fragments.capture import capture_fragment, _is_reminder_turn
    assert _is_reminder_turn("Astra can you set a reminder in 5 minutes that I am amazing")
    assert not _is_reminder_turn("I live in Manchester and work as a PT")
    out = capture_fragment("set a reminder for 4pm that the van needs its MOT")
    assert out == {"skipped": "reminder_turn"}


def test_after_user_message_guard_present():
    from app.memory import integration
    src = inspect.getsource(integration.after_user_message)
    assert "_reminder_turn" in src and "remind_or_schedule" in src
