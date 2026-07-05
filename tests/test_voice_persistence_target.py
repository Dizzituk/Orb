# FILE: tests/test_voice_persistence_target.py
# Purpose: Unit tests for voice-turn persistence targeting (2026-07-03 chat-UI
#          mirroring): record_turn writes to the OPEN conversation when
#          chat_project_id is given and valid, and falls back to the Voice
#          project (with its session markers) otherwise.
# Called-by: pytest
# Depends-on: app.voice_ambient.persistence
# Last-renovated: 2026-07-03
from __future__ import annotations

from app.voice_ambient import persistence as vp


class _Proj:
    def __init__(self, pid: int):
        self.id = pid


def _capture_messages(monkeypatch):
    created = []

    def _create_message(db, data):
        created.append(data)
        return data

    monkeypatch.setattr(vp.memory_service, "create_message", _create_message)
    return created


def test_record_turn_targets_open_chat_project(monkeypatch):
    created = _capture_messages(monkeypatch)
    monkeypatch.setattr(
        vp.memory_service, "get_project",
        lambda db, pid: _Proj(pid) if pid == 42 else None,
    )
    vp._marked_sessions.clear()

    pid = vp.record_turn(
        db=None, session_id="sess-a", user_text="what's the time",
        assistant_text="just gone nine", provider="openai", model="gpt-x",
        chat_project_id=42,
    )

    assert pid == 42
    # No session marker inline in a normal conversation - just the turn pair.
    assert [(m.project_id, m.role) for m in created] == [(42, "user"), (42, "assistant")]
    assert created[0].provider == "voice"          # user row badged as voice
    assert created[1].provider == "openai"         # assistant row keeps real provider
    assert "sess-a" not in vp._marked_sessions     # marker path never ran


def test_record_turn_missing_chat_project_falls_back_to_voice(monkeypatch):
    created = _capture_messages(monkeypatch)
    monkeypatch.setattr(vp.memory_service, "get_project", lambda db, pid: None)
    monkeypatch.setattr(vp, "_get_or_create_voice_project", lambda db: 7)
    vp._marked_sessions.clear()

    pid = vp.record_turn(
        db=None, session_id="sess-b", user_text="hello",
        assistant_text="hi", chat_project_id=999,
    )

    assert pid == 7
    # Voice project keeps its session-start separator + the turn pair.
    assert [(m.project_id, m.role) for m in created] == [
        (7, "system"), (7, "user"), (7, "assistant"),
    ]


def test_record_turn_default_unchanged_voice_path(monkeypatch):
    created = _capture_messages(monkeypatch)
    monkeypatch.setattr(vp, "_get_or_create_voice_project", lambda db: 7)
    vp._marked_sessions.clear()

    pid = vp.record_turn(
        db=None, session_id="sess-c", user_text="ping", assistant_text="pong",
    )

    assert pid == 7
    assert [(m.project_id, m.role) for m in created] == [
        (7, "system"), (7, "user"), (7, "assistant"),
    ]
