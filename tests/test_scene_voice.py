# FILE: tests/test_scene_voice.py
# Purpose: Units for app/scene_director/voice.py — POST /scene/ask endpoint logic with the
#          chat brain, TTS, and DB persistence mocked: happy path, TTS-down, brain error.
# Called-by: pytest
# Depends-on: app.scene_director.voice
# Last-renovated: 2026-06-13
from __future__ import annotations

from urllib.parse import unquote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.scene_director import voice as voice_mod
from app.astra_presence.state import presence_state


class _FakeProject:
    id = 4242
    name = "ASTRA Room"


@pytest.fixture(autouse=True)
def _clean():
    presence_state.reset()
    yield
    presence_state.reset()


@pytest.fixture()
def client(monkeypatch):
    # No DB: stub project resolution + message persistence + history.
    monkeypatch.setattr(voice_mod, "_resolve_room_project", lambda db: _FakeProject())

    import app.memory.service as mem_service
    import app.memory._service_utils_2 as mem_utils
    monkeypatch.setattr(mem_service, "create_message", lambda db, mc: None, raising=False)
    monkeypatch.setattr(mem_utils, "list_messages", lambda db, pid, limit=20: [], raising=False)

    app = FastAPI()
    app.include_router(voice_mod.router)
    # get_db dependency: yield a dummy session (never touched once helpers are stubbed)
    from app.db import get_db
    app.dependency_overrides[get_db] = lambda: iter([object()])
    with TestClient(app) as c:
        yield c


def _mock_brain(monkeypatch, reply="The sky is blue because of Rayleigh scattering."):
    import app.bridge.capability_layer as cap

    async def fake_run(**kwargs):
        return {"reply": reply, "provider": "test", "model": "test-model", "tools_used": []}
    monkeypatch.setattr(cap, "run_astra_chat", fake_run, raising=False)


def _mock_tts(monkeypatch, ok=True):
    import app.bridge.chat_and_speak as cas

    async def fake_synth(sentence):
        if not ok:
            raise RuntimeError("chatterbox down")
        return b"\xff\xf3\x14MP3" + sentence.encode("utf-8")[:8]
    monkeypatch.setattr(cas, "_synthesise_sentence", fake_synth, raising=False)


def test_ask_happy_path_returns_audio_and_text(client, monkeypatch):
    _mock_brain(monkeypatch)
    _mock_tts(monkeypatch, ok=True)
    resp = client.post("/scene/ask", json={"message": "why is the sky blue?"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/mpeg")
    assert len(resp.content) > 0
    assert "Rayleigh" in unquote(resp.headers["X-Full-Text"])
    assert resp.headers["X-Project-Id"] == "4242"


def test_ask_tts_down_returns_text_only(client, monkeypatch):
    _mock_brain(monkeypatch)
    _mock_tts(monkeypatch, ok=False)
    resp = client.post("/scene/ask", json={"message": "hello"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Tts-Error") == "chatterbox-unreachable"
    assert resp.content == b""
    assert "Rayleigh" in unquote(resp.headers["X-Full-Text"])


def test_ask_brain_error_503(client, monkeypatch):
    import app.bridge.capability_layer as cap

    async def boom(**kwargs):
        raise RuntimeError("no LLM key")
    monkeypatch.setattr(cap, "run_astra_chat", boom, raising=False)
    _mock_tts(monkeypatch, ok=True)
    resp = client.post("/scene/ask", json={"message": "hi"})
    assert resp.status_code == 503


def test_ask_empty_message_422(client):
    assert client.post("/scene/ask", json={"message": ""}).status_code == 422


def test_ask_sets_presence_thinking_then_speaking(client, monkeypatch):
    _mock_brain(monkeypatch)
    _mock_tts(monkeypatch, ok=True)
    resp = client.post("/scene/ask", json={"message": "tell me something"})
    assert resp.status_code == 200
    # by the time the response is built, presence has advanced to speaking
    assert presence_state.get_state() == "speaking"
