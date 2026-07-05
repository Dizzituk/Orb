# FILE: tests/test_voice_ambient_session.py
# Purpose: Regression tests for app/voice_ambient/session.py run-loop exception
#          handling (2026-07-03 incident: WebSocketDisconnect caught without an
#          import -> NameError on every ambient-voice disconnect) + the live
#          partial_transcript feed (2026-07-03: as-you-speak preview frames).
# Called-by: pytest
# Depends-on: app.voice_ambient.session
# Last-renovated: 2026-07-03
from __future__ import annotations

import ast
import asyncio
import base64
import pathlib
import warnings

import numpy as np
from starlette.websockets import WebSocketDisconnect as StarletteWSDisconnect

from app.voice_ambient import session as session_mod


class _DummySpotter:
    """Stands in for the whisper keyword spotter - no model load in unit tests."""

    def reset(self):
        pass


class _FakeWebSocket:
    """Minimal websocket double: raises a scripted exception on receive, records sends."""

    def __init__(self, receive_exc: Exception):
        self._receive_exc = receive_exc
        self.sent = []

    async def send_json(self, payload: dict):
        self.sent.append(payload)

    async def receive_json(self):
        raise self._receive_exc


def _run_session(monkeypatch, receive_exc: Exception) -> _FakeWebSocket:
    monkeypatch.setattr(session_mod, "get_keyword_spotter", lambda: _DummySpotter())
    ws = _FakeWebSocket(receive_exc)
    asyncio.run(session_mod.AmbientVoiceSession(ws).run())
    return ws


def test_websocket_disconnect_is_the_starlette_exception():
    # The except clause must reference the same class starlette raises from
    # receive_json(), via fastapi's re-export (house import style).
    assert getattr(session_mod, "WebSocketDisconnect", None) is StarletteWSDisconnect


def test_run_handles_client_disconnect_cleanly(monkeypatch):
    # Before the fix this raised NameError from the except clause itself.
    ws = _run_session(monkeypatch, StarletteWSDisconnect(1005))
    assert [p.get("type") for p in ws.sent] == ["ready"]  # no error frame


def test_run_unexpected_error_sends_error_frame(monkeypatch):
    ws = _run_session(monkeypatch, RuntimeError("boom"))
    assert [p.get("type") for p in ws.sent] == ["ready", "error"]
    assert "boom" in ws.sent[1]["message"]


# ─── Live partial transcripts (2026-07-03) ────────────────────────────────

class _SpottingDummy:
    """Wakes on the first feed; then serves scripted partial snippets."""

    def __init__(self, partials):
        self._partials = list(partials)

    def reset(self):
        pass

    def feed(self, _audio):
        class _R:
            detected = True
            transcript = "astra"
            matched_keyword = "astra"
        return _R()

    def transcribe_snippet(self, _audio):
        return self._partials.pop(0) if self._partials else ""


def _pcm_frame(seconds: float) -> str:
    """Base64 int16 mono PCM of the given duration (quiet — content irrelevant,
    the dummy spotter scripts what 'was heard')."""
    samples = np.zeros(int(seconds * session_mod.KW_SAMPLE_RATE), dtype="<i2")
    return base64.b64encode(samples.tobytes()).decode("ascii")


async def _wait_for_frame(ws: _FakeWebSocket, frame_type: str, timeout: float = 2.0) -> bool:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if any(p.get("type") == frame_type for p in ws.sent):
            return True
        await asyncio.sleep(0.02)
    return False


def _make_recording_session(monkeypatch, partials):
    monkeypatch.setattr(
        session_mod, "get_keyword_spotter", lambda: _SpottingDummy(partials)
    )
    ws = _FakeWebSocket(RuntimeError("unused"))
    return ws, session_mod.AmbientVoiceSession(ws)


def test_partials_stream_while_recording(monkeypatch):
    ws, sess = _make_recording_session(monkeypatch, ["set a timer", "set a timer for five"])
    monkeypatch.setattr(session_mod, "PARTIAL_MIN_INTERVAL_SECONDS", 0.0)

    async def drive():
        await sess._handle_audio({"pcm16": _pcm_frame(0.2)})   # wake -> RECORDING
        await sess._handle_audio({"pcm16": _pcm_frame(0.7)})   # crosses min audio
        assert await _wait_for_frame(ws, "partial_transcript")
        await sess._handle_audio({"pcm16": _pcm_frame(0.1)})   # second pass
        deadline = asyncio.get_event_loop().time() + 2.0
        while asyncio.get_event_loop().time() < deadline:
            if sum(1 for p in ws.sent if p.get("type") == "partial_transcript") >= 2:
                break
            await asyncio.sleep(0.02)

    asyncio.run(drive())
    partial_frames = [p for p in ws.sent if p.get("type") == "partial_transcript"]
    assert [p["text"] for p in partial_frames] == ["set a timer", "set a timer for five"]
    # Wake/recording frames still precede the previews
    assert [p.get("type") for p in ws.sent[:2]] == ["wake", "recording_started"]


def test_partials_dedupe_identical_text(monkeypatch):
    ws, sess = _make_recording_session(monkeypatch, ["same text", "same text"])
    monkeypatch.setattr(session_mod, "PARTIAL_MIN_INTERVAL_SECONDS", 0.0)

    async def drive():
        await sess._handle_audio({"pcm16": _pcm_frame(0.2)})
        await sess._handle_audio({"pcm16": _pcm_frame(0.7)})
        assert await _wait_for_frame(ws, "partial_transcript")
        await sess._handle_audio({"pcm16": _pcm_frame(0.1)})
        await asyncio.sleep(0.2)  # give the second (deduped) pass time to run

    asyncio.run(drive())
    partial_frames = [p for p in ws.sent if p.get("type") == "partial_transcript"]
    assert [p["text"] for p in partial_frames] == ["same text"]


def test_partials_skip_when_buffer_exceeds_cap(monkeypatch):
    # Passes re-read the whole buffer, so beyond the cap they stop — previews
    # must never lag further and further behind on very long utterances.
    ws, sess = _make_recording_session(monkeypatch, ["too late for previews"])
    monkeypatch.setattr(session_mod, "PARTIAL_MIN_INTERVAL_SECONDS", 0.0)

    async def drive():
        await sess._handle_audio({"pcm16": _pcm_frame(0.2)})   # wake -> RECORDING
        over_cap = session_mod.PARTIAL_MAX_BUFFER_SECONDS + 1.0
        await sess._handle_audio({"pcm16": _pcm_frame(over_cap)})
        await asyncio.sleep(0.2)

    asyncio.run(drive())
    assert not any(p.get("type") == "partial_transcript" for p in ws.sent)


def test_partials_kill_switch(monkeypatch):
    monkeypatch.setenv("ASTRA_AMBIENT_PARTIALS", "0")
    ws, sess = _make_recording_session(monkeypatch, ["should never appear"])
    monkeypatch.setattr(session_mod, "PARTIAL_MIN_INTERVAL_SECONDS", 0.0)

    async def drive():
        await sess._handle_audio({"pcm16": _pcm_frame(0.2)})
        await sess._handle_audio({"pcm16": _pcm_frame(0.7)})
        await asyncio.sleep(0.2)

    asyncio.run(drive())
    assert not any(p.get("type") == "partial_transcript" for p in ws.sent)


def _catches_wsdisconnect_without_binding(src: str) -> bool:
    """True if the module catches bare-name WebSocketDisconnect but never binds it."""
    with warnings.catch_warnings():
        # Pre-existing bad escape sequences in swept files are not this test's business.
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(src)
    caught = bound = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is not None:
            for sub in ast.walk(node.type):
                if isinstance(sub, ast.Name) and sub.id == "WebSocketDisconnect":
                    caught = True
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if (alias.asname or alias.name) == "WebSocketDisconnect":
                    bound = True
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "WebSocketDisconnect":
                    bound = True
    return caught and not bound


def test_no_app_module_catches_wsdisconnect_without_importing_it():
    # Sweep app/ for the exact gap class this incident exposed.
    app_root = pathlib.Path(session_mod.__file__).resolve().parents[1]
    offenders = []
    for py in app_root.rglob("*.py"):
        try:
            src = py.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        try:
            if _catches_wsdisconnect_without_binding(src):
                offenders.append(str(py))
        except SyntaxError:
            continue  # compile sweep owns syntax errors
    assert offenders == []
