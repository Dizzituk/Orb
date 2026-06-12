# FILE: app/voice_ambient/session.py
# Purpose: Ambient voice session over WebSocket.
# Called-by: app.voice_ambient, app.voice_ambient.router
# Depends-on: app.services.faster_whisper_service, app.voice_ambient.keyword_spotter
# Last-renovated: 2026-06-11
"""
Ambient voice session over WebSocket.

Protocol:
  Client -> Server (JSON frames):
    {"type":"start","sample_rate":16000}
    {"type":"audio","pcm16":"<base64 of int16 LE mono PCM>"}
    {"type":"stop"}                   # end user utterance early
    {"type":"abort"}                  # abandon current turn completely
    {"type":"keep_alive"}

  Server -> Client (JSON frames):
    {"type":"ready"}
    {"type":"wake","transcript":"...","keyword":"astra"}
    {"type":"recording_started"}
    {"type":"partial_transcript","text":"...partial..."}  (reserved)
    {"type":"transcript","text":"..."}
    {"type":"error","message":"..."}

State machine:
  IDLE -> (audio streaming) -> DETECTED (wake heard)
                                   |
                                   v
                              RECORDING (capturing user utterance)
                                   |
                                   v  (silence OR client stop)
                              TRANSCRIBING
                                   |
                                   v  (transcript sent)
                              IDLE

All Whisper work happens in a thread executor to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import base64
import enum
import io
import logging
import time
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from app.voice_ambient.keyword_spotter import (
    get_keyword_spotter,
    SAMPLE_RATE as KW_SAMPLE_RATE,
)

logger = logging.getLogger(__name__)

# Recording stops after this many seconds of sub-threshold audio
SILENCE_AFTER_SECONDS = 1.2
# Absolute cap on a single recorded utterance
MAX_RECORDING_SECONDS = 20.0
# Energy threshold for "speech" in normalised float32 audio (0..1).
# Deliberately simple — VAD could be upgraded later to webrtcvad.
SPEECH_RMS_THRESHOLD = 0.01
# Maximum idle time without any audio before the session shuts itself down.
MAX_IDLE_SECONDS = 120.0


class _Phase(str, enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    CLOSED = "closed"


@dataclass
class _RecordingBuffer:
    chunks: List[np.ndarray] = field(default_factory=list)
    samples: int = 0
    started_at: float = 0.0
    last_speech_at: float = 0.0

    def append(self, audio: np.ndarray):
        self.chunks.append(audio)
        self.samples += audio.size

    def concat(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks)

    def duration(self) -> float:
        return self.samples / KW_SAMPLE_RATE if self.samples else 0.0


def _decode_pcm16_b64(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return arr


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int = KW_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        wf.writeframes(pcm16)
    return buf.getvalue()


class AmbientVoiceSession:
    """Per-WebSocket state machine that spots wake words and captures utterances."""

    def __init__(self, websocket):
        self.ws = websocket
        self.phase: _Phase = _Phase.IDLE
        self.spotter = get_keyword_spotter()
        self.buffer = _RecordingBuffer()
        self.sample_rate: int = KW_SAMPLE_RATE
        self.last_audio_at: float = time.monotonic()

    # ─── Sending helpers ──────────────────────────────────────────
    async def _send(self, payload: dict):
        try:
            await self.ws.send_json(payload)
        except Exception as e:
            logger.warning("[voice_session] send failed: %s", e)

    # ─── Main loop ────────────────────────────────────────────────
    async def run(self):
        await self._send({"type": "ready"})
        try:
            while self.phase != _Phase.CLOSED:
                try:
                    msg = await asyncio.wait_for(
                        self.ws.receive_json(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # Idle-out guard
                    if time.monotonic() - self.last_audio_at > MAX_IDLE_SECONDS:
                        logger.info("[voice_session] idle timeout, closing")
                        self.phase = _Phase.CLOSED
                        break
                    continue

                t = msg.get("type")
                if t == "audio":
                    await self._handle_audio(msg)
                elif t == "stop":
                    await self._handle_stop()
                elif t == "abort":
                    await self._handle_abort()
                elif t == "start":
                    self.sample_rate = int(msg.get("sample_rate") or KW_SAMPLE_RATE)
                    self.spotter.reset()
                elif t == "keep_alive":
                    pass
                else:
                    logger.debug("[voice_session] unknown msg type=%s", t)
        except WebSocketDisconnect:
            # Client closed the socket; normal termination, not an error.
            logger.info("[voice_session] client disconnected cleanly")
        except Exception as e:
            logger.exception("[voice_session] run loop error")
            # Best-effort error broadcast; will no-op if socket is dead.
            await self._send({"type": "error", "message": str(e)})

    # ─── Message handlers ─────────────────────────────────────────
    async def _handle_audio(self, msg: dict):
        b64 = msg.get("pcm16")
        if not b64:
            return
        try:
            audio = _decode_pcm16_b64(b64)
        except Exception as e:
            await self._send({"type": "error", "message": f"bad audio frame: {e}"})
            return

        self.last_audio_at = time.monotonic()

        if self.phase == _Phase.IDLE:
            # Feed into spotter. Keyword detection runs in a thread.
            result = await asyncio.to_thread(self.spotter.feed, audio)
            if result.detected:
                self.phase = _Phase.RECORDING
                self.buffer = _RecordingBuffer()
                self.buffer.started_at = time.monotonic()
                self.buffer.last_speech_at = self.buffer.started_at
                await self._send({
                    "type": "wake",
                    "transcript": result.transcript,
                    "keyword": result.matched_keyword,
                })
                await self._send({"type": "recording_started"})
            return

        if self.phase == _Phase.RECORDING:
            self.buffer.append(audio)

            # Energy-based silence detection
            rms = _rms(audio)
            now = time.monotonic()
            if rms > SPEECH_RMS_THRESHOLD:
                self.buffer.last_speech_at = now

            # Stop conditions
            since_speech = now - self.buffer.last_speech_at
            total = now - self.buffer.started_at
            if total >= MAX_RECORDING_SECONDS:
                logger.info("[voice_session] recording hit max duration")
                await self._finalise()
            elif since_speech >= SILENCE_AFTER_SECONDS and total > 0.8:
                logger.info("[voice_session] silence detected, finalising")
                await self._finalise()

    async def _handle_stop(self):
        if self.phase == _Phase.RECORDING:
            await self._finalise()
        else:
            # Nothing to stop
            await self._send({"type": "transcript", "text": ""})

    async def _handle_abort(self):
        self.phase = _Phase.IDLE
        self.buffer = _RecordingBuffer()
        self.spotter.reset()
        await self._send({"type": "ready"})

    async def _finalise(self):
        if self.phase != _Phase.RECORDING:
            return
        self.phase = _Phase.TRANSCRIBING
        audio = self.buffer.concat()
        if audio.size == 0:
            await self._send({"type": "transcript", "text": ""})
            self.phase = _Phase.IDLE
            return

        try:
            text = await asyncio.to_thread(self._transcribe_utterance, audio)
        except Exception as e:
            logger.exception("[voice_session] transcribe failed")
            await self._send({"type": "error", "message": str(e)})
            self.phase = _Phase.IDLE
            return

        await self._send({"type": "transcript", "text": text})
        self.phase = _Phase.IDLE

    def _transcribe_utterance(self, audio: np.ndarray) -> str:
        """Use the main transcription service for the full utterance."""
        from app.services.faster_whisper_service import FasterWhisperService
        svc = FasterWhisperService()
        wav = _float32_to_wav_bytes(audio, self.sample_rate)
        result = svc.transcribe(wav, language="en", vad_filter=True)
        return (result.text or "").strip()