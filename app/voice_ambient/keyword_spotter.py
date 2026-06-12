# FILE: app/voice_ambient/keyword_spotter.py
# Purpose: Whisper-based keyword spotter for ambient wake detection.
# Called-by: app.voice_ambient, app.voice_ambient.session
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Whisper-based keyword spotter for ambient wake detection.

Runs faster-whisper (tiny.en by default) on rolling audio windows and
fires a detection event whenever one of a configured keyword set appears
in the transcription.

Design notes:
- Model loaded ONCE (singleton). Default CPU/int8 — tiny.en is ~300ms on CPU.
- Throttled: we only transcribe every MIN_INTERVAL_BETWEEN_TRANSCRIBES seconds,
  not on every audio chunk. This prevents whisper backlog when chunks arrive
  every 100ms.
- We skip transcription on near-silence (RMS below SILENCE_RMS_SKIP) — no point
  running whisper on silent buffers, and it often hallucinates on silence.
- Known whisper hallucinations ('you', 'thanks for watching', 'thank you',
  single-word fragments that often appear on silence) are filtered as
  non-detections unless they happen to contain a keyword.
- Every transcription is logged at DEBUG, so you can see what whisper hears
  by cranking log level. Non-silence, non-empty transcripts log at INFO.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BUFFER_SECONDS = 2.0
MIN_AUDIO_FOR_TRANSCRIBE = 0.6
COOLDOWN_SECONDS = 2.5
# How often we actually run whisper, regardless of chunk rate
MIN_INTERVAL_BETWEEN_TRANSCRIBES = 0.35
# Skip transcription on buffers quieter than this
SILENCE_RMS_SKIP = 0.005

DEFAULT_KEYWORDS: Set[str] = frozenset({
    "astra",
    "astro",
    "aster",
    "astor",
    "asta",
    "hey astra",
    "ok astra",
})

# Common whisper hallucinations - usually appear on silence or noise.
# We only flag these as "probably hallucination" - they're allowed to
# contain a keyword; we just strip them from logs to reduce noise.
_HALLUCINATION_PATTERNS = {
    "you",
    "you.",
    "thank you.",
    "thanks for watching!",
    "thank you for watching.",
    "bye.",
    ".",
}

_word_re = re.compile(r"[a-z0-9']+")


def _normalise(text: str) -> str:
    tokens = _word_re.findall((text or "").lower())
    return " ".join(tokens)


def _is_hallucination(text: str) -> bool:
    return (text or "").strip().lower() in _HALLUCINATION_PATTERNS


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


@dataclass
class SpotResult:
    detected: bool
    transcript: str
    matched_keyword: Optional[str] = None
    confidence: float = 0.0


@dataclass
class _State:
    buffer: List[np.ndarray] = field(default_factory=list)
    buffer_samples: int = 0
    last_detection_ts: float = 0.0
    last_transcribe_ts: float = 0.0


class WhisperKeywordSpotter:
    _instance: Optional["WhisperKeywordSpotter"] = None
    _cls_lock = threading.Lock()

    def __init__(
        self,
        model_size: str = "tiny.en",
        keywords: Optional[Set[str]] = None,
    ):
        self._model_size = model_size
        self._keywords: Set[str] = set(keywords or DEFAULT_KEYWORDS)
        self._state = _State()
        self._state_lock = threading.Lock()
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "WhisperKeywordSpotter":
        if cls._instance is None:
            with cls._cls_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            from faster_whisper import WhisperModel
            device = os.environ.get("KEYWORD_SPOTTER_DEVICE", "cpu").lower()
            compute_type = "float16" if device == "cuda" else "int8"
            try:
                self._model = WhisperModel(
                    self._model_size, device=device, compute_type=compute_type,
                )
                logger.info("[keyword_spotter] loaded %s on %s/%s",
                            self._model_size, device, compute_type)
            except Exception as e:
                logger.warning("[keyword_spotter] %s load failed (%s), falling back to CPU", device, e)
                self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
                logger.info("[keyword_spotter] loaded %s on cpu/int8", self._model_size)
        return self._model

    def reset(self):
        with self._state_lock:
            self._state = _State()

    def set_keywords(self, keywords: Set[str]):
        self._keywords = {_normalise(k) for k in keywords if k}

    def get_keywords(self) -> Set[str]:
        return set(self._keywords)

    def feed(self, audio_float32: np.ndarray) -> SpotResult:
        if audio_float32 is None or audio_float32.size == 0:
            return SpotResult(detected=False, transcript="")

        now = time.monotonic()

        if now - self._state.last_detection_ts < COOLDOWN_SECONDS:
            return SpotResult(detected=False, transcript="")

        with self._state_lock:
            self._state.buffer.append(audio_float32)
            self._state.buffer_samples += audio_float32.size
            max_samples = int(BUFFER_SECONDS * SAMPLE_RATE)
            while self._state.buffer_samples > max_samples and len(self._state.buffer) > 1:
                dropped = self._state.buffer.pop(0)
                self._state.buffer_samples -= dropped.size
            if self._state.buffer_samples < int(MIN_AUDIO_FOR_TRANSCRIBE * SAMPLE_RATE):
                return SpotResult(detected=False, transcript="")
            # Throttle transcription rate
            if now - self._state.last_transcribe_ts < MIN_INTERVAL_BETWEEN_TRANSCRIBES:
                return SpotResult(detected=False, transcript="")
            self._state.last_transcribe_ts = now
            buffer_snapshot = np.concatenate(self._state.buffer)

        # Skip if buffer is near-silent
        rms = _rms(buffer_snapshot)
        if rms < SILENCE_RMS_SKIP:
            logger.debug("[keyword_spotter] skip silence rms=%.5f", rms)
            return SpotResult(detected=False, transcript="")

        transcript = self._transcribe(buffer_snapshot)
        normalised = _normalise(transcript)

        # Log EVERY non-empty transcript so we can see what whisper hears.
        # This is the critical visibility we were missing.
        if transcript.strip():
            logger.info("[keyword_spotter] HEARD rms=%.4f text=%r", rms, transcript)

        if not normalised:
            return SpotResult(detected=False, transcript=transcript)

        matched = None
        for kw in self._keywords:
            if kw and kw in normalised:
                matched = kw
                break

        if not matched:
            if _is_hallucination(transcript):
                logger.debug("[keyword_spotter] hallucination ignored: %r", transcript)
            return SpotResult(detected=False, transcript=transcript)

        with self._state_lock:
            self._state.last_detection_ts = now
            self._state.buffer = []
            self._state.buffer_samples = 0

        logger.info("[keyword_spotter] *** DETECTED *** keyword=%r transcript=%r",
                    matched, transcript)
        return SpotResult(detected=True, transcript=transcript,
                          matched_keyword=matched, confidence=0.8)

    def _transcribe(self, audio_float32: np.ndarray) -> str:
        model = self._ensure_model()
        try:
            segments_iter, _info = model.transcribe(
                audio_float32,
                language="en",
                beam_size=1,
                vad_filter=False,
                # Lowered from 0.4 so whisper is less eager to mark audio as "no speech"
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                # Bias toward "Astra" with initial_prompt
                initial_prompt="Astra.",
            )
            parts = [seg.text for seg in segments_iter]
            return " ".join(p.strip() for p in parts if p).strip()
        except Exception as e:
            logger.error("[keyword_spotter] transcription error: %s", e)
            return ""


def get_keyword_spotter() -> WhisperKeywordSpotter:
    return WhisperKeywordSpotter.get_instance()