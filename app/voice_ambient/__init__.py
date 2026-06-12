# FILE: app/voice_ambient/__init__.py
# Purpose: Ambient voice module - wake word + utterance capture over WebSocket.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: app.voice_ambient.keyword_spotter, app.voice_ambient.router, app.voice_ambient.session
# Last-renovated: 2026-06-11
"""Ambient voice module - wake word + utterance capture over WebSocket."""

from app.voice_ambient.keyword_spotter import (
    WhisperKeywordSpotter,
    get_keyword_spotter,
    SpotResult,
    DEFAULT_KEYWORDS,
    SAMPLE_RATE,
)
from app.voice_ambient.session import AmbientVoiceSession
from app.voice_ambient.router import router

__all__ = [
    "WhisperKeywordSpotter",
    "get_keyword_spotter",
    "SpotResult",
    "DEFAULT_KEYWORDS",
    "SAMPLE_RATE",
    "AmbientVoiceSession",
    "router",
]