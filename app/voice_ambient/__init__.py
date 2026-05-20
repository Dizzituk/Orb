# FILE: app/voice_ambient/__init__.py
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