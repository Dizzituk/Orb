# FILE: app/briefing/__init__.py
"""
Morning Briefing System — Package Init.

Delivers curated daily and weekly news digests covering configurable
topic categories, with optional dual-voice audio delivery using
Google Cloud TTS.

Modules:
- briefing_config:    Topic categories, sources, schedule settings
- briefing_collector: Gathers news via multi-topic web searches
- briefing_compiler:  Compiles stories into structured briefing digest
- briefing_audio:     Dual-voice TTS for podcast-style audio output
- briefing_scheduler: Schedule runner (daily / weekly)
- briefing_router:    FastAPI endpoints for triggering and retrieving briefings
"""
from __future__ import annotations

__all__ = [
    "briefing_config",
    "briefing_collector",
    "briefing_compiler",
    "briefing_audio",
    "briefing_scheduler",
    "briefing_router",
]
