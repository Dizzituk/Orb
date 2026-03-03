# FILE: app/briefing/briefing_config.py
"""
Briefing Configuration — Topic categories, sources, and schedule settings.

Configurable via environment variables and the defaults below.
Each topic category defines search queries, priority, and
optional source preferences.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Briefing frequency
# =========================================================================

class BriefingFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


# =========================================================================
# Topic category definition
# =========================================================================

@dataclass
class TopicConfig:
    """Configuration for a briefing topic category."""
    name: str                            # Display name: "Financial Markets"
    key: str                             # Internal key: "finance"
    enabled: bool = True
    priority: int = 1                    # Lower = higher priority in briefing order
    search_queries: list = field(default_factory=list)  # Queries to run
    max_stories: int = 5                 # Max stories per category
    freshness_hint: str = "today"        # "today", "this week", "this month"
    description: str = ""                # Short description for the briefing header
    astra_relevant: bool = False         # Flag stories relevant to ASTRA's own domain


# =========================================================================
# Default topic categories
# =========================================================================

_DEFAULT_TOPICS: list[TopicConfig] = [
    TopicConfig(
        name="Financial Markets & Crypto",
        key="finance",
        priority=1,
        search_queries=[
            "financial markets today summary",
            "cryptocurrency news today",
            "TAO bittensor price news",
            "FTSE 100 today",
            "global economy news today",
        ],
        max_stories=5,
        freshness_hint="today",
        description="Markets, crypto, and economic indicators",
    ),
    TopicConfig(
        name="AI & Technology",
        key="ai_tech",
        priority=2,
        search_queries=[
            "artificial intelligence news today",
            "new AI model release announcement",
            "AI technology breakthrough",
            "LLM benchmark leaderboard latest",
        ],
        max_stories=5,
        freshness_hint="this week",
        description="AI developments, model releases, and tech industry moves",
        astra_relevant=True,
    ),
    TopicConfig(
        name="UK & World Affairs",
        key="world_affairs",
        priority=3,
        search_queries=[
            "UK news today headlines",
            "world news today top stories",
            "UK economy news today",
        ],
        max_stories=4,
        freshness_hint="today",
        description="UK domestic and international headlines",
    ),
    TopicConfig(
        name="Geopolitics",
        key="geopolitics",
        priority=4,
        search_queries=[
            "geopolitics news today",
            "international relations latest",
            "global conflict update",
        ],
        max_stories=3,
        freshness_hint="today",
        description="International relations, conflicts, and diplomacy",
    ),
    TopicConfig(
        name="Surf & Weather",
        key="surf_weather",
        priority=5,
        search_queries=[
            "surf forecast Devon Cornwall today",
            "Plymouth weather forecast",
            "Portugal surf forecast Nazare Peniche",
        ],
        max_stories=3,
        freshness_hint="today",
        description="Local and Portugal surf conditions and weather",
    ),
]


def _load_topics_from_env() -> Optional[list[TopicConfig]]:
    """Load topic config override from BRIEFING_TOPICS_JSON env var."""
    raw = os.getenv("BRIEFING_TOPICS_JSON", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        topics = []
        for item in data:
            topics.append(TopicConfig(**item))
        logger.info("[briefing_config] Loaded %d topics from env", len(topics))
        return topics
    except Exception as e:
        logger.warning("[briefing_config] Failed to parse BRIEFING_TOPICS_JSON: %s", e)
        return None


def get_topics() -> list[TopicConfig]:
    """Get the active topic categories, sorted by priority."""
    env_topics = _load_topics_from_env()
    topics = env_topics if env_topics is not None else _DEFAULT_TOPICS
    return sorted(
        [t for t in topics if t.enabled],
        key=lambda t: t.priority,
    )


# =========================================================================
# Schedule configuration
# =========================================================================

@dataclass
class ScheduleConfig:
    """Schedule settings for briefing generation."""
    daily_hour: int = 6           # Hour (24h) to generate daily briefing
    daily_minute: int = 0
    weekly_day: int = 0           # 0=Monday
    weekly_hour: int = 7
    weekly_minute: int = 0
    auto_generate: bool = True    # Whether to auto-generate on schedule
    audio_enabled: bool = True    # Whether to generate audio version


def get_schedule() -> ScheduleConfig:
    """Get schedule config with env overrides."""
    return ScheduleConfig(
        daily_hour=int(os.getenv("BRIEFING_DAILY_HOUR", "6")),
        daily_minute=int(os.getenv("BRIEFING_DAILY_MINUTE", "0")),
        weekly_day=int(os.getenv("BRIEFING_WEEKLY_DAY", "0")),
        weekly_hour=int(os.getenv("BRIEFING_WEEKLY_HOUR", "7")),
        weekly_minute=int(os.getenv("BRIEFING_WEEKLY_MINUTE", "0")),
        auto_generate=os.getenv("BRIEFING_AUTO_GENERATE", "true").lower() in ("true", "1"),
        audio_enabled=os.getenv("BRIEFING_AUDIO_ENABLED", "true").lower() in ("true", "1"),
    )


# =========================================================================
# Audio voice config
# =========================================================================

@dataclass
class VoiceConfig:
    """Dual-voice configuration for audio briefings."""
    voice_headlines: str = "en-GB-Chirp3-HD-Achird"  # Voice A: headlines/summaries
    voice_analysis: str = "en-GB-Chirp3-HD-Fenrir"   # Voice B: analysis/context
    speed: float = 1.0
    pause_between_stories_ms: int = 800
    pause_between_sections_ms: int = 1200


def get_voice_config() -> VoiceConfig:
    """Get voice config with env overrides."""
    return VoiceConfig(
        voice_headlines=os.getenv("BRIEFING_VOICE_HEADLINES", "en-GB-Chirp3-HD-Achird"),
        voice_analysis=os.getenv("BRIEFING_VOICE_ANALYSIS", "en-GB-Chirp3-HD-Fenrir"),
        speed=float(os.getenv("BRIEFING_VOICE_SPEED", "1.0")),
    )


__all__ = [
    "BriefingFrequency",
    "TopicConfig",
    "ScheduleConfig",
    "VoiceConfig",
    "get_topics",
    "get_schedule",
    "get_voice_config",
]
