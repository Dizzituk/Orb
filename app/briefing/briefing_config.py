from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from app.briefing.news_profiles import get_profile_topics

logger = logging.getLogger(__name__)


class BriefingFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"


@dataclass
class TopicConfig:
    name: str
    key: str
    enabled: bool = True
    priority: int = 1
    search_queries: list = field(default_factory=list)
    max_stories: int = 5
    freshness_hint: str = "today"
    description: str = ""
    astra_relevant: bool = False


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
    raw = os.getenv("BRIEFING_TOPICS_JSON", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return [TopicConfig(**item) for item in data]
    except Exception as exc:
        logger.warning("[briefing_config] Failed to parse BRIEFING_TOPICS_JSON: %s", exc)
        return None


def get_topics(profile: str = "default") -> list[TopicConfig]:
    if profile != "default":
        profile_topics = get_profile_topics(profile)
        if profile_topics:
            return profile_topics
    env_topics = _load_topics_from_env()
    topics = env_topics if env_topics is not None else _DEFAULT_TOPICS
    return sorted([topic for topic in topics if topic.enabled], key=lambda topic: topic.priority)


@dataclass
class ScheduleConfig:
    daily_hour: int = 6
    daily_minute: int = 0
    weekly_day: int = 0
    weekly_hour: int = 7
    weekly_minute: int = 0
    auto_generate: bool = True
    audio_enabled: bool = True


@dataclass
class VoiceConfig:
    voice_headlines: str = "en-GB-Chirp3-HD-Achird"
    voice_analysis: str = "en-GB-Chirp3-HD-Fenrir"
    speed: float = 1.0
    pause_between_stories_ms: int = 800
    pause_between_sections_ms: int = 1200


def get_schedule() -> ScheduleConfig:
    return ScheduleConfig(
        daily_hour=int(os.getenv("BRIEFING_DAILY_HOUR", "6")),
        daily_minute=int(os.getenv("BRIEFING_DAILY_MINUTE", "0")),
        weekly_day=int(os.getenv("BRIEFING_WEEKLY_DAY", "0")),
        weekly_hour=int(os.getenv("BRIEFING_WEEKLY_HOUR", "7")),
        weekly_minute=int(os.getenv("BRIEFING_WEEKLY_MINUTE", "0")),
        auto_generate=os.getenv("BRIEFING_AUTO_GENERATE", "true").lower() in ("true", "1"),
        audio_enabled=os.getenv("BRIEFING_AUDIO_ENABLED", "true").lower() in ("true", "1"),
    )


def get_voice_config() -> VoiceConfig:
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
