# Purpose: news profiles
# Called-by: app.briefing.briefing_compiler, app.briefing.briefing_config
# Depends-on: app.briefing.briefing_config
# Last-renovated: 2026-06-11
from __future__ import annotations

from dataclasses import dataclass

from app.briefing.briefing_config import TopicConfig


@dataclass(frozen=True)
class BriefingProfile:
    key: str
    label: str
    title_prefix: str
    intro_line: str
    closing_line: str
    topics: tuple[TopicConfig, ...]


_COMMAND_CENTRE_NEWS_TOPICS: tuple[TopicConfig, ...] = (
    TopicConfig(
        name="AI Capability Frontier",
        key="ai_capability_frontier",
        priority=1,
        search_queries=[
            "AI model release today frontier model benchmark",
            "new AI capability breakthrough today model release",
            "agentic AI system launch benchmark today",
            "frontier AI lab announcement today",
        ],
        max_stories=4,
        freshness_hint="today",
        description="Model releases, capability jumps, benchmark movement, and serious frontier announcements.",
        astra_relevant=True,
    ),
    TopicConfig(
        name="Robotics & Embodied AI",
        key="robotics_embodied_ai",
        priority=2,
        search_queries=[
            "robotics AI breakthrough today humanoid robot",
            "embodied AI robotics news today",
            "warehouse robot autonomy update today",
            "robotics foundation model today",
        ],
        max_stories=4,
        freshness_hint="today",
        description="Humanoids, industrial robotics, embodied models, and real-world autonomy.",
        astra_relevant=True,
    ),
    TopicConfig(
        name="Science & AI Discovery",
        key="science_ai_discovery",
        priority=3,
        search_queries=[
            "AI scientific discovery breakthrough today",
            "science breakthrough today AI researchers",
            "materials science breakthrough today battery material",
            "drug discovery AI breakthrough today",
        ],
        max_stories=4,
        freshness_hint="this week",
        description="Scientific discoveries, AI-assisted research, materials, biotech, and high-signal lab results.",
        astra_relevant=True,
    ),
    TopicConfig(
        name="Energy, Batteries & Infrastructure",
        key="energy_batteries_infra",
        priority=4,
        search_queries=[
            "battery technology breakthrough today",
            "grid scale energy storage breakthrough today",
            "semiconductor manufacturing breakthrough today",
            "fusion energy update today",
        ],
        max_stories=3,
        freshness_hint="this week",
        description="Battery chemistry, energy systems, semiconductors, and enabling infrastructure.",
        astra_relevant=False,
    ),
    TopicConfig(
        name="Space & Frontier Engineering",
        key="space_frontier_engineering",
        priority=5,
        search_queries=[
            "space technology breakthrough today",
            "launch system update today reusable rocket",
            "satellite manufacturing breakthrough today",
            "space science mission result today",
        ],
        max_stories=3,
        freshness_hint="this week",
        description="Launch systems, spacecraft engineering, missions, and high-leverage space developments.",
        astra_relevant=False,
    ),
    TopicConfig(
        name="Labour, Economy & Society",
        key="labour_economy_society",
        priority=6,
        search_queries=[
            "AI labour market impact today",
            "automation jobs report today AI",
            "AI regulation workplace today",
            "productivity impact AI economy today",
        ],
        max_stories=3,
        freshness_hint="this week",
        description="How AI and automation are changing work, regulation, incentives, and real economic behaviour.",
        astra_relevant=True,
    ),
)


_PROFILE_MAP: dict[str, BriefingProfile] = {
    "default": BriefingProfile(
        key="default",
        label="Morning Briefing",
        title_prefix="ASTRA Morning Briefing",
        intro_line="Good morning. Here is your daily briefing.",
        closing_line="That's your daily briefing. Have a good one.",
        topics=(),
    ),
    "command_centre_news": BriefingProfile(
        key="command_centre_news",
        label="Command Centre News",
        title_prefix="ASTRA Command Centre News",
        intro_line="Here is your Command Centre news briefing.",
        closing_line="That's the current Command Centre news picture.",
        topics=_COMMAND_CENTRE_NEWS_TOPICS,
    ),
}


def get_briefing_profile(profile_key: str = "default") -> BriefingProfile:
    return _PROFILE_MAP.get(profile_key, _PROFILE_MAP["default"])


def get_profile_topics(profile_key: str = "default") -> list[TopicConfig]:
    profile = get_briefing_profile(profile_key)
    if profile.topics:
        return sorted([topic for topic in profile.topics if topic.enabled], key=lambda topic: topic.priority)
    return []


__all__ = ["BriefingProfile", "get_briefing_profile", "get_profile_topics"]
