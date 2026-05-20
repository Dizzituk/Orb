from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.briefing.briefing_collector import BriefingCollection
from app.briefing.news_profiles import get_briefing_profile

logger = logging.getLogger(__name__)


@dataclass
class BriefingItem:
    headline: str
    summary: str
    source_name: str = ""
    source_url: str = ""
    credibility: str = "unknown"
    topic_key: str = ""
    astra_flag: str = ""


@dataclass
class BriefingSection:
    topic_name: str
    topic_key: str
    description: str = ""
    items: list[BriefingItem] = field(default_factory=list)


@dataclass
class AudioSegment:
    text: str
    voice_role: str = "headlines"
    pause_after_ms: int = 800


@dataclass
class CompiledBriefing:
    title: str = ""
    generated_at: str = ""
    sections: list[BriefingSection] = field(default_factory=list)
    audio_script: list[AudioSegment] = field(default_factory=list)
    astra_alerts: list[str] = field(default_factory=list)
    total_items: int = 0
    text_digest: str = ""


def _snippet_summary(title: str, snippet: str) -> str:
    if snippet and len(snippet) > 20:
        return snippet[:300].strip()
    return title


def _extract_source_name(url: str) -> str:
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace("www.", "")
        parts = domain.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return domain
    except Exception:
        return ""


def compile_briefing(collection: BriefingCollection, frequency: str = "daily", profile: str = "default") -> CompiledBriefing:
    now = datetime.now(timezone.utc)
    profile_def = get_briefing_profile(profile)
    title = f"{profile_def.title_prefix} — {now.strftime('%A %d %B %Y')}"

    briefing = CompiledBriefing(title=title, generated_at=now.isoformat())
    briefing.audio_script.append(AudioSegment(text=profile_def.intro_line, voice_role="headlines", pause_after_ms=1200))

    text_parts = [f"# {title}", ""]
    for topic_col in collection.topics:
        if not topic_col.stories:
            continue
        section = BriefingSection(topic_name=topic_col.topic_name, topic_key=topic_col.topic_key, description=topic_col.description)
        briefing.audio_script.append(AudioSegment(text=f"Now to {topic_col.topic_name}.", voice_role="headlines", pause_after_ms=800))
        text_parts.append(f"## {topic_col.topic_name}")
        if topic_col.description:
            text_parts.append(topic_col.description)
            text_parts.append("")

        for story in topic_col.stories:
            source_name = _extract_source_name(story.url) or story.source_type
            summary = _snippet_summary(story.title, story.snippet)
            astra_flag = ""
            if story.astra_relevant:
                lower = f"{story.title} {story.snippet}".lower()
                if any(keyword in lower for keyword in ["model", "benchmark", "agent", "robot", "automation", "battery", "discovery", "labour"]):
                    astra_flag = "Command Centre relevant — track this"
                    briefing.astra_alerts.append(f"[{topic_col.topic_name}] {story.title}")

            item = BriefingItem(
                headline=story.title,
                summary=summary,
                source_name=source_name,
                source_url=story.url,
                credibility=story.credibility_label,
                topic_key=topic_col.topic_key,
                astra_flag=astra_flag,
            )
            section.items.append(item)
            briefing.audio_script.append(AudioSegment(text=story.title, voice_role="headlines", pause_after_ms=500))
            briefing.audio_script.append(AudioSegment(text=summary, voice_role="analysis", pause_after_ms=800))
            cred_tag = f" [{story.credibility_label}]" if story.credibility_label != "unknown" else ""
            text_parts.append(f"- **{story.title}**{cred_tag}")
            text_parts.append(f"  {summary}")
            text_parts.append(f"  *Source: {source_name}* — [link]({story.url})")
            if astra_flag:
                text_parts.append(f"  🔔 *{astra_flag}*")
            text_parts.append("")

        briefing.sections.append(section)

    if briefing.astra_alerts:
        text_parts.append("## ASTRA-Relevant Alerts")
        text_parts.extend(f"- {alert}" for alert in briefing.astra_alerts)
        text_parts.append("")

    briefing.audio_script.append(AudioSegment(text=profile_def.closing_line, voice_role="headlines", pause_after_ms=0))
    briefing.text_digest = "\n".join(text_parts).strip()
    briefing.total_items = sum(len(section.items) for section in briefing.sections)

    logger.info(
        "[briefing_compiler] Compiled profile=%s: sections=%d, items=%d",
        profile, len(briefing.sections), briefing.total_items,
    )
    return briefing


__all__ = [
    "BriefingItem",
    "BriefingSection",
    "AudioSegment",
    "CompiledBriefing",
    "compile_briefing",
]
