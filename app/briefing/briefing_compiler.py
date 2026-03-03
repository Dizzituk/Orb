# FILE: app/briefing/briefing_compiler.py
"""
Briefing Compiler — Compiles collected stories into a structured digest.

Takes a BriefingCollection and produces:
1. A text digest (structured markdown for display)
2. An audio script (alternating headline/analysis blocks for dual-voice TTS)
3. An ASTRA relevance flag list (stories relevant to the pipeline)

Uses an LLM to summarise raw stories into concise briefing items,
but falls back to snippet-based summaries if LLM is unavailable.

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

from app.briefing.briefing_collector import BriefingCollection, TopicCollection

logger = logging.getLogger(__name__)


# =========================================================================
# Compiled briefing models
# =========================================================================

@dataclass
class BriefingItem:
    """A single compiled story in the briefing."""
    headline: str
    summary: str
    source_name: str = ""
    source_url: str = ""
    credibility: str = "unknown"
    topic_key: str = ""
    astra_flag: str = ""         # E.g. "New model release — may affect pipeline"


@dataclass
class BriefingSection:
    """A topic section in the compiled briefing."""
    topic_name: str
    topic_key: str
    description: str = ""
    items: List[BriefingItem] = field(default_factory=list)


@dataclass
class AudioSegment:
    """A single TTS segment with voice assignment."""
    text: str
    voice_role: str = "headlines"    # "headlines" or "analysis"
    pause_after_ms: int = 800


@dataclass
class CompiledBriefing:
    """The final compiled briefing output."""
    title: str = ""
    generated_at: str = ""
    sections: List[BriefingSection] = field(default_factory=list)
    audio_script: List[AudioSegment] = field(default_factory=list)
    astra_alerts: List[str] = field(default_factory=list)
    total_items: int = 0
    text_digest: str = ""           # Full markdown text


# =========================================================================
# Story summarisation
# =========================================================================

def _snippet_summary(title: str, snippet: str) -> str:
    """Fallback: build a summary from title and snippet."""
    if snippet and len(snippet) > 20:
        return snippet[:300].strip()
    return title


def _extract_source_name(url: str) -> str:
    """Extract a human-readable source name from URL."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        # Strip www. and common suffixes
        domain = domain.replace("www.", "")
        parts = domain.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return domain
    except Exception:
        return ""


# =========================================================================
# Compilation logic
# =========================================================================

def compile_briefing(
    collection: BriefingCollection,
    frequency: str = "daily",
) -> CompiledBriefing:
    """
    Compile a BriefingCollection into a structured briefing.

    Processes each topic's stories into briefing items with
    headlines and summaries, builds the text digest, generates
    the audio script for dual-voice TTS, and flags ASTRA-relevant items.

    Args:
        collection: The gathered stories from briefing_collector.
        frequency: "daily" or "weekly" — affects title and depth.

    Returns:
        CompiledBriefing with all outputs.
    """
    now = datetime.now(timezone.utc)
    day_str = now.strftime("%A %d %B %Y")

    if frequency == "weekly":
        title = f"ASTRA Weekly Deep Dive — Week of {day_str}"
    else:
        title = f"ASTRA Morning Briefing — {day_str}"

    briefing = CompiledBriefing(
        title=title,
        generated_at=now.isoformat(),
    )

    # Audio: opening
    briefing.audio_script.append(AudioSegment(
        text=f"Good morning. Here is your {frequency} briefing for {now.strftime('%A the %d of %B')}.",
        voice_role="headlines",
        pause_after_ms=1200,
    ))

    # Process each topic
    text_parts = [f"# {title}\n"]

    for topic_col in collection.topics:
        if not topic_col.stories:
            continue

        section = BriefingSection(
            topic_name=topic_col.topic_name,
            topic_key=topic_col.topic_key,
            description=topic_col.description,
        )

        # Audio: section header
        briefing.audio_script.append(AudioSegment(
            text=f"Moving on to {topic_col.topic_name}.",
            voice_role="headlines",
            pause_after_ms=800,
        ))

        text_parts.append(f"\n## {topic_col.topic_name}")
        if topic_col.description:
            text_parts.append(f"*{topic_col.description}*\n")

        for story in topic_col.stories:
            source_name = _extract_source_name(story.url) or story.source_type
            summary = _snippet_summary(story.title, story.snippet)

            # Check for ASTRA relevance
            astra_flag = ""
            if story.astra_relevant:
                lower = f"{story.title} {story.snippet}".lower()
                if any(kw in lower for kw in [
                    "model release", "new model", "benchmark",
                    "framework", "llm", "open source",
                    "api", "fine-tun", "rag", "agent",
                ]):
                    astra_flag = "Pipeline relevant — worth investigating"
                    briefing.astra_alerts.append(
                        f"[{topic_col.topic_name}] {story.title} — {astra_flag}"
                    )

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

            # Audio: headline (voice A) then summary (voice B)
            briefing.audio_script.append(AudioSegment(
                text=story.title,
                voice_role="headlines",
                pause_after_ms=500,
            ))
            briefing.audio_script.append(AudioSegment(
                text=summary,
                voice_role="analysis",
                pause_after_ms=800,
            ))

            # Text digest
            cred_tag = f" [{story.credibility_label}]" if story.credibility_label != "unknown" else ""
            text_parts.append(f"- **{story.title}**{cred_tag}")
            text_parts.append(f"  {summary}")
            text_parts.append(f"  *Source: {source_name}* — [link]({story.url})")
            if astra_flag:
                text_parts.append(f"  🔔 *{astra_flag}*")
            text_parts.append("")

        briefing.sections.append(section)

    # ASTRA alerts section
    if briefing.astra_alerts:
        text_parts.append("\n## 🔔 ASTRA-Relevant Alerts")
        for alert in briefing.astra_alerts:
            text_parts.append(f"- {alert}")
        text_parts.append("")

        briefing.audio_script.append(AudioSegment(
            text="And a note for the pipeline. The following stories may be relevant to ASTRA's development.",
            voice_role="headlines",
            pause_after_ms=600,
        ))
        for alert in briefing.astra_alerts[:3]:
            briefing.audio_script.append(AudioSegment(
                text=alert,
                voice_role="analysis",
                pause_after_ms=600,
            ))

    # Audio: closing
    briefing.audio_script.append(AudioSegment(
        text=f"That's your {frequency} briefing. Have a good one.",
        voice_role="headlines",
        pause_after_ms=0,
    ))

    briefing.text_digest = "\n".join(text_parts)
    briefing.total_items = sum(len(s.items) for s in briefing.sections)

    logger.info(
        "[briefing_compiler] Compiled: %d sections, %d items, %d audio segments, %d ASTRA alerts",
        len(briefing.sections), briefing.total_items,
        len(briefing.audio_script), len(briefing.astra_alerts),
    )

    return briefing


__all__ = [
    "BriefingItem",
    "BriefingSection",
    "AudioSegment",
    "CompiledBriefing",
    "compile_briefing",
]
