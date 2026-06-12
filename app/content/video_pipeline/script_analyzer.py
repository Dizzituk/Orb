# FILE: app/content/video_pipeline/script_analyzer.py
# Purpose: Script Analyzer — breaks a script into structured scene segments.
# Called-by: app.content.video_pipeline.orchestrator
# Depends-on: app.content.video_pipeline.models, app.content.video_pipeline.prompts
# Last-renovated: 2026-06-11
"""
Script Analyzer — breaks a script into structured scene segments.

Uses GPT-5.4 for script analysis (superior linguistics and
sentence boundary handling) via the OpenAI API.
Falls back to Gemini if OpenAI is unavailable.
"""
import json
import logging
import os
from typing import Optional

from app.content.video_pipeline.models import (
    ScenePlan, SceneSegment, StyleProfile,
)
from app.content.video_pipeline.prompts import (
    SCRIPT_ANALYZER_SYSTEM, SCRIPT_ANALYZER_USER,
)

logger = logging.getLogger(__name__)

# GPT-5.4 for script analysis — best at linguistics and
# understanding sentence boundaries. Gemini as fallback.
DEFAULT_MODEL = "gpt-5.4"


def _get_model_name() -> str:
    """Get analyzer model."""
    return os.getenv("VIDEO_PIPELINE_ANALYZER_MODEL", DEFAULT_MODEL)


def _build_style_notes(style: Optional[StyleProfile]) -> str:
    """Convert style profile into brief notes for the analyzer."""
    if not style:
        return "No style profile set. Use sensible defaults."

    return (
        f"Pacing: {style.segment_rhythm} rhythm, "
        f"avg cut ~{style.avg_cut_duration_s}s. "
        f"Intro: ~{style.intro_length_s}s, Outro: ~{style.outro_length_s}s. "
        f"Tone: {style.overall_tone}, energy: {style.energy_level}. "
        f"B-roll density: {style.b_roll_density}. "
        f"Avatar usage: {style.avatar_frequency}. "
        f"Mood: {', '.join(style.visual_mood_keywords)}."
    )


async def analyze_script(
    script_text: str,
    title: str,
    target_platform: str = "youtube_longform",
    target_duration_s: Optional[int] = None,
    style_profile: Optional[StyleProfile] = None,
) -> ScenePlan:
    """
    Analyse a script and produce a structured scene plan.

    Args:
        script_text: The full script text
        title: Video title
        target_platform: Target format (youtube_longform, youtube_short, etc.)
        target_duration_s: Desired total duration, or None to auto-estimate
        style_profile: Style parameters to guide segmentation

    Returns:
        ScenePlan with segments ready for asset resolution
    """
    model_name = _get_model_name()
    style_notes = _build_style_notes(style_profile)

    # Build the user prompt
    user_prompt = SCRIPT_ANALYZER_USER.format(
        title=title,
        target_platform=target_platform,
        target_duration_s=target_duration_s or "auto (estimate from script length)",
        style_notes=style_notes,
        script_text=script_text,
    )

    logger.info(
        f"[script_analyzer] Analyzing script: '{title}' "
        f"({len(script_text)} chars, model={model_name})"
    )

    # Use GPT-5.4 via OpenAI API (falls back to Gemini)
    use_openai = model_name.startswith("gpt")

    if use_openai:
        from openai import AsyncOpenAI
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        client = AsyncOpenAI(api_key=api_key)
        response = await client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SCRIPT_ANALYZER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_text = response.choices[0].message.content.strip()
    else:
        # Gemini fallback
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SCRIPT_ANALYZER_SYSTEM,
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )
        response = model.generate_content(user_prompt)
        raw_text = response.text.strip()

    # Parse JSON response
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code block
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0].strip()
            data = json.loads(json_str)
        elif "```" in raw_text:
            json_str = raw_text.split("```")[1].split("```")[0].strip()
            data = json.loads(json_str)
        else:
            raise ValueError(
                f"[script_analyzer] Failed to parse response as JSON: "
                f"{raw_text[:200]}..."
            )

    # Build ScenePlan from response
    segments = []
    for seg_data in data.get("segments", []):
        segments.append(SceneSegment(**seg_data))

    plan = ScenePlan(
        title=data.get("title", title),
        total_segments=len(segments),
        estimated_total_duration_s=data.get(
            "estimated_total_duration_s",
            sum(s.estimated_duration_s for s in segments),
        ),
        segments=segments,
        metadata={
            "model": model_name,
            "target_platform": target_platform,
            "script_length_chars": len(script_text),
        },
    )

    logger.info(
        f"[script_analyzer] Scene plan ready: {plan.total_segments} segments, "
        f"~{plan.estimated_total_duration_s:.0f}s total"
    )
    return plan
