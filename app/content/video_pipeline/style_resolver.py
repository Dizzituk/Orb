# FILE: app/content/video_pipeline/style_resolver.py
# Purpose: Style Resolver — loads and blends style profiles.
# Called-by: app.content.video_pipeline.orchestrator, app.content.video_pipeline.router, main
# Depends-on: app.content.video_pipeline.models, app.content.video_pipeline.prompts, app.db
# Last-renovated: 2026-06-11
"""
Style Resolver — loads and blends style profiles.

Manages the StyleProfile DB table and provides resolved
style parameters for the pipeline.
"""
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional, List

import google.generativeai as genai
from sqlalchemy import Column, String, DateTime, Text, Float
from sqlalchemy.orm import Session

from app.db import Base
from app.content.video_pipeline.models import StyleProfile
from app.content.video_pipeline.prompts import (
    STYLE_EXTRACTOR_SYSTEM, STYLE_EXTRACTOR_USER,
)

logger = logging.getLogger(__name__)


def _now():
    return datetime.now(timezone.utc)


class StyleProfileRecord(Base):
    """Stored style profile extracted from a reference video."""
    __tablename__ = "video_style_profiles"

    profile_id = Column(String, primary_key=True)
    source_filename = Column(String, nullable=False)
    profile_json = Column(Text, nullable=False)
    weight = Column(Float, default=1.0)
    created_at = Column(DateTime, default=_now)


def get_style_profile(
    db: Session,
    profile_id: Optional[str] = None,
) -> StyleProfile:
    """
    Get a resolved style profile.

    If profile_id is given, load that specific profile.
    If None, blend all profiles with recency weighting.
    If no profiles exist, return defaults.
    """
    if profile_id:
        record = db.query(StyleProfileRecord).get(profile_id)
        if record:
            data = json.loads(record.profile_json)
            return StyleProfile(profile_id=profile_id, **data)
        logger.warning(
            f"[style_resolver] Profile '{profile_id}' not found, "
            f"using defaults"
        )
        return StyleProfile()

    # Blend all profiles with recency weighting
    records = (
        db.query(StyleProfileRecord)
        .order_by(StyleProfileRecord.created_at.desc())
        .all()
    )

    if not records:
        logger.info("[style_resolver] No style profiles found, using defaults")
        return StyleProfile()

    if len(records) == 1:
        data = json.loads(records[0].profile_json)
        return StyleProfile(profile_id=records[0].profile_id, **data)

    # Weighted blend: newer profiles carry more weight
    return _blend_profiles(records)


def _blend_profiles(records: List[StyleProfileRecord]) -> StyleProfile:
    """
    Blend multiple style profiles with recency weighting.
    Numeric values are weighted averaged.
    String values use the most recent profile.
    """
    profiles = []
    weights = []
    for i, record in enumerate(records):
        data = json.loads(record.profile_json)
        profiles.append(data)
        # Recency weight: newest = 1.0, each older = 0.7x previous
        weights.append(record.weight * (0.7 ** i))

    total_weight = sum(weights)
    if total_weight == 0:
        total_weight = 1.0

    # Blend numeric fields
    blended = {}
    numeric_fields = [
        "avg_cut_duration_s", "intro_length_s", "outro_length_s",
        "font_size_px", "music_volume_ratio", "voice_volume",
    ]
    for field in numeric_fields:
        weighted_sum = sum(
            p.get(field, 0) * w for p, w in zip(profiles, weights)
        )
        blended[field] = round(weighted_sum / total_weight, 2)

    # String fields: use most recent (first in list)
    string_fields = [
        "segment_rhythm", "primary_transition", "secondary_transition",
        "transition_frequency", "colour_temperature", "saturation_level",
        "contrast_level", "lut_reference", "caption_style", "font_family",
        "caption_position", "caption_animation", "sfx_frequency",
        "music_genre_preference", "aspect_ratio_preference", "zoom_usage",
        "b_roll_density", "avatar_frequency", "overall_tone", "energy_level",
    ]
    for field in string_fields:
        blended[field] = profiles[0].get(field, "")

    # List fields: merge unique values from all profiles
    all_keywords = set()
    for p in profiles:
        all_keywords.update(p.get("visual_mood_keywords", []))
    blended["visual_mood_keywords"] = list(all_keywords)[:8]

    blended["profile_id"] = "blended"
    return StyleProfile(**blended)


def _is_youtube_url(url: str) -> bool:
    """Check if a string is a YouTube URL."""
    return any(
        domain in url.lower()
        for domain in ("youtube.com/watch", "youtu.be/", "youtube.com/shorts")
    )


async def extract_style_from_video(
    db: Session,
    video_path: str,
    profile_id: str,
    context: str = "",
) -> StyleProfile:
    """
    Extract style parameters from a reference video using Gemini.

    Accepts either:
    - A YouTube URL (analysed directly by Gemini, no download needed)
    - A local file path (uploaded via File API)

    Stores the result in the DB.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    genai.configure(api_key=api_key)

    is_youtube = _is_youtube_url(video_path)

    if is_youtube:
        # YouTube URLs are passed directly to Gemini — no upload needed.
        # Gemini natively analyses public YouTube videos via the Google
        # ecosystem. Only public videos are supported.
        # Uses the google-generativeai SDK's protos.Part (not google-genai).
        logger.info(
            f"[style_resolver] Analysing YouTube video directly: {video_path}"
        )
        from google.generativeai import protos
        video_part = protos.Part(
            file_data=protos.FileData(
                file_uri=video_path,
                mime_type="video/mp4",
            )
        )
        source_name = video_path
    else:
        # Local file — upload via File API
        logger.info(
            f"[style_resolver] Uploading local video for analysis: {video_path}"
        )
        video_file = genai.upload_file(video_path)

        # Wait for processing
        import time
        while video_file.state.name == "PROCESSING":
            time.sleep(5)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise RuntimeError(f"Video processing failed: {video_path}")

        video_part = video_file
        source_name = os.path.basename(video_path)

    user_prompt = STYLE_EXTRACTOR_USER.format(
        filename=source_name,
        context=context,
    )

    from app.content.video_pipeline.models import PIPELINE_GEMINI_MODEL
    model = genai.GenerativeModel(
        model_name=PIPELINE_GEMINI_MODEL,
        system_instruction=STYLE_EXTRACTOR_SYSTEM,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json",
        },
    )

    response = model.generate_content([video_part, user_prompt])
    raw_text = response.text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0].strip()
            data = json.loads(json_str)
        else:
            raise ValueError(f"Failed to parse style extraction: {raw_text[:200]}")

    # Store in DB
    record = StyleProfileRecord(
        profile_id=profile_id,
        source_filename=source_name,
        profile_json=json.dumps(data),
    )
    db.merge(record)
    db.commit()

    profile = StyleProfile(profile_id=profile_id, **data)
    logger.info(f"[style_resolver] Style profile saved: {profile_id}")

    # Clean up uploaded file (only for local uploads)
    if not is_youtube:
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

    return profile
