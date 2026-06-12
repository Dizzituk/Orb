# FILE: app/content/seed.py
# Purpose: Seed data for the Content Creation Pipeline.
# Called-by: main
# Depends-on: app.content.models, app.content.service
# Last-renovated: 2026-06-11
"""
Seed data for the Content Creation Pipeline.

Creates the initial content series defined in Spec Section 10.1
and the default style profile from Spec Section 11.1.
Called once at startup (idempotent).
"""
import logging
from sqlalchemy.orm import Session
from app.content.models import ContentSeries, StyleProfile
from app.content.service import ensure_default_style_profile

logger = logging.getLogger(__name__)

INITIAL_SERIES = [
    {
        "name": "Man in the Van",
        "description": (
            "Raw, unfiltered takes on AI, economics, society. "
            "Anchor footage with minimal cutaways. Authenticity is the brand."
        ),
        "categories": ["opinion"],
        "target_formats": ["youtube_short", "instagram_reel", "tiktok"],
        "target_platforms": ["youtube", "instagram", "tiktok"],
        "posting_cadence": "3_per_week",
    },
    {
        "name": "The Abundance Question",
        "description": (
            "Deep dives into economic transition: UBI vs UBD, "
            "capitalism's breaking point, post-labour economics. "
            "Heavy use of animated explainers."
        ),
        "categories": ["educational"],
        "target_formats": ["youtube_longform", "blog_post"],
        "target_platforms": ["youtube", "blog"],
        "posting_cadence": "1_per_week",
    },
    {
        "name": "AI for Humans",
        "description": (
            "Accessible explanations of AI concepts for "
            "non-technical audiences. What is AGI? How does "
            "an LLM work? What does this mean for your job?"
        ),
        "categories": ["educational"],
        "target_formats": ["youtube_short", "instagram_carousel"],
        "target_platforms": ["youtube", "instagram"],
        "posting_cadence": "2_per_week",
    },
    {
        "name": "The Build Log",
        "description": (
            "Documenting the ASTRA build process. Raw development "
            "footage, problem-solving, architecture decisions. "
            "Aimed at aspiring builders."
        ),
        "categories": ["documentary"],
        "target_formats": ["youtube_longform"],
        "target_platforms": ["youtube"],
        "posting_cadence": "1_per_week",
    },
    {
        "name": "From Van to Vision",
        "description": (
            "Personal journey content. Transitioning from delivery "
            "driving to AI development. Aimed at people considering "
            "similar transitions."
        ),
        "categories": ["documentary", "tutorial"],
        "target_formats": [
            "youtube_short", "youtube_longform",
            "instagram_reel", "blog_post",
        ],
        "target_platforms": ["youtube", "instagram", "blog"],
        "posting_cadence": "1_per_week",
    },
]


def seed_content_data(db: Session) -> dict:
    """
    Seed initial series and style profile. Idempotent.
    Returns counts of created items.
    """
    created_series = 0
    skipped_series = 0

    for series_data in INITIAL_SERIES:
        existing = (
            db.query(ContentSeries)
            .filter(ContentSeries.name == series_data["name"])
            .first()
        )
        if existing:
            skipped_series += 1
            continue

        series = ContentSeries(**series_data)
        db.add(series)
        created_series += 1

    db.commit()

    # Ensure default style profile exists
    ensure_default_style_profile(db)

    result = {
        "series_created": created_series,
        "series_skipped": skipped_series,
    }
    logger.info(f"[content] Seed complete: {result}")
    return result
