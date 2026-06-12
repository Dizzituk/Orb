# FILE: app/content/production/video_analyser.py
# Purpose: Video Analyser — Gemini watches a video and extracts full context.
# Called-by: app.content.distribution.youtube_router
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Video Analyser — Gemini watches a video and extracts full context.

Used for direct uploads where ASTRA needs to understand what
a video is about before generating metadata for YouTube.
Uploads to Gemini File API, then asks for comprehensive analysis.
"""
import os
import json
import time
import logging
from typing import Optional, Dict, Any

import google.generativeai as genai

logger = logging.getLogger(__name__)


async def analyse_video(
    video_path: str,
    user_hint: str = "",
) -> Dict[str, Any]:
    """
    Upload a video to Gemini and get full contextual analysis.

    Returns a dict with:
    - summary: 2-3 sentence overview
    - topics: list of key topics covered
    - suggested_title: AI-generated title optimised for YouTube
    - suggested_description: full description with keywords
    - suggested_tags: list of relevant tags
    - suggested_hashtags: list of hashtags for description
    - content_type: 'educational', 'entertainment', 'tutorial', etc.
    - key_moments: list of notable timestamps
    - target_audience: who this video is for
    - suggested_shorts: list of 2-3 sections for YouTube Shorts
    - category_id: YouTube category ID
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.error("[video_analyser] GOOGLE_API_KEY not set")
        return {"error": "GOOGLE_API_KEY not set"}

    genai.configure(api_key=api_key)

    # Step 1: Upload video to Gemini File API
    logger.info("[video_analyser] Uploading %s to Gemini...", os.path.basename(video_path))

    video_file = genai.upload_file(path=video_path)

    # Wait for processing
    while video_file.state.name == "PROCESSING":
        time.sleep(5)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        logger.error("[video_analyser] Upload failed: state=%s", video_file.state.name)
        return {"error": f"Video processing failed: {video_file.state.name}"}

    logger.info("[video_analyser] Video ready: %s", video_file.uri)

    # Step 2: Ask Gemini to analyse
    model = genai.GenerativeModel("gemini-3.1-pro-preview")

    hint_section = f"\nCreator's note: {user_hint}" if user_hint else ""

    prompt = f"""Watch this entire video carefully and provide a comprehensive analysis.
{hint_section}

You are a YouTube content strategist. Analyse this video and generate metadata
that will maximise its reach on YouTube's algorithm.

Respond in this exact JSON format (no markdown, no backticks, just JSON):
{{
    "summary": "2-3 sentence overview of what this video covers",
    "topics": ["topic1", "topic2", "topic3"],
    "suggested_title": "YouTube-optimised title (max 60 chars, front-load keywords, create curiosity)",
    "suggested_description": "Full YouTube description with keywords woven in naturally. First 2 lines are critical (visible above Show More). Include a brief hook, what the viewer will learn, and a call to action.",
    "suggested_tags": ["tag1", "tag2", "up to 25 relevant tags ordered by importance"],
    "suggested_hashtags": ["#Hashtag1", "#Hashtag2", "#Hashtag3", "#Hashtag4", "#Hashtag5"],
    "content_type": "educational or entertainment or tutorial or commentary or review or vlog",
    "target_audience": "who this video is for",
    "category_id": "YouTube category ID (28=Science&Tech, 22=People&Blogs, 27=Education, 24=Entertainment)",
    "key_moments": [
        {{"time": "0:00", "description": "Introduction"}},
        {{"time": "1:30", "description": "Key point"}}
    ],
    "mood": "informative, inspiring, serious, casual, etc.",
    "suggested_thumbnail_timestamp": 0.0,
    "suggested_shorts": [
        {{
            "start_seconds": 45.0,
            "end_seconds": 95.0,
            "title": "Catchy short title under 60 chars",
            "caption": "Brief hook caption for the short",
            "reason": "Why this section works as a standalone short"
        }}
    ]
}}

RULES:
- Title must be under 60 characters, front-load the most compelling keyword
- Use power words in the title (truth, secret, future, why, how)
- Description first 2 lines are everything — hook the viewer
- Tags: most important first, mix of broad and specific, 15-25 total
- Hashtags: 5 relevant ones for the description
- Category ID must match the content
- SHORTS: Identify 2-3 sections that work as standalone YouTube Shorts.
  CRITICAL RULES FOR SHORTS:
  * Each short MUST be between 15 and 60 seconds. Never under 15 seconds.
  * The end timestamp MUST fall at the END of a complete sentence.
    NEVER cut mid-word or mid-sentence. Listen to the audio carefully.
  * The start timestamp must begin at the START of a sentence.
  * Each short must make complete sense on its own without any other context.
  * Prefer sections with surprising facts, bold claims, or relatable analogies.
  * A short that is 30-45 seconds with a complete thought is better than
    a 10 second clip that cuts off mid-sentence.
- Return ONLY valid JSON"""

    try:
        response = model.generate_content([video_file, prompt])
        raw = response.text.strip()

        # Clean up response
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        result = json.loads(raw)
        logger.info(
            "[video_analyser] Analysis complete: '%s' (%d tags, %d topics)",
            result.get("suggested_title", "?"),
            len(result.get("suggested_tags", [])),
            len(result.get("topics", [])),
        )

        # Cleanup the uploaded file
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

        return result

    except json.JSONDecodeError as e:
        logger.error("[video_analyser] JSON parse failed: %s", e)
        logger.error("[video_analyser] Raw response: %s", raw[:500])

        # Cleanup
        try:
            genai.delete_file(video_file.name)
        except Exception:
            pass

        return {
            "error": "JSON parse failed",
            "raw_response": raw[:1000],
            "summary": "Analysis completed but response was malformed",
        }

    except Exception as e:
        logger.error("[video_analyser] Analysis failed: %s", e)
        return {"error": str(e)}
