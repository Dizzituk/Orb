# FILE: app/content/distribution/youtube_optimiser.py
"""
YouTube Metadata Optimiser.

Uses an LLM to analyse video title/description and generate
optimised tags, improved descriptions, and SEO suggestions.

Called automatically when manually queueing a video, or
on demand for existing videos.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


async def optimise_metadata(
    title: str,
    description: str = "",
    existing_tags: Optional[List[str]] = None,
    category: str = "science_tech",
) -> Dict[str, Any]:
    """
    Generate optimised YouTube metadata using an LLM.

    Returns:
        {
            "tags": [...],
            "optimised_title": "...",
            "optimised_description": "...",
            "hashtags": [...],
            "suggested_category_id": "28",
        }
    """
    import httpx

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Fall back to basic tag generation
        return _fallback_tags(title, description, existing_tags)

    prompt = _build_optimisation_prompt(
        title, description, existing_tags, category
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/"
                "models/gemini-2.0-flash:generateContent",
                params={"key": api_key},
                json={
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {
                        "temperature": 0.4,
                        "maxOutputTokens": 1024,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )

            return _parse_llm_response(text, title, description)

    except Exception as e:
        logger.error("[youtube-optimiser] LLM call failed: %s", e)
        return _fallback_tags(title, description, existing_tags)


def _build_optimisation_prompt(
    title: str,
    description: str,
    existing_tags: Optional[List[str]],
    category: str,
) -> str:
    """Build the prompt for the LLM optimisation call."""
    tags_section = ""
    if existing_tags:
        tags_section = f"\nExisting tags: {', '.join(existing_tags)}"

    return f"""You are a YouTube SEO expert. Analyse the following video metadata and generate optimised tags, title, and description for maximum discoverability.

Video Title: {title}
Description: {description[:500]}
Category: {category}{tags_section}

Respond in this exact JSON format (no markdown, no backticks):
{{
    "tags": ["tag1", "tag2", "tag3", ...],
    "optimised_title": "improved title if needed, or original if good",
    "optimised_description": "improved description with keywords",
    "hashtags": ["#hashtag1", "#hashtag2", "#hashtag3"],
    "suggested_category_id": "28"
}}

Rules:
- Generate 15-30 relevant tags mixing broad and specific terms
- Tags should include: topic keywords, related searches, trending terms
- Keep the title under 100 characters, front-load keywords
- Description should include key terms naturally in the first 2 lines
- Include 3-5 hashtags that are relevant and searchable
- Category IDs: 28=Science&Tech, 27=Education, 24=Entertainment, 22=People&Blogs
- Return ONLY valid JSON, no other text"""


def _parse_llm_response(
    text: str,
    original_title: str,
    original_description: str,
) -> Dict[str, Any]:
    """Parse the LLM JSON response with fallback handling."""
    # Strip markdown fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)

        # Validate and sanitise
        return {
            "tags": result.get("tags", [])[:30],
            "optimised_title": (
                result.get("optimised_title", original_title)[:100]
            ),
            "optimised_description": result.get(
                "optimised_description", original_description
            ),
            "hashtags": result.get("hashtags", [])[:5],
            "suggested_category_id": result.get(
                "suggested_category_id", "28"
            ),
        }

    except json.JSONDecodeError:
        logger.warning(
            "[youtube-optimiser] Failed to parse LLM response"
        )
        return _fallback_tags(
            original_title, original_description, None
        )


def _fallback_tags(
    title: str,
    description: str,
    existing_tags: Optional[List[str]],
) -> Dict[str, Any]:
    """Generate basic tags without LLM (keyword extraction)."""
    import re

    # Simple keyword extraction from title and description
    text = f"{title} {description}".lower()
    # Remove common stop words and short words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in",
        "on", "at", "to", "for", "of", "and", "or", "but",
        "with", "this", "that", "from", "how", "what", "why",
        "when", "where", "who", "it", "its", "my", "your",
    }
    words = re.findall(r"[a-z]+", text)
    keywords = [
        w for w in words
        if w not in stop_words and len(w) > 3
    ]

    # Deduplicate preserving order
    seen = set()
    unique = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)

    tags = (existing_tags or []) + unique[:20]

    return {
        "tags": tags[:30],
        "optimised_title": title,
        "optimised_description": description,
        "hashtags": [f"#{w}" for w in unique[:3]],
        "suggested_category_id": "28",
    }

