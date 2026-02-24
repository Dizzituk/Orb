# FILE: app/content/production/draft_writer.py
"""
Draft Writer — AI-powered text content (Spec Section 7.2).

Produces first drafts for text-based content in the user's
authentic voice. Operates from voice profile in style settings.

Output formats: blog posts, Instagram captions, thread breakdowns,
video scripts with cutaway placement markers.
"""
import json
import logging
from typing import Dict, Any, Optional, List

from sqlalchemy.orm import Session

from app.content.models import ContentPiece, StyleProfile
from app.content.service import ensure_default_style_profile

logger = logging.getLogger(__name__)


# ─── PROMPTS ───

DRAFT_SYSTEM = """You are a ghostwriter producing content in a specific voice.

Voice Profile:
{voice_profile}

CRITICAL RULES:
- Write EXACTLY as this person speaks — casual, direct, passionate
- Short sentences. Rhetorical questions. Build arguments step by step.
- Use everyday analogies (delivery driving, gym, cooking) not academic references
- Occasional mild profanity for emphasis is fine — never forced
- Never use corporate buzzwords, marketing speak, or LinkedIn-style platitudes
- The reader should feel like someone is talking to them, not lecturing them

You are writing a {format_type} about: {topic}"""

BLOG_USER = """Write a blog post based on this content:

Title: {title}
Key arguments:
{key_arguments}

Key excerpts from the original conversation:
{key_excerpts}

Requirements:
- 800-2000 words
- Conversational tone throughout
- Clear headings that sound natural (not SEO-stuffed)
- Open with a hook that grabs attention
- Close with a specific thought or question (not a generic CTA)
- Include the strongest original quotes naturally"""

CAPTION_USER = """Write an Instagram caption based on this content:

Title: {title}
Key point: {key_arguments}

Requirements:
- 150-300 words
- Strong opening hook (first line must stop the scroll)
- Line breaks for readability
- End with a specific question to drive comments
- Include 5-8 relevant hashtags at the end
- No emojis unless they add genuine meaning"""

THREAD_USER = """Write a thread breakdown based on this content:

Title: {title}
Key arguments:
{key_arguments}

Requirements:
- 5-15 connected posts
- Each post makes ONE point and can stand alone
- First post is the hook — must be compelling enough to click
- Number each post (1/, 2/, etc.)
- Build to a conclusion
- Final post: summary + specific engagement prompt
- Each post under 280 characters"""

SCRIPT_USER = """Write a video script based on this content:

Title: {title}
Duration target: {duration} minutes
Key arguments:
{key_arguments}

Key excerpts:
{key_excerpts}

Requirements:
- Structured as talking points, not word-for-word script
- Mark [CUTAWAY: description] where visual supports would enhance the point
- Open with the strongest hook in the first 5 seconds
- Include natural transitions between sections
- Mark [CHAPTER: title] for YouTube chapter markers
- Close with specific engagement prompt"""


FORMAT_TEMPLATES = {
    "blog_post": BLOG_USER,
    "instagram_caption": CAPTION_USER,
    "twitter_thread": THREAD_USER,
    "video_script": SCRIPT_USER,
}


async def generate_draft(
    db: Session,
    piece_id: str,
    format_type: str = "blog_post",
    duration_minutes: int = 5,
) -> Optional[str]:
    """
    Generate a text draft for a content piece.
    Returns the draft text, also stores it on the piece.
    """
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise ValueError(f"Content piece {piece_id} not found")

    # Get voice profile from active style
    profile = ensure_default_style_profile(db)
    voice = json.dumps(profile.voice_profile or {}, indent=2)

    # Build system prompt with voice
    system = DRAFT_SYSTEM.format(
        voice_profile=voice,
        format_type=format_type,
        topic=piece.title,
    )

    # Build format-specific user prompt
    template = FORMAT_TEMPLATES.get(format_type, BLOG_USER)
    key_args = "\n".join(
        f"- {e}" for e in (piece.key_excerpts or [])[:8]
    ) or "No key arguments available"

    user_prompt = template.format(
        title=piece.title,
        key_arguments=key_args,
        key_excerpts=json.dumps(piece.key_excerpts or [], indent=2),
        duration=duration_minutes,
    )

    # Call LLM
    from app.content.scout import _llm_call_json
    # Draft writing needs text, not JSON — use raw LLM call
    from app.providers.registry import llm_call

    try:
        result = await llm_call(
            provider_id="google",
            model_id="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
            temperature=0.7,  # Slightly creative for writing
        )

        if not result or not result.text:
            logger.warning(f"[draft_writer] Empty response for {piece_id}")
            return None

        draft = result.text.strip()

    except Exception as e:
        logger.error(f"[draft_writer] LLM call failed: {e}")
        return None

    # Store draft on piece
    piece.draft_text = draft
    db.commit()

    logger.info(
        f"[draft_writer] Generated {format_type} draft for "
        f"'{piece.title}' ({len(draft)} chars)"
    )
    return draft


async def refine_draft(
    db: Session,
    piece_id: str,
    feedback: str,
) -> Optional[str]:
    """
    Refine an existing draft based on user feedback.
    Uses the current draft + feedback to produce an improved version.
    """
    piece = db.query(ContentPiece).get(piece_id)
    if not piece or not piece.draft_text:
        raise ValueError(f"No draft found for piece {piece_id}")

    profile = ensure_default_style_profile(db)
    voice = json.dumps(profile.voice_profile or {}, indent=2)

    system = (
        f"You are refining a content draft. Maintain the original voice.\n"
        f"Voice Profile:\n{voice}\n\n"
        f"Apply the user's feedback while keeping the core message and "
        f"authentic voice intact."
    )

    user_prompt = (
        f"Current draft:\n\n{piece.draft_text}\n\n"
        f"---\n\nFeedback to apply:\n{feedback}\n\n"
        f"Produce the refined version."
    )

    from app.providers.registry import llm_call

    try:
        result = await llm_call(
            provider_id="google",
            model_id="gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
            temperature=0.5,
        )

        if not result or not result.text:
            return None

        refined = result.text.strip()
        piece.draft_text = refined
        db.commit()

        logger.info(
            f"[draft_writer] Refined draft for '{piece.title}'"
        )
        return refined

    except Exception as e:
        logger.error(f"[draft_writer] Refinement failed: {e}")
        return None
