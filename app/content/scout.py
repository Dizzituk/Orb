# FILE: app/content/scout.py
"""
Content Scout — AI-powered conversation analysis (Spec Section 4).

Analyses conversation transcripts to identify content opportunities.
Two modes:
1. Realtime tagging: lightweight pass during conversation
2. Deep analysis: comprehensive pass after conversation ends

Uses Gemini Flash for realtime (cheap, fast) and Gemini Pro for deep analysis.
"""
import json
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session

from app.content.models import (
    ContentConversation, ContentTag, ContentTopic, ContentPiece,
)
from app.content.service import (
    get_or_create_topic, create_content_piece,
    check_topic_coverage, record_position_evolution,
)
from app.content.scout_prompts import (
    REALTIME_TAG_SYSTEM, REALTIME_TAG_USER,
    DEEP_ANALYSIS_SYSTEM, DEEP_ANALYSIS_USER,
    TOPIC_CLASSIFY_SYSTEM, TOPIC_CLASSIFY_USER,
    EVOLUTION_DETECT_SYSTEM, EVOLUTION_DETECT_USER,
)

logger = logging.getLogger(__name__)


# ─── LLM CALL HELPER ───

async def _llm_call_json(
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-2.5-flash",
    provider: str = "google",
    max_tokens: int = 4000,
) -> Optional[Any]:
    """
    Make an LLM call expecting JSON response.
    Returns parsed JSON or None on failure.
    """
    from app.providers.registry import llm_call

    try:
        result = await llm_call(
            provider_id=provider,
            model_id=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )

        if not result or not result.text:
            logger.warning("[scout] LLM returned empty response")
            return None

        # Strip markdown code fences if present
        text = result.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        return json.loads(text)

    except json.JSONDecodeError as e:
        logger.error(f"[scout] Failed to parse LLM JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"[scout] LLM call failed: {e}")
        return None


# ═══════════════════════════════════════════════════
# REALTIME TAGGING (Spec Section 4.1)
# ═══════════════════════════════════════════════════

async def tag_transcript_segment(
    db: Session,
    conversation_id: str,
    transcript_segment: str,
    offset_seconds: float = 0.0,
) -> List[ContentTag]:
    """
    Lightweight realtime tagging of a conversation segment.
    Uses Gemini Flash for speed and cost efficiency.
    Returns list of created ContentTag objects.
    """
    if not transcript_segment or len(transcript_segment.strip()) < 50:
        return []

    prompt = REALTIME_TAG_USER.format(
        transcript_segment=transcript_segment
    )
    result = await _llm_call_json(
        REALTIME_TAG_SYSTEM, prompt,
        model="gemini-2.5-flash",
        max_tokens=2000,
    )

    if not result or not isinstance(result, list):
        return []

    tags = []
    for tag_data in result:
        try:
            tag = ContentTag(
                conversation_id=conversation_id,
                timestamp_offset_seconds=offset_seconds,
                tag_type=tag_data.get("tag_type", "quotable_moment"),
                excerpt=tag_data.get("excerpt", ""),
                strength_score=float(tag_data.get("strength_score", 0.5)),
            )
            db.add(tag)
            tags.append(tag)
        except Exception as e:
            logger.warning(f"[scout] Failed to create tag: {e}")

    if tags:
        db.commit()
        logger.info(
            f"[scout] Tagged {len(tags)} moments in conversation "
            f"{conversation_id}"
        )

    return tags


# ═══════════════════════════════════════════════════
# DEEP ANALYSIS (Spec Section 4.2)
# ═══════════════════════════════════════════════════

async def deep_analyse_conversation(
    db: Session,
    conversation_id: str,
) -> List[ContentPiece]:
    """
    Comprehensive post-conversation analysis using Gemini Pro.
    Identifies all content opportunities, classifies topics,
    scores pieces, and creates ContentPiece records.
    
    Returns list of created ContentPiece objects (status: identified).
    """
    conv = db.query(ContentConversation).get(conversation_id)
    if not conv:
        raise ValueError(f"Conversation {conversation_id} not found")

    if not conv.transcript_raw:
        logger.warning(
            f"[scout] No transcript for conversation {conversation_id}"
        )
        return []

    # Get known topics for context
    known_topics = db.query(ContentTopic).all()
    topic_names = [t.name for t in known_topics] if known_topics else []

    # Calculate duration
    duration_min = (conv.duration_seconds or 0) // 60

    # Run deep analysis
    prompt = DEEP_ANALYSIS_USER.format(
        full_transcript=conv.transcript_raw,
        duration_minutes=duration_min,
        known_topics=", ".join(topic_names) if topic_names else "None yet",
    )

    result = await _llm_call_json(
        DEEP_ANALYSIS_SYSTEM, prompt,
        model="gemini-2.5-pro",
        max_tokens=8000,
    )

    if not result or "opportunities" not in result:
        logger.warning(
            f"[scout] Deep analysis returned no opportunities for "
            f"conversation {conversation_id}"
        )
        conv.deep_analysis_done = True
        db.commit()
        return []

    pieces = []
    for opp in result["opportunities"]:
        try:
            piece = await _process_opportunity(
                db, conv, opp, topic_names
            )
            if piece:
                pieces.append(piece)
        except Exception as e:
            logger.error(f"[scout] Failed to process opportunity: {e}")

    # Mark conversation as analysed
    conv.deep_analysis_done = True
    conv.scout_processed = True
    db.commit()

    logger.info(
        f"[scout] Deep analysis complete for {conversation_id}: "
        f"{len(pieces)} opportunities identified"
    )
    return pieces


async def _process_opportunity(
    db: Session,
    conv: ContentConversation,
    opp: Dict[str, Any],
    known_topic_names: List[str],
) -> Optional[ContentPiece]:
    """
    Process a single content opportunity from deep analysis.
    Handles topic assignment, coverage checking, and piece creation.
    """
    title = opp.get("title", "Untitled")
    category = opp.get("content_category", "opinion")
    topics = opp.get("topics", [])
    scores = opp.get("scores", {})

    # Assign primary topic
    topic_id = None
    if topics:
        primary_topic_name = topics[0]
        topic = get_or_create_topic(db, primary_topic_name)
        topic_id = topic.id

        # Check for duplicate coverage
        coverage = check_topic_coverage(db, topic_id)
        if coverage.get("recommendation") == "wait_or_find_new_angle":
            logger.info(
                f"[scout] Topic '{primary_topic_name}' recently covered, "
                f"flagging for review"
            )
            # Still create the piece but note it in description
            opp_desc = opp.get("description", "")
            opp["description"] = (
                f"[NOTE: Topic recently covered — "
                f"last published {coverage.get('days_since', '?')} "
                f"days ago] {opp_desc}"
            )

    # Detect series match
    series_id = None
    series_suggestion = opp.get("series_suggestion", "none")
    if series_suggestion and series_suggestion != "none":
        from app.content.models import ContentSeries
        series = (
            db.query(ContentSeries)
            .filter(ContentSeries.name == series_suggestion)
            .first()
        )
        if series:
            series_id = series.id

    # Create the content piece
    piece = create_content_piece(
        db,
        title=title,
        content_category=category,
        description=opp.get("description"),
        topic_id=topic_id,
        series_id=series_id,
        source_conversation_ids=[conv.id],
        recommended_formats=opp.get("recommended_formats", []),
        suggested_hooks=opp.get("suggested_hooks", []),
        key_excerpts=opp.get("key_excerpts", []),
        scores={
            "originality": scores.get("originality", 0.5),
            "audience_relevance": scores.get("audience_relevance", 0.5),
            "emotional_impact": scores.get("emotional_impact", 0.5),
            "educational_value": scores.get("educational_value", 0.5),
            "overall": scores.get("overall", 0.5),
        },
    )

    return piece


# ═══════════════════════════════════════════════════
# TOPIC CLASSIFICATION
# ═══════════════════════════════════════════════════

async def classify_topic(
    db: Session,
    excerpt: str,
) -> Dict[str, Any]:
    """
    Classify an excerpt into an existing or new topic.
    Returns {"topic_name": str, "is_new": bool, "confidence": float}
    """
    known_topics = db.query(ContentTopic).all()
    topic_list = "\n".join(
        f"- {t.name}: {t.description or 'No description'}"
        for t in known_topics
    ) if known_topics else "No existing topics."

    prompt = TOPIC_CLASSIFY_USER.format(
        excerpt=excerpt[:2000],  # Limit excerpt size
        topic_list=topic_list,
    )

    result = await _llm_call_json(
        TOPIC_CLASSIFY_SYSTEM, prompt,
        model="gemini-2.5-flash",
        max_tokens=500,
    )

    if not result:
        return {"topic_name": "Uncategorised", "is_new": True, "confidence": 0.0}

    return result


# ═══════════════════════════════════════════════════
# POSITION EVOLUTION DETECTION (Spec Section 5.3)
# ═══════════════════════════════════════════════════

async def detect_position_evolution(
    db: Session,
    topic_id: str,
    current_excerpt: str,
) -> Dict[str, Any]:
    """
    Compare current discussion with historical positions on a topic.
    Returns evolution status and summary.
    """
    topic = db.query(ContentTopic).get(topic_id)
    if not topic:
        return {"status": "new_topic", "summary_of_change": "First discussion"}

    history = topic.position_history or []
    if not history:
        return {"status": "first_position", "summary_of_change": "No prior positions recorded"}

    # Format history for prompt
    history_text = "\n\n".join(
        f"[{h.get('date', 'unknown')}]: {h.get('summary', 'No summary')}"
        for h in history[-5:]  # Last 5 positions max
    )

    prompt = EVOLUTION_DETECT_USER.format(
        topic_name=topic.name,
        position_history=history_text,
        current_excerpt=current_excerpt[:3000],
    )

    result = await _llm_call_json(
        EVOLUTION_DETECT_SYSTEM, prompt,
        model="gemini-2.5-flash",
        max_tokens=1000,
    )

    if not result:
        return {"status": "unknown", "summary_of_change": "Analysis failed"}

    # If evolved or refined, record the new position
    status = result.get("status", "unknown")
    if status in ("evolved", "refined", "reversed"):
        record_position_evolution(
            db, topic_id,
            summary=result.get("summary_of_change", "Position updated"),
            key_arguments=result.get("new_elements", []),
        )

    return result
