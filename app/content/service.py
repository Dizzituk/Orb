# FILE: app/content/service.py
# Purpose: Content Pipeline Service — Core business logic.
# Called-by: app.content.distribution.publisher, app.content.production.cutaway_gen, app.content.production.draft_writer, app.content.production.format_converter (+3 more)
# Depends-on: app.content.models
# Last-renovated: 2026-06-11
"""
Content Pipeline Service — Core business logic.

Handles conversation tracking, topic management, content piece
lifecycle, and series operations. The deterministic backbone
of the content creation pipeline.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.content.models import (
    ContentConversation, ContentTag, ContentTopic,
    ContentPiece, ContentOutput, ContentAnalytics,
    ContentSeries, ContentApproval, StyleProfile,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# CONVERSATION MANAGEMENT
# ═══════════════════════════════════════════════════

def start_conversation(
    db: Session,
    linked_video_path: Optional[str] = None,
) -> ContentConversation:
    """Begin tracking a new conversation session."""
    conv = ContentConversation(
        linked_video_path=linked_video_path,
    )
    db.add(conv)
    db.commit()
    db.refresh(conv)
    logger.info(f"[content] Started conversation {conv.id}")
    return conv


def end_conversation(
    db: Session,
    conversation_id: str,
    transcript_raw: Optional[str] = None,
) -> ContentConversation:
    """
    Mark a conversation as ended.
    Stores the raw transcript and calculates duration.
    """
    conv = db.query(ContentConversation).get(conversation_id)
    if not conv:
        raise ValueError(f"Conversation {conversation_id} not found")

    conv.timestamp_end = datetime.now(timezone.utc)
    if conv.timestamp_start:
        delta = conv.timestamp_end - conv.timestamp_start
        conv.duration_seconds = int(delta.total_seconds())

    if transcript_raw:
        conv.transcript_raw = transcript_raw

    db.commit()
    db.refresh(conv)
    logger.info(
        f"[content] Ended conversation {conv.id} "
        f"(duration: {conv.duration_seconds}s)"
    )
    return conv


def get_conversations_for_date(
    db: Session,
    date: datetime,
) -> List[ContentConversation]:
    """Get all conversations from a specific date."""
    start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)
    return (
        db.query(ContentConversation)
        .filter(
            ContentConversation.timestamp_start >= start,
            ContentConversation.timestamp_start < end,
        )
        .order_by(ContentConversation.timestamp_start.asc())
        .all()
    )


# ═══════════════════════════════════════════════════
# TOPIC MANAGEMENT
# ═══════════════════════════════════════════════════

def get_or_create_topic(
    db: Session,
    name: str,
    description: Optional[str] = None,
) -> ContentTopic:
    """
    Find existing topic by name or create a new one.
    If found, increments discussion_count and updates last_discussed.
    """
    # Case-insensitive search
    topic = (
        db.query(ContentTopic)
        .filter(func.lower(ContentTopic.name) == name.lower())
        .first()
    )

    if topic:
        topic.discussion_count += 1
        topic.last_discussed = datetime.now(timezone.utc)
        db.commit()
        db.refresh(topic)
        logger.info(f"[content] Updated topic '{name}' (count: {topic.discussion_count})")
        return topic

    topic = ContentTopic(
        name=name,
        description=description,
    )
    db.add(topic)
    db.commit()
    db.refresh(topic)
    logger.info(f"[content] Created topic '{name}'")
    return topic


def record_position_evolution(
    db: Session,
    topic_id: str,
    summary: str,
    key_arguments: List[str] = None,
    evidence_cited: List[str] = None,
) -> ContentTopic:
    """
    Record an evolution in the user's position on a topic.
    Appends to position_history and updates maturity_score.
    """
    topic = db.query(ContentTopic).get(topic_id)
    if not topic:
        raise ValueError(f"Topic {topic_id} not found")

    history = topic.position_history or []
    history.append({
        "date": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "key_arguments": key_arguments or [],
        "evidence_cited": evidence_cited or [],
    })
    topic.position_history = history

    # Maturity increases with each position recording (capped at 1.0)
    topic.maturity_score = min(1.0, topic.maturity_score + 0.1)
    topic.updated_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(topic)
    return topic


def check_topic_coverage(
    db: Session,
    topic_id: str,
    duplicate_window_days: int = 30,
) -> Dict[str, Any]:
    """
    Check if a topic has been recently covered.
    Returns coverage status and recommendations.
    (Spec Section 5.3 — Duplicate and Evolution Detection)
    """
    topic = db.query(ContentTopic).get(topic_id)
    if not topic:
        return {"status": "new_topic", "recommendation": "proceed"}

    # Find published pieces on this topic
    published = (
        db.query(ContentPiece)
        .filter(
            ContentPiece.topic_id == topic_id,
            ContentPiece.status == "published",
        )
        .order_by(ContentPiece.published_at.desc())
        .all()
    )

    if not published:
        return {
            "status": "never_published",
            "recommendation": "proceed",
            "discussion_count": topic.discussion_count,
        }

    last_published = published[0]
    days_since = (
        datetime.now(timezone.utc) - last_published.published_at
    ).days if last_published.published_at else 999

    if days_since < duplicate_window_days:
        return {
            "status": "recently_published",
            "recommendation": "wait_or_find_new_angle",
            "last_published_at": last_published.published_at.isoformat(),
            "days_since": days_since,
            "published_count": len(published),
        }

    return {
        "status": "eligible_for_repost",
        "recommendation": "proceed_with_refined_take",
        "last_published_at": last_published.published_at.isoformat(),
        "days_since": days_since,
        "published_count": len(published),
    }


# ═══════════════════════════════════════════════════
# CONTENT PIECE LIFECYCLE
# ═══════════════════════════════════════════════════

def create_content_piece(
    db: Session,
    title: str,
    content_category: str,
    description: Optional[str] = None,
    topic_id: Optional[str] = None,
    series_id: Optional[str] = None,
    source_conversation_ids: List[str] = None,
    source_tag_ids: List[str] = None,
    recommended_formats: List[str] = None,
    suggested_hooks: List[str] = None,
    key_excerpts: List[str] = None,
    scores: Optional[Dict[str, float]] = None,
) -> ContentPiece:
    """Create a new content piece (status: identified)."""
    piece = ContentPiece(
        title=title,
        description=description,
        content_category=content_category,
        topic_id=topic_id,
        series_id=series_id,
        source_conversation_ids=source_conversation_ids or [],
        source_tag_ids=source_tag_ids or [],
        recommended_formats=recommended_formats or [],
        suggested_hooks=suggested_hooks or [],
        key_excerpts=key_excerpts or [],
    )

    if scores:
        piece.originality_score = scores.get("originality")
        piece.audience_relevance_score = scores.get("audience_relevance")
        piece.emotional_impact_score = scores.get("emotional_impact")
        piece.educational_value_score = scores.get("educational_value")
        piece.overall_score = scores.get("overall")

    db.add(piece)
    db.commit()
    db.refresh(piece)
    logger.info(f"[content] Created piece '{title}' ({content_category})")
    return piece


def transition_piece_status(
    db: Session,
    piece_id: str,
    new_status: str,
    user_notes: Optional[str] = None,
    rejection_reason: Optional[str] = None,
) -> ContentPiece:
    """Move a content piece to a new lifecycle status."""
    piece = db.query(ContentPiece).get(piece_id)
    if not piece:
        raise ValueError(f"Content piece {piece_id} not found")

    old_status = piece.status
    piece.status = new_status
    piece.updated_at = datetime.now(timezone.utc)

    if user_notes:
        piece.user_notes = user_notes
    if rejection_reason:
        piece.rejection_reason = rejection_reason

    if new_status == "approved":
        piece.approved_at = datetime.now(timezone.utc)
    elif new_status == "published":
        piece.published_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(piece)
    logger.info(
        f"[content] Piece '{piece.title}': {old_status} → {new_status}"
    )
    return piece


def get_pieces_by_status(
    db: Session,
    status: str,
    limit: int = 50,
) -> List[ContentPiece]:
    """Get content pieces filtered by status."""
    return (
        db.query(ContentPiece)
        .filter(ContentPiece.status == status)
        .order_by(ContentPiece.updated_at.desc())
        .limit(limit)
        .all()
    )


# ═══════════════════════════════════════════════════
# APPROVAL TRACKING (Preference Learning)
# ═══════════════════════════════════════════════════

def log_approval(
    db: Session,
    piece_id: str,
    approval_type: str,
    decision: str,
    proposed: Optional[Dict] = None,
    modifications: Optional[Dict] = None,
    reason: Optional[str] = None,
) -> ContentApproval:
    """Log an approval decision for preference learning."""
    approval = ContentApproval(
        piece_id=piece_id,
        approval_type=approval_type,
        decision=decision,
        proposed=proposed,
        modifications=modifications,
        reason=reason,
    )
    db.add(approval)
    db.commit()
    db.refresh(approval)
    logger.info(
        f"[content] Approval logged: {approval_type} → {decision} "
        f"(piece: {piece_id})"
    )
    return approval


def get_approval_stats(
    db: Session,
    approval_type: Optional[str] = None,
    days: int = 90,
) -> Dict[str, Any]:
    """
    Get approval rate statistics for preference learning.
    Used to determine when to transition to autonomous mode.
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    query = db.query(ContentApproval).filter(
        ContentApproval.created_at >= since
    )
    if approval_type:
        query = query.filter(ContentApproval.approval_type == approval_type)

    approvals = query.all()
    total = len(approvals)
    if total == 0:
        return {"total": 0, "approval_rate": 0.0}

    approved = sum(1 for a in approvals if a.decision == "approved")
    rejected = sum(1 for a in approvals if a.decision == "rejected")
    modified = sum(1 for a in approvals if a.decision == "modified")

    return {
        "total": total,
        "approved": approved,
        "rejected": rejected,
        "modified": modified,
        "approval_rate": approved / total,
        "autonomous_ready": (approved / total) >= 0.9 and total >= 20,
    }


# ═══════════════════════════════════════════════════
# SERIES MANAGEMENT
# ═══════════════════════════════════════════════════

def create_series(
    db: Session,
    name: str,
    description: Optional[str] = None,
    categories: List[str] = None,
    target_formats: List[str] = None,
    target_platforms: List[str] = None,
    posting_cadence: Optional[str] = None,
) -> ContentSeries:
    """Create a new content series."""
    series = ContentSeries(
        name=name,
        description=description,
        categories=categories or [],
        target_formats=target_formats or [],
        target_platforms=target_platforms or [],
        posting_cadence=posting_cadence,
    )
    db.add(series)
    db.commit()
    db.refresh(series)
    logger.info(f"[content] Created series '{name}'")
    return series


def get_all_series(db: Session, active_only: bool = True) -> List[ContentSeries]:
    """Get all content series."""
    query = db.query(ContentSeries)
    if active_only:
        query = query.filter(ContentSeries.active == True)
    return query.order_by(ContentSeries.name).all()


# ═══════════════════════════════════════════════════
# STYLE PROFILE
# ═══════════════════════════════════════════════════

def get_active_style_profile(db: Session) -> Optional[StyleProfile]:
    """Get the currently active style profile."""
    return (
        db.query(StyleProfile)
        .filter(StyleProfile.active == True)
        .first()
    )


def ensure_default_style_profile(db: Session) -> StyleProfile:
    """
    Create default style profile if none exists.
    Based on Spec Section 11.1 defaults.
    """
    existing = get_active_style_profile(db)
    if existing:
        return existing

    profile = StyleProfile(
        name="default",
        video_params={
            "cut_frequency_shorts_seconds": [2, 4],
            "cut_frequency_longform_seconds": [5, 8],
            "max_same_shot_sentences": 3,
            "caption_font": "bold_sans_serif",
            "caption_size_pt": 21,
            "caption_colour": "white_dark_outline",
            "caption_position": "centre_bottom",
            "colour_grading": "neutral_warm",
            "background_music_style": "lofi_ambient",
            "background_music_volume": 0.15,
            "hook_placement": "strongest_moment_first_2s",
            "engagement_prompt_style": "specific_question",
        },
        voice_profile={
            "vocabulary": "conversational_direct",
            "jargon_handling": "explain_when_used",
            "sentence_structure": "short_rhetorical_building",
            "tone": "passionate_grounded_occasional_profanity",
            "analogy_sources": [
                "delivery_driving", "gym_fitness",
                "cooking_catering", "everyday_experience",
            ],
            "signature_phrases": [],
        },
        active=True,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    logger.info("[content] Created default style profile")
    return profile
