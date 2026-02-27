# FILE: app/content/engagement/responder.py
"""
Template-Based Auto-Response Engine.

Deterministic response selection from pre-approved templates.
Rotates through templates to feel natural and human.
Staggers response times to avoid bot-like patterns.

Design principle: Taz controls the voice. The system only
picks from templates he's approved and times the delivery.
"""
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.content.engagement.models import (
    EngagementComment, EngagementResponse, EngagementTemplate,
)

logger = logging.getLogger(__name__)

# Response timing constraints
MIN_DELAY_MINUTES = 15      # Don't respond instantly (looks bot-like)
MAX_DELAY_MINUTES = 180     # Respond within 3 hours
MAX_RESPONSES_PER_HOUR = 4  # Cap to look natural
MAX_RESPONSES_PER_DAY = 15  # Daily limit across all platforms


# ═══════════════════════════════════════════════════
# TEMPLATE SELECTION
# ═══════════════════════════════════════════════════

def select_template(
    db: Session,
    sentiment: str,
    platform: str,
) -> Optional[EngagementTemplate]:
    """
    Select the best template for a response.
    Uses least-recently-used rotation to vary responses.
    """
    templates = (
        db.query(EngagementTemplate)
        .filter(
            and_(
                EngagementTemplate.sentiment_tier == sentiment,
                EngagementTemplate.active.is_(True),
            )
        )
        .all()
    )

    # Filter by platform compatibility
    compatible = [
        t for t in templates
        if platform in (t.platforms or [])
    ]

    if not compatible:
        logger.warning(
            f"[responder] No templates for {sentiment}/{platform}"
        )
        return None

    # Sort by least recently used, then pick from top 3 randomly
    compatible.sort(key=lambda t: t.last_used_at or datetime.min)
    pool = compatible[:max(3, len(compatible) // 2)]

    return random.choice(pool)


def render_template(
    template: EngagementTemplate,
    author_name: Optional[str] = None,
) -> str:
    """
    Render a template with variable substitution.
    Supported placeholders: {author}
    """
    text = template.text

    if "{author}" in text and author_name:
        text = text.replace("{author}", author_name)
    elif "{author}" in text:
        # Remove the placeholder if no author name available
        text = text.replace("{author} ", "").replace("{author}", "")

    return text.strip()


# ═══════════════════════════════════════════════════
# RATE LIMITING
# ═══════════════════════════════════════════════════

def check_rate_limits(db: Session) -> Dict[str, Any]:
    """
    Check if we're within auto-response rate limits.
    Returns dict with 'allowed' bool and current counts.
    """
    now = datetime.now(timezone.utc)
    hour_ago = now - timedelta(hours=1)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    hourly_count = (
        db.query(EngagementResponse)
        .filter(EngagementResponse.sent_at >= hour_ago)
        .count()
    )

    daily_count = (
        db.query(EngagementResponse)
        .filter(EngagementResponse.sent_at >= day_start)
        .count()
    )

    return {
        "allowed": (
            hourly_count < MAX_RESPONSES_PER_HOUR
            and daily_count < MAX_RESPONSES_PER_DAY
        ),
        "hourly_count": hourly_count,
        "hourly_limit": MAX_RESPONSES_PER_HOUR,
        "daily_count": daily_count,
        "daily_limit": MAX_RESPONSES_PER_DAY,
    }


# ═══════════════════════════════════════════════════
# RESPONSE SCHEDULING
# ═══════════════════════════════════════════════════

def calculate_response_time() -> datetime:
    """
    Calculate a natural-looking response time.
    Randomised delay between MIN and MAX to look human.
    """
    delay_minutes = random.randint(MIN_DELAY_MINUTES, MAX_DELAY_MINUTES)
    return datetime.now(timezone.utc) + timedelta(minutes=delay_minutes)


# ═══════════════════════════════════════════════════
# AUTO-RESPOND
# ═══════════════════════════════════════════════════

def prepare_auto_response(
    db: Session,
    comment: EngagementComment,
) -> Optional[EngagementResponse]:
    """
    Prepare an auto-response for a positive comment.
    Does NOT send it — just creates the DB record.
    Actual sending is handled by the dispatcher.

    Returns None if:
    - No suitable template found
    - Rate limits exceeded
    - Comment already responded to
    """
    if comment.auto_responded:
        return None

    # Only auto-respond to positive comments
    if comment.sentiment != "positive":
        return None

    # Check rate limits
    limits = check_rate_limits(db)
    if not limits["allowed"]:
        logger.info(
            f"[responder] Rate limit reached: "
            f"{limits['hourly_count']}/hr, {limits['daily_count']}/day"
        )
        return None

    # Select and render template
    template = select_template(db, "positive", comment.platform)
    if not template:
        return None

    response_text = render_template(template, comment.author_name)

    # Create response record
    response = EngagementResponse(
        comment_id=comment.id,
        response_text=response_text,
        template_id=template.id,
        sent_at=calculate_response_time(),
        send_status="pending",
    )
    db.add(response)

    # Update template usage
    template.use_count += 1
    template.last_used_at = datetime.now(timezone.utc)

    # Mark comment as responded
    comment.auto_responded = True
    comment.response_id = response.id

    db.commit()
    db.refresh(response)

    logger.info(
        f"[responder] Prepared auto-response for {comment.platform} "
        f"comment {comment.platform_comment_id}"
    )
    return response


def get_pending_responses(db: Session) -> List[EngagementResponse]:
    """Get responses that are pending and past their scheduled send time."""
    now = datetime.now(timezone.utc)
    return (
        db.query(EngagementResponse)
        .filter(
            and_(
                EngagementResponse.send_status == "pending",
                EngagementResponse.sent_at <= now,
            )
        )
        .order_by(EngagementResponse.sent_at.asc())
        .all()
    )
