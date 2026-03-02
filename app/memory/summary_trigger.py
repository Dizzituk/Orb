# FILE: app/memory/summary_trigger.py
"""
Determines when a conversation summary should be generated.

Trigger conditions (any one is sufficient):
  1. Message count since last summary >= threshold (default 5)
  2. Model switch detected (different provider:model on consecutive messages)
  3. Explicit user request (placeholder for future UI button)

Non-blocking — callers should check should_generate_summary() and
spawn the actual generation in a background task.

Part of CONV-MEMORY-001: Phase 2 (Summary Generation Service).
"""

import logging
from typing import Optional

from sqlalchemy.orm import Session as DbSession

from app.memory.conversation_models import ConversationSession, ConversationSummary
from app.memory.conversation_service import get_latest_summary

logger = logging.getLogger(__name__)

# Minimum messages since last summary before triggering a new one.
# 5 keeps summaries fresh without burning LLM budget on every message.
MESSAGE_THRESHOLD = 5


def _count_unsummarised_messages(
    db: DbSession,
    session: ConversationSession,
) -> int:
    """
    Count messages in this session that arrived after the last summary.

    If no summary exists, all messages in the session are unsummarised.
    """
    from app.memory.models import Message

    latest = get_latest_summary(db, session.id)

    query = db.query(Message).filter(
        Message.session_id == session.id,
    )

    if latest and latest.summarised_through_message_id:
        query = query.filter(
            Message.id > latest.summarised_through_message_id,
        )

    return query.count()


def _detect_model_switch(
    db: DbSession,
    session: ConversationSession,
) -> bool:
    """
    Check whether the last two assistant messages used different models.

    A model switch means the new model has zero context about what the
    previous model discussed — summary injection is critical here.
    """
    from app.memory.models import Message

    recent_assistant = (
        db.query(Message)
        .filter(
            Message.session_id == session.id,
            Message.role == "assistant",
            Message.provider.isnot(None),
            Message.model.isnot(None),
        )
        .order_by(Message.id.desc())
        .limit(2)
        .all()
    )

    if len(recent_assistant) < 2:
        return False

    newest = recent_assistant[0]
    previous = recent_assistant[1]

    newest_key = f"{newest.provider}:{newest.model}"
    previous_key = f"{previous.provider}:{previous.model}"

    if newest_key != previous_key:
        logger.info(
            "[summary_trigger] Model switch detected: %s → %s",
            previous_key, newest_key,
        )
        return True

    return False


def should_generate_summary(
    db: DbSession,
    session: ConversationSession,
    force: bool = False,
) -> bool:
    """
    Determine whether a new summary should be generated.

    Args:
        db: Database session.
        session: The active conversation session.
        force: If True, always returns True (for explicit requests).

    Returns:
        True if summary generation should be triggered.
    """
    if force:
        logger.info(
            "[summary_trigger] Forced summary for session %d", session.id,
        )
        return True

    # Check 1: Model switch (highest priority — context gap is immediate)
    if _detect_model_switch(db, session):
        return True

    # Check 2: Message count threshold
    unsummarised = _count_unsummarised_messages(db, session)
    if unsummarised >= MESSAGE_THRESHOLD:
        logger.info(
            "[summary_trigger] Threshold reached: %d unsummarised messages "
            "in session %d",
            unsummarised, session.id,
        )
        return True

    return False
