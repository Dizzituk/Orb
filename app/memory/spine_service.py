# FILE: app/memory/spine_service.py
# Purpose: Within-conversation memory spine — per-message topic index + live recall.
# Called-by: app.memory.service (write), app.endpoints.chat (recall/injection)
# Depends-on: app.astra_memory.topic_tagger, app.memory.conversation_models, app.memory.models, app.memory.summary_injection
# Last-renovated: 2026-06-15 (Job 1: attach_spine_to_session keeps spine.session_id in lockstep with messages)
"""
Within-conversation memory spine (Build Request B).

The chat window only carries the most recent ~50 messages; everything older
scrolls out of the model's view. This module keeps a lightweight index of the
ENTIRE conversation so a question like "what was I telling you about Anthropic"
can search that index, find the matching older messages, and pull just those
ORIGINAL messages back into context — without widening the window.

Design (reuse, not reinvent):
  - Tagging is the existing app.astra_memory.topic_tagger: pure keyword/entity
    matching, no LLM, ~1ms a call — cheap enough to run synchronously on every
    message as it is stored.
  - One tiny ConversationSpine row per message holds {project, session, message
    id, role, tags, entities, timestamp}. No message text is duplicated here:
    recall fetches the real (encrypted) Message rows by id, so nothing weakens
    the Security-Level-4 content store.
  - Retrieval mirrors the hot-index stage-1 pattern: a project-scoped LIKE over
    the plaintext tag/entity columns, newest-first, excluding the messages that
    are already in the live window.

The rolling ConversationSummary (app.memory.summary_injection) is the always-on
backstop; build_within_conversation_context() stitches recall + summary into a
single block for the chat path that previously injected neither.

Part of CONV-MEMORY-002: within-conversation spine.
"""
from __future__ import annotations

import json
import logging
from typing import List, Sequence, Tuple

from sqlalchemy import or_
from sqlalchemy.orm import Session as DbSession

logger = logging.getLogger(__name__)

# Recall sizing — keep the injected block bounded so it never bloats the prompt.
_MAX_RECALL_MSGS = 4
_RECALL_CHAR_BUDGET = 2400
_PER_MSG_CHARS = 700


def _tags_entities_for(text: str) -> Tuple[List[str], List[str]]:
    """Tag a message with the shared topic_tagger. Never raises.

    Drops the 'general' fallback tag — as a filter it carries no signal.
    """
    try:
        from app.astra_memory.topic_tagger import extract_tags, extract_entities
        tags = [t for t in extract_tags(text or "") if t != "general"]
        entities = extract_entities(text or "")
        return tags, entities
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("[spine] tagging failed: %s", exc)
        return [], []


def _like_term(value: str) -> str:
    """LIKE pattern matching one element of a JSON array column.

    Stored as e.g. '["TAO","Anthropic"]'; json.dumps("TAO") == '"TAO"', so the
    pattern '%"TAO"%' hits the exact element and not a substring of a longer one.
    """
    return "%" + json.dumps(value) + "%"


def record_message_spine(db: DbSession, message) -> None:
    """Index one stored Message into the conversation spine.

    Called at the end of memory.service.create_message. Failure-proof: any error
    here must never break message creation (the caller also guards). Idempotent:
    a second call for the same message_id is a no-op.
    """
    from app.memory.conversation_models import ConversationSpine
    try:
        mid = getattr(message, "id", None)
        if mid is None:
            return

        already = (
            db.query(ConversationSpine.id)
            .filter(ConversationSpine.message_id == mid)
            .first()
        )
        if already:
            return

        tags, entities = _tags_entities_for(getattr(message, "content", "") or "")

        kwargs = dict(
            project_id=getattr(message, "project_id", None),
            session_id=getattr(message, "session_id", None),
            message_id=mid,
            role=getattr(message, "role", "") or "",
            tags=json.dumps(tags),
            entities=json.dumps(entities),
        )
        created = getattr(message, "created_at", None)
        if created is not None:
            kwargs["created_at"] = created

        db.add(ConversationSpine(**kwargs))
        db.commit()
    except Exception as exc:
        logger.debug(
            "[spine] record failed for message %s: %s",
            getattr(message, "id", "?"), exc,
        )
        try:
            db.rollback()
        except Exception:
            pass


def attach_spine_to_session(
    db: DbSession,
    message_ids: Sequence[int],
    session_id: int,
) -> int:
    """Claim already-written spine rows into a conversation session.

    Spine rows are written at the create_message chokepoint (see
    record_message_spine) carrying whatever session_id the Message had at that
    moment -- which is NULL on every live chat path, because the session is
    only resolved by the post-message hook AFTER the row is stored. The hook's
    orphan claim (integration._backfill_orphan_messages) fills in the Message's
    session_id but never the spine's, so without this the spine could not be
    scoped by session. This reconciles the spine to match its messages once the
    session is known. Only fills NULLs -- it never moves a row that already
    belongs to a session. Returns the number of rows updated. Never raises.
    """
    if not message_ids:
        return 0
    from app.memory.conversation_models import ConversationSpine
    try:
        updated = (
            db.query(ConversationSpine)
            .filter(
                ConversationSpine.message_id.in_(list(message_ids)),
                ConversationSpine.session_id.is_(None),
            )
            .update(
                {ConversationSpine.session_id: session_id},
                synchronize_session=False,
            )
        )
        db.commit()
        return int(updated or 0)
    except Exception as exc:
        logger.debug("[spine] session attach failed: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass
        return 0


def search_spine(
    db: DbSession,
    project_id: int,
    query_tags: Sequence[str],
    query_entities: Sequence[str],
    exclude_message_ids: Sequence[int] = (),
    limit: int = 12,
) -> List:
    """Project-scoped search of the spine by topic tags / named entities.

    Newest-first (spine id ≈ message recency). Excludes message ids that are
    already in the live window so recall only ever surfaces scrolled-out
    history. Returns ConversationSpine rows.
    """
    from app.memory.conversation_models import ConversationSpine

    conds = []
    for ent in (query_entities or []):
        conds.append(ConversationSpine.entities.like(_like_term(ent)))
    for tag in (query_tags or []):
        conds.append(ConversationSpine.tags.like(_like_term(tag)))
    if not conds:
        return []

    q = (
        db.query(ConversationSpine)
        .filter(ConversationSpine.project_id == project_id)
        .filter(or_(*conds))
    )
    if exclude_message_ids:
        q = q.filter(~ConversationSpine.message_id.in_(list(exclude_message_ids)))

    return q.order_by(ConversationSpine.id.desc()).limit(limit).all()


def build_conversation_recall(
    db: DbSession,
    project_id: int,
    user_message: str,
    window_message_ids: Sequence[int] = (),
    max_msgs: int = _MAX_RECALL_MSGS,
) -> str:
    """Build a [CONVERSATION_RECALL] block of scrolled-out messages on this topic.

    Returns "" when the question carries no specific topic, or when nothing
    older than the window matches. Never raises.
    """
    try:
        tags, entities = _tags_entities_for(user_message or "")
        if not tags and not entities:
            return ""

        rows = search_spine(
            db, project_id, tags, entities,
            exclude_message_ids=window_message_ids,
            limit=max_msgs * 3,
        )
        if not rows:
            return ""

        msg_ids = [r.message_id for r in rows][:max_msgs]

        from app.memory.models import Message
        msgs = (
            db.query(Message)
            .filter(Message.id.in_(msg_ids))
            .order_by(Message.id.asc())  # oldest first for reading order
            .all()
        )
        if not msgs:
            return ""

        lines = [
            "[CONVERSATION_RECALL]",
            "Earlier in THIS conversation (now scrolled out of the visible "
            "window) the user and you discussed things related to the current "
            "question. Use these original earlier messages to answer; you may "
            "note that the topic came up before. Oldest first:",
        ]
        used = 0
        for m in msgs:
            content = (m.content or "").strip().replace("\r", " ")
            if len(content) > _PER_MSG_CHARS:
                content = content[:_PER_MSG_CHARS] + " […]"
            stamp = m.created_at.strftime("%Y-%m-%d %H:%M") if m.created_at else "?"
            speaker = "User" if m.role == "user" else "You (ASTRA)"
            block = f"- [{stamp}] {speaker}: {content}"
            if used + len(block) > _RECALL_CHAR_BUDGET:
                break
            lines.append(block)
            used += len(block)

        if len(lines) <= 2:
            return ""
        lines.append("[/CONVERSATION_RECALL]")
        return "\n".join(lines)
    except Exception as exc:
        logger.debug("[spine] recall failed: %s", exc)
        return ""


def build_within_conversation_context(
    db: DbSession,
    project_id: int,
    user_message: str,
    window_message_ids: Sequence[int] = (),
) -> str:
    """Combined within-conversation context for the chat path.

    Topic recall of scrolled-out messages + the rolling conversation summary
    backstop (reusing app.memory.summary_injection — the canonical formatter).
    Returns "" if neither produces anything. Never raises.
    """
    parts: List[str] = []

    recall = build_conversation_recall(
        db, project_id, user_message, window_message_ids,
    )
    if recall:
        parts.append(recall)

    try:
        from app.memory.summary_injection import build_summary_context
        summary = build_summary_context(db, project_id)
        if summary:
            parts.append(summary)
    except Exception as exc:
        logger.debug("[spine] summary backstop skipped: %s", exc)

    return "\n\n".join(parts)


__all__ = [
    "record_message_spine",
    "attach_spine_to_session",
    "search_spine",
    "build_conversation_recall",
    "build_within_conversation_context",
]
