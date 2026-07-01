# FILE: app/memory/conversation_overview.py
# Purpose: Durable chronological overview ([CONVERSATION_TIMELINE]) served on
#          whole-conversation meta questions — pinned opener + evenly-spaced
#          waypoints + newest scrolled-out turns, read straight from the
#          messages table so it survives rolling-summary shedding.
# Called-by: app.memory.spine_service (recall source -> every chat path),
#            app.endpoints.chat (belt-and-braces fallback)
# Depends-on: app.memory.conversation_meta, app.memory.models
# Last-renovated: 2026-07-01 (Lane B Tasks 1+2: whole-conversation recall)
"""
Whole-conversation overview (Lane B, 2026-07-01).

The evidence from the 2026-07-01 chat log: "give me a full lowdown of the whole
day" only covered the afternoon (the rolling summary had shed the morning), and
"what were the very first messages" was answered with an invented opener. Both
questions are topicless, so the tag-gated spine recall returned "" exactly when
recall mattered most.

This module serves those questions from the one source that never sheds:
the session's own ordered message index. The block pins the REAL opening
messages (Task 2), adds coarse chronological waypoints across the scope, and
closes with the newest scrolled-out turns — assembled oldest-first. The rolling
summary and [CONVERSATION_COVERAGE] blocks travel alongside it in the same
assembled memory block, so this deliberately does NOT duplicate them.

Scope mirrors the proven coverage.py pattern: the user's local calendar day
first (an all-day conversation spans sessions — they rotate after 2h idle),
then the current session, then a recent-messages fallback.

Failure-proof: any error returns "". Kill-switch: ASTRA_CONV_OVERVIEW=0.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import List, Sequence, Tuple

logger = logging.getLogger(__name__)

OVERVIEW_MARKER = "[CONVERSATION_TIMELINE]"
_OVERVIEW_END = "[/CONVERSATION_TIMELINE]"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _enabled() -> bool:
    return os.getenv("ASTRA_CONV_OVERVIEW", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _today_start_utc() -> datetime:
    """Naive-UTC datetime for local-midnight today (Message.created_at is naive
    UTC). Mirrors coverage._today_start_utc — the codebase's local-day idiom."""
    local_now = datetime.now().astimezone()
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc).replace(tzinfo=None)


def _scoped_query(db, project_id: int) -> Tuple[object, str]:
    """(query, scope_label) for the messages the overview should span.

    Day first (sessions rotate after 2h idle, so one day can hold several),
    then the latest message's session, then plain project recency.
    """
    from app.memory.models import Message

    base = db.query(Message).filter(Message.project_id == project_id)

    today = base.filter(Message.created_at >= _today_start_utc())
    if db.query(today.exists()).scalar():
        return today, "today"

    latest = base.order_by(Message.id.desc()).first()
    sid = getattr(latest, "session_id", None) if latest else None
    if sid is not None:
        in_session = base.filter(Message.session_id == sid)
        if db.query(in_session.exists()).scalar():
            return in_session, "this session"

    return base, "recently"


def _fmt(m, scope: str, per_msg: int) -> str:
    content = (m.content or "").strip().replace("\r", " ")
    if len(content) > per_msg:
        content = content[:per_msg] + " […]"
    if m.created_at:
        stamp = m.created_at.strftime("%H:%M" if scope == "today" else "%Y-%m-%d %H:%M")
    else:
        stamp = "?"
    speaker = "User" if m.role == "user" else "You (ASTRA)"
    return f"- [{stamp}] {speaker}: {content}"


def build_conversation_overview(
    db,
    project_id: int,
    user_message: str,
    window_message_ids: Sequence[int] = (),
) -> str:
    """[CONVERSATION_TIMELINE] block for whole-conversation meta questions.

    Returns "" unless the question is a conversation-meta question (see
    app.memory.conversation_meta), the kill-switch is on, and the scope holds
    messages. Never raises.
    """
    block, _ = build_conversation_overview_detail(
        db, project_id, user_message, window_message_ids,
    )
    return block


def build_conversation_overview_detail(
    db,
    project_id: int,
    user_message: str,
    window_message_ids: Sequence[int] = (),
    skip_ids: Sequence[int] = (),
) -> Tuple[str, frozenset]:
    """(block, rendered_message_ids).

    `skip_ids` = messages another channel (topic recall) already renders this
    turn; they are kept out of the waypoint/tail sections so the same message
    never appears twice with different truncations (review 2026-07-01). The
    pinned opener is EXEMPT — anti-confabulation outranks the rare
    opener-overlap duplication. Never raises; failure returns ("", frozenset()).
    """
    try:
        if not project_id or not _enabled():
            return "", frozenset()
        from app.memory.conversation_meta import is_conversation_meta_question
        if not is_conversation_meta_question(user_message or ""):
            return "", frozenset()

        from app.memory.models import Message

        first_n = max(1, _env_int("ASTRA_CONV_OVERVIEW_FIRST_N", 5))
        last_n = max(0, _env_int("ASTRA_CONV_OVERVIEW_LAST_N", 5))
        anchors_n = max(0, _env_int("ASTRA_CONV_OVERVIEW_ANCHORS", 6))
        per_msg = max(60, _env_int("ASTRA_CONV_OVERVIEW_PER_MSG_CHARS", 280))
        max_chars = max(600, _env_int("ASTRA_CONV_OVERVIEW_MAX_CHARS", 3600))

        q, scope = _scoped_query(db, project_id)

        # Opening messages: ALWAYS pinned (Task 2) — even when still inside the
        # visible window, the explicit "authoritative first messages" framing is
        # what stops a confabulated opener.
        first = q.order_by(Message.id.asc()).limit(first_n).all()
        if not first:
            return "", frozenset()
        first_ids = {m.id for m in first}

        # Newest scrolled-out turns: the visible window already shows the tail,
        # so window ids are pure duplication here — exclude them (and any ids
        # another channel is already rendering this turn).
        excluded = set(window_message_ids or ()) | first_ids | set(skip_ids or ())
        last_q = q.order_by(Message.id.desc())
        if excluded:
            last_q = last_q.filter(~Message.id.in_(list(excluded)))
        last = list(reversed(last_q.limit(last_n).all())) if last_n else []
        last_ids = {m.id for m in last}

        # Coarse waypoints: evenly spaced ids strictly between the pinned opener
        # and the tail sections (ids only — cheap even on a long day).
        anchors: List = []
        if anchors_n:
            lo = max(first_ids)
            hi = min(last_ids) if last_ids else None
            id_q = q.with_entities(Message.id).filter(Message.id > lo)
            if hi is not None:
                id_q = id_q.filter(Message.id < hi)
            mid_ids = [row[0] for row in id_q.order_by(Message.id.asc()).all()]
            mid_ids = [i for i in mid_ids if i not in excluded and i not in last_ids]
            if mid_ids:
                if len(mid_ids) <= anchors_n:
                    picked = mid_ids
                else:
                    step = len(mid_ids) / anchors_n
                    picked = [mid_ids[int(i * step)] for i in range(anchors_n)]
                anchors = (
                    db.query(Message)
                    .filter(Message.id.in_(picked))
                    .order_by(Message.id.asc())
                    .all()
                )

        lines: List[str] = [
            OVERVIEW_MARKER,
            f"Durable chronological index of this conversation ({scope}), read "
            "verbatim from the stored messages — it survives summary compression. "
            "For 'how did this start' / 'what were the first messages' questions, "
            "the OPENING MESSAGES below are the authoritative first messages: "
            "quote them, never invent an opener.",
            "Opening messages (the actual start):",
        ]
        used = sum(len(l) for l in lines)

        def _append(block_lines: List[str]) -> bool:
            nonlocal used
            for line in block_lines:
                if used + len(line) > max_chars:
                    return False
                lines.append(line)
                used += len(line)
            return True

        _append([_fmt(m, scope, per_msg) for m in first])
        if anchors:
            if _append([f"Waypoints across the conversation ({scope}, evenly spaced):"]):
                _append([_fmt(m, scope, per_msg) for m in anchors])
        if last:
            if _append(["Most recent turns before the visible window:"]):
                _append([_fmt(m, scope, per_msg) for m in last])

        lines.append(_OVERVIEW_END)
        rendered = frozenset(
            first_ids | last_ids | {m.id for m in anchors}
        )
        return "\n".join(lines), rendered
    except Exception as exc:  # noqa: BLE001 — must never break a reply
        logger.debug("[conv_overview] build failed: %s", exc)
        return "", frozenset()


__all__ = [
    "build_conversation_overview",
    "build_conversation_overview_detail",
    "OVERVIEW_MARKER",
]
