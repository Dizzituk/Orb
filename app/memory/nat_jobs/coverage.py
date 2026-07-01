# FILE: app/memory/nat_jobs/coverage.py
# Purpose: Job 1 consumption side — build the [CONVERSATION_COVERAGE] block from
#          the keyword trail when the user asks a recall question. Pure DB read,
#          no model call. Pairs with the lossy summary: summary = synthesis,
#          coverage = enumeration. Together they fix full-conversation recall.
#          Scopes to TODAY (the user's local day) or the current session, and
#          surfaces the threads that recurred most so the reply can elaborate on
#          the parts that stood out — all behind the scenes.
# Called-by: app.llm.routing.memory_injection_sources.collect_coverage
# Depends-on: app.memory.nat_jobs.keyword_models.MessageKeywords, app.memory.models.Message
# Last-renovated: 2026-06-20
from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# Shared recall-intent predicate — the SAME one the web-search + grounding gates
# use to SUPPRESS a search — re-exported so existing callers keep using
# coverage.is_recall_question.
from app.memory.recall_intent import is_recall_question

logger = logging.getLogger(__name__)

_MAX_TOPICS = int(os.getenv("NAT_COVERAGE_MAX_TOPICS", "60"))
_MAX_ROWS = 400
_STANDOUT_MIN = int(os.getenv("NAT_COVERAGE_STANDOUT_MIN", "2"))  # appears in >=N msgs
_MAX_STANDOUTS = int(os.getenv("NAT_COVERAGE_MAX_STANDOUTS", "8"))

# "Today" intent — scope the trail to the user's local calendar day rather than
# the latest session, because an all-day conversation can span several sessions.
_TODAY_RE = re.compile(
    r"\b(?:today|so far today|the day so far|this (?:morning|afternoon|evening))\b",
    re.IGNORECASE,
)


def _mentions_today(text: str) -> bool:
    return bool(text) and bool(_TODAY_RE.search(text))


def _today_start_utc() -> datetime:
    """Naive-UTC datetime for local-midnight today (trail created_at is naive UTC).

    Mirrors the codebase's local-time convention (datetime.now().astimezone()
    binds to the machine tz, Europe/London here)."""
    local_now = datetime.now().astimezone()
    local_midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return local_midnight.astimezone(timezone.utc).replace(tzinfo=None)


def build_coverage_block(
    db,
    project_id: int,
    user_message: str = "",
    session_id: Optional[int] = None,
) -> str:
    """Return a [CONVERSATION_COVERAGE] block, or "" when not applicable.

    Failure-proof: any error returns "".
    """
    try:
        if not project_id:
            return ""
        # Gate on recall intent when we have the question text.
        if user_message and not is_recall_question(user_message):
            return ""

        from app.memory.nat_jobs.keyword_models import MessageKeywords

        want_today = _mentions_today(user_message) if user_message else False
        rows, scope_label = _select_rows(
            db, project_id, MessageKeywords, want_today, session_id
        )
        if not rows:
            return ""

        ordered, standouts = _aggregate(rows)
        if not ordered:
            return ""

        # Observability — the user wants to KNOW this fired and worked, without a
        # UI. One INFO line per rundown: scope + how much it pulled in.
        logger.info(
            "[NAT][coverage] scope=%s rows=%d topics=%d standouts=%d",
            scope_label, len(rows), len(ordered), len(standouts),
        )

        lines: List[str] = [
            "[CONVERSATION_COVERAGE]",
            f"Every topic discussed {scope_label}, in order (use this for full "
            "recall — it does not forget the start of the conversation):",
            ", ".join(ordered),
        ]
        if standouts:
            lines += [
                "",
                "Threads that came up most (the parts that stood out — elaborate "
                "on these first if asked for highlights):",
                ", ".join(f"{name} ({count}x)" for name, count in standouts),
            ]
        return "\n".join(lines)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[coverage] build failed: %s", exc)
        return ""


def _select_rows(
    db,
    project_id: int,
    MK,
    want_today: bool,
    session_id: Optional[int],
) -> Tuple[list, str]:
    """Pick the in-scope keyword rows + a human scope label.

    Scoping is the part that was broken: the trail's own `session_id` is written
    the instant the user message is saved, BEFORE the conversation service
    attaches the message to a session, so it is almost always NULL. So:
      - "today" scopes on the trail's own `created_at` (no join needed),
      - "this session" resolves the session from the AUTHORITATIVE messages table
        (which is backfilled) and scopes through a join,
      - a trail-only fallback covers tests / legacy rows with no joinable message.
    """
    base = db.query(MK).filter(MK.project_id == project_id)

    # "Today" — every topic across ALL of today's conversations. Each phone chat is
    # its OWN project, so scoping to the current project would miss the rest of
    # today's chats (the exact "it pulled old/empty stuff in a new chat" bug). So
    # "today" deliberately spans projects (single user), keyed on the trail's own
    # created_at (survives NULL session_id).
    if want_today:
        try:
            rows = (
                db.query(MK)
                .filter(MK.created_at >= _today_start_utc())
                .order_by(MK.message_id.asc())
                .limit(_MAX_ROWS)
                .all()
            )
            return rows, "today"
        except Exception as exc:  # noqa: BLE001
            logger.debug("[coverage] today-scoped select failed: %s", exc)

    # "This session" — resolve the session from the messages table, scope by join.
    try:
        from app.memory.models import Message

        sid = session_id
        if sid is None:
            latest = (
                db.query(Message)
                .filter(Message.project_id == project_id)
                .order_by(Message.id.desc())
                .first()
            )
            sid = getattr(latest, "session_id", None) if latest else None
        if sid is not None:
            rows = (
                base.join(Message, MK.message_id == Message.id)
                .filter(Message.session_id == sid)
                .order_by(MK.message_id.asc())
                .limit(_MAX_ROWS)
                .all()
            )
            if rows:
                return rows, "this session"
    except Exception as exc:  # noqa: BLE001
        logger.debug("[coverage] session-scoped select failed: %s", exc)

    # Fallback: the trail's own session_id (populated in tests / legacy data),
    # else project-wide recent.
    sid = session_id
    if sid is None:
        latest = base.order_by(MK.message_id.desc()).first()
        if latest is not None and getattr(latest, "session_id", None) is not None:
            sid = latest.session_id
    q = base
    label = "recently"
    if sid is not None:
        q = q.filter(MK.session_id == sid)
        label = "this session"
    rows = q.order_by(MK.message_id.asc()).limit(_MAX_ROWS).all()
    return rows, label


def _aggregate(rows) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Build the ordered, deduped topic list AND the recurrence standouts.

    A topic's "recurrence" = the number of messages it appeared in (counted once
    per message), so threads the conversation kept returning to bubble up as the
    parts that stood out. Frequency is counted across ALL scoped rows even past
    the display cap, so the standouts stay accurate.
    """
    ordered: List[str] = []
    seen = set()
    orig = {}            # lower -> first-seen original casing
    freq: Counter = Counter()
    for r in rows:
        try:
            kws = json.loads(r.keywords or "[]")
        except Exception:
            continue
        row_seen = set()
        for kw in kws:
            if not isinstance(kw, str):
                continue
            k = kw.strip()
            if not k:
                continue
            low = k.lower()
            orig.setdefault(low, k)
            if low not in seen:
                seen.add(low)
                ordered.append(k)
            if low not in row_seen:
                row_seen.add(low)
                freq[low] += 1
    ordered = ordered[:_MAX_TOPICS]
    standouts = [
        (orig[low], c)
        for low, c in freq.most_common()
        if c >= _STANDOUT_MIN
    ][:_MAX_STANDOUTS]
    return ordered, standouts
