# FILE: app/memory/nat_jobs/recurring_themes.py
# Purpose: Cross-CONVERSATION recurring-theme recognition. Counts how many DISTINCT
#          conversations (= projects; the phone opens a new project per chat) each
#          keyword-trail topic appears in, and surfaces the themes the user returns
#          to across many separate conversations -- the
#          "you keep coming back to investments / Portugal" channel that neither
#          within-day coverage (today/session only) nor the cross-DOMAIN graph walk
#          (a link drawn once) provides.
# Called-by: app.llm.routing.memory_injection_sources.collect_recurring
# Depends-on: app.memory.nat_jobs.keyword_models.MessageKeywords, app.memory.models.Message,
#             app.memory.recall_intent.is_recall_question, app.astra_memory.topic_tagger
# Last-renovated: 2026-06-24
"""
Recurring themes (cross-conversation pattern recognition).

The keyword trail already tags every user message with topics, but nothing ever
asked "which topics recur across MANY conversations?" — so ASTRA could not notice
that the user keeps circling back to the same handful of preoccupations. This
module reads the trail, counts distinct SESSIONS per topic (joining the
authoritative messages table, because the trail's own session_id is written NULL),
and surfaces the durable recurring themes.

Two triggers, both behind the one D0-D4 gate that already governs the front door:
  - a recall question ("what do I keep talking about", "what have we discussed")
    -> the top recurring themes, as an overview block;
  - any turn whose own topic IS one of the recurring themes -> a one-line nudge so
    the model can connect today's mention to the bigger pattern.

The distinct-session frequency map is cached per project (slow-changing) so the
per-turn cost is ~0. Failure-proof: every path returns "" on error, never raises.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("ASTRA_RECURRING_THEMES", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _min_conversations() -> int:
    """A theme must recur across at least this many DISTINCT conversations
    (projects) to count as recurring. Env name kept (ASTRA_RECURRING_MIN_SESSIONS)
    for back-compat."""
    try:
        return max(2, int(os.getenv("ASTRA_RECURRING_MIN_SESSIONS", "3")))
    except Exception:
        return 3


def _max_themes() -> int:
    try:
        return max(1, int(os.getenv("ASTRA_RECURRING_MAX_THEMES", "6")))
    except Exception:
        return 6


def _max_rows() -> int:
    try:
        return max(100, int(os.getenv("ASTRA_RECURRING_MAX_ROWS", "5000")))
    except Exception:
        return 5000


def _horizon_days() -> int:
    """Look-back window for 'recurring' — long enough to be genuinely cross-time,
    not just recent activity (the whole point of this channel)."""
    try:
        return max(7, int(os.getenv("ASTRA_RECURRING_HORIZON_DAYS", "180")))
    except Exception:
        return 180


def _cache_ttl_s() -> float:
    try:
        return max(0.0, float(os.getenv("ASTRA_RECURRING_CACHE_TTL_S", "600")))
    except Exception:
        return 600.0


# project_id -> (monotonic_ts, {topic: distinct_session_count}). In-process,
# cleared on restart; refreshed every TTL so per-turn membership checks are free.
_FREQ_CACHE: Dict[str, Tuple[float, Dict[str, int]]] = {}


def _conversation_frequency(db) -> Dict[str, int]:
    """Distinct-CONVERSATION count per topic from the keyword trail.

    Each bridge conversation is its OWN project (the phone opens a new project per
    chat), so 'a separate conversation' = a distinct project_id — NOT a session
    (projects here hold a single session each, so session-counting always yields 1
    and never surfaces anything). Counts distinct project_id per topic across the
    user's whole trail within a long-horizon window. No join needed:
    message_keywords carries project_id directly (NOT NULL).
    """
    from app.memory.nat_jobs.keyword_models import MessageKeywords

    # created_at is indexed; long horizon + generous row cap as a backstop.
    cutoff = datetime.utcnow() - timedelta(days=_horizon_days())
    rows = (
        db.query(MessageKeywords.keywords, MessageKeywords.project_id)
        .filter(MessageKeywords.created_at >= cutoff)
        .order_by(MessageKeywords.message_id.desc())
        .limit(_max_rows())
        .all()
    )
    topic_convos: Dict[str, set] = {}
    for keywords_json, pid in rows:
        if pid is None:
            continue
        try:
            kws = json.loads(keywords_json or "[]")
        except Exception:
            continue
        for kw in kws:
            if not isinstance(kw, str):
                continue
            k = kw.strip().lower()
            if not k or k == "general":
                continue
            topic_convos.setdefault(k, set()).add(pid)
    return {t: len(s) for t, s in topic_convos.items()}


def _frequency_cached(db) -> Dict[str, int]:
    # Whole-trail (all conversations) frequency — identical regardless of the
    # current project — so cache under one key. In-process, TTL-refreshed.
    now = time.monotonic()
    hit = _FREQ_CACHE.get("all")
    if hit and (now - hit[0]) < _cache_ttl_s():
        return hit[1]
    freq = _conversation_frequency(db)
    _FREQ_CACHE["all"] = (now, freq)
    return freq


def _current_topics(user_message: str) -> set:
    if not user_message:
        return set()
    try:
        from app.astra_memory.topic_tagger import extract_tags
        return {t for t in extract_tags(user_message) if t and t != "general"}
    except Exception:
        return set()


def build_recurring_themes_block(db, project_id, user_message: str = "") -> str:
    """Return a [RECURRING THEMES] / [RECURRING THEME] block, or "".

    Fires on a recall question (overview of what the user keeps returning to) or
    when the current turn's own topic is itself a recurring theme. Failure-proof.
    """
    try:
        if not _enabled() or not project_id:
            return ""
        from app.memory.recall_intent import is_recall_question
        recall = is_recall_question(user_message) if user_message else False
        # A "today" / this-conversation question is answered by coverage (today-
        # scoped, cross-conversation), NOT by all-time recurring themes — otherwise
        # "what did we talk about today" returns OLD cross-conversation topics.
        today_q = False
        if user_message:
            try:
                from app.memory.nat_jobs.coverage import _mentions_today
                today_q = _mentions_today(user_message)
            except Exception:
                today_q = False
        overview = recall and not today_q
        topics = _current_topics(user_message)
        # Cheap exit only when there's nothing at all to match against.
        if not overview and not (user_message or "").strip():
            return ""

        freq = _frequency_cached(db)
        if not freq:
            return ""
        threshold = _min_conversations()
        recurring = sorted(
            ((t, n) for t, n in freq.items() if n >= threshold),
            key=lambda tn: tn[1], reverse=True,
        )
        if not recurring:
            return ""

        if overview:
            top = recurring[:_max_themes()]
            body = ", ".join(f"{t} ({n} conversations)" for t, n in top)
            logger.info("[recurring_themes] recall: surfaced %d theme(s)", len(top))
            return (
                "[RECURRING THEMES — subjects the user returns to across many "
                "separate conversations; use to answer 'what do I keep talking "
                "about' and to connect today to the bigger picture]\n" + body
            )

        # Non-recall turn: surface only if the CURRENT message is about a recurring
        # theme. Match each recurring key by domain tag OR a word-boundary substring
        # — the trail's keys are Nat free-form strings ("index fund", "pasta") that
        # the coarse domain tagger alone would miss.
        msg_low = (user_message or "").lower()
        hits = []
        for t, n in recurring:
            if t in topics or re.search(r"\b" + re.escape(t) + r"\b", msg_low):
                hits.append((t, n))
        if not hits:
            return ""
        hits.sort(key=lambda tn: tn[1], reverse=True)
        body = ", ".join(f"{t} ({n} conversations)" for t, n in hits[:_max_themes()])
        logger.info("[recurring_themes] topical: %d recurring topic(s) in turn", len(hits))
        return (
            "[RECURRING THEME] This connects to something the user returns to "
            "often across separate conversations: " + body
        )
    except Exception as exc:  # noqa: BLE001 — never break a reply on memory
        logger.debug("[recurring_themes] build failed: %s", exc)
        return ""


__all__ = ["build_recurring_themes_block"]
