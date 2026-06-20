# FILE: app/memory/nat_jobs/keyword_trail.py
# Purpose: Job 1 extraction side — on each USER message, extract 3-8 topic
#          keywords (via Nat; deterministic tagger as fallback) and store them
#          in message_keywords. Background, post-persist, failure-proof.
# Called-by: app.memory.service.create_message (spawn), app.db.init_db (none)
# Depends-on: app.llm.workers.nat_worker.run_nat_job, app.astra_memory.topic_tagger.extract_tags
# Last-renovated: 2026-06-19
"""
Keyword recall trail — extraction.

Fires a background task when a user message is saved. Never blocks message
creation; any failure leaves the message simply un-tagged (the deterministic
tagger is tried before giving up, so that's rare). This is the exact clean-JSON
extraction Nat is good at, thinking OFF (~sub-second).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Extract 3-8 short topic keywords capturing what this message is about. "
    "Reply ONLY with a JSON array of strings, nothing else. If the message has "
    "no real topic (a greeting, 'ok', 'thanks'), reply []."
)
_MAX_INPUT_CHARS = 2000


def keywords_enabled() -> bool:
    return os.getenv("NAT_KEYWORDS_ENABLED", "1").strip() == "1"


def maybe_spawn_keyword_extraction(msg) -> None:
    """Fire-and-forget keyword extraction for a freshly-persisted USER message.

    Sync + safe to call from create_message (inside the request's event loop).
    Mirrors integration._spawn_summary_task: if no loop is running we skip.
    """
    try:
        if not keywords_enabled():
            return
        if getattr(msg, "role", None) != "user":
            return
        content = getattr(msg, "content", "") or ""
        if not content.strip():
            return
        # Capture primitives now — the ORM object is bound to the request session.
        message_id = int(getattr(msg, "id"))
        project_id = getattr(msg, "project_id", None)
        session_id = getattr(msg, "session_id", None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[keyword_trail] spawn precheck failed: %s", exc)
        return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_extract_and_store(message_id, project_id, session_id, content))
    except RuntimeError:
        logger.debug("[keyword_trail] no running loop — keywords deferred for msg %s", message_id)


async def _extract_and_store(message_id: int, project_id, session_id, content: str) -> None:
    from app.db import get_db_session
    db = get_db_session()
    try:
        keywords, source = await extract_keywords(content)
        if keywords:
            _store(db, message_id, project_id, session_id, keywords, source)
    except Exception as exc:  # noqa: BLE001 — must never surface
        logger.debug("[keyword_trail] extract/store failed for msg %s: %s", message_id, exc)
    finally:
        try:
            db.close()
        except Exception:
            pass


async def extract_keywords(text: str) -> Tuple[List[str], str]:
    """Return (keywords, source). source in {"nat","fallback","none"}."""
    snippet = (text or "")[:_MAX_INPUT_CHARS]
    # 1) Nat (clean JSON, thinking off).
    try:
        from app.llm.workers.nat_worker import run_nat_job, coerce_json
        max_tok = int(os.getenv("NAT_JOB_MAX_TOKENS", "200"))
        reply = await run_nat_job(
            _SYSTEM_PROMPT, snippet,
            enable_thinking=False, max_tokens=max_tok, timeout_seconds=15,
        )
        kws = _clean(coerce_json(reply) if isinstance(reply, str) else None)
        if kws:
            return kws, "nat"
    except Exception as exc:  # noqa: BLE001
        logger.debug("[keyword_trail] Nat extraction failed: %s", exc)

    # 2) Deterministic fallback — never leave a message un-tagged.
    try:
        from app.astra_memory.topic_tagger import extract_tags
        kws = _clean(extract_tags(snippet))
        if kws:
            return kws, "fallback"
    except Exception as exc:  # noqa: BLE001
        logger.debug("[keyword_trail] fallback tagger failed: %s", exc)

    return [], "none"


def _clean(value) -> List[str]:
    """Coerce Nat/tagger output into <=8 short, deduped, non-empty strings."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen = set()
    for item in value:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s or len(s) > 40:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= 8:
            break
    return out


def _store(db, message_id: int, project_id, session_id, keywords: List[str], source: str) -> None:
    """Idempotent upsert keyed on message_id (unique)."""
    from app.memory.nat_jobs.keyword_models import MessageKeywords
    payload = json.dumps(keywords, ensure_ascii=False)
    existing = db.query(MessageKeywords).filter(
        MessageKeywords.message_id == message_id
    ).first()
    if existing is not None:
        existing.keywords = payload
        existing.source = source
        if session_id is not None and existing.session_id is None:
            existing.session_id = session_id
    else:
        db.add(MessageKeywords(
            message_id=message_id,
            project_id=project_id,
            session_id=session_id,
            keywords=payload,
            source=source,
        ))
    db.commit()
    logger.debug("[keyword_trail] stored %d keywords (%s) for msg %s",
                 len(keywords), source, message_id)
