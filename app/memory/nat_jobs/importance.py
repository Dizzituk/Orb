# FILE: app/memory/nat_jobs/importance.py
# Purpose: Job 3 — after a reply is sent, Nat extracts durable facts from the
#          exchange and PROPOSES them into ASTRA's existing write path (Tier-1
#          shadow->enforce arbiter for canonical biographical fields; Tier-2
#          preference store otherwise). Nat never commits; the existing
#          machinery decides. Complements knowledge_extractor (the on-archive
#          safety net) by catching importance live, per exchange.
# Called-by: app.memory.service.create_message (assistant message spawn)
# Depends-on: app.llm.workers.nat_worker.run_nat_job,
#             app.self_model.write_arbiter.propose (canonical bio facts),
#             app.astra_memory.preference_service.create_preference (everything else)
# Last-renovated: 2026-06-20
"""
Importance extraction & save (background, post-reply, not latency-critical).

Hard rule (spec): Nat PROPOSES writes; the existing arbiter/preference machinery
decides whether to commit. We never bypass it, and Nat is instructed never to
invent — only extract what was explicitly stated. Failure-proof throughout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You extract DURABLE FACTS ABOUT THE USER from a chat exchange — the kind a "
    "personal assistant should still know months from now.\n"
    "INCLUDE only stable facts the USER stated about THEMSELVES: their identity and "
    "circumstances, lasting preferences, and firm personal decisions or commitments "
    "they actually made (e.g. 'is vegetarian', 'works night shifts as a delivery "
    "driver', 'lives in <place>', 'decided to buy two RTX 3090s').\n"
    "EXCLUDE everything else — when in doubt, leave it out. In particular do NOT "
    "record: topics or subjects you merely discussed; the user's opinions about the "
    "world, news, politics, economics or technology; anything the ASSISTANT said; "
    "hypotheticals, predictions or general musings; one-off events with no lasting "
    "relevance. A near-empty result is correct for a chat that was just conversation.\n"
    'Output JSON: a list of {"type","key","value"} objects, where type is one of '
    '"preference","biographical","decision","commitment", key is a short snake_case '
    'name for the fact, and value is a crisp phrase (not a sentence). '
    "Reply ONLY with the JSON list, or [] if nothing durable about the user was stated."
)

# Opinion / prediction / musing markers — a durable FACT about the user reads as a
# crisp state ("vegetarian", "lives in X"), not a sentence carrying these. Used by
# _is_durable_user_fact as a conservative guard against topic/opinion leakage.
_OPINION_MARKERS = (
    " should ", " would ", " ought ", " might ", " could ", " will ", " won't ",
    " i think", " i believe", " i reckon", "likely", "in the future", "probably",
    "concerned that", "believes that", "in my opinion", "the concept of",
)
_MAX_VALUE_CHARS = 120
_MAX_FACTS = 10
_MAX_INPUT_CHARS = 4000
_BIO_TYPES = {"biographical", "identity", "bio", "personal_fact", "personal"}


def importance_enabled() -> bool:
    return os.getenv("NAT_IMPORTANCE_ENABLED", "1").strip() == "1"


def maybe_spawn_importance_extraction(msg) -> None:
    """Fire-and-forget importance extraction for a freshly-persisted ASSISTANT
    message. Sync + safe to call from create_message (inside the event loop)."""
    try:
        if not importance_enabled():
            return
        if getattr(msg, "role", None) != "assistant":
            return
        assistant_text = getattr(msg, "content", "") or ""
        if not assistant_text.strip():
            return
        project_id = getattr(msg, "project_id", None)
        session_id = getattr(msg, "session_id", None)
        message_id = int(getattr(msg, "id"))
    except Exception as exc:  # noqa: BLE001
        logger.debug("[importance] spawn precheck failed: %s", exc)
        return

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_extract_and_save(message_id, project_id, session_id, assistant_text))
    except RuntimeError:
        logger.debug("[importance] no running loop — importance deferred for msg %s", message_id)


async def _extract_and_save(message_id, project_id, session_id, assistant_text: str) -> None:
    from app.db import get_db_session
    db = get_db_session()
    try:
        user_text = _latest_user_message(db, project_id)
        facts = await extract_durable_facts(user_text, assistant_text)
        if facts:
            saved = _save_facts(db, project_id, facts)
            logger.debug("[importance] proposed %d/%d durable fact(s) from msg %s",
                         saved, len(facts), message_id)
    except Exception as exc:  # noqa: BLE001 — never surface
        logger.debug("[importance] extract/save failed for msg %s: %s", message_id, exc)
    finally:
        try:
            db.close()
        except Exception:
            pass


def _latest_user_message(db, project_id) -> str:
    """The user half of the just-completed exchange (newest user message)."""
    try:
        from app.memory import service as memory_service
        for m in memory_service.list_messages(db, project_id, limit=6):
            if getattr(m, "role", None) == "user":
                return getattr(m, "content", "") or ""
    except Exception as exc:  # noqa: BLE001
        logger.debug("[importance] could not load paired user message: %s", exc)
    return ""


async def extract_durable_facts(user_text: str, assistant_text: str) -> List[Dict[str, Any]]:
    """Ask Nat for durable facts. Returns a cleaned list (possibly empty)."""
    payload = (f"User: {user_text}\nAssistant: {assistant_text}")[:_MAX_INPUT_CHARS]
    try:
        from app.llm.workers.nat_worker import run_nat_job, coerce_json
        reply = await run_nat_job(
            _SYSTEM_PROMPT, payload,
            enable_thinking=False, max_tokens=300, timeout_seconds=20,
        )
        return _clean_facts(coerce_json(reply) if isinstance(reply, str) else None)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[importance] Nat extraction failed: %s", exc)
        return []


def _is_durable_user_fact(key: str, value) -> bool:
    """Conservative guard so the trail stores facts ABOUT THE USER, not subjects
    that were merely discussed. Errs toward keeping — the strict prompt does the
    main filtering; this just catches obvious essays/opinions. Tunable."""
    v = str(value).strip()
    if not v or len(v) > _MAX_VALUE_CHARS:   # an essay is a musing, not a fact
        return False
    low = " " + v.lower() + " "
    if any(m in low for m in _OPINION_MARKERS):
        return False
    return True


def _clean_facts(value) -> List[Dict[str, Any]]:
    """Keep only well-formed {type,key,value} dicts that read as durable facts
    about the user; dedupe phrasing variants by canonical key; capped."""
    if not isinstance(value, list):
        return []
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        key = str(item.get("key", "")).strip()
        val = item.get("value")
        if not key or val in (None, "", [], {}):
            continue
        ftype = str(item.get("type", "fact")).strip().lower() or "fact"
        if not _is_durable_user_fact(key, val):
            continue
        canon = _pref_key(ftype, key)   # collapse "place_of_residence"/"placeofresidence"
        if canon in seen_keys:
            continue
        seen_keys.add(canon)
        out.append({"type": ftype, "key": key, "value": val})
        if len(out) >= _MAX_FACTS:
            break
    return out


def _canon(s: str) -> str:
    """Lowercase + strip ALL separators so phrasing variants collapse to one key."""
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _pref_key(ftype: str, key: str) -> str:
    """Canonical, separator-free preference key so phrasing variants collapse to a
    SINGLE preference (e.g. "place_of_residence" and "placeofresidence")."""
    return f"nat_importance:{_canon(ftype) or 'fact'}:{_canon(key) or 'fact'}"


def _save_facts(db, project_id, facts: List[Dict[str, Any]]) -> int:
    """Propose each fact into the EXISTING write path. Never commits directly.

    - Canonical biographical fields -> Tier-1 shadow->enforce arbiter (queues for
      the human; the arbiter decides). Non-canonical keys fall through.
    - Everything else -> Tier-2 preference store (idempotent create/update with
      its own confidence machinery). source='nat_importance' for traceability.
    """
    saved = 0
    for f in facts:
        ftype = f["type"]
        key = f["key"]
        value = f["value"]

        # Tier-1: only for canonical biographical fields (human review).
        if ftype in _BIO_TYPES:
            try:
                from app.self_model.canonical_schema import get_field_schema
                if get_field_schema(key) is not None:
                    from app.self_model.write_arbiter import propose
                    propose(
                        field_name=key,
                        proposed_value=value,
                        source=f"nat_importance:project:{project_id}",
                        evidence={"origin": "conversation_importance"},
                    )
                    saved += 1
                    continue
            except Exception as exc:  # noqa: BLE001
                logger.debug("[importance] arbiter propose failed for %s: %s", key, exc)

        # Tier-2: preference store (idempotent; create_preference updates if exists).
        try:
            from app.astra_memory.preference_service import create_preference
            from app.astra_memory.preference_models import PreferenceStrength
            create_preference(
                db,
                preference_key=_pref_key(ftype, key),
                preference_value=str(value),
                strength=PreferenceStrength.SOFT,  # inferred -> low, decays normally
                source="nat_importance",
                namespace="user_personal",
                context_pointer=f"project:{project_id}",
            )
            saved += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("[importance] preference write failed for %s: %s", key, exc)
    return saved
