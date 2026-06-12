# FILE: app/memory/integration.py
# Purpose: Memory system integration hooks.
# Called-by: app.bridge.router, app.llm.routing.core, app.llm.routing.prompt_builders, app.llm.stream_router (+1 more)
# Depends-on: app.db, app.memory.complexity_router, app.memory.conversation_service, app.memory.domains.confidence_learning (+9 more)
# Last-renovated: 2026-06-11
"""
Memory system integration hooks.

Wires the memory stores (preferences, confidence, context, complexity)
into the existing ASTRA pipeline at their natural call sites.

This module provides thin wrappers that the stream router, translation
layer, and LLM router import. Each hook is safe to call — if the
underlying store isn't available, it logs and returns gracefully.

Integration points:
    1. on_intent_confirmed()     — Translation confirmation → confidence learning
    2. on_intent_corrected()     — Translation correction → confidence learning
    3. after_user_message()      — Chat/command → preference capture + context extraction
    4. enrich_routing()          — Job classification → complexity-aware model selection
    5. inject_memory_context()   — Pre-LLM call → RAG memory injection

Usage:
    # In stream_router.py after confirmed_intent bypass:
    from app.memory.integration import on_intent_confirmed
    on_intent_confirmed(req.message, confirmed_intent)

    # In chat_routing.py after building messages:
    from app.memory.integration import after_user_message
    after_user_message(req.message, project_id=req.project_id)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# 1. Confidence learning: confirmation
# =========================================================================

def on_intent_confirmed(
    phrase: str,
    intent: str,
    user_id: str = "default",
) -> None:
    """
    Called when a user confirms a proposed intent.

    Wires: stream_router.py confirmed_intent bypass
           + translation confirmation gate passed

    Safe to call even if confidence store is unavailable.
    """
    try:
        from app.memory.domains.confidence_learning import on_confirmation
        result = on_confirmation(phrase, intent)
        logger.debug(
            "[integration] Confirmed: '%s' → %s (conf=%.3f)",
            phrase[:40], intent, result.get("new_confidence", 0),
        )
    except Exception as e:
        logger.debug("[integration] Confidence confirm skipped: %s", e)


# =========================================================================
# 2. Confidence learning: correction
# =========================================================================

def on_intent_corrected(
    phrase: str,
    wrong_intent: str,
    right_intent: str,
    user_id: str = "default",
) -> None:
    """
    Called when a user corrects ASTRA's intent interpretation.

    Wires: translation feedback handler (false_negative)

    Safe to call even if confidence store is unavailable.
    """
    try:
        from app.memory.domains.confidence_learning import on_correction
        result = on_correction(phrase, wrong_intent, right_intent)
        logger.debug(
            "[integration] Corrected: '%s' %s → %s",
            phrase[:40], wrong_intent, right_intent,
        )
    except Exception as e:
        logger.debug("[integration] Confidence correct skipped: %s", e)


# =========================================================================
# 3. Post-message: preference capture + context extraction
# =========================================================================

def after_user_message(
    message: str,
    project_id: str = "astra-core",
    user_id: str = "default",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    db_session=None,
) -> None:
    """
    Called after processing a user message.

    Runs three checks:
      a) Does the message contain a preference statement?
      b) Does the message contain extractable facts/decisions?
      c) Should a conversation summary be generated? (v10.0)

    v10.0: Added conversation session management and summary triggering.
           Accepts provider/model for session tracking and db_session for
           session operations.

    Wires: chat_routing.handle_chat_mode()
           + stream_router command mode message save

    Non-blocking — failures are logged and swallowed.
    """
    # a) Preference capture (lightweight detection only)
    # We detect preference-like language and log it.
    # Full key/value extraction requires an LLM pass which we
    # don't do inline — we just flag it so the inferred channel
    # can accumulate evidence from repeated patterns.
    try:
        from app.memory.domains.preference_capture import (
            detect_preference_intent,
            capture_inferred,
        )
        detection = detect_preference_intent(message)
        if detection.get("has_preference"):
            # Store the raw preference statement for later extraction.
            # The inferred channel accumulates these — when the same
            # preference appears 2+ times, it gets promoted.
            capture_inferred(
                domain="general",
                key="_raw_preference_statement",
                observed_value=message[:200],
                count=1,
                context=f"auto_capture:{project_id}",
            )
            logger.info(
                "[integration] Preference language detected: %s",
                detection.get("triggers_found", []),
            )
    except Exception as e:
        logger.debug("[integration] Preference capture skipped: %s", e)

    # b) Real-time biographical fact capture (v0.14.0)
    # Detects explicit personal statements ("I live in X", "I work at Y")
    # and writes them to permanent memory immediately — no LLM needed.
    try:
        from app.memory.realtime_fact_capture import capture_biographical_facts
        _bio_count = capture_biographical_facts(message, db=db_session)
        if _bio_count > 0:
            logger.info("[integration] Captured %d biographical fact(s) from chat", _bio_count)
    except Exception as e:
        logger.debug("[integration] Biographical capture skipped: %s", e)

    # b2) Phase 3: Tier 1 identity capture — independent of SQL preference path.
    # Writes to data/self_model/identity.json (the always-injected block).
    try:
        from app.self_model.identity_capture import capture_and_persist as _id_cap
        _id_result = _id_cap(message, source=f"chat:{project_id}")
        if _id_result.get("written"):
            logger.info(
                "[integration] Identity facts captured: %s",
                ", ".join(w.get("field", "?") for w in _id_result["written"]),
            )
    except Exception as e:
        logger.debug("[integration] Identity capture skipped: %s", e)

    # b3) Phase 7: fragment capture — lifetime associative memory.
    # Every user message → classified → embedded → clustered into themes.
    # Never deleted, decays over time, hot themes inject, cold themes surface on match.
    try:
        from app.self_model.fragments.capture import capture_fragment
        _frag_result = capture_fragment(
            message,
            source=f"chat:{project_id}",
            project_id=int(project_id) if str(project_id).isdigit() else None,
        )
        if _frag_result.get("fragment_id"):
            logger.info(
                "[integration] fragment captured: signal=%s claim=%s theme=%s",
                _frag_result.get("signal"),
                _frag_result.get("claim"),
                _frag_result.get("theme_id", "(none)"),
            )
    except Exception as e:
        logger.debug("[integration] fragment capture skipped: %s", e)

    # c) Context extraction (facts and decisions)
    try:
        from app.memory.domains.context import ContextStore
        _extract_context_facts(message, project_id)
    except Exception as e:
        logger.debug("[integration] Context extraction skipped: %s", e)

    # c) Conversation session management + summary triggering (v10.0)
    _handle_conversation_session(
        project_id=project_id,
        provider=provider,
        model=model,
        db=db_session,
    )


def record_session_activity(
    project_id,
    provider=None,
    model=None,
    db_session=None,
) -> None:
    """Session + summary handling ONLY (no preference/identity/fragment
    capture). For callers like the Bridge that already run their own
    capture hooks (identity_hook.capture_from_bridge_message etc.) and
    would double-count preference evidence if they called the full
    after_user_message(). v2026-06-10: added so phone conversations get
    sessions and rolling summaries like the desktop path.

    Non-blocking -- failures are logged and swallowed downstream.
    """
    _handle_conversation_session(
        project_id=str(project_id),
        provider=provider,
        model=model,
        db=db_session,
    )


def _handle_conversation_session(
    project_id: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    db=None,
) -> None:
    """
    Manage conversation session lifecycle and trigger summary generation.

    v10.0: CONV-MEMORY-001 Phase 2 integration.

    Steps:
      1. Get or create the active session for this project.
      2. Bump session activity (message count, model tracking).
      3. Check if a summary should be generated.
      4. If yes, spawn summary generation as a background task.
    """
    if db is None:
        logger.debug("[integration] No db_session for conversation session management")
        return

    try:
        # Convert project_id to int (may be passed as string from some callers)
        pid = int(project_id) if isinstance(project_id, str) and project_id.isdigit() else None
        if pid is None:
            # Non-numeric project_id (e.g. "astra-core") — look up by project_key
            from app.memory.models import Project
            proj = db.query(Project).filter(
                Project.project_key == project_id
            ).first()
            pid = proj.id if proj else None

        if pid is None:
            logger.debug("[integration] Could not resolve project_id=%s", project_id)
            return

        from app.memory.conversation_service import (
            get_or_create_active_session,
            bump_session_activity,
        )
        from app.memory.summary_trigger import should_generate_summary

        session = get_or_create_active_session(db, pid)

        # v10.0 FIX: Backfill orphan messages that were created before
        # the session hook ran (message is saved in handle_chat_mode
        # BEFORE after_user_message fires). Also catches pre-existing
        # messages from before the conv memory layer was deployed.
        _backfill_orphan_messages(db, pid, session.id)

        bump_session_activity(db, session, provider=provider, model=model)

        if should_generate_summary(db, session):
            _spawn_summary_task(db, session)

    except Exception as e:
        logger.debug("[integration] Conversation session handling failed: %s", e)


def _backfill_orphan_messages(
    db, project_id: int, session_id: int,
) -> None:
    """
    Claim messages that have no session_id into the current session.

    This handles two cases:
      a) The current request's message — saved by handle_chat_mode()
         before after_user_message() runs, so it has session_id=NULL.
      b) Pre-existing messages from before the conv memory layer existed.

    Only backfills recent messages (last 20) to avoid claiming ancient
    history into a new session.
    """
    try:
        from app.memory.models import Message

        orphans = (
            db.query(Message)
            .filter(
                Message.project_id == project_id,
                Message.session_id.is_(None),
            )
            .order_by(Message.id.desc())
            .limit(20)
            .all()
        )

        if orphans:
            for msg in orphans:
                msg.session_id = session_id
            db.commit()
            logger.debug(
                "[integration] Backfilled %d orphan messages into session %d",
                len(orphans), session_id,
            )
    except Exception as e:
        logger.debug("[integration] Orphan backfill failed: %s", e)


def _spawn_summary_task(db, session) -> None:
    """
    Spawn summary generation as a non-blocking background task.

    Uses asyncio.create_task if we're inside an event loop,
    otherwise logs and skips (summary will catch up on next trigger).
    """
    import asyncio

    async def _run_summary():
        try:
            # Use a fresh DB session for the background task to avoid
            # SQLAlchemy thread-safety issues with the request session.
            from app.db import get_db_session
            bg_db = get_db_session()
            try:
                from app.memory.summary_generator import generate_summary
                from app.memory.conversation_service import get_session
                # Re-fetch session in the new DB session
                bg_session = get_session(bg_db, session.id)
                if bg_session:
                    await generate_summary(bg_db, bg_session)
            finally:
                bg_db.close()
        except Exception as e:
            logger.warning("[integration] Background summary failed: %s", e)

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_run_summary())
        logger.info(
            "[integration] Summary generation spawned for session %d",
            session.id,
        )
    except RuntimeError:
        # No running event loop — can't spawn async task
        logger.debug(
            "[integration] No event loop — summary deferred for session %d",
            session.id,
        )


def _extract_context_facts(message: str, project_id: str) -> None:
    """
    Extract key facts and decisions from a user message
    and store them in the ephemeral context.

    Uses simple pattern matching — not LLM-based.
    Keeps it fast and deterministic.
    """
    import re
    from app.memory.domains.context import ContextStore

    store = ContextStore()
    msg_lower = message.lower().strip()

    # Decision patterns: "let's go with X", "I've decided to X"
    decision_patterns = [
        r"(?:let'?s|i'?ll|we should|i'?ve decided to|going to)\s+(.{10,80})",
        r"(?:the plan is|the approach is)\s+(.{10,80})",
    ]
    for pattern in decision_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            store.set_decision(
                decision=match.group(1).strip().rstrip(".!"),
                project_id=project_id,
            )
            return  # One extraction per message is enough

    # Fact patterns: "I'm working on X", "the file is X"
    fact_patterns = [
        (r"(?:i'?m|we'?re) working on\s+(.{5,60})", "current_task"),
        (r"(?:the|this) (?:file|module|component) is\s+(.{5,60})", "current_file"),
        (r"(?:focus|priority) (?:is|should be)\s+(.{5,60})", "current_priority"),
    ]
    for pattern, key in fact_patterns:
        match = re.search(pattern, msg_lower)
        if match:
            store.set_fact(
                key=key,
                value=match.group(1).strip().rstrip(".!"),
                project_id=project_id,
            )
            return


# =========================================================================
# 4. Complexity-aware routing
# =========================================================================

def enrich_routing(
    query: str,
    job_type: Optional[str] = None,
    intent: Optional[str] = None,
    attachments: Optional[list] = None,
) -> dict:
    """
    Enrich a routing decision with complexity analysis.

    Wires: call_llm_async() in routing/core.py after classify_and_route

    Returns dict with model_tier, complexity info, and rag_needed flag.
    Returns empty dict if complexity module unavailable.
    """
    try:
        from app.memory.complexity_router import (
            route_with_complexity,
            should_use_rag,
            get_rag_depth_for_tier,
        )
        result = route_with_complexity(
            query=query,
            job_type=job_type,
            intent=intent,
            attachments=attachments,
        )
        complexity = result.get("complexity")
        result["rag_needed"] = should_use_rag(complexity, job_type)
        result["rag_depth"] = get_rag_depth_for_tier(
            complexity.tier if complexity else "reasoning"
        )
        return result
    except Exception as e:
        logger.debug("[integration] Complexity routing skipped: %s", e)
        return {}


# =========================================================================
# 5. RAG memory injection
# =========================================================================

def inject_memory_context(
    query: str,
    project_id: str = "astra-core",
    domains: Optional[list[str]] = None,
    limit: int = 10,
) -> str:
    """
    Query the memory router and format results for LLM context injection.

    Wires: prompt_builders.build_system_prompt() or build_full_context()

    Returns formatted string for system prompt injection,
    or empty string if no results or router unavailable.
    """
    try:
        from app.memory.router import memory_router

        results = memory_router.query(
            text=query,
            project_id=project_id,
            domains=domains,
            limit=limit,
            min_relevance=0.3,
        )
        if not results:
            return ""

        lines = ["[MEMORY CONTEXT]"]
        for r in results:
            domain_tag = f"[{r.domain}]" if r.domain else ""
            lines.append(f"  {domain_tag} {r.content[:200]}")
        lines.append("[/MEMORY CONTEXT]")

        return "\n".join(lines)

    except Exception as e:
        logger.debug("[integration] Memory injection skipped: %s", e)
        return ""
