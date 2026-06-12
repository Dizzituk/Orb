# FILE: app/self_model/hooks.py
# Purpose: Observer Hooks — Integration Points
# Called-by: app.grounding.grounding_gate, app.llm.stream_handlers, app.memory.knowledge_extractor, app.transparency.router
# Depends-on: app.self_model.chat_integration, app.self_model.identity, app.self_model.journal, app.self_model.models (+3 more)
# Last-renovated: 2026-06-11
"""
Observer Hooks — Integration Points

Connects the Self-Model pattern observer to existing ASTRA subsystems.
These are lightweight, fire-and-forget hooks that record events without
blocking or slowing down the main request path.

Hook points:
- Routing: records classification decisions and misroutes
- Memory: records retrieval hits and misses
- Corrections: records when the user corrects ASTRA
- Subsystem failures: records errors for trend detection
- Session: records learning events from knowledge extraction
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _get_observer():
    """Lazy import to avoid circular dependencies."""
    try:
        from app.self_model.observer import get_observer
        return get_observer()
    except Exception:
        return None


def _get_journal():
    """Lazy import to avoid circular dependencies."""
    try:
        from app.self_model.journal import get_journal
        return get_journal()
    except Exception:
        return None


def _get_user_model():
    """Lazy import to avoid circular dependencies."""
    try:
        from app.self_model.user_model import get_user_model
        return get_user_model()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════
# ROUTING HOOKS
# ══════════════════════════════════════════════════════════

def on_routing_decision(
    user_message: str,
    tier: str,
    provider: str,
    model: str,
    category: str = "",
    confidence: float = 0.0,
) -> None:
    """Called after a routing decision is made."""
    try:
        observer = _get_observer()
        if not observer:
            return
        # Track low-confidence classifications as potential misroutes
        if confidence > 0 and confidence < 0.5:
            observer.record_event(
                domain="routing",
                event_type="low_confidence_classification",
                detail=f"Message '{user_message[:60]}...' classified as {category} with only {confidence:.0%} confidence (tier={tier}, model={model})",
            )
    except Exception as e:
        logger.debug("[self_model] routing hook error: %s", e)


def on_routing_correction(
    user_message: str,
    original_domain: str,
    corrected_domain: str,
) -> None:
    """Called when the user indicates routing was wrong."""
    try:
        observer = _get_observer()
        if not observer:
            return
        observer.record_routing_miss(
            intended_domain=corrected_domain,
            actual_domain=original_domain,
            user_input=user_message,
        )
        journal = _get_journal()
        if journal:
            journal.record_observation(
                f"Routing sent a {corrected_domain} message to {original_domain}",
                f"User message: {user_message[:100]}",
            )
    except Exception as e:
        logger.debug("[self_model] routing correction hook error: %s", e)


# ══════════════════════════════════════════════════════════
# MEMORY HOOKS
# ══════════════════════════════════════════════════════════

def on_memory_retrieval(
    query: str,
    results_count: int,
    top_score: float = 0.0,
) -> None:
    """Called after memory retrieval completes."""
    try:
        observer = _get_observer()
        if not observer:
            return
        # Track queries that return no results
        if results_count == 0:
            observer.record_memory_miss(
                query=query,
                expected_topic="unknown",
            )
    except Exception as e:
        logger.debug("[self_model] memory hook error: %s", e)


# ══════════════════════════════════════════════════════════
# CORRECTION HOOKS
# ══════════════════════════════════════════════════════════

def on_user_correction(
    domain: str,
    user_comment: str,
    stage_name: str = "",
) -> None:
    """Called when the user submits a correction via transparency."""
    try:
        observer = _get_observer()
        if not observer:
            return
        observer.record_user_correction(
            domain=domain or "general",
            what_was_wrong=user_comment,
        )
        journal = _get_journal()
        if journal:
            journal.record_observation(
                f"User correction in {domain or 'general'}: {user_comment[:80]}",
                f"Stage: {stage_name}" if stage_name else "",
            )
    except Exception as e:
        logger.debug("[self_model] correction hook error: %s", e)


# ══════════════════════════════════════════════════════════
# SUBSYSTEM FAILURE HOOKS
# ══════════════════════════════════════════════════════════

def on_subsystem_error(
    subsystem: str,
    error_summary: str,
) -> None:
    """Called when a subsystem encounters an error."""
    try:
        observer = _get_observer()
        if not observer:
            return
        observer.record_subsystem_failure(
            subsystem=subsystem,
            error_summary=error_summary,
        )
    except Exception as e:
        logger.debug("[self_model] subsystem error hook error: %s", e)


# ══════════════════════════════════════════════════════════
# KNOWLEDGE EXTRACTION HOOKS
# ══════════════════════════════════════════════════════════

# Keys from knowledge extractor that should ALSO be written to the
# Tier 1 identity store (these are hard biographical facts that
# belong in the always-injected identity block, not only in the
# confidence-scored self-model).
_IDENTITY_ROUTED_KEYS = {
    "name", "preferred_name", "date_of_birth", "birthplace",
    "current_location", "nationality", "occupation", "email", "phone",
}


def on_knowledge_extracted(
    category: str,
    key: str,
    value: str,
    source: str = "",
) -> None:
    """Called when the knowledge extractor promotes a fact to memory."""
    try:
        from app.self_model.models import UserFact
        user_model = _get_user_model()
        if not user_model:
            return
        fact = UserFact(
            category=category,
            key=key,
            value=value,
            source=source,
        )
        user_model.add_fact(fact)
        journal = _get_journal()
        if journal:
            journal.record_learning(
                f"Learned {category} fact: {key} = {value[:60]}",
                source,
            )

        # Phase 5: mirror biographical hard facts into the Tier 1
        # identity store so they end up in the always-injected block.
        #
        # Phase 1 audit (2026-05-02): this mirror is the path that
        # wrote `knowledge_extractor:session:NN` corruptions like
        # current_location = "Grew up in Lagos, Algarve, Portugal."
        # into identity.json. The LLM extractor is fallible and there
        # is no validation between its output and the Tier 1 store.
        #
        # Default behaviour is now LOG-ONLY: record what the extractor
        # would have written so Phase 3's arbiter has training data,
        # but do not actually mutate identity.json. Set
        # ASTRA_KNOWLEDGE_EXTRACTOR_IDENTITY_MIRROR=true to restore
        # the old behaviour (not recommended).
        if category == "biographical" and key in _IDENTITY_ROUTED_KEYS:
            # Phase 3 audit (2026-05-02): observe via arbiter regardless of
            # mirror flag — this gives us a record of every extractor decision
            # that targeted Tier 1, even when the mirror is suppressed.
            try:
                from app.self_model.write_arbiter import propose as _arbiter_propose
                _arbiter_propose(
                    field_name=key,
                    proposed_value=value,
                    source=f"knowledge_extractor:{source}",
                    evidence={"category": category},
                )
            except Exception as _arb_err:
                logger.debug("[self_model] arbiter observe failed: %s", _arb_err)

            _mirror_enabled = os.getenv(
                "ASTRA_KNOWLEDGE_EXTRACTOR_IDENTITY_MIRROR", "false",
            ).lower() in ("1", "true", "yes")
            if not _mirror_enabled:
                logger.info(
                    "[self_model] identity mirror SUPPRESSED (Phase 1): "
                    "would have written %s = %r (source=knowledge_extractor:%s)",
                    key, value, source,
                )
            else:
                try:
                    from app.self_model.identity import get_identity_store
                    r = get_identity_store().set(
                        key, value,
                        source=f"knowledge_extractor:{source}",
                    )
                    logger.info(
                        "[self_model] mirrored to identity store: %s -> %s",
                        key, r.get("action"),
                    )
                except Exception as id_err:
                    logger.debug("[self_model] identity mirror failed: %s", id_err)
    except Exception as e:
        logger.debug("[self_model] knowledge hook error: %s", e)


# ══════════════════════════════════════════════════════════
# SESSION HOOKS
# ══════════════════════════════════════════════════════════

def on_session_start() -> Optional[str]:
    """Called at session start. Returns greeting context if suggestions are pending."""
    try:
        from app.self_model.chat_integration import get_session_greeting_context
        return get_session_greeting_context()
    except Exception as e:
        logger.debug("[self_model] session start hook error: %s", e)
        return None
