# FILE: app/memory/integration.py
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
) -> None:
    """
    Called after processing a user message.

    Runs two checks:
      a) Does the message contain a preference statement?
      b) Does the message contain extractable facts/decisions?

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

    # b) Context extraction (facts and decisions)
    try:
        from app.memory.domains.context import ContextStore
        _extract_context_facts(message, project_id)
    except Exception as e:
        logger.debug("[integration] Context extraction skipped: %s", e)


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
