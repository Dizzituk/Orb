# FILE: app/llm/routing/memory_injection.py
# Purpose: Memory Context Injection for LLM Routing
# Called-by: app.llm.routing.envelope
# Depends-on: app.astra_memory, app.astra_memory.confidence_config, app.astra_memory.topic_tagger, app.lifestyle.nudges (+1 more)
# Last-renovated: 2026-06-15 (Job 2: per-turn front-door block observability)
"""
Memory Context Injection for LLM Routing

Injects relevant memory context into LLM calls based on:
1. Intent depth classification (D0-D4)
2. Applicable preferences for the job type
3. Hot index retrieval results

This module is called during envelope synthesis to add memory
context to the system prompt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Lazy memory-system import (2026-06-12). app.astra_memory's own import
# cascade reaches back into this module via app.llm.routing, so a module-
# level `from app.astra_memory import ...` here hits the partially-
# initialised package and latched MEMORY_AVAILABLE=False for the whole
# process whenever astra_memory happened to be imported first — silently
# killing ALL memory injection. Resolve the API at call time instead.
MEMORY_AVAILABLE = True  # kept for envelope's gate; resolution is lazy

_MEMORY_API: Optional[Dict[str, Any]] = None


def _memory_api() -> Dict[str, Any]:
    """Resolve the astra_memory API on first use (post-init, cycle-safe)."""
    global _MEMORY_API
    if _MEMORY_API is None:
        from app.astra_memory import (
            classify_intent_depth,
            retrieve_for_query,
            get_applicable_preferences,
            IntentDepth,
        )
        _MEMORY_API = {
            "classify_intent_depth": classify_intent_depth,
            "retrieve_for_query": retrieve_for_query,
            "get_applicable_preferences": get_applicable_preferences,
            "IntentDepth": IntentDepth,
        }
    return _MEMORY_API


@dataclass
class MemoryContext:
    """Memory context to inject into LLM call."""
    depth: str
    preferences_text: str
    facts_text: str
    token_estimate: int
    preferences_applied: List[str]
    records_retrieved: int
    # Job 2 (2026-06-12): repo-scan chunks for code/architecture queries,
    # already carrying their staleness header (see rag.retrieval.chat_injection)
    repo_context: str = ""
    # Job 0 (2026-06-15): conversation-level + legacy sources folded into the
    # single front door so a fact surfaces once. Off for the envelope (no
    # project_id), on for the project-aware paths via build_memory_block.
    summary_text: str = ""
    recall_text: str = ""
    router_text: str = ""
    # Job 4 (2026-06-15): bounded knowledge-graph walk — strong cross-domain
    # links (e.g. "investments fund the Portugal move") that vectors can't
    # represent. Opt-in (include_graph), so the envelope path is unchanged.
    graph_text: str = ""

    def is_empty(self) -> bool:
        """Check if there's anything to inject."""
        return not any((
            self.preferences_text, self.facts_text, self.repo_context,
            self.summary_text, self.recall_text, self.router_text,
            self.graph_text,
        ))

    def format_for_system_prompt(self) -> str:
        """Format memory context for system prompt injection.

        Job 0 (2026-06-15): assembly + cross-source de-dup live in
        memory_injection_sources.assemble_block — the single place that
        guarantees one fact appears at most once across all sources.
        """
        if self.is_empty():
            return ""
        from app.llm.routing.memory_injection_sources import assemble_block
        return assemble_block(
            preferences_text=self.preferences_text,
            facts_text=self.facts_text,
            graph_text=self.graph_text,
            summary_text=self.summary_text,
            recall_text=self.recall_text,
            repo_context=self.repo_context,
            router_text=self.router_text,
        )


def _extract_user_message_text(messages: List[Dict[str, Any]]) -> str:
    """Extract text from the last user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Multimodal: extract text parts
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                return " ".join(text_parts)
    return ""


def _format_preferences(preferences: List[Any]) -> str:
    """Format preferences as readable text."""
    if not preferences:
        return ""
    
    lines = []
    for pref in preferences:
        key = pref.preference_key
        value = pref.preference_value
        strength = pref.strength.value if hasattr(pref.strength, 'value') else str(pref.strength)
        
        # Format based on value type
        if isinstance(value, bool):
            value_str = "enabled" if value else "disabled"
        elif isinstance(value, dict):
            value_str = ", ".join(f"{k}={v}" for k, v in value.items())
        else:
            value_str = str(value)
        
        # Add strength indicator for hard rules
        if strength == "hard_rule":
            lines.append(f"• {key}: {value_str} [REQUIRED]")
        else:
            lines.append(f"• {key}: {value_str}")
    
    return "\n".join(lines)


def _format_retrieved_records(result: Any, max_per_record: int = 2000) -> str:
    """
    Format retrieved records as readable text.
    
    For hot-only (D0/D1): Shows title + one-liner
    For cold (D2+): Shows full content (already depth-truncated in retrieval)
    """
    if not result or not result.records:
        return ""
    
    lines = []
    total_chars = 0
    max_total = 16000  # Cap total context size
    
    for record in result.records:
        if total_chars >= max_total:
            lines.append(f"\n[...{len(result.records) - len(lines)} more records omitted...]")
            break
            
        title = record.title
        content = record.content
        summary_level = getattr(record, 'summary_level', 0)
        
        # For hot-only (L0), keep brief
        if summary_level == 0:
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"• [{record.record_type}] {title}: {content}")
        else:
            # For cold content (L1+), include full retrieved content
            # Already truncated by depth in retrieval layer
            if len(content) > max_per_record:
                content = content[:max_per_record] + "..."
            
            lines.append(f"\n### [{record.record_type.upper()}] {title}\n{content}")
        
        total_chars += len(lines[-1])
    
    return "\n".join(lines)


def build_memory_context(
    db: Session,
    messages: List[Dict[str, Any]],
    job_type: Optional[str] = None,
    component: str = "llm_router",
    *,
    project_id: Optional[int] = None,
    window_message_ids: Optional[List[int]] = None,
    include_conversation: bool = False,
    include_router: bool = False,
    include_graph: bool = False,
) -> MemoryContext:
    """
    Build memory context for injection.

    Args:
        db: Database session
        messages: The conversation messages
        job_type: Optional job type for preference filtering
        component: Component name for preference scoping
        project_id: Project for the conversation-scoped sources (summary/recall).
            The envelope path leaves this None (LLMTask carries no project_id),
            so those sources stay off there — see MEMORY_MAP.md §2b.
        window_message_ids: Live-window message ids to exclude from recall.
            Loaded from project_id when omitted.
        include_conversation: Add the rolling summary + within-conversation
            recall (Job 0 front-door consolidation).
        include_router: Add the legacy MemoryRouter keyword block.
        include_graph: Add the bounded cross-domain graph walk (Job 4 task 2).
            Opt-in and defaulted off so the envelope path is byte-for-byte
            unchanged; the full front-door path (build_memory_block) turns it on.

    Returns:
        MemoryContext with formatted text ready for injection
    """
    try:
        api = _memory_api()
    except Exception as _api_exc:
        logger.warning("[memory_injection] ASTRA memory system not available: %s", _api_exc)
        return MemoryContext(
            depth="unavailable",
            preferences_text="",
            facts_text="",
            token_estimate=0,
            preferences_applied=[],
            records_retrieved=0,
        )

    IntentDepth = api["IntentDepth"]

    # Extract user message for depth classification
    user_message = _extract_user_message_text(messages)

    # Classify intent depth
    depth = api["classify_intent_depth"](user_message)

    # D0: No memory at all
    if depth == IntentDepth.D0:
        return MemoryContext(
            depth="D0",
            preferences_text="",
            facts_text="",
            token_estimate=0,
            preferences_applied=[],
            records_retrieved=0,
        )
    
    # Get applicable preferences
    preferences = []
    preferences_applied = []
    try:
        # Get preferences for this component
        prefs = api["get_applicable_preferences"](db, component)

        # Also get preferences for "all" and the specific job type
        if job_type:
            job_prefs = api["get_applicable_preferences"](db, job_type)
            prefs.extend(job_prefs)
        
        # Deduplicate by key
        seen_keys = set()
        for pref in prefs:
            if pref.preference_key not in seen_keys:
                preferences.append(pref)
                preferences_applied.append(pref.preference_key)
                seen_keys.add(pref.preference_key)
                
    except Exception as e:
        logger.warning(f"[memory_injection] Failed to get preferences: {e}")
    
    preferences_text = _format_preferences(preferences)
    
    # Retrieve facts based on depth
    facts_text = ""
    records_retrieved = 0
    query_tags = None
    query_entities = None
    retrieved_records: List[Any] = []  # captured for the Job 4 graph walk

    if depth != IntentDepth.D0:
        try:
            # Phase 9 (2026-05-13): extract tags/entities from the user
            # message and pass them through. Previously this was hardcoded
            # to None, None, which meant stage1_candidate_selection had no
            # semantic filter and fell back to "top-N by static priority +
            # recency" — i.e. retrieval ignored what the question was
            # actually ABOUT. The topic_tagger uses the same vocab the
            # indexer uses to tag records, so tags from the query match
            # tags on the records.
            try:
                from app.astra_memory.topic_tagger import (
                    extract_tags as _q_extract_tags,
                    extract_entities as _q_extract_entities,
                )
                _qt = _q_extract_tags(user_message)
                # Drop the 'general' fallback when used as a query filter —
                # filtering on 'general' adds no signal.
                query_tags = [t for t in _qt if t != "general"] or None
                _qe = _q_extract_entities(user_message)
                query_entities = _qe or None
            except Exception as _tag_err:
                logger.debug(
                    "[memory_injection] topic_tagger failed, falling back "
                    "to unfiltered retrieval: %s", _tag_err,
                )
                query_tags = None
                query_entities = None

            result = api["retrieve_for_query"](
                db=db,
                user_message=user_message,
                query_tags=query_tags,
                query_entities=query_entities,
                depth_override=depth,
            )
            
            facts_text = _format_retrieved_records(result)
            records_retrieved = result.records_expanded
            retrieved_records = list(getattr(result, "records", []) or [])

        except Exception as e:
            logger.warning(f"[memory_injection] Failed to retrieve facts: {e}")
    
    # ── Job 2 (2026-06-12): repo-scan retrieval — when the question is about
    # ASTRA's own code/architecture, surface arch-scan chunks with an honest
    # staleness header (host vs sandbox snapshot; offline marker when the
    # sandbox is unreachable). Gated so everyday chat is never polluted.
    repo_context = ""
    if depth != IntentDepth.D0:
        try:
            from app.rag.retrieval.chat_injection import build_repo_context
            repo_context = build_repo_context(db, user_message, query_tags)
        except Exception as _repo_exc:
            logger.debug(f"[memory_injection] repo context skipped: {_repo_exc}")

    # ── Job 5 (2026-06-10): daily coach nudge — if the lifestyle scheduler has
    # produced an active nudge, weave it into context once (cooldown handled
    # inside get_nudge_for_injection so it doesn't repeat every message).
    try:
        from app.lifestyle.nudges import get_nudge_for_injection
        _nudge = get_nudge_for_injection()
        if _nudge:
            facts_text = (facts_text + "\n\n" if facts_text else "") + (
                "[DAILY COACH NOTE — mention once, naturally, if the moment fits]: " + _nudge
            )
    except Exception as _nudge_exc:
        logger.debug(f"[memory_injection] nudge injection skipped: {_nudge_exc}")

    # ── ASTRA Sentinel (2026-06-12): active security alert — weave the top
    # unacknowledged alert into context once (cooldown handled inside
    # get_alert_for_injection, same nudges pattern as above).
    try:
        from app.sentinel.alerts import get_alert_for_injection
        _sec_alert = get_alert_for_injection()
        if _sec_alert:
            facts_text = (facts_text + "\n\n" if facts_text else "") + (
                "[ACTIVE SECURITY ALERT — tell Taz about this proactively and plainly, "
                "without scaremongering; he can review/act in the Security tab]: " + _sec_alert
            )
    except Exception as _sentinel_exc:
        logger.debug(f"[memory_injection] sentinel alert injection skipped: {_sentinel_exc}")

    # ── Job 1b (2026-06-10): pending identity confirmations — surface queued
    # arbiter proposals conversationally (max once per 6h per process) so
    # Tier 1 facts get confirmed instead of expiring unseen after 30 days.
    try:
        import time as _time
        _state = globals().setdefault("_PROPOSAL_INJECT_STATE", {"ts": 0.0})
        if _time.time() - _state["ts"] > 6 * 3600:
            from app.self_model.proposed_facts import get_proposed_facts_store
            _queued = get_proposed_facts_store().list(status="queued", limit=3)
            if _queued:
                _state["ts"] = _time.time()
                _p = _queued[0]
                facts_text = (facts_text + "\n\n" if facts_text else "") + (
                    f"[PENDING IDENTITY CONFIRMATION — {len(_queued)} queued]: ASTRA noticed a "
                    f"possible fact about the user: {_p.field_name} = {_p.proposed_value!r} "
                    f"(currently {_p.current_value!r}). At a natural moment, ask once whether "
                    f"that's correct; applying or rejecting it is one tap in Self Model → "
                    f"Identity on the desktop."
                )
    except Exception as _prop_exc:
        logger.debug(f"[memory_injection] proposal injection skipped: {_prop_exc}")

    # ── Job 0 (2026-06-15): conversation-level + legacy sources, folded into
    # the single front door behind the SAME D0-D4 gate (we only reach here for
    # D1+). project_id is required for the project-scoped summary/recall, so
    # the envelope path (no project_id) leaves them empty by construction.
    summary_text = ""
    recall_text = ""
    router_text = ""
    if project_id is not None and (include_conversation or include_router):
        from app.llm.routing import memory_injection_sources as _src
        win = window_message_ids
        if win is None:
            win = _src.recent_window_ids(db, project_id)
        if include_conversation:
            summary_text = _src.collect_summary(db, project_id)
            recall_text = _src.collect_recall(db, project_id, user_message, win)
        if include_router:
            router_text = _src.collect_router(user_message, project_id)

    # ── Job 4 (2026-06-15, task 2): bounded cross-domain graph walk. When the
    # topics in play (query tags + the tags of the records just retrieved) sit
    # on a strong consolidated edge, surface the connected node — one or two
    # hops, strength-thresholded, small cap. This is the relationship channel
    # ("investments fund the Portugal move") that the similarity channel can't
    # represent. Behind the same D0-D4 gate (we only reach here for D1+);
    # opt-in so the envelope path stays unchanged. The walk is in-process and
    # cheap (no LLM, no embedding), so it is safe on the live voice path.
    graph_text = ""
    if include_graph:
        try:
            from app.llm.routing.graph_context import build_graph_context
            graph_text = build_graph_context(query_tags, retrieved_records)
        except Exception as _graph_exc:
            logger.debug(f"[memory_injection] graph walk skipped: {_graph_exc}")

    # Estimate tokens (rough: 4 chars per token)
    total_text = (
        preferences_text + facts_text + repo_context
        + summary_text + recall_text + router_text + graph_text
    )
    token_estimate = len(total_text) // 4

    return MemoryContext(
        depth=depth.value,
        preferences_text=preferences_text,
        facts_text=facts_text,
        token_estimate=token_estimate,
        preferences_applied=preferences_applied,
        records_retrieved=records_retrieved,
        repo_context=repo_context,
        summary_text=summary_text,
        recall_text=recall_text,
        router_text=router_text,
        graph_text=graph_text,
    )


def inject_memory_into_system_prompt(
    system_prompt: Optional[str],
    memory_context: MemoryContext,
) -> str:
    """
    Inject memory context into system prompt.
    
    Memory is prepended to the system prompt so it's available
    as context for the entire conversation.
    """
    if memory_context.is_empty():
        return system_prompt or ""
    
    memory_block = memory_context.format_for_system_prompt()
    
    if system_prompt:
        return f"{memory_block}\n\n{system_prompt}"
    else:
        return memory_block


def get_memory_injection_stats(memory_context: MemoryContext) -> Dict[str, Any]:
    """Get stats about memory injection for logging/debugging."""
    return {
        "depth": memory_context.depth,
        "token_estimate": memory_context.token_estimate,
        "preferences_applied": memory_context.preferences_applied,
        "records_retrieved": memory_context.records_retrieved,
        "is_empty": memory_context.is_empty(),
    }


def build_memory_block(
    db: Session,
    *,
    user_message: str,
    project_id: Optional[int] = None,
    window_message_ids: Optional[List[int]] = None,
    job_type: Optional[str] = None,
    component: str = "llm_router",
    conversation_only: bool = False,
) -> str:
    """THE single memory front door (MEMORY_MAP.md §2).

    Returns one de-duped memory block (or "") for the project-aware chat paths
    to splice into their context. Every source sits behind the one D0-D4 gate.

    Two callers:
      - build_full_context (desktop streaming + phone): full block — semantic
        facts + preferences + repo + nudges/sentinel/proposals + summary +
        recall + MemoryRouter, all de-duped.
      - endpoints/chat.py (/chat): conversation_only=True — summary + recall
        only, because that path already gets the semantic core from the
        envelope (build_memory_context) via call_llm; this avoids doubling it.

    Failure-proof: any error returns "" so a chat turn never breaks on memory.
    """
    try:
        if conversation_only:
            # Apply the D0-D4 gate here (we don't go through build_memory_context
            # on this path) so the silence-on-D0 rule still governs it.
            try:
                api = _memory_api()
                depth = api["classify_intent_depth"](user_message or "")
                if depth == api["IntentDepth"].D0:
                    return ""
            except Exception:
                pass  # gate unavailable — proceed (sources are themselves safe)
            if project_id is None:
                return ""
            from app.llm.routing import memory_injection_sources as _src
            win = window_message_ids
            if win is None:
                win = _src.recent_window_ids(db, project_id)
            summary_text = _src.collect_summary(db, project_id)
            recall_text = _src.collect_recall(db, project_id, user_message, win)
            block = _src.assemble_block(
                summary_text=summary_text, recall_text=recall_text,
            )
            # Job 2 (task 3): make the voice-path lost-middle signature
            # observable — recall=n summary=n block=0 on a long session = the bug.
            _src.log_block_stats(
                "conv", project_id, block,
                summary=bool(summary_text), recall=bool(recall_text),
            )
            return block

        ctx = build_memory_context(
            db,
            messages=[{"role": "user", "content": user_message or ""}],
            job_type=job_type,
            component=component,
            project_id=project_id,
            window_message_ids=window_message_ids,
            include_conversation=True,
            include_router=True,
            include_graph=True,
        )
        block = ctx.format_for_system_prompt()
        from app.llm.routing import memory_injection_sources as _src
        _src.log_block_stats(
            "full", project_id, block,
            summary=bool(ctx.summary_text), recall=bool(ctx.recall_text),
            facts=bool(ctx.facts_text), depth=ctx.depth,
            graph=bool(ctx.graph_text),
        )
        return block
    except Exception as exc:
        logger.warning("[memory_injection] front door failed: %s", exc)
        return ""


__all__ = [
    "MemoryContext",
    "build_memory_context",
    "build_memory_block",
    "inject_memory_into_system_prompt",
    "get_memory_injection_stats",
    "MEMORY_AVAILABLE",
]
