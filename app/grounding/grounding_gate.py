# FILE: app/grounding/grounding_gate.py
# Purpose: Grounding Gate — Core orchestrator for the Grounded Intelligence System.
# Called-by: app.grounding.chat_integration
# Depends-on: app.grounding.freshness_policy, app.grounding.perspective_engine, app.grounding.perspective_formatter, app.grounding.perspective_prompts (+4 more)
# Last-renovated: 2026-06-11
"""
Grounding Gate — Core orchestrator for the Grounded Intelligence System.

Intercepts chat-mode messages BEFORE the LLM generates a response.
When a message is classified as claims-dependent or contested, the gate:

1. Runs topic classification
2. Checks freshness policy for the detected domain
3. Performs one or more web searches to gather evidence
4. Injects the evidence + grounding system prompt into the LLM context
5. Returns the augmented system prompt and evidence for the streaming handler

The gate does NOT replace the existing web_search intent routing —
that handles explicit "search the web for X" requests. The grounding
gate handles the implicit case: the user asks a factual question in
normal conversation and expects a grounded answer without having to
say "search the web."

v1.0 (2026-03): Initial implementation.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# =========================================================================
# Internal imports
# =========================================================================

from app.grounding.topic_classifier import (
    TopicCategory,
    ClassificationResult,
    classify_topic,
)
from app.grounding.freshness_policy import (
    get_freshness_rule,
    is_search_required,
)
from app.grounding.source_attribution import (
    GROUNDING_SYSTEM_PROMPT,
    CONTESTED_SYSTEM_PROMPT,
    format_evidence_block,
    build_confidence_note,
)


# =========================================================================
# Gate result
# =========================================================================

@dataclass
class GroundingResult:
    """Output of the grounding gate."""
    grounding_applied: bool = False
    category: TopicCategory = TopicCategory.PERSONAL
    confidence: float = 0.0
    domain_hint: str = ""
    system_prompt_injection: str = ""     # Extra prompt block to prepend
    evidence_block: str = ""              # [GROUNDING EVIDENCE] text
    confidence_note: str = ""             # Thin/stale evidence disclaimer
    source_count: int = 0
    search_queries_used: list[str] = field(default_factory=list)
    classification_signals: list[str] = field(default_factory=list)


# =========================================================================
# Configuration
# =========================================================================

def _is_grounding_enabled() -> bool:
    """Check if grounding gate is enabled (default: True)."""
    val = os.getenv("GROUNDING_GATE_ENABLED", "true").strip().lower()
    return val in ("true", "1", "yes")


def _min_confidence_threshold() -> float:
    """Minimum classification confidence to trigger grounding."""
    val = os.getenv("GROUNDING_MIN_CONFIDENCE", "0.5").strip()
    try:
        return max(0.0, min(1.0, float(val)))
    except ValueError:
        return 0.5

def _get_int_env(key: str, default: int, lo: int = 1, hi: int = 30) -> int:
    """Bounded int from env. Falls back to default on invalid or missing."""
    val = os.getenv(key, "").strip()
    if not val:
        return default
    try:
        return max(lo, min(hi, int(val)))
    except ValueError:
        return default


def _max_queries_standard() -> int:
    """Max queries per claims-grounding pass. v17: default raised 3 -> 6."""
    return _get_int_env("GROUNDING_MAX_QUERIES", 6)


def _max_results_standard() -> int:
    """Max results per query for claims grounding. v17: default raised 5 -> 8."""
    return _get_int_env("GROUNDING_MAX_RESULTS_PER_QUERY", 8)


def _max_queries_balanced() -> int:
    """Max queries for balanced (contested) grounding. v17: 5 -> 8."""
    return _get_int_env("GROUNDING_MAX_BALANCED_QUERIES", 8)


def _max_results_balanced() -> int:
    """Max results per query for balanced grounding. v17: 4 -> 6."""
    return _get_int_env("GROUNDING_MAX_BALANCED_RESULTS", 6)



# =========================================================================
# Search execution
# =========================================================================

async def _perform_grounding_search(
    queries: list[str],
    max_results_per_query: Optional[int] = None,
    context: Optional[dict] = None,
) -> list[dict]:
    """
    Execute web searches for grounding evidence.
    
    Returns a flat list of source dicts with: title, url, snippet,
    credibility_label, source_type, bias_note.
    
    Uses the existing web search infrastructure (Brave primary,
    DuckDuckGo fallback).
    """
    try:
        from app.llm.web_search import WebSearchRequest, search_and_answer
    except ImportError:
        logger.error("[grounding_gate] web_search module not available")
        return []
    
    all_sources: list[dict] = []
    seen_urls: set[str] = set()
    
    # v17: env-configurable caps. Defaults raised from 3 queries / 5 results.
    if max_results_per_query is None:
        max_results_per_query = _max_results_standard()
    for query in queries[:_max_queries_standard()]:
        try:
            req = WebSearchRequest(
                query=query,
                max_results=max_results_per_query,
            )
            resp = await search_and_answer(req, context=context)
            
            if resp.ok and resp.sources:
                for src in resp.sources:
                    if src.url in seen_urls:
                        continue
                    seen_urls.add(src.url)
                    all_sources.append({
                        "title": src.title,
                        "url": src.url,
                        "snippet": src.snippet,
                        "credibility_label": src.credibility_label,
                        "source_type": src.source_type,
                        "bias_note": src.bias_note,
                    })
        except Exception as e:
            logger.warning("[grounding_gate] Search failed for query '%s': %s", query, e)
            continue
    
    return all_sources


async def _perform_balanced_search(
    message: str,
    context: Optional[dict] = None,
) -> tuple[list[dict], "DebateAnalysis"]:
    """
    Execute multi-perspective searches for contested topics using
    the Balanced Perspective Engine.
    
    1. Analyses the debate type and identifies perspectives
    2. Generates targeted search queries per perspective
    3. Runs searches and returns sources + debate analysis
    
    Returns:
        Tuple of (sources_list, DebateAnalysis).
    """
    try:
        from app.grounding.perspective_engine import analyse_debate
    except ImportError:
        logger.warning("[grounding_gate] perspective_engine not available, using basic search")
        queries = [message.strip()[:200], f"{message.strip()[:150]} arguments for", f"{message.strip()[:150]} arguments against"]
        sources = await _perform_grounding_search(queries=queries, max_results_per_query=_max_results_balanced(), context=context)
        return sources, None
    
    analysis = analyse_debate(message)
    
    # Use the engine's targeted queries (neutral + per-perspective)
    sources = await _perform_grounding_search(
        queries=analysis.search_queries[:_max_queries_balanced()],
        max_results_per_query=_max_results_balanced(),
        context=context,
    )
    
    return sources, analysis


# =========================================================================
# Main gate function
# =========================================================================

async def evaluate_grounding(
    message: str,
    context: Optional[dict] = None,
) -> GroundingResult:
    """
    Evaluate whether a message needs grounding and perform it.
    
    This is the main entry point. Call it before the LLM generates
    a response in chat mode.
    
    Args:
        message: The user's raw message text.
        context: Optional context dict (user_id etc.) for search.
    
    Returns:
        GroundingResult with system prompt injection and evidence.
    """
    # Check if grounding is enabled
    if not _is_grounding_enabled():
        return GroundingResult()

    # Recall guard (2026-06-24): a question about OUR OWN conversation
    # ("what was I mentioning to you about anthropic", "what have we talked about
    # today") must NOT trigger a grounding web search — the topic_classifier reads
    # "anthropic"/"today" as a CLAIM to verify, but the answer lives in memory.
    # Shares the one predicate the Tier-0 + coverage paths use. A genuine "what's
    # the latest on anthropic" is not a recall question and still grounds normally.
    try:
        from app.memory.recall_intent import is_recall_question
        if is_recall_question(message):
            return GroundingResult(
                category=TopicCategory.PERSONAL,
                classification_signals=["recall_question"],
            )
    except Exception:
        pass

    # Classify the topic
    classification = classify_topic(message)
    
    logger.debug(
        "[grounding_gate] Classification: category=%s, confidence=%.2f, "
        "domain=%s, signals=%s",
        classification.category.value,
        classification.confidence,
        classification.domain_hint,
        classification.matched_signals[:3],
    )

    # Notify Self-Model of classification decision
    try:
        from app.self_model.hooks import on_routing_decision
        on_routing_decision(
            user_message=message[:100],
            tier="grounding",
            provider="",
            model="",
            category=classification.category.value,
            confidence=classification.confidence,
        )
    except Exception:
        pass
    
    # If personal/conversational, no grounding needed
    if classification.category == TopicCategory.PERSONAL:
        return GroundingResult(
            category=TopicCategory.PERSONAL,
            confidence=classification.confidence,
            classification_signals=classification.matched_signals,
        )
    
    # Check confidence threshold
    if classification.confidence < _min_confidence_threshold():
        logger.info(
            "[grounding_gate] Confidence %.2f below threshold %.2f — skipping",
            classification.confidence, _min_confidence_threshold(),
        )
        return GroundingResult(
            category=classification.category,
            confidence=classification.confidence,
            classification_signals=classification.matched_signals,
        )
    
    # Check freshness policy
    domain = classification.domain_hint or "general"
    rule = get_freshness_rule(domain)
    
    # Build search queries
    queries = classification.suggested_search_queries
    if not queries:
        # Fallback: use the message itself as a search query
        queries = [message.strip()[:200]]
    
    # Execute search based on category
    if classification.category == TopicCategory.CONTESTED:
        # v1.1: Enhanced balanced search via Perspective Engine
        sources, debate_analysis = await _perform_balanced_search(message, context=context)
        
        # Build perspective-aware prompt and evidence
        try:
            from app.grounding.perspective_prompts import build_balanced_prompt
            from app.grounding.perspective_formatter import (
                assign_perspectives,
                format_balanced_evidence,
            )
            
            if debate_analysis and debate_analysis.perspectives:
                system_injection = build_balanced_prompt(debate_analysis)
                tagged = assign_perspectives(sources, debate_analysis.perspectives)
                evidence_block = format_balanced_evidence(tagged, debate_analysis.perspectives)
            else:
                system_injection = CONTESTED_SYSTEM_PROMPT
                evidence_block = format_evidence_block(sources)
        except ImportError:
            logger.warning("[grounding_gate] Perspective modules not available, using basic prompt")
            system_injection = CONTESTED_SYSTEM_PROMPT
            evidence_block = format_evidence_block(sources)
    else:
        # Standard claims grounding
        sources = await _perform_grounding_search(queries, context=context)
        system_injection = GROUNDING_SYSTEM_PROMPT
        evidence_block = format_evidence_block(sources)
    
    # Confidence note
    confidence_note = build_confidence_note(len(sources))
    
    # Build the full injection (system prompt + evidence)
    full_injection = f"\n\n{system_injection}\n\n{evidence_block}"
    
    logger.info(
        "[grounding_gate] Grounding applied: category=%s, domain=%s, "
        "sources=%d, queries=%s",
        classification.category.value,
        domain,
        len(sources),
        queries[:2],
    )
    
    return GroundingResult(
        grounding_applied=True,
        category=classification.category,
        confidence=classification.confidence,
        domain_hint=domain,
        system_prompt_injection=full_injection,
        evidence_block=evidence_block,
        confidence_note=confidence_note,
        source_count=len(sources),
        search_queries_used=queries,
        classification_signals=classification.matched_signals,
    )


__all__ = [
    "GroundingResult",
    "evaluate_grounding",
]
