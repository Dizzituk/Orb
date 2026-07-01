# FILE: app/rag/_answerer_model_selection.py
# Purpose: Complexity-aware, env-driven model selection for RAG queries.
# Called-by: app.rag.answerer
# Depends-on: app.memory.complexity
# Last-renovated: 2026-07-01
"""
Complexity-aware model selection for RAG queries.

Maps the complexity classifier's tier to a provider + model resolved
entirely from environment variables — no model ID is ever hardcoded.

Resolution order per tier:
    1. ORB_RAG_MODEL              — manual override, bypasses complexity
    2. RAG_<TIER>_PROVIDER/MODEL  — tier-specific env vars:
           ping_pong, lookup      → RAG_LOOKUP_PROVIDER  / RAG_LOOKUP_MODEL
           reasoning              → RAG_REASONING_PROVIDER / RAG_REASONING_MODEL
           deep, multimodal       → RAG_DEEP_PROVIDER    / RAG_DEEP_MODEL
    3. DEFAULT_PROVIDER / DEFAULT_MODEL — global env fallback
    4. Nothing. If no env var resolves, an error is logged and empty
       strings are returned; the downstream LLM call fails visibly
       instead of silently running a rotted hardcoded model.

History: until 2026-07-01 this file hardcoded a tier map
(gpt-4.1-mini / claude-sonnet-4-5-20250929 / claude-opus-4-5) — all
three were stale, superseded models by the time they were removed,
which is exactly why model IDs live in .env now.

v10.0 (2026-02-23): Job 10B — complexity-aware model selection.
v11.0 (2026-07-01): Lane C — de-hardcoded; tiers resolve from .env.
"""

import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# Complexity tier → env var names (values live in .env, never here)
# =========================================================================

RAG_TIER_ENV_VARS: dict[str, Tuple[str, str]] = {
    "ping_pong":  ("RAG_LOOKUP_PROVIDER",    "RAG_LOOKUP_MODEL"),
    "lookup":     ("RAG_LOOKUP_PROVIDER",    "RAG_LOOKUP_MODEL"),
    "reasoning":  ("RAG_REASONING_PROVIDER", "RAG_REASONING_MODEL"),
    "deep":       ("RAG_DEEP_PROVIDER",      "RAG_DEEP_MODEL"),
    "multimodal": ("RAG_DEEP_PROVIDER",      "RAG_DEEP_MODEL"),
}


def _env_default() -> Tuple[str, str]:
    """
    Global fallback: DEFAULT_PROVIDER / DEFAULT_MODEL from env.

    If DEFAULT_MODEL is missing too, returns empty strings and logs an
    error — the LLM call downstream will fail loudly, which is the
    intended behaviour (a misconfigured .env should be visible, not
    papered over by a hardcoded model that rots).
    """
    provider = os.getenv("DEFAULT_PROVIDER", "").strip()
    model = os.getenv("DEFAULT_MODEL", "").strip()
    if not model:
        logger.error(
            "[rag:model_selection] No RAG tier model vars and no DEFAULT_MODEL "
            "set — configure RAG_LOOKUP/REASONING/DEEP_MODEL or DEFAULT_MODEL "
            "in .env"
        )
        return ("", "")
    if not provider:
        provider = _infer_provider(model)
    return (provider, model)


def _resolve_tier(tier: str) -> Tuple[str, str]:
    """Resolve (provider, model) for a complexity tier from env vars."""
    provider_var, model_var = RAG_TIER_ENV_VARS.get(tier, ("", ""))
    if not model_var:
        return _env_default()

    model = os.getenv(model_var, "").strip()
    if not model:
        logger.warning(
            "[rag:model_selection] %s not set — falling back to DEFAULT_MODEL",
            model_var,
        )
        return _env_default()

    provider = os.getenv(provider_var, "").strip() or _infer_provider(model)
    return (provider, model)


def select_rag_model(question: str) -> Tuple[str, str, str]:
    """
    Select provider and model for a RAG codebase query.

    Priority:
        1. ORB_RAG_MODEL env var (manual override — bypasses complexity)
        2. Complexity classifier tier → RAG_<TIER>_PROVIDER/MODEL env vars
        3. DEFAULT_PROVIDER/DEFAULT_MODEL env vars

    Args:
        question: The user's codebase question.

    Returns:
        (provider, model, tier) tuple.
        If overridden by env var, tier is "override".
    """
    # Check manual override first
    override_model = os.getenv("ORB_RAG_MODEL", "").strip()
    if override_model:
        # Infer provider from model name
        provider = _infer_provider(override_model)
        logger.info(
            "[rag:model_selection] Override: ORB_RAG_MODEL=%s → %s/%s",
            override_model, provider, override_model,
        )
        return (provider, override_model, "override")

    # Classify complexity
    try:
        from app.memory.complexity import classify_complexity

        result = classify_complexity(query=question)
        tier = result.tier
        provider, model = _resolve_tier(tier)

        logger.info(
            "[rag:model_selection] Complexity: %s (conf=%.2f) → %s/%s",
            tier, result.confidence, provider, model,
        )
        return (provider, model, tier)

    except Exception as e:
        provider, model = _env_default()
        logger.warning(
            "[rag:model_selection] Complexity classification failed: %s — "
            "falling back to %s/%s",
            e, provider, model,
        )
        return (provider, model, "fallback")


def _infer_provider(model_name: str) -> str:
    """
    Infer provider from model name for manual overrides.

    Simple heuristic based on model naming conventions:
        - claude-* → anthropic
        - gemini-* → google
        - everything else → openai
    """
    m = model_name.lower()
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    return "openai"
