# FILE: app/rag/_answerer_model_selection.py
# Purpose: Complexity-aware model selection for RAG queries.
# Called-by: app.rag.answerer
# Depends-on: app.memory.complexity
# Last-renovated: 2026-06-11
"""
Complexity-aware model selection for RAG queries.

Maps the complexity classifier's tier to a specific provider + model
for codebase question answering.

The ORB_RAG_MODEL env var acts as a manual override — if set, it
bypasses complexity routing entirely (same pattern used elsewhere
in the LLM router for manual overrides).

v10.0 (2026-02-23): Job 10B — complexity-aware model selection.
"""

import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)


# =========================================================================
# Complexity tier → (provider, model) mapping for RAG queries
# =========================================================================

RAG_COMPLEXITY_MODEL_MAP: dict[str, Tuple[str, str]] = {
    "ping_pong": ("openai", "gpt-4.1-mini"),
    "lookup":    ("openai", "gpt-4.1-mini"),
    "reasoning": ("anthropic", "claude-sonnet-4-5-20250929"),
    "deep":      ("anthropic", "claude-opus-4-5"),
    "multimodal": ("anthropic", "claude-opus-4-5"),
}

# Fallback if tier is unknown
_DEFAULT_PROVIDER = "openai"
_DEFAULT_MODEL = "gpt-4.1-mini"


def select_rag_model(question: str) -> Tuple[str, str, str]:
    """
    Select provider and model for a RAG codebase query.

    Priority:
        1. ORB_RAG_MODEL env var (manual override — bypasses complexity)
        2. Complexity classifier tier → model map

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
        provider, model = RAG_COMPLEXITY_MODEL_MAP.get(
            tier, (_DEFAULT_PROVIDER, _DEFAULT_MODEL)
        )

        logger.info(
            "[rag:model_selection] Complexity: %s (conf=%.2f) → %s/%s",
            tier, result.confidence, provider, model,
        )
        return (provider, model, tier)

    except Exception as e:
        logger.warning(
            "[rag:model_selection] Complexity classification failed: %s — "
            "falling back to %s/%s",
            e, _DEFAULT_PROVIDER, _DEFAULT_MODEL,
        )
        return (_DEFAULT_PROVIDER, _DEFAULT_MODEL, "fallback")


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
