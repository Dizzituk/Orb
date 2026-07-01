# FILE: app/memory/model_env.py
# Purpose: ONE shared env-only resolver for the cheap summariser-tier model.
#          No model ID is ever hardcoded here or in callers (Lane B, principle 1).
# Called-by: app.memory.summary_generator, app.memory.consolidation_graph,
#            app.memory.document_knowledge_promoter, app.memory.enrichment.media_enricher
# Depends-on: stdlib only
# Last-renovated: 2026-07-01 (Lane B Task 4: de-hardcode the summariser model)
"""
Resolve (provider, model) for the cheap/fast summariser tier from .env ONLY.

Precedence per field (first non-empty wins):
  provider: SUMMARIZER_PROVIDER > SUMMARY_PROVIDER > DEFAULT_PROVIDER
  model:    SUMMARIZER_MODEL    > SUMMARY_MODEL    > DEFAULT_MODEL

The live .env (2026-07-01) defines SUMMARY_PROVIDER / SUMMARY_MODEL; the
SUMMARIZER_* aliases are accepted first so a Lane-A rename to the SUMMARIZER_*
convention needs no code change here. There is deliberately NO literal
fallback: when nothing resolves, callers must SKIP their LLM step (and log)
rather than silently run a baked-in model.
"""
from __future__ import annotations

import logging
import os
from typing import Tuple

logger = logging.getLogger(__name__)

# Ensure .env is loaded even when this module is reached outside the app boot
# path (unit tests, scripts) — same pattern as seed_tiers/gemini_vision.
# override=False: real process env always wins over .env.
try:  # pragma: no cover - environment plumbing
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _first_env(*names: str) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def summary_model_from_env() -> Tuple[str, str]:
    """Return (provider, model) for summary-tier LLM calls, from env only.

    Either field may come back "" when no env var defines it — callers must
    treat that as "summariser unconfigured" and skip the call gracefully.
    """
    provider = _first_env("SUMMARIZER_PROVIDER", "SUMMARY_PROVIDER", "DEFAULT_PROVIDER")
    model = _first_env("SUMMARIZER_MODEL", "SUMMARY_MODEL", "DEFAULT_MODEL")
    if not provider or not model:
        logger.warning(
            "[model_env] summariser model unresolved (provider=%r model=%r) — "
            "set SUMMARY_PROVIDER/SUMMARY_MODEL (or SUMMARIZER_*/DEFAULT_*) in .env; "
            "callers will skip their LLM step",
            provider, model,
        )
    return provider, model


__all__ = ["summary_model_from_env"]
