# FILE: app/scene_director/research/__init__.py
# Purpose: The agentic research rail — gate (needs facts?) + researcher (reuse web search) →
#          a fact pack the director composes from and records as provenance.
# Called-by: app.scene_director.router (compose path) via maybe_research
# Depends-on: app.scene_director.research.gate/researcher
# Last-renovated: 2026-06-13
"""maybe_research(intent, era) -> fact_pack | None.

intent → gate → (research → fact pack) → injected into the director prompt + recorded as
SceneDoc.provenance. Conservative: imaginative scenes skip research (no latency). Behind
SCENE_RESEARCH_ENABLED (default on). Never raises — a failure means compose without research.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from app.scene_director.research.gate import should_research
from app.scene_director.research.researcher import research

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return os.getenv("SCENE_RESEARCH_ENABLED", "true").strip().lower() not in ("false", "0", "no")


async def maybe_research(intent: str, era: Optional[str] = None) -> Optional[dict]:
    """Gate the intent; if it needs facts, run the bounded research rail. Returns a fact
    pack dict or None (compose imaginatively). Never raises."""
    if not _enabled():
        return None
    try:
        needs, reason = should_research(intent, era)
    except Exception as exc:  # pragma: no cover — gate is pure heuristic
        logger.debug("[scene.research] gate error: %s", exc)
        return None
    if not needs:
        logger.debug("[scene.research] gate: skip (%s)", reason)
        return None
    logger.info("[scene.research] gate: research (%s)", reason)
    return await research(intent, era)


__all__ = ["maybe_research", "should_research", "research"]
