# FILE: app/llm/research_context.py
# Purpose: Findings-injection contract — build_research_context() for the conversational layer.
# Called-by: app.llm.research_task (get_research_findings tool); conversational injection (Lane A consumer)
# Depends-on: app.llm.research_models
# Last-renovated: 2026-07-01
"""
Same contract style as app/rag/answerer.build_codebase_context (the A<->C
precedent): the findings pack is NEVER the user-facing reply — this returns a
delimited context block the conversational layer injects, and Astra speaks
the synthesis in her own voice with a freshness line.

Contract:
    async def build_research_context(research_id: str, db) -> str
        - returns "" when the run is missing, not completed, or empty
        - NEVER raises — any internal error is logged and returns ""
        - output <= 6000 characters, plain text, delimited:
          [RESEARCH_FINDINGS] ... [/RESEARCH_FINDINGS]
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RESEARCH_BLOCK_CHARS = 6000


def latest_completed_research_id(db) -> Optional[str]:
    """Most recently completed run's id, or None. Never raises."""
    try:
        from app.llm.research_models import RESEARCH_COMPLETED, ResearchRun

        run = (
            db.query(ResearchRun)
            .filter(ResearchRun.status == RESEARCH_COMPLETED)
            .order_by(ResearchRun.completed_at.desc())
            .first()
        )
        return run.id if run is not None else None
    except Exception as exc:
        logger.warning("[research_context] latest lookup failed: %s", exc)
        return None


async def build_research_context(research_id: str, db) -> str:
    """Delimited findings block for prompt injection. See module contract."""
    try:
        from app.llm.research_models import RESEARCH_COMPLETED, ResearchRun

        run = db.query(ResearchRun).filter(ResearchRun.id == str(research_id)).first()
        if run is None or run.status != RESEARCH_COMPLETED:
            return ""

        try:
            findings = json.loads(run.findings_json or "[]")
        except Exception:
            findings = []
        try:
            sources = json.loads(run.sources_json or "[]")
        except Exception:
            sources = []
        if not findings and not (run.synthesis or "").strip():
            return ""

        completed = run.completed_at.strftime("%Y-%m-%d %H:%M UTC") if run.completed_at else "unknown"
        lines = [
            "[RESEARCH_FINDINGS]",
            f"Deep research: {run.query.strip()}",
            f"Research completed: {completed} (background run — speak the synthesis in your own voice and mention this freshness).",
        ]

        if findings:
            lines.append("Key findings:")
            for i, f in enumerate(findings[:12], 1):
                claim = str(f.get("claim", "")).strip()
                src = str(f.get("source_url", "")).strip()
                lines.append(f"  {i}. {claim}" + (f" (source: {src})" if src else ""))

        synthesis = (run.synthesis or "").strip()
        if synthesis:
            lines.append("Draft synthesis (working notes, re-synthesise naturally):")
            lines.append(synthesis)

        if sources:
            lines.append("Sources:")
            for i, s in enumerate(sources[:10], 1):
                title = str(s.get("title", "")).strip()[:90]
                url = str(s.get("url", "")).strip()
                cred = str(s.get("credibility_label", "")).strip()
                tag = f" [{cred}]" if cred else ""
                lines.append(f"  [{i}]{tag} {title} — {url}")

        footer = "[/RESEARCH_FINDINGS]"
        block = "\n".join(lines)
        if len(block) + len(footer) + 1 > MAX_RESEARCH_BLOCK_CHARS:
            block = block[: MAX_RESEARCH_BLOCK_CHARS - len(footer) - 2] + "…"
        return f"{block}\n{footer}"
    except Exception as exc:
        logger.warning("[research_context] build failed for %s: %s", research_id, exc)
        return ""
