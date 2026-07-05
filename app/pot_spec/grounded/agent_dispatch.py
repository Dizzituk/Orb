# FILE: app/pot_spec/grounded/agent_dispatch.py
# Purpose: Derek phase 4 — SpecGate research agents: HANDS-tier distillation of fetched evidence for the APEX main.
# Called-by: app.pot_spec.grounded._simple_create_evidence
# Depends-on: app.llm.stage_roles, app.providers.registry, app.pipeline_v2.ledger
# Last-renovated: 2026-07-04
"""
SpecGate agent dispatch: the main (APEX) model synthesises; agents fetch.

The evidence loop reads files deterministically. Raw dumps (up to 60KB per
round) then flood the APEX main's context. When dispatch is enabled, each
large fetched file is DISTILLED by a SPECGATE_AGENT (HANDS tier) into
exactly what the evidence request asked for — signatures kept verbatim,
prose summarised — before the re-prompt. Every distilled result also lands
in the job's Decision Ledger as file-read evidence (source-of-truth
discipline), cost-attributed as stage="specgate_agent".

Env:
    ASTRA_SPECGATE_AGENT_DISPATCH   default "0" (off — old path preserved)
    ASTRA_SPECGATE_AGENT_MAX_TASKS  default 8 per SpecGate run (bounded)
    ASTRA_SPECGATE_AGENT_MIN_CHARS  default 3000 (small files stay raw)

Failure policy: any agent error keeps the RAW evidence — distillation can
only save tokens, never lose ground truth.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DISTILL_SYSTEM = (
    "You are a code-research agent inside a build pipeline. You receive one "
    "file and one stated need. Distil the file into exactly what matters for "
    "that need: keep class/function signatures, constants, route paths and "
    "interface shapes VERBATIM; summarise implementation prose in one line "
    "each; note line-relevant identifiers. Never invent content that is not "
    "in the file. Output plain text, no preamble."
)


def dispatch_enabled() -> bool:
    return os.getenv("ASTRA_SPECGATE_AGENT_DISPATCH", "0").strip().lower() in ("1", "true", "yes")


def _max_tasks() -> int:
    try:
        return max(0, int(os.getenv("ASTRA_SPECGATE_AGENT_MAX_TASKS", "8")))
    except ValueError:
        return 8


def _min_chars() -> int:
    try:
        return max(0, int(os.getenv("ASTRA_SPECGATE_AGENT_MIN_CHARS", "3000")))
    except ValueError:
        return 3000


class AgentBudget:
    """Bounded task count for one SpecGate run."""

    def __init__(self, max_tasks: Optional[int] = None):
        self.remaining = _max_tasks() if max_tasks is None else max_tasks
        self.dispatched = 0

    def take(self) -> bool:
        if self.remaining <= 0:
            return False
        self.remaining -= 1
        self.dispatched += 1
        return True


async def _distill_one(content: str, label: str, need: str, goal: str) -> Optional[str]:
    """One HANDS-tier distillation call. None on any failure (keep raw)."""
    try:
        from app.llm.stage_roles import resolve_stage_role
        role = resolve_stage_role("SPECGATE_AGENT")
        from app.providers.registry import llm_call

        prompt = (
            f"JOB GOAL: {goal[:400]}\n"
            f"EVIDENCE NEED: {need[:400]}\n\n"
            f"FILE: {label}\n"
            f"--- FILE CONTENT ---\n{content}\n--- END FILE ---\n\n"
            "Distil now (keep signatures verbatim; target under 120 lines)."
        )
        result = await llm_call(
            provider_id=role.provider,
            model_id=role.model,
            messages=[{"role": "user", "content": prompt}],
            system_prompt=_DISTILL_SYSTEM,
            max_tokens=2000,
            timeout_seconds=90,
            stage="specgate_agent",
        )
        if result.is_success() and (result.content or "").strip():
            return result.content.strip()
        logger.warning("[agent_dispatch] distil failed for %s: %s",
                       label, getattr(result, "error_message", "?"))
    except Exception as exc:
        logger.warning("[agent_dispatch] distil error for %s: %s", label, exc)
    return None


def _record_to_ledger(job_id: str, label: str, need: str, distilled: str) -> None:
    """Land the agent's evidence in the job's Decision Ledger (best effort)."""
    if not job_id:
        return
    try:
        from app.pot_spec.grounded._spec_runner_utils_11 import _get_job_dir_for_segmentation
        from app.pipeline_v2.ledger import load_or_create_ledger, ledger_append, save_ledger

        job_dir = _get_job_dir_for_segmentation(job_id)
        os.makedirs(job_dir, exist_ok=True)
        ledger = load_or_create_ledger(job_id, job_dir)
        ledger_append(
            ledger,
            entry_type="file_read",
            stage="spec_gate",
            relevant_to=["spec_gate", "agentic_builder"],
            summary=f"SpecGate agent distilled {label} (need: {need[:120]})",
            path=label,
            category="specgate_agent",
            description=distilled[:4000],
        )
        save_ledger(ledger, job_dir)
        logger.info("[agent_dispatch] Ledger evidence recorded for %s (job %s)", label, job_id)
    except Exception as exc:
        logger.warning("[agent_dispatch] ledger write skipped for %s: %s", label, exc)


async def distill_evidence_results(
    er_results: List[Dict[str, Any]],
    need: str,
    goal: str,
    job_id: Optional[str] = None,
    budget: Optional[AgentBudget] = None,
) -> List[Dict[str, Any]]:
    """Distil large successful read results in place. No-op when disabled.

    Returns the (possibly modified) result list — same shape the evidence
    formatter already consumes, so the old path is preserved bit-for-bit
    when ASTRA_SPECGATE_AGENT_DISPATCH is off.
    """
    if not dispatch_enabled():
        return er_results
    budget = budget or AgentBudget()
    min_chars = _min_chars()

    for r in er_results:
        content = r.get("content")
        if not (r.get("success") and content and len(content) >= min_chars):
            continue
        if not budget.take():
            logger.info("[agent_dispatch] Task budget exhausted (%d dispatched) — raw evidence from here on",
                        budget.dispatched)
            break
        label = r.get("file_path") or r.get("path") or "unknown"
        distilled = await _distill_one(content, str(label), need, goal)
        if distilled:
            r["content"] = (
                f"[agent-distilled by SPECGATE_AGENT — raw was {len(content)} chars]\n"
                + distilled
            )
            _record_to_ledger(job_id or "", str(label), need, distilled)
    return er_results


__all__ = ["AgentBudget", "dispatch_enabled", "distill_evidence_results"]
