# FILE: app/overwatcher/cost_recorder.py
"""
Cost recording hook for the provider registry.

Called after every successful llm_call() to record costs.
Called before every llm_call() to check budget.

This module is the bridge between the provider layer and the
cost tracking system. It deliberately has no business logic —
it just calls cost_pricing, cost_ledger, and cost_budget.

Usage (in _registry_utils_3.py):
    from app.cost.cost_recorder import record_llm_cost, pre_call_budget_check

    # Before call:
    allowed, reason = pre_call_budget_check(stage="implementer")

    # After call:
    record_llm_cost(provider, model, prompt_tokens, completion_tokens, stage)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def record_llm_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    stage: Optional[str] = None,
    job_id: Optional[str] = None,
) -> float:
    """
    Record the cost of an LLM call.

    Calculates cost from the pricing table, writes to the persistent
    ledger, and updates the in-memory CostGuard.

    Returns the calculated cost in USD.
    Never raises — cost tracking must not break the pipeline.
    """
    try:
        from app.cost.cost_pricing import calculate_cost
        from app.cost.cost_ledger import append_record
        from app.cost.cost_guard import record_usage, ModelRole

        cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)

        # Write to persistent ledger
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": cost_usd,
            "stage": stage or "unknown",
            "job_id": job_id or "",
        }
        append_record(record)

        # Update in-memory CostGuard (best effort)
        try:
            role = _stage_to_role(stage)
            record_usage(
                job_id=job_id or "untracked",
                role=role,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_estimate=cost_usd,
                stage=stage,
            )
        except Exception as e:
            logger.debug("[cost_recorder] CostGuard update skipped: %s", e)

        if cost_usd > 0.01:
            logger.info(
                "[cost_recorder] %s/%s stage=%s tokens=%d+%d cost=$%.4f",
                provider, model, stage or "?",
                prompt_tokens, completion_tokens, cost_usd,
            )

        return cost_usd

    except Exception as e:
        logger.error("[cost_recorder] Failed to record cost: %s", e)
        return 0.0


def pre_call_budget_check(
    stage: Optional[str] = None,
    break_glass: bool = False,
) -> Tuple[bool, str]:
    """
    Check if an LLM call is allowed given current budget.

    Returns: (allowed, reason)
    Never raises — budget failures should not block the pipeline.
    """
    try:
        from app.cost.cost_budget import is_call_allowed
        return is_call_allowed(stage or "unknown", break_glass)
    except Exception as e:
        logger.debug("[cost_recorder] Budget check skipped: %s", e)
        return True, "budget check unavailable"


def _stage_to_role(stage: Optional[str]) -> "ModelRole":
    """Map a pipeline stage name to a CostGuard ModelRole."""
    from app.cost.cost_guard import ModelRole

    stage_role_map = {
        "spec_gate": ModelRole.SPEC_GATE,
        "overwatcher": ModelRole.OVERWATCHER,
        "implementer": ModelRole.IMPLEMENTER,
        "architecture": ModelRole.ARCHITECT,
        "critique": ModelRole.CRITIQUE,
        "revision": ModelRole.ARCHITECT,
        "mediator": ModelRole.MEDIATOR,
        "phase_checkout": ModelRole.OVERWATCHER,
        "final_checkout": ModelRole.OVERWATCHER,
        "planner": ModelRole.ARCHITECT,
        "weaver": ModelRole.SPEC_GATE,
        "coherence_guardian": ModelRole.CRITIQUE,
        "segment_enrichment": ModelRole.IMPLEMENTER,
    }
    return stage_role_map.get(stage or "", ModelRole.IMPLEMENTER)


__all__ = [
    "record_llm_cost",
    "pre_call_budget_check",
]
