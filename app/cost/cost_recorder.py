# FILE: app/cost/cost_recorder.py
# Purpose: Cost recording hook for the provider registry and instrumented call paths.
# Called-by: app.debug.debug_chat, app.providers._registry_utils_3, app.llm.streaming, app.pipeline_v2.llm_tools, app.llm.weaver_stream
# Depends-on: app.cost.cost_budget, app.cost.cost_guard, app.cost.cost_ledger, app.cost.cost_pricing
# Last-renovated: 2026-07-04 (Derek phase 1: null cost for unpriced models, stage map extended)
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
    ledger, and updates the in-memory CostGuard. An unpriced model records
    cost_usd=null in the ledger (cost_pricing WARNs) — never a silent 0.00.

    Returns the calculated cost in USD (0.0 when unpriced — the ledger row
    keeps the honest null; the float return is for legacy callers only).
    Never raises — cost tracking must not break the pipeline.
    """
    try:
        from app.cost.cost_pricing import calculate_cost
        from app.cost.cost_ledger import append_record
        from app.cost.cost_guard import record_usage, ModelRole

        cost_usd = calculate_cost(model, prompt_tokens, completion_tokens)  # None = unpriced

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
                cost_estimate=cost_usd or 0.0,
                stage=stage,
            )
        except Exception as e:
            logger.debug("[cost_recorder] CostGuard update skipped: %s", e)

        if cost_usd is not None and cost_usd > 0.01:
            logger.info(
                "[cost_recorder] %s/%s stage=%s tokens=%d+%d cost=$%.4f",
                provider, model, stage or "?",
                prompt_tokens, completion_tokens, cost_usd,
            )

        return cost_usd if cost_usd is not None else 0.0

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
        # Derek tier map (phase 1, 2026-07-04)
        "specgate_agent": ModelRole.SPEC_GATE,
        "specgate_vision": ModelRole.SPEC_GATE,
        "builder_main": ModelRole.IMPLEMENTER,
        "builder_worker": ModelRole.IMPLEMENTER,
        "builder_integrator": ModelRole.IMPLEMENTER,
        "builder_escalation": ModelRole.IMPLEMENTER,
        "checkout_eyes": ModelRole.OVERWATCHER,
        "checkout_judge": ModelRole.OVERWATCHER,
        "verifier": ModelRole.OVERWATCHER,
        "bvl_diagnostic": ModelRole.OVERWATCHER,
        "bvl_test_generator": ModelRole.IMPLEMENTER,
        "spec_review": ModelRole.OVERWATCHER,
        "SPEC_REVIEW": ModelRole.OVERWATCHER,
        "chat": ModelRole.IMPLEMENTER,
    }
    return stage_role_map.get(stage or "", ModelRole.IMPLEMENTER)


__all__ = [
    "record_llm_cost",
    "pre_call_budget_check",
]
