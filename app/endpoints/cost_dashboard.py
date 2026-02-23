# FILE: app/endpoints/cost_dashboard.py
"""
Cost dashboard API endpoint.

Provides a GET endpoint for viewing current spend data.
Used by the frontend dashboard and for manual inspection.

Routes:
    GET /api/cost/summary   — Full spend summary (daily, monthly, by stage, by model)
    GET /api/cost/today     — Today's spend only
    GET /api/cost/budget    — Budget status (daily + monthly)
"""

from __future__ import annotations

import logging
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cost", tags=["cost"])


@router.get("/summary")
async def get_cost_summary() -> dict:
    """
    Full spend summary with breakdown by stage and model.

    Returns daily and monthly spend, budget status, call counts,
    and top cost contributors.
    """
    try:
        from app.overwatcher.cost_budget import get_spend_summary

        summary = get_spend_summary()

        # Sort by_stage and by_model by cost descending, take top 5
        top_stages = dict(
            sorted(summary.by_stage.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        top_models = dict(
            sorted(summary.by_model.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        return {
            "daily": {
                "level": summary.daily.level.value,
                "spent_gbp": summary.daily.spent_gbp,
                "budget_gbp": summary.daily.budget_gbp,
                "remaining_gbp": summary.daily.remaining_gbp,
                "percentage_used": summary.daily.percentage_used,
                "message": summary.daily.message,
            },
            "monthly": {
                "level": summary.monthly.level.value,
                "spent_gbp": summary.monthly.spent_gbp,
                "budget_gbp": summary.monthly.budget_gbp,
                "remaining_gbp": summary.monthly.remaining_gbp,
                "percentage_used": summary.monthly.percentage_used,
                "message": summary.monthly.message,
            },
            "today_calls": summary.today_calls,
            "month_calls": summary.month_calls,
            "top_stages_gbp": top_stages,
            "top_models_gbp": top_models,
        }

    except Exception as e:
        logger.error("[cost_dashboard] Summary failed: %s", e)
        return {"error": str(e)}


@router.get("/today")
async def get_today_spend() -> dict:
    """Quick view of today's spend."""
    try:
        from app.overwatcher.cost_budget import check_daily_budget

        daily = check_daily_budget()
        return {
            "level": daily.level.value,
            "spent_gbp": daily.spent_gbp,
            "budget_gbp": daily.budget_gbp,
            "remaining_gbp": daily.remaining_gbp,
            "percentage_used": daily.percentage_used,
            "message": daily.message,
        }

    except Exception as e:
        logger.error("[cost_dashboard] Today spend failed: %s", e)
        return {"error": str(e)}


@router.get("/budget")
async def get_budget_status() -> dict:
    """Budget status for both daily and monthly."""
    try:
        from app.overwatcher.cost_budget import check_daily_budget, check_monthly_budget

        daily = check_daily_budget()
        monthly = check_monthly_budget()

        return {
            "daily": {
                "level": daily.level.value,
                "exceeded": daily.exceeded,
                "spent_gbp": daily.spent_gbp,
                "budget_gbp": daily.budget_gbp,
                "remaining_gbp": daily.remaining_gbp,
            },
            "monthly": {
                "level": monthly.level.value,
                "exceeded": monthly.exceeded,
                "spent_gbp": monthly.spent_gbp,
                "budget_gbp": monthly.budget_gbp,
                "remaining_gbp": monthly.remaining_gbp,
            },
        }

    except Exception as e:
        logger.error("[cost_dashboard] Budget status failed: %s", e)
        return {"error": str(e)}


@router.get("/caches")
async def get_cache_stats() -> dict:
    """Statistics for spec and architecture caches."""
    result = {}
    try:
        from app.pot_spec.spec_cache import get_cache_stats as _spec_stats
        result["spec_cache"] = _spec_stats()
    except Exception as e:
        result["spec_cache"] = {"error": str(e)}

    try:
        from app.orchestrator.architecture_cache import get_cache_stats as _arch_stats
        result["architecture_cache"] = _arch_stats()
    except Exception as e:
        result["architecture_cache"] = {"error": str(e)}

    return result
