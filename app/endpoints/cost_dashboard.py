# FILE: app/endpoints/cost_dashboard.py
# Purpose: Cost dashboard API endpoint.
# Called-by: main
# Depends-on: app.cost.cost_budget, app.cost.cost_ledger, app.orchestrator.architecture_cache, app.pot_spec.spec_cache
# Last-renovated: 2026-07-04 (Derek phase 1: /report rollup + /pricing, null-safe tally)
"""
Cost dashboard API endpoint.

Provides a GET endpoint for viewing current spend data.
Used by the frontend dashboard and for manual inspection.

Routes:
    GET /api/cost/summary   — Full spend summary (daily, monthly, by stage, by model)
    GET /api/cost/today     — Today's spend only
    GET /api/cost/budget    — Budget status (daily + monthly)
    GET /api/cost/report    — Rollup by stage|model|job|day|provider (Derek phase 1)
    GET /api/cost/pricing   — Active pricing table + its source file
    POST /api/cost/probe    — Localhost-only: fire one cheap labelled call per stage (Gate 1)
    POST /api/cost/probe/segmented — Localhost-only: two-file toy through the segmented builder (Gate 5)
"""

from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException, Request

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
        from app.cost.cost_budget import get_spend_summary

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
        from app.cost.cost_budget import check_daily_budget

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
        from app.cost.cost_budget import check_daily_budget, check_monthly_budget

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


@router.get("/tally")
async def get_cost_tally() -> dict:
    """Compact cost tally for the header UI — daily, weekly, monthly in GBP."""
    try:
        from app.cost.cost_ledger import read_records_since
        from datetime import datetime, timezone, timedelta
        import os

        usd_to_gbp = float(os.getenv("ORB_USD_TO_GBP_RATE", "0.79"))
        now = datetime.now(timezone.utc)

        # Daily: today
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        daily_records = read_records_since(today_start)
        daily_usd = sum(r.get("cost_usd") or 0 for r in daily_records)

        # Weekly: last 7 days
        week_start = now - timedelta(days=7)
        weekly_records = read_records_since(week_start)
        weekly_usd = sum(r.get("cost_usd") or 0 for r in weekly_records)

        # Monthly: this calendar month
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_records = read_records_since(month_start)
        monthly_usd = sum(r.get("cost_usd") or 0 for r in monthly_records)

        return {
            "daily_gbp": round(daily_usd * usd_to_gbp, 2),
            "weekly_gbp": round(weekly_usd * usd_to_gbp, 2),
            "monthly_gbp": round(monthly_usd * usd_to_gbp, 2),
            "daily_calls": len(daily_records),
            "weekly_calls": len(weekly_records),
            "monthly_calls": len(monthly_records),
        }

    except Exception as e:
        logger.error("[cost_dashboard] Tally failed: %s", e)
        return {"daily_gbp": 0, "weekly_gbp": 0, "monthly_gbp": 0, "error": str(e)}


@router.get("/report")
async def get_cost_report(
    by: str = "stage",
    days: int = 7,
    job_id: str = "",
) -> dict:
    """Rollup of ledger spend by stage, model, job, day or provider (Derek phase 1)."""
    try:
        from app.cost.cost_report import rollup

        return rollup(
            by=by,
            days=days if days > 0 else None,
            job_id=job_id or None,
        )
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error("[cost_dashboard] Report failed: %s", e)
        return {"error": str(e)}


@router.post("/probe")
async def run_cost_probe(request: Request) -> dict:
    """Derek phase 1 gate probe — one cheap real call per stage label.

    Localhost-only (it spends real money — well under a cent total) and runs
    in-process, where the encrypted key store has been synced into the env.
    """
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "Probe is localhost-only")
    try:
        from app.cost.cost_probe import run_probe
        return await run_probe()
    except Exception as e:
        logger.error("[cost_dashboard] Probe failed: %s", e)
        return {"green": False, "error": str(e)}


@router.post("/probe/segmented")
async def run_segmented_probe(request: Request) -> dict:
    """Derek phase 5 gate probe — a real two-file CLI toy built end-to-end
    by the segmented builder on the configured stand-ins. Localhost-only
    (spends real money — a few HANDS-tier worker calls, pennies)."""
    client_host = request.client.host if request.client else ""
    if client_host not in ("127.0.0.1", "::1", "localhost"):
        raise HTTPException(403, "Probe is localhost-only")
    try:
        from app.pipeline_v2.segmented.probe import run_toy_probe
        return await run_toy_probe()
    except Exception as e:
        logger.error("[cost_dashboard] Segmented probe failed: %s", e)
        return {"green": False, "error": str(e)}


@router.get("/pricing")
async def get_pricing_table() -> dict:
    """The active pricing table and where it was loaded from."""
    try:
        from app.cost.cost_pricing import get_all_pricing, get_pricing_source

        table = {
            model: {
                "input_per_million": p.input_per_million,
                "output_per_million": p.output_per_million,
                "note": p.note,
            }
            for model, p in sorted(get_all_pricing().items())
        }
        return {"source": get_pricing_source(), "count": len(table), "models": table}
    except Exception as e:
        logger.error("[cost_dashboard] Pricing dump failed: %s", e)
        return {"error": str(e)}