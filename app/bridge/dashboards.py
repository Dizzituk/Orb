# FILE: app/bridge/dashboards.py
"""
Bridge dashboard endpoints -- compact Work / Investments / Health summaries
for the Android companion app's drawer tabs.

Single responsibility: expose data the desktop already computes (work ledger,
investment portfolio, lifestyle/health) over the bridge's own auth
(require_bridge_auth), in mobile-shaped JSON. Each handler just calls the
existing service and trims for a phone screen -- no business logic here.

Kept out of bridge/router.py (already ~27 KB, near the 30 KB ceiling) as a
separate sub-router, mirroring tts_proxy.py and missed_replies.py.

v1.0 (2026-05-30): initial -- GET /bridge/work, /bridge/investments,
/bridge/health.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.lifestyle.schemas import NutritionLogIn, NutritionLogUpdate
from .schemas import (
    require_bridge_auth,
    HealthIngestRequest,
    HealthIngestResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/bridge", tags=["Bridge"])

# How many recent days the phone's Work tab lists.
_RECENT_DAYS = 14


@router.get("/work")
async def bridge_work(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
) -> dict:
    """Work-ledger summary: current tax-year stats + recent days."""
    from app.finance.work_router import list_tax_years, build_dashboard, days_for
    years = list_tax_years(db)
    ty = years.get("current")
    dash = build_dashboard(db, ty)
    recent = days_for(db, ty)[:_RECENT_DAYS]
    return {
        "tax_year": ty,
        "stats": dash.get("stats", {}),
        "recent_days": recent,
    }


@router.get("/investments")
async def bridge_investments(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Portfolio summary (Trading 212 + crypto) for the Investments tab.
    Returns the same PortfolioSummary the desktop uses; FastAPI serialises it."""
    from app.investments.service import fetch_portfolio_summary
    return await fetch_portfolio_summary(db)


@router.get("/health/summary")
async def bridge_health(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Lifestyle / fitness dashboard summary for the Health & Fitness tab.

    NOTE: served at /bridge/health/summary, not /bridge/health — the latter is
    the bridge liveness probe (returns {status: ok}) defined in router.py and
    would otherwise shadow this route, so the phone received the ping payload
    and rendered empty defaults.
    """
    from app.lifestyle import service as lifestyle_service
    return lifestyle_service.get_dashboard_summary(db)


# ── Nutrition: phone logs food, sees today's items tally, can delete ──────────
# Manual quick-log path (permissive on macros), separate from the chat tool's
# hard-rule path. Phone is just an interface; all data lives in the PC backend.

@router.get("/health/nutrition")
async def bridge_health_nutrition_today(
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Today's nutrition: per-item logs + day totals (for the tally list)."""
    from app.lifestyle import service as lifestyle_service
    return lifestyle_service.get_daily_nutrition(db)


@router.get("/health/nutrition/day")
async def bridge_health_nutrition_day(
    date: str,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Nutrition for a specific day (diary back/forward scroll). date=YYYY-MM-DD."""
    from datetime import date as _d
    from app.lifestyle import service as lifestyle_service
    try:
        d = _d.fromisoformat(date)
    except ValueError:
        d = None
    return lifestyle_service.get_daily_nutrition(db, d)


@router.get("/health/workouts/day")
async def bridge_health_workouts_day(
    date: str,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Strength sets for a specific day, grouped by exercise. date=YYYY-MM-DD.

    Pulls the day-grouped strength log and returns just the requested day so
    the phone's Fitness diary can show that day's exercises/sets.
    """
    from app.lifestyle import service as lifestyle_service
    log = lifestyle_service.get_strength_log(db, days=120)
    day = next((d for d in log.days if d.date == date), None)
    return {"date": date, "exercises": (day.exercises if day else [])}


@router.post("/health/nutrition")
async def bridge_health_nutrition_log(
    entry: NutritionLogIn,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Log a food item from the phone. Returns the created entry."""
    from app.lifestyle import service as lifestyle_service
    return lifestyle_service.log_nutrition(
        db,
        description=entry.description,
        meal_type=entry.meal_type or "meal",
        calories=entry.calories,
        protein_g=entry.protein_g,
        carbs_g=entry.carbs_g,
        fat_g=entry.fat_g,
        sugar_g=entry.sugar_g,
    )


@router.delete("/health/nutrition/{log_id}")
async def bridge_health_nutrition_delete(
    log_id: int,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Delete a food item by id (recomputes the day's totals)."""
    from app.lifestyle import service as lifestyle_service
    ok = lifestyle_service.delete_nutrition(db, log_id)
    return {"ok": ok, "deleted_id": log_id if ok else None}


@router.put("/health/nutrition/{log_id}")
async def bridge_health_nutrition_update(
    log_id: int,
    body: NutritionLogUpdate,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
):
    """Edit a food item from the phone — change the portion (re-scales macros
    from the per-100g basis where known) or override macros directly."""
    from app.lifestyle.nutrition_edit import update_nutrition
    out = update_nutrition(db, log_id, body)
    if out is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Nutrition log {log_id} not found")
    return out


@router.post("/health/ingest", response_model=HealthIngestResponse)
async def bridge_health_ingest(
    req: HealthIngestRequest,
    db: Session = Depends(get_db),
    _auth: bool = Depends(require_bridge_auth),
) -> HealthIngestResponse:
    """Ingest a batch of Health Connect data relayed by AstraBridge.

    The phone reads Health Connect on-device (Garmin writes there natively),
    aggregates daily metrics, and posts discrete weigh-ins + workouts plus the
    daily roll-ups here. Weigh-ins feed the weight log, workouts become activity
    sessions, daily metrics land on the day summary. Dedup is by each record's
    Health Connect UID, so re-syncing overlapping windows never doubles up.
    """
    from app.lifestyle import health_sync
    payload = req.model_dump()
    result = health_sync.ingest_health_batch(
        db,
        weights=payload.get("weights", []),
        workouts=payload.get("workouts", []),
        days=payload.get("days", []),
    )
    return HealthIngestResponse(**result)
