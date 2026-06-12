# FILE: app/lifestyle/energy_router.py
# Purpose: HTTP surface for the energy engine — today's burn + dynamic target,
#          the weekly ledger, history, the calibration readout, and the debrief.
# Called-by: main.py (mounted under /lifestyle/energy)
# Depends-on: app.lifestyle.energy, app.lifestyle.energy_ledger, app.db
"""Endpoints:
  GET  /lifestyle/energy/today        — burn breakdown + dynamic intake target
  GET  /lifestyle/energy/ledger       — the week so far (the contract)
  GET  /lifestyle/energy/history      — per-day rows for trends (?days=30)
  GET  /lifestyle/energy/calibration  — ledger vs scales readout (?days=14)
  POST /lifestyle/energy/debrief      — {"effort": "heavy", "date"?, "notes"?}
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import get_db
from app.lifestyle.energy import (
    EFFORT_LEVELS, estimate_day_energy, normalise_effort, set_day_effort,
)
from app.lifestyle.energy_ledger import (
    calibration_readout, compute_today_target, get_energy_history,
    get_week_ledger,
)

logger = logging.getLogger(__name__)

# The energy engine selects DailySummary/WorkoutSession, which now carry new
# columns (total_calories_burned, avg_hr, ...). The ALTER migration normally
# runs on first ingest — run it at mount too so reads never outpace it.
try:
    from app.lifestyle.health_sync import ensure_health_schema
    ensure_health_schema()
except Exception as _mig_exc:  # pragma: no cover - defensive
    logger.warning("[energy_router] schema ensure skipped: %s", _mig_exc)

router = APIRouter(prefix="/lifestyle/energy", tags=["lifestyle-energy"])


class DebriefIn(BaseModel):
    effort: str = Field(..., description="light | normal | heavy | very_heavy "
                                         "(natural synonyms accepted)")
    date: Optional[str] = Field(None, description="YYYY-MM-DD, default today")
    notes: Optional[str] = None


@router.get("/today")
def energy_today(db: Session = Depends(get_db)) -> Dict[str, Any]:
    return {
        "energy": estimate_day_energy(db),
        "target": compute_today_target(db),
    }


@router.get("/ledger")
def energy_ledger(db: Session = Depends(get_db)) -> Dict[str, Any]:
    return get_week_ledger(db)


@router.get("/history")
def energy_history(days: int = Query(30, ge=1, le=120),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    return {"days": get_energy_history(db, days)}


@router.get("/calibration")
def energy_calibration(days: int = Query(14, ge=7, le=60),
                       db: Session = Depends(get_db)) -> Dict[str, Any]:
    return calibration_readout(db, days)


@router.post("/debrief")
def energy_debrief(body: DebriefIn, db: Session = Depends(get_db)) -> Dict[str, Any]:
    level = normalise_effort(body.effort)
    if level is None:
        raise HTTPException(422, f"effort must be one of {list(EFFORT_LEVELS)} "
                                 f"or a recognised synonym")
    try:
        day = date.fromisoformat(body.date) if body.date else date.today()
    except ValueError:
        raise HTTPException(422, "date must be YYYY-MM-DD")
    rec = set_day_effort(day, level, source="api", notes=body.notes)
    return {
        "ok": True,
        "recorded": rec,
        "date": day.isoformat(),
        "energy": estimate_day_energy(db, day),
        "target": compute_today_target(db) if day == date.today() else None,
    }
