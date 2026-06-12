# FILE: app/lifestyle/energy_ledger.py
# Purpose: Weekly energy ledger — the contract is the week, not the day.
#          Rolling balance vs a weekly deficit target, a dynamic daily intake
#          target, and a scale-feedback calibration readout.
# Called-by: app.lifestyle.energy_router, app.lifestyle.energy (debrief reply),
#            app.lifestyle.nudges
# Depends-on: app.lifestyle.energy, app.lifestyle.metabolic, app.lifestyle.models
"""
PT principle (Taz's own): the weeks and months must be bang on; the day-to-day
can flex. So instead of scoring each day pass/fail, this keeps a running weekly
balance — burn minus intake per day, banked against a weekly deficit target —
and derives TODAY's intake target from what's left of the week, smoothed across
the remaining days. A big Saturday surf raises headroom; a heavy Monday graft
banks deficit early so Friday can breathe.

Days with no food logged are "no data", not "zero eaten" — they are excluded
from the banked sum and flagged, never punished or rewarded.

Calibration: over a window, compare the cumulative ledger deficit against the
actual scale movement (7,700 kcal ~= 1 kg of fat). The ratio exposes whether
the engine systematically over- or under-prices the days; the readout is
surfaced rather than silently auto-applied for now — trust first, magic later.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.lifestyle import metabolic
from app.lifestyle.energy import estimate_day_energy, _profile_inputs
from app.lifestyle.models import DailySummary, WeightEntry

logger = logging.getLogger(__name__)

KCAL_PER_KG = 7700.0
# A day's intake target never drops below BMR and never demands more than this
# multiple of the configured pace deficit — one bad Tuesday can't order a fast.
MAX_DAILY_DEFICIT_FACTOR = 1.8


def _week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday


def _daily_deficit(db: Session) -> int:
    pace = _profile_inputs(db).get("pace") or metabolic.DEFAULT_PACE
    return metabolic.PACE_DEFICITS.get(pace, metabolic.PACE_DEFICITS[metabolic.DEFAULT_PACE])


def _intake_for(db: Session, day: date) -> Optional[float]:
    row = db.query(DailySummary).filter(DailySummary.date == day).first()
    if row is None or not row.total_calories or row.total_calories <= 0:
        return None  # nothing logged is no-data, not zero eaten
    return float(row.total_calories)


def get_week_ledger(db: Session, ref: Optional[date] = None) -> Dict[str, Any]:
    """The week so far: per-day rows + the running balance vs target."""
    ref = ref or date.today()
    start = _week_start(ref)
    daily_deficit = _daily_deficit(db)
    weekly_target = daily_deficit * 7

    days: List[Dict[str, Any]] = []
    banked = 0.0
    days_logged = 0
    for i in range((ref - start).days + 1):
        d = start + timedelta(days=i)
        est = estimate_day_energy(db, d)
        intake = _intake_for(db, d)
        balance = None
        if intake is not None and est.get("total_burn"):
            balance = round(est["total_burn"] - intake)
            banked += balance
            days_logged += 1
        days.append({
            "date": d.isoformat(),
            "burn": est.get("total_burn"),
            "burn_method": est.get("method"),
            "is_work_day": est.get("is_work_day"),
            "effort": est.get("effort"),
            "intake": round(intake) if intake is not None else None,
            "balance": balance,
            "logged": intake is not None,
            "provisional": d == date.today(),
        })

    pct = round(100.0 * banked / weekly_target) if weekly_target else 0
    elapsed = (ref - start).days + 1
    pace_due = daily_deficit * days_logged  # fair share over logged days only
    if days_logged == 0:
        message = "No logged days yet this week — the ledger starts when food logs land."
    elif banked >= pace_due:
        message = (f"{pct}% of the weekly deficit banked across {days_logged} logged "
                   f"day{'s' if days_logged != 1 else ''} — ahead of pace, cruise in.")
    elif banked >= pace_due * 0.7:
        message = (f"{pct}% of the weekly deficit banked — close to pace, "
                   f"a steady finish covers it.")
    else:
        message = (f"{pct}% of the weekly deficit banked over {days_logged} logged "
                   f"day{'s' if days_logged != 1 else ''} — behind pace, lean the "
                   f"remaining days slightly rather than slashing one.")
    return {
        "week_start": start.isoformat(),
        "ref_date": ref.isoformat(),
        "days_elapsed": elapsed,
        "days_logged": days_logged,
        "daily_deficit_target": daily_deficit,
        "weekly_deficit_target": weekly_target,
        "banked_deficit": round(banked),
        "banked_pct": pct,
        "on_track": days_logged > 0 and banked >= pace_due * 0.7,
        "message": message,
        "days": days,
    }


def compute_today_target(db: Session) -> Dict[str, Any]:
    """Today's dynamic intake target: today's burn minus today's fair share of
    what's left of the weekly deficit, smoothed over the remaining days,
    clamped so the day stays humane (never below BMR, never a crash demand)."""
    today = date.today()
    est = estimate_day_energy(db, today)
    prof = _profile_inputs(db)
    bmr = prof.get("bmr")
    daily_deficit = _daily_deficit(db)
    weekly_target = daily_deficit * 7

    # Banked balance over the week BEFORE today (today is still moving).
    start = _week_start(today)
    banked_prior = 0.0
    d = start
    while d < today:
        intake = _intake_for(db, d)
        e = estimate_day_energy(db, d)
        if intake is not None and e.get("total_burn"):
            banked_prior += e["total_burn"] - intake
        d += timedelta(days=1)

    remaining_days = 7 - today.weekday()  # today included
    needed_today = (weekly_target - banked_prior) / max(1, remaining_days)
    needed_today = max(0.0, min(needed_today, daily_deficit * MAX_DAILY_DEFICIT_FACTOR))

    target = None
    floored = False
    if est.get("total_burn"):
        target = est["total_burn"] - needed_today
        if bmr and target < bmr:
            target, floored = bmr, True
        target = round(target)

    ahead = banked_prior - daily_deficit * today.weekday()  # vs pace through yesterday
    if today.weekday() == 0:
        week_note = "fresh week"
    elif ahead >= daily_deficit * 0.5:
        week_note = "ahead of the week — a little headroom today"
    elif ahead <= -daily_deficit * 0.5:
        week_note = "behind the week — leaning today slightly"
    else:
        week_note = "on pace for the week"

    return {
        "date": today.isoformat(),
        "target_calories": target,
        "today_burn": est.get("total_burn"),
        "burn_method": est.get("method"),
        "burn_confidence": est.get("confidence"),
        "needed_deficit_today": round(needed_today),
        "banked_prior": round(banked_prior),
        "weekly_deficit_target": weekly_target,
        "remaining_days": remaining_days,
        "floored_at_bmr": floored,
        "week_note": week_note,
        "is_work_day": est.get("is_work_day"),
        "effort": est.get("effort"),
    }


def get_energy_history(db: Session, days: int = 30) -> List[Dict[str, Any]]:
    """Per-day energy picture, newest first — the long-trend feedstock."""
    today = date.today()
    out: List[Dict[str, Any]] = []
    for i in range(max(1, min(days, 120))):
        d = today - timedelta(days=i)
        est = estimate_day_energy(db, d)
        intake = _intake_for(db, d)
        row = db.query(DailySummary).filter(DailySummary.date == d).first()
        out.append({
            "date": d.isoformat(),
            "burn": est.get("total_burn"),
            "method": est.get("method"),
            "is_work_day": est.get("is_work_day"),
            "effort": est.get("effort"),
            "intake": round(intake) if intake is not None else None,
            "balance": (round(est["total_burn"] - intake)
                        if intake is not None and est.get("total_burn") else None),
            # Wearable witnesses for the dashboards (2026-06-11)
            "steps": row.steps if row else None,
            "resting_hr": row.resting_hr if row else None,
            "sleep_minutes": row.sleep_minutes if row else None,
            "active_calories": row.active_calories if row else None,
        })
    return out


def calibration_readout(db: Session, window_days: int = 14) -> Dict[str, Any]:
    """Scale-feedback audit: predicted loss from the ledger vs actual scale
    movement over the window. Ratio < 1 means the engine is over-pricing burn
    (or intake is under-logged); > 1 means it is conservative — by design it
    should start slightly conservative. Surfaced, not yet auto-applied."""
    today = date.today()
    start = today - timedelta(days=window_days)

    total_deficit = 0.0
    days_logged = 0
    d = start
    while d <= today:
        intake = _intake_for(db, d)
        est = estimate_day_energy(db, d)
        if intake is not None and est.get("total_burn"):
            total_deficit += est["total_burn"] - intake
            days_logged += 1
        d += timedelta(days=1)

    def _weight_near(target: date) -> Optional[float]:
        rows = (db.query(WeightEntry)
                .filter(WeightEntry.recorded_at >= target - timedelta(days=3),
                        WeightEntry.recorded_at <= target + timedelta(days=4))
                .order_by(WeightEntry.recorded_at.asc()).all())
        if not rows:
            return None
        return float(sum(r.weight_kg for r in rows) / len(rows))

    w_start, w_end = _weight_near(start), _weight_near(today)
    predicted_kg = round(total_deficit / KCAL_PER_KG, 2)
    actual_kg = round(w_start - w_end, 2) if (w_start and w_end) else None
    ratio = None
    if actual_kg is not None and predicted_kg:
        try:
            ratio = round(actual_kg / predicted_kg, 2)
        except ZeroDivisionError:
            ratio = None

    if actual_kg is None:
        note = "Need weigh-ins near both ends of the window to calibrate."
    elif days_logged < window_days * 0.5:
        note = "Too few logged days in the window for a fair audit yet."
    elif ratio is None:
        note = "Ledger deficit is ~zero over the window — nothing to compare."
    elif ratio >= 1.15:
        note = ("Losing faster than the ledger predicts — the engine is running "
                "conservative (by design); headroom exists to ease intake up.")
    elif ratio <= 0.7:
        note = ("Losing slower than predicted — burn may be over-priced or some "
                "intake unlogged; tighten logging before trusting the numbers.")
    else:
        note = "Ledger and scales agree within tolerance — the model is honest."
    return {
        "window_days": window_days,
        "days_logged": days_logged,
        "ledger_deficit_kcal": round(total_deficit),
        "predicted_loss_kg": predicted_kg,
        "actual_loss_kg": actual_kg,
        "agreement_ratio": ratio,
        "note": note,
    }
