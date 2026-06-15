# FILE: app/lifestyle/nutrition_edit.py
# Purpose: Edit / manual-override a logged food row, re-scaling macros from its per-100g basis.
# Called-by: app.lifestyle.router (PUT /nutrition/{id}), app.bridge.dashboards (Phase 2b)
# Depends-on: app.lifestyle.models, app.lifestyle.service, app.lifestyle.schemas
# Last-renovated: 2026-06-14
"""
Per-item edit / override for the food log (food overhaul Phase 2).

A logged NutritionLog row stores ABSOLUTE macros (so the existing daily-summary
func.sum() stays correct). To make a portion editable, the row also carries a
`quantity_g` and a `per_100g_json` basis. This module re-scales the absolute
macros from that basis when the user changes the quantity, and applies direct
macro overrides (which win, and mark the row user-verified). `scale_per_100g` is
the shared scaler the product/resolver phases reuse.

Kept in its own module because app/lifestyle/service.py is already over the 30KB
file cap — this is the natural home for the edit path and its scaler.
"""
from __future__ import annotations

import json
import logging
from datetime import date
from typing import Optional

from sqlalchemy.orm import Session

from app.lifestyle.models import NutritionLog
from app.lifestyle.schemas import NutritionLogOut, NutritionLogUpdate

logger = logging.getLogger(__name__)

_MACRO_KEYS = ("calories", "protein_g", "carbs_g", "fat_g", "fibre_g", "sugar_g")

# Micros that are product-level LABELS (a fixed classification), never scaled by
# portion — must pass through a re-scale unchanged.
_MICRO_LABELS = ("nutri_score", "nova_group")


def scale_per_100g(per_100g: dict, grams: float) -> dict:
    """Scale a per-100g macro/micro dict to an absolute portion of `grams`.

    factor = grams / 100. Numeric values are scaled and rounded to 1dp;
    non-numeric values (and a nested 'micros' key) are skipped. Returns {} on
    bad input — callers treat an empty result as "no basis to scale".
    """
    try:
        factor = float(grams) / 100.0
    except (TypeError, ValueError):
        return {}
    out: dict = {}
    for k, v in (per_100g or {}).items():
        if k == "micros":
            continue
        if k in _MICRO_LABELS:  # nutri_score / nova_group are labels, not amounts
            out[k] = v
            continue
        try:
            out[k] = round(float(v) * factor, 1)
        except (TypeError, ValueError):
            continue
    return out


def _load_json(s) -> Optional[dict]:
    if not s:
        return None
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def update_nutrition(
    db: Session, log_id: int, data: NutritionLogUpdate,
) -> Optional[NutritionLogOut]:
    """Edit a logged food row in place, then recompute that day's summary.

    Returns the updated row, or None if `log_id` doesn't exist. Behaviour:
      - per_100g / micros set or replace the stored basis;
      - a new quantity_g re-scales the macros from the basis, UNLESS explicit
        macros are also supplied (a manual override always wins);
      - any explicit macro marks the row verified (is_estimate False) unless the
        caller says otherwise.
    """
    # service helpers are imported lazily to avoid any import-order surprises
    from app.lifestyle.service import _update_daily_summary, _nutrition_to_out

    entry = db.query(NutritionLog).filter(NutritionLog.id == log_id).first()
    if not entry:
        return None

    explicit_macros = {
        k: getattr(data, k) for k in _MACRO_KEYS if getattr(data, k) is not None
    }

    # 1. Set/replace the per-100g basis and absolute micros if provided.
    if data.per_100g is not None:
        entry.per_100g_json = json.dumps(data.per_100g)
    if data.micros is not None:
        entry.micros_json = json.dumps(data.micros)

    # 2. Quantity change → re-scale from the basis (only when we have one and the
    #    caller didn't also hand us explicit macros to use instead).
    if data.quantity_g is not None:
        entry.quantity_g = data.quantity_g
        basis = _load_json(entry.per_100g_json)
        if basis and not explicit_macros:
            scaled = scale_per_100g(basis, data.quantity_g)
            for k in _MACRO_KEYS:
                if scaled.get(k) is not None:
                    setattr(entry, k, scaled[k])
            micros_basis = basis.get("micros")
            if isinstance(micros_basis, dict):
                entry.micros_json = json.dumps(scale_per_100g(micros_basis, data.quantity_g))

    # 3. Explicit macro + metadata overrides win.
    for k in _MACRO_KEYS:
        v = getattr(data, k)
        if v is not None:
            setattr(entry, k, v)
    if data.description is not None:
        entry.description = data.description
    if data.meal_type is not None:
        entry.meal_type = data.meal_type
    if data.food_product_id is not None:
        entry.food_product_id = data.food_product_id
    if data.estimate_note is not None:
        entry.estimate_note = data.estimate_note
    if data.source is not None:
        entry.source = data.source

    # A manual macro override marks the row user-verified unless told otherwise.
    if explicit_macros and data.is_estimate is None:
        entry.is_estimate = False
        if data.confidence is None:
            entry.confidence = 0.99
    if data.is_estimate is not None:
        entry.is_estimate = data.is_estimate
    if data.confidence is not None:
        entry.confidence = data.confidence

    entry_date = entry.logged_at.date() if entry.logged_at else date.today()
    db.commit()
    db.refresh(entry)
    _update_daily_summary(db, entry_date)
    logger.info("[lifestyle] Nutrition log %s edited", log_id)
    return _nutrition_to_out(entry)


__all__ = ["update_nutrition", "scale_per_100g"]
