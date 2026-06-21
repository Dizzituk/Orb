# FILE: app/lifestyle/service_nutrition.py
# Purpose: Food logging, itemisation, prep-splitting, and daily/history nutrition reads.
# Called-by: app.lifestyle.service (re-export facade)
# Depends-on: app.lifestyle.models, app.lifestyle.schemas, app.lifestyle.service_activity
# Last-renovated: 2026-06-21
"""
Nutrition layer for the Lifestyle Engine: logging meals (single + itemised),
validated itemisation, batch-cook prep-splitting across days, and the per-day /
multi-day nutrition reads. Split out of service.py; the facade re-exports every
public name here for back-compat. The daily-summary upsert lives in
service_activity (shared with weight/workouts) and is imported from there.
"""
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List

from sqlalchemy.orm import Session

from app.lifestyle.models import NutritionLog
from app.lifestyle.schemas import NutritionLogOut, DailyNutrition

from app.lifestyle.service_activity import _update_daily_summary

logger = logging.getLogger(__name__)


def log_nutrition(db: Session, description: str, meal_type: str = "meal",





                  calories: Optional[float] = None, protein_g: Optional[float] = None,





                  carbs_g: Optional[float] = None, fat_g: Optional[float] = None,





                  sugar_g: Optional[float] = None) -> NutritionLogOut:





    """





    Log a meal/food entry.











    If macros aren't provided, they'll be None until the AI estimation





    service fills them in (future: nutrition.py estimate_macros).





    """





    entry = NutritionLog(





        description=description,





        meal_type=meal_type,





        calories=calories,





        protein_g=protein_g,





        carbs_g=carbs_g,





        fat_g=fat_g,





        sugar_g=sugar_g,





        is_estimate=calories is None,  # if user didn't provide, it's an estimate





        confidence=0.9 if calories is not None else 0.0,





        source="text",





    )





    db.add(entry)





    db.commit()





    db.refresh(entry)





    logger.info(f"[lifestyle] Nutrition logged: {description} ({meal_type})")











    _update_daily_summary(db, date.today())





    return _nutrition_to_out(entry)


def log_nutrition_items(db: Session, items: List[dict], meal_type: str = "meal",





                        on_date: Optional[date] = None) -> List[NutritionLogOut]:





    """Log several distinct foods as SEPARATE rows in one transaction, then





    update the day summary once. Each item dict carries its own description +





    calories/protein_g/carbs_g/fat_g(+sugar_g). Itemising like this lets the





    user see every food with its own macros and check each estimate, rather





    than one merged blob. `on_date` (default today) stamps the rows — used by





    prep-splitting to write meals into future days. Returns the created entries.





    """





    target = on_date or date.today()





    logged_at = (




        datetime.combine(target, datetime.min.time()).replace(tzinfo=timezone.utc)




        + timedelta(hours=12)




    )  # noon, so the row sits cleanly inside the day window





    created = []





    for it in items:





        cal = it.get("calories")





        entry = NutritionLog(





            description=str(it.get("description") or "").strip(),





            meal_type=meal_type,





            calories=cal,





            protein_g=it.get("protein_g"),





            carbs_g=it.get("carbs_g"),





            fat_g=it.get("fat_g"),





            sugar_g=it.get("sugar_g"),





            is_estimate=bool(it.get("is_estimate", True)),





            confidence=0.9 if cal is not None else 0.0,





            source=it.get("source") or "text",





            logged_at=logged_at,





        )





        db.add(entry)





        created.append(entry)





    db.commit()





    for e in created:





        db.refresh(e)





    _update_daily_summary(db, target)





    logger.info(f"[lifestyle] Logged {len(created)} itemised nutrition rows for {target.isoformat()}")





    return [_nutrition_to_out(e) for e in created]


def _coerce_num(v):





    if v is None:





        return None





    try:





        return float(v)





    except (TypeError, ValueError):





        return None


def log_nutrition_items_checked(db: Session, raw_items: list, meal_type: str = "meal") -> dict:





    """Validate then log a list of food items as separate rows.











    Hard rule per item: each needs a description + calories/protein/carbs/fat.





    Returns {'ok': True, 'rows': [NutritionLogOut, ...]} on success, or





    {'ok': False, 'incomplete': [{'item','missing'}, ...]} if any item is





    missing required fields (nothing is logged in that case).





    """





    req = ("calories", "protein_g", "carbs_g", "fat_g")





    norm, incomplete = [], []





    for idx, it in enumerate(raw_items):





        if not isinstance(it, dict):





            continue





        desc = str(it.get("description") or "").strip()





        vals = {k: _coerce_num(it.get(k)) for k in





                ("calories", "protein_g", "carbs_g", "fat_g", "sugar_g")}





        miss = [k for k in req if vals[k] is None]





        if not desc:





            miss = ["description"] + miss





        if miss:





            incomplete.append({"item": desc or f"item {idx + 1}", "missing": miss})





        else:





            norm.append({"description": desc, **vals})





    if incomplete:





        return {"ok": False, "incomplete": incomplete}





    return {"ok": True, "rows": log_nutrition_items(db, norm, meal_type)}


def prep_split(db: Session, description: str, total: dict, days: int,





               start_date: Optional[date] = None, meal_type: str = "meal") -> dict:





    """Divide a batch cook's TOTAL macros evenly across `days` and write one





    portion row into each day, starting at `start_date` (default tomorrow).











    `total` is the whole-batch {calories, protein_g, carbs_g, fat_g[, sugar_g]}.





    Each day gets total/days, so the per-day portion is what the user actually





    eats. Rows are tagged source='prep' and the description notes the portion





    (e.g. 'Chicken curry batch (1/3)'). Future days then show the prepped meal





    in the diary. Returns {'ok', 'days': [{date, item}], 'per_day': {...}}.





    """





    n = max(1, min(14, int(days)))





    req = ("calories", "protein_g", "carbs_g", "fat_g")





    vals = {k: _coerce_num(total.get(k)) for k in





            ("calories", "protein_g", "carbs_g", "fat_g", "sugar_g")}





    missing = [k for k in req if vals[k] is None]





    if not str(description or "").strip():





        missing = ["description"] + missing





    if missing:





        return {"ok": False, "missing": missing}











    per_day = {k: (round(v / n, 1) if v is not None else None) for k, v in vals.items()}





    base = (start_date or (date.today() + timedelta(days=1)))











    written = []





    for i in range(n):





        d = base + timedelta(days=i)





        item = {





            "description": f"{description.strip()} (1/{n} prepped portion)",





            "source": "prep",





            **per_day,





        }





        rows = log_nutrition_items(db, [item], meal_type, on_date=d)





        written.append({"date": d.isoformat(), "item": _to_dict_safe(rows[0]) if rows else None})











    logger.info(f"[lifestyle] Prep-split '{description}' over {n} days from {base.isoformat()}")





    return {"ok": True, "days": written, "per_day": per_day, "n": n,





            "start_date": base.isoformat()}


def _to_dict_safe(obj):





    """Pydantic v1/v2 -> dict (local helper to avoid importing the tools layer)."""





    if hasattr(obj, "model_dump"):





        return obj.model_dump()





    if hasattr(obj, "dict"):





        return obj.dict()





    return dict(obj)


def get_daily_nutrition(db: Session, target_date: Optional[date] = None) -> DailyNutrition:





    """Get all nutrition logs for a given day with totals."""





    target = target_date or date.today()





    start = datetime.combine(target, datetime.min.time()).replace(tzinfo=timezone.utc)





    end = start + timedelta(days=1)











    logs = (





        db.query(NutritionLog)





        .filter(NutritionLog.logged_at >= start, NutritionLog.logged_at < end)





        .order_by(NutritionLog.logged_at.asc())





        .all()





    )











    items = [_nutrition_to_out(l) for l in logs]











    return DailyNutrition(





        date=target.isoformat(),





        total_calories=sum(l.calories or 0 for l in logs),





        total_protein_g=sum(l.protein_g or 0 for l in logs),





        total_carbs_g=sum(l.carbs_g or 0 for l in logs),





        total_fat_g=sum(l.fat_g or 0 for l in logs),





        total_sugar_g=sum(l.sugar_g or 0 for l in logs),





        meal_count=len(logs),





        logs=items,





    )


def get_nutrition_history(db: Session, days: int = 30) -> List[DailyNutrition]:





    """Get daily nutrition summaries for the last N days."""





    result = []





    today = date.today()





    for i in range(days):





        d = today - timedelta(days=i)





        result.append(get_daily_nutrition(db, d))





    result.reverse()





    return result


def delete_nutrition(db: Session, log_id: int) -> bool:





    """Delete a nutrition log by id and recompute that day's summary."""





    entry = db.query(NutritionLog).filter(NutritionLog.id == log_id).first()





    if not entry:





        return False





    entry_date = entry.logged_at.date() if entry.logged_at else date.today()





    db.delete(entry)





    db.commit()





    _update_daily_summary(db, entry_date)





    logger.info(f"[lifestyle] Nutrition log {log_id} deleted")





    return True


def _safe_json(s):





    """Parse a JSON text column to a dict, or None. Never raises."""





    if not s:





        return None





    try:





        import json





        return json.loads(s)





    except Exception:





        return None


def _nutrition_to_out(entry: NutritionLog) -> NutritionLogOut:





    return NutritionLogOut(





        id=entry.id,





        logged_at=entry.logged_at.isoformat(),





        meal_type=entry.meal_type or "meal",





        description=entry.description,





        calories=entry.calories,





        protein_g=entry.protein_g,





        carbs_g=entry.carbs_g,





        fat_g=entry.fat_g,





        fibre_g=entry.fibre_g,





        sugar_g=entry.sugar_g,





        is_estimate=entry.is_estimate,





        confidence=entry.confidence or 0.6,





        source=entry.source or "text",





        quantity_g=entry.quantity_g,





        per_100g=_safe_json(entry.per_100g_json),





        micros=_safe_json(entry.micros_json),





        food_product_id=entry.food_product_id,





        estimate_note=entry.estimate_note,





    )
