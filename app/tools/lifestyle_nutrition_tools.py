# FILE: app/tools/lifestyle_nutrition_tools.py
# Purpose: Nutrition tool handlers that don't fit in lifestyle_tools.py (kept under the
# Called-by: app.debug.gemini_lifestyle_tools, app.tools.lifestyle_tools
# Depends-on: app.lifestyle.nutrition_copy, app.lifestyle.service, app.tools.lifestyle_tools
# Last-renovated: 2026-06-11
"""
Nutrition tool handlers that don't fit in lifestyle_tools.py (kept under the
size ceiling). Currently: meal-prep splitting — divide a batch cook's macros
across N future days and write a portion into each.

Re-exported from lifestyle_tools so the Gemini executor adapters can import it
from the usual place. Session/dict helpers come from lifestyle_tools (defined
above its re-export line, so no import cycle).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

from app.tools.lifestyle_tools import _db_session, _to_dict

logger = logging.getLogger(__name__)


def _parse_iso_date(v) -> Optional[date]:
    if not v:
        return None
    try:
        return datetime.strptime(str(v).strip()[:10], "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


async def prep_split_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Split a batch cook's total macros evenly across N days, writing a portion
    into each day's diary. Defaults to starting tomorrow.

    Args: description, days, total {calories,protein_g,carbs_g,fat_g[,sugar_g]}
    (or the macros passed flat), optional start_date (YYYY-MM-DD), meal_type.
    """
    description = str(input_data.get("description") or "").strip()
    try:
        days = int(input_data.get("days"))
    except (TypeError, ValueError):
        return {"ok": False, "error": "days must be an integer (how many days to split over)"}

    # Accept either a nested `total` object or flat macro keys.
    total = input_data.get("total")
    if not isinstance(total, dict):
        total = {k: input_data.get(k) for k in
                 ("calories", "protein_g", "carbs_g", "fat_g", "sugar_g")}

    start = _parse_iso_date(input_data.get("start_date"))
    meal_type = str(input_data.get("meal_type") or "meal").strip() or "meal"

    try:
        from app.lifestyle.service import prep_split
        with _db_session() as db:
            r = prep_split(db, description, total, days, start_date=start, meal_type=meal_type)
    except Exception as exc:
        logger.exception("[lifestyle_tools] prep_split failed")
        return {"ok": False, "error": str(exc)}

    if not r.get("ok"):
        miss = r.get("missing") or []
        return {
            "ok": False, "needs": miss,
            "error": (
                "NOT split — need the whole batch's " + ", ".join(miss) + ". Give the "
                "TOTAL macros for everything you cooked (calories, protein, carbs, fat) "
                "plus a description and how many days to split over. Ask the user or "
                "web_search to estimate the batch total, then call prep_split once."
            ),
        }
    return {
        "ok": True,
        "days_written": r.get("n"),
        "start_date": r.get("start_date"),
        "per_day": r.get("per_day"),
        "dates": [d.get("date") for d in r.get("days", [])],
    }


async def copy_nutrition_day_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Copy one day's food diary onto another day (2026-06-11).

    Args: source_date, target_date — each YYYY-MM-DD or
    yesterday/today/tomorrow/weekday name. Defaults: yesterday -> today.
    Idempotent: items already on the target day are skipped.
    """
    from app.lifestyle.nutrition_copy import copy_day_nutrition, resolve_day_word

    try:
        source_day = resolve_day_word(input_data.get("source_date"), default="yesterday")
        target_day = resolve_day_word(
            input_data.get("target_date"), default="today", future_bias=True,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    try:
        with _db_session() as db:
            result = copy_day_nutrition(db, source_day, target_day)
    except Exception as exc:
        logger.exception("[lifestyle_tools] copy_nutrition_day failed")
        return {"ok": False, "error": str(exc)}
    return result


def _resolve_and_log_sync(name, brand, barcode, url, grams, meal_type, do_log) -> dict:
    """Sync resolve + log, run in a worker thread (the network calls and the
    per-host rate-limit sleep are blocking). Owns its OWN DB session so nothing
    crosses the event-loop/worker-thread boundary."""
    from app.lifestyle.product_service import resolve_nutrition, record_product_use
    from app.lifestyle.product_models import FoodProduct
    with _db_session() as db:
        res = resolve_nutrition(db, name=name, brand=brand, barcode=barcode, grams=grams, url=url)
        if not res.get("found"):
            return {"ok": False, "found": False, "ask_user": True, "message": res.get("message")}
        logged = None
        if do_log:
            product = (
                db.query(FoodProduct).filter(FoodProduct.id == res["food_product_id"]).first()
            )
            if product is not None:
                row = record_product_use(
                    db, product, grams=grams, meal_type=meal_type,
                    source=res["source"], is_estimate=res["is_estimate"],
                    confidence=res["confidence"], estimate_note=res.get("estimate_note"),
                )
                # record_product_use ALREADY committed `row` (an ORM NutritionLog).
                # Convert via the same Pydantic mapper the service uses — NOT _to_dict,
                # which would fall through to dict(<ORM row>) and raise
                # "'NutritionLog' object is not iterable". That crash (live 2026-06-15)
                # left the row committed but reported the tool as failed, so the model
                # re-logged it = duplicate diary entries.
                from app.lifestyle.service import _nutrition_to_out
                logged = _nutrition_to_out(row).model_dump()
        return {
            "ok": True, "found": True, "exact": res["exact"], "source": res["source"],
            "confidence": res["confidence"], "is_estimate": res["is_estimate"],
            "estimate_note": res.get("estimate_note"), "display_name": res["display_name"],
            "quantity_g": res["quantity_g"], "macros": res["scaled"],
            "micros": res.get("micros"), "logged": logged,
        }


async def resolve_nutrition_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Look up ONE food's nutrition online (the exact→similar→ask ladder) and,
    when a portion is known, log it as one row carrying its source, confidence,
    quantity and micronutrients. Use this for branded / ready-made items
    (e.g. 'Aldi Chicken, Tomato & Basil Topped Pasta') instead of guessing.

    Args: name OR barcode (one required); optional brand, grams (portion),
    url (a retailer product page you found via web_search), meal_type,
    log (default true — set false to just look up without logging).
    """
    name = str(input_data.get("name") or "").strip() or None
    brand = str(input_data.get("brand") or "").strip() or None
    barcode = str(input_data.get("barcode") or "").strip() or None
    url = str(input_data.get("url") or "").strip() or None
    meal_type = str(input_data.get("meal_type") or "meal").strip() or "meal"
    do_log = input_data.get("log", True) is not False
    try:
        grams = float(input_data.get("grams")) if input_data.get("grams") is not None else None
    except (TypeError, ValueError):
        grams = None

    if not (name or barcode):
        return {"ok": False, "error": "Give a food name or a barcode to resolve."}

    try:
        import asyncio
        return await asyncio.to_thread(
            _resolve_and_log_sync, name, brand, barcode, url, grams, meal_type, do_log,
        )
    except Exception as exc:
        logger.exception("[lifestyle_tools] resolve_nutrition failed")
        return {"ok": False, "error": str(exc)}


async def save_recipe_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Remember a recipe / meal the user cooks repeatedly — its ingredients and
    typical volumes — so it can be logged later in one shot by name.

    Args: name, ingredients (list of {name, grams, calories?, protein_g?,
    carbs_g?, fat_g?, sugar_g?}); optional meal_type, notes.
    """
    name = str(input_data.get("name") or "").strip()
    ingredients = input_data.get("ingredients")
    if not name or not isinstance(ingredients, list) or not ingredients:
        return {"ok": False, "error": "Give the recipe a name and a list of ingredients."}
    meal_type = str(input_data.get("meal_type") or "meal").strip() or "meal"
    notes = str(input_data.get("notes") or "").strip() or None
    try:
        from app.lifestyle.recipe_service import save_recipe, recipe_to_dict
        with _db_session() as db:
            r = save_recipe(db, name=name, ingredients=ingredients, meal_type=meal_type, notes=notes)
            return {"ok": True, "recipe": recipe_to_dict(r)}
    except Exception as exc:
        logger.exception("[lifestyle_tools] save_recipe failed")
        return {"ok": False, "error": str(exc)}


async def log_recipe_handler(input_data: dict, context: Optional[dict]) -> dict:
    """Log a previously-saved recipe in one go — expands it into one row per
    ingredient at the saved volumes ('found it before, same items, done').

    Args: name (the recipe) OR ingredient_names (match by ingredients); optional
    meal_type, on_date (YYYY-MM-DD).
    """
    name = str(input_data.get("name") or "").strip() or None
    ingredient_names = input_data.get("ingredient_names")
    if isinstance(ingredient_names, str):
        ingredient_names = [ingredient_names]
    meal_type = str(input_data.get("meal_type") or "").strip() or None
    on_date = _parse_iso_date(input_data.get("on_date"))
    if not name and not ingredient_names:
        return {"ok": False, "error": "Give the recipe name (or its ingredients) to log."}
    try:
        from app.lifestyle.recipe_service import find_recipe, expand_recipe_to_logs
        with _db_session() as db:
            recipe = find_recipe(db, name=name, ingredient_names=ingredient_names)
            if recipe is None:
                return {
                    "ok": False, "found": False,
                    "message": (
                        f"No saved recipe matches '{name or ingredient_names}'. Save it "
                        "first with save_recipe, or log the items individually."
                    ),
                }
            rows = expand_recipe_to_logs(db, recipe, meal_type=meal_type, on_date=on_date)
            return {
                "ok": True, "found": True, "recipe": recipe.name,
                "logged": len(rows), "items": [_to_dict(x) for x in rows],
            }
    except Exception as exc:
        logger.exception("[lifestyle_tools] log_recipe failed")
        return {"ok": False, "error": str(exc)}
