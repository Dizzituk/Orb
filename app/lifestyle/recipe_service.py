# FILE: app/lifestyle/recipe_service.py
# Purpose: Save / match / expand saved recipes into itemised food logs (Phase 6).
# Called-by: app.tools.lifestyle_nutrition_tools (save_recipe / log_recipe tools)
# Depends-on: app.lifestyle.recipe_models, app.lifestyle.product_service, app.lifestyle.service
# Last-renovated: 2026-06-14
"""
Saved-recipe service (food overhaul Phase 6).

save_recipe stores a named dish + its ingredient list (typical grams, optional
product link or macro override). find_recipe matches a repeated meal by name,
then by signature (sorted normalised ingredient set), then by fuzzy token
overlap. expand_recipe_to_logs turns a matched recipe into one NutritionLog row
PER ingredient via service.log_nutrition_items — the same writer per-item
logging and prep-split use, so everything converges on one path.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.lifestyle.recipe_models import (
    FoodRecipe, RecipeIngredient, ensure_recipe_tables,
)
from app.lifestyle.product_service import normalise_name

logger = logging.getLogger(__name__)

_FUZZY_THRESHOLD = 0.6  # Jaccard overlap of ingredient signatures to call it a match


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _signature(names: List[str]) -> str:
    toks = sorted({normalise_name(n) for n in names if n and normalise_name(n)})
    return "|".join(toks)


def _ing_name(i: Dict[str, Any]) -> str:
    return str(i.get("display_name") or i.get("name") or "").strip()


def save_recipe(
    db: Session, *, name: str, ingredients: List[Dict[str, Any]],
    meal_type: str = "meal", notes: Optional[str] = None,
) -> FoodRecipe:
    """Create or update a named recipe and its ingredient list (dedupe by name)."""
    ensure_recipe_tables()
    norm = normalise_name(name)
    signature = _signature([_ing_name(i) for i in ingredients])

    recipe = db.query(FoodRecipe).filter(FoodRecipe.norm_name == norm).first()
    if recipe is None:
        recipe = FoodRecipe(name=name.strip(), norm_name=norm)
        db.add(recipe)
    recipe.signature = signature
    recipe.meal_type = meal_type or "meal"
    if notes:
        recipe.notes = notes
    recipe.updated_at = _now()

    recipe.ingredients.clear()
    db.flush()
    for i in ingredients:
        recipe.ingredients.append(RecipeIngredient(
            display_name=_ing_name(i),
            food_product_id=i.get("food_product_id"),
            grams=_num(i.get("grams")),
            calories=_num(i.get("calories")), protein_g=_num(i.get("protein_g")),
            carbs_g=_num(i.get("carbs_g")), fat_g=_num(i.get("fat_g")),
            sugar_g=_num(i.get("sugar_g")),
        ))
    db.commit()
    db.refresh(recipe)
    logger.info("[recipe] Saved '%s' (%d ingredients)", recipe.name, len(recipe.ingredients))
    return recipe


def find_recipe(
    db: Session, *, name: Optional[str] = None,
    ingredient_names: Optional[List[str]] = None,
) -> Optional[FoodRecipe]:
    """Match a recipe by name, then exact signature, then fuzzy ingredient overlap."""
    ensure_recipe_tables()
    if name:
        norm = normalise_name(name)
        r = db.query(FoodRecipe).filter(FoodRecipe.norm_name == norm).first()
        if r:
            return r
        # Substring fallback ONLY when unambiguous (exactly one candidate) — an
        # arbitrary .first() could expand a different, more specific recipe.
        cands = db.query(FoodRecipe).filter(FoodRecipe.norm_name.like(f"%{norm}%")).all()
        if len(cands) == 1:
            return cands[0]
    if ingredient_names:
        sig = _signature(ingredient_names)
        if sig:
            r = db.query(FoodRecipe).filter(FoodRecipe.signature == sig).first()
            if r:
                return r
            want = set(sig.split("|"))
            best, best_score = None, 0.0
            for cand in db.query(FoodRecipe).all():
                have = set((cand.signature or "").split("|")) - {""}
                # Require the candidate to COVER every logged ingredient, so a
                # fuzzy match never silently drops an ingredient the user ate.
                if not have or not want.issubset(have):
                    continue
                score = len(want & have) / len(want | have)
                if score > best_score:
                    best, best_score = cand, score
            if best and best_score >= _FUZZY_THRESHOLD:
                return best
    return None


def _ingredient_macros(db: Session, ing: RecipeIngredient) -> Dict[str, Any]:
    """Macros for one ingredient: explicit override, else scaled from its product."""
    if any(v is not None for v in (
        ing.calories, ing.protein_g, ing.carbs_g, ing.fat_g, ing.sugar_g,
    )):
        return {
            "calories": ing.calories, "protein_g": ing.protein_g,
            "carbs_g": ing.carbs_g, "fat_g": ing.fat_g, "sugar_g": ing.sugar_g,
        }
    if ing.food_product_id and ing.grams:
        from app.lifestyle.product_models import FoodProduct
        p = db.query(FoodProduct).filter(FoodProduct.id == ing.food_product_id).first()
        if p is not None:
            factor = float(ing.grams) / 100.0

            def sc(v):
                return round(v * factor, 1) if v is not None else None

            return {
                "calories": sc(p.calories), "protein_g": sc(p.protein_g),
                "carbs_g": sc(p.carbs_g), "fat_g": sc(p.fat_g), "sugar_g": sc(p.sugar_g),
            }
    return {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None, "sugar_g": None}


def expand_recipe_to_logs(
    db: Session, recipe: FoodRecipe,
    meal_type: Optional[str] = None, on_date: Optional[date] = None,
) -> List[Any]:
    """Log one row PER ingredient via the shared itemised writer, bump usage."""
    from app.lifestyle.service import log_nutrition_items

    mt = meal_type or recipe.meal_type or "meal"
    items = [
        {"description": ing.display_name, **_ingredient_macros(db, ing)}
        for ing in recipe.ingredients
    ]
    rows = log_nutrition_items(db, items, mt, on_date=on_date)
    recipe.times_logged = (recipe.times_logged or 0) + 1
    recipe.last_logged_at = _now()
    db.commit()
    logger.info("[recipe] Logged '%s' → %d rows", recipe.name, len(rows))
    return rows


def list_recipes(db: Session) -> List[FoodRecipe]:
    ensure_recipe_tables()
    return (
        db.query(FoodRecipe)
        .order_by(FoodRecipe.times_logged.desc(), FoodRecipe.updated_at.desc())
        .all()
    )


def recipe_to_dict(recipe: FoodRecipe) -> Dict[str, Any]:
    return {
        "id": recipe.id,
        "name": recipe.name,
        "meal_type": recipe.meal_type,
        "times_logged": recipe.times_logged,
        "ingredients": [
            {
                "display_name": ing.display_name,
                "grams": ing.grams,
                "food_product_id": ing.food_product_id,
            }
            for ing in recipe.ingredients
        ],
    }
