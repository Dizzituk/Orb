# FILE: app/lifestyle/itemisation.py
# Purpose: Detect when a single food-log description actually names 2+ distinct foods (Bug 6).
# Called-by: app.tools.lifestyle_tools (log_nutrition_handler itemisation gate)
# Depends-on: stdlib only
# Last-renovated: 2026-06-13
"""
Itemisation gate for food logging (Bug 6).

ASTRA's log_nutrition tool already supports an `items` array that writes one
NutritionLog row per food (the same writer prep_split uses to split a batch
cook across days). The defect is the MODEL: it sometimes calls log_nutrition
once with a combined description ("Dinner: Aldi Chicken...; fish sushi; ice
cream") and merged macros, bunching a whole meal into a single row.

This module is the deterministic safety net. `looks_multi_food` decides whether
a single-row description clearly names two or more distinct foods; if so the
handler rejects it with `multi_food_rejection` so the model re-issues itemised.

Design goal: HIGH PRECISION — fire on real multi-food meals, but NEVER on a
single food whose text merely contains commas. The two hard cases it must get
right (both from the live bug report):
  - "480g pork loin (oven-baked, lean, drained)"  -> ONE food (commas in parens)
  - "Aldi Chicken, Tomato & Basil Topped Pasta"   -> ONE product (comma in name)
The actual splitting is left to the model (it has the context to know the Aldi
pasta is one product despite its comma); this module only judges multi-vs-single.
"""
from __future__ import annotations

import re

# Quantity-led fragment, e.g. "100g", "250 g", "1.5kg", "400ml", "2 slices",
# "1 pot", "3 scoops", "2 eggs". Two or more of these = a list of foods.
_QTY_UNIT = (
    r"(?:g|kg|mg|ml|l|cl|oz|lb|tbsp|tsp|cup|cups|slice|slices|piece|pieces|"
    r"pot|pots|can|cans|tin|tins|scoop|scoops|egg|eggs|bar|bars|pack|packs|"
    r"serving|servings|handful|portion|portions)"
)
_QTY_RE = re.compile(r"\b\d+(?:\.\d+)?\s*" + _QTY_UNIT + r"\b", re.IGNORECASE)

# A leading meal label we strip before counting ("Dinner:", "Lunch -", ...).
_MEAL_LABEL_RE = re.compile(
    r"^\s*(?:breakfast|brunch|lunch|dinner|supper|snack|meal|drinks?)\s*[:\-]\s*",
    re.IGNORECASE,
)

# Parenthetical groups whose internal commas must not count as separators.
_PARENS_RE = re.compile(r"\([^)]*\)")


def looks_multi_food(description: str) -> bool:
    """True when `description` clearly names 2+ distinct foods to itemise.

    Conservative by design — see module docstring. Fires on:
      1. two or more quantity-led fragments (100g X, 250g Y, ...),
      2. a semicolon list of 2+ segments,
      3. a comma list of 3+ segments (a list, not an aside).
    Parenthetical content is removed first so internal commas don't count.
    """
    if not description:
        return False

    text = _MEAL_LABEL_RE.sub("", description.strip())
    no_parens = _PARENS_RE.sub("", text)

    # 1. Multiple explicit quantities -> a list of weighed foods.
    if len(_QTY_RE.findall(no_parens)) >= 2:
        return True

    # 2. Semicolon-delimited list of 2+ non-trivial segments.
    semis = [s for s in no_parens.split(";") if len(s.strip()) >= 3]
    if len(semis) >= 2:
        return True

    # 3. Comma-delimited list of 3+ non-trivial segments. The >=3 threshold
    #    avoids firing on a single food with one descriptive comma
    #    (e.g. "chicken, lean") or a product name with one comma.
    commas = [s for s in no_parens.split(",") if len(s.strip()) >= 2]
    if len(commas) >= 3:
        return True

    return False


def multi_food_rejection(description: str) -> dict:
    """The actionable rejection the log tool returns for a bunched meal.

    Mirrors the shape/style of the macro hard-rule rejection so the model
    re-issues a single itemised call instead of writing a merged blob.
    """
    return {
        "ok": False,
        "logged": 0,
        "needs_itemisation": True,
        "description": description,
        "error": (
            "NOT logged — this names several distinct foods, so it must be "
            "logged as one row PER food, not merged into a single entry. Re-call "
            "log_nutrition ONCE with the `items` array: one object per food, each "
            "with its OWN description and its own calories, protein_g, carbs_g, "
            "fat_g (and sugar_g where known) — ask the user or web_search to "
            "estimate per item. "
            "BUT if this is actually ONE dish or product — a recipe whose "
            "ingredients you listed (e.g. 'stir fry made with chicken, peppers, "
            "onion'), or a product NAME that contains commas (e.g. 'Aldi Chicken, "
            "Tomato & Basil Topped Pasta') — do NOT split it: re-call with a SHORT "
            "single `description` naming just that one item (drop the comma list) "
            "plus its total calories, protein_g, carbs_g and fat_g."
        ),
    }


__all__ = ["looks_multi_food", "multi_food_rejection"]
