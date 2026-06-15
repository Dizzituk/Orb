# FILE: tests/test_nutrition_itemisation.py
# Purpose: Unit tests for the Bug 6 itemisation gate (food logging one row per food).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.lifestyle.itemisation, app.tools.lifestyle_tools
# Last-renovated: 2026-06-13
"""
Unit tests for the food-logging itemisation gate (Bug 6).

The detector must FIRE on genuine multi-food meals (so the model re-issues an
itemised log) but NEVER on a single food whose text merely contains commas.
The two hard false-positive traps come straight from the live bug screenshot:
  - "480g pork loin (oven-baked, lean, drained)"  -> ONE food
  - "Aldi Chicken, Tomato & Basil Topped Pasta"   -> ONE product

Run with: pytest tests/test_nutrition_itemisation.py -v
"""

import asyncio

from app.lifestyle.itemisation import looks_multi_food, multi_food_rejection


# =========================================================================
# Detector — multi-food (must return True)
# =========================================================================

class TestMultiFoodTrue:
    def test_screenshot_row1_six_quantities(self):
        # The real bunched "Meal" row from the screenshot.
        assert looks_multi_food(
            "100g blueberries, 130g strawberries, 93g broccoli, 250g 0% Greek "
            "yoghurt, 60g vanilla whey protein, 480g pork loin (oven-baked, "
            "lean, drained)"
        ) is True

    def test_screenshot_row2_semicolon_list(self):
        # The real bunched "Dinner" row — note item 1 has an internal comma.
        assert looks_multi_food(
            "Dinner: Aldi Chicken, Tomato & Basil Topped Pasta; Aldi Fish Sushi "
            "Selection; caramelised biscuit slices; 400g strawberries; Aldi "
            "cookie dough ice cream pot"
        ) is True

    def test_two_quantities_joined_by_and(self):
        assert looks_multi_food("200g chicken and 150g rice") is True

    def test_comma_list_without_quantities(self):
        assert looks_multi_food(
            "blueberries, strawberries, broccoli, greek yoghurt, pork loin"
        ) is True

    def test_three_item_plate(self):
        assert looks_multi_food("eggs on toast, beans, bacon") is True


# =========================================================================
# Detector — single food (must return False; the false-positive traps)
# =========================================================================

class TestSingleFoodFalse:
    def test_pork_loin_internal_paren_commas(self):
        assert looks_multi_food("480g pork loin (oven-baked, lean, drained)") is False

    def test_branded_product_with_comma_in_name(self):
        assert looks_multi_food("Aldi Chicken, Tomato & Basil Topped Pasta 400g") is False

    def test_single_food_one_descriptor_comma(self):
        assert looks_multi_food("chicken breast, grilled") is False

    def test_snickers_quantity_in_parens(self):
        assert looks_multi_food("Snickers bar (58.7g)") is False

    def test_single_homemade_dish_one_quantity(self):
        assert looks_multi_food("500g beef chilli") is False

    def test_single_pot_yoghurt(self):
        assert looks_multi_food("1 pot 0% Greek yoghurt") is False

    def test_empty_and_blank(self):
        assert looks_multi_food("") is False
        assert looks_multi_food("   ") is False


# =========================================================================
# Rejection payload
# =========================================================================

class TestRejection:
    def test_rejection_shape(self):
        r = multi_food_rejection("a, b, c, d")
        assert r["ok"] is False
        assert r["needs_itemisation"] is True
        assert r["logged"] == 0
        assert "items" in r["error"]
        assert r["description"] == "a, b, c, d"


# =========================================================================
# Handler gate (both DB-free paths — gate returns before any DB session)
# =========================================================================

class TestHandlerGate:
    def test_multi_food_single_description_is_rejected(self):
        from app.tools.lifestyle_tools import log_nutrition_handler
        r = asyncio.run(log_nutrition_handler(
            {"description": "100g rice, 200g chicken, 150g broccoli"}, None,
        ))
        assert r["ok"] is False
        assert r.get("needs_itemisation") is True

    def test_single_food_passes_gate_to_macro_rule(self):
        # Gate must NOT fire; the next guard (macro completeness) fires instead,
        # proving the single-food path is untouched by Bug 6's gate.
        from app.tools.lifestyle_tools import log_nutrition_handler
        r = asyncio.run(log_nutrition_handler(
            {"description": "Snickers bar (58.7g)"}, None,
        ))
        assert r["ok"] is False
        assert r.get("needs_itemisation") is not True
        assert "needs" in r  # macro-completeness hard rule, not itemisation
