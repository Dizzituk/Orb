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

    def test_pork_loin_descriptor_commas_with_trailing_qty(self):
        # Live 2026-06-15: one quantity token + descriptor commas = ONE food.
        assert looks_multi_food(
            "Pork loin steaks, oven cooked, fat drained, 470 g"
        ) is False

    def test_green_olives_descriptor_commas_no_qty(self):
        # Live 2026-06-15: descriptor commas, no quantity unit = ONE food.
        assert looks_multi_food("Green olives, pitted in brine, about 25") is False

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
# Accepted false negatives (Bug-6 hotfix 2026-06-15)
# =========================================================================

class TestAcceptedFalseNegatives:
    """Removing the >=3-comma signal was the only way to stop single foods with
    descriptor commas being false-rejected. The cost: a bare comma list with NO
    quantities is no longer flagged. That's the accepted trade — the strengthened
    tool description and the items[] path push itemisation for these instead."""

    def test_comma_list_without_quantities_now_single(self):
        assert looks_multi_food(
            "blueberries, strawberries, broccoli, greek yoghurt, pork loin"
        ) is False

    def test_three_item_plate_without_quantities_now_single(self):
        assert looks_multi_food("eggs on toast, beans, bacon") is False


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

    def test_macro_complete_single_with_descriptor_commas_logs(self):
        # Bug-6 hotfix 2026-06-15: a single description with descriptor commas BUT
        # complete macros must SKIP the itemisation gate (the model committed to
        # one item). Patch the service writer so no DB/network is touched.
        from unittest.mock import patch
        import app.lifestyle.service as svc
        from app.tools.lifestyle_tools import log_nutrition_handler

        class _FakeOut:
            def model_dump(self):
                return {
                    "id": 7,
                    "description": "Pork loin steaks, oven cooked, fat drained, 470 g",
                    "calories": 847, "protein_g": 120, "carbs_g": 0,
                    "fat_g": 39, "sugar_g": 0,
                }

        with patch.object(svc, "log_nutrition", return_value=_FakeOut()):
            r = asyncio.run(log_nutrition_handler({
                "description": "Pork loin steaks, oven cooked, fat drained, 470 g",
                "calories": 847, "protein_g": 120, "carbs_g": 0, "fat_g": 39,
            }, None))
        assert r["ok"] is True
        assert r.get("needs_itemisation") is not True
        assert r["log_id"] == 7
