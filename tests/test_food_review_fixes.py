# FILE: tests/test_food_review_fixes.py
# Purpose: Regression tests for the adversarial-review fixes (food overhaul Phases 3-6).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.lifestyle.retailer_scrape, product_service, recipe_service, nutrition_edit
# Last-renovated: 2026-06-14
"""
Regression tests locking in the adversarial-review fixes:
  1. scraper only re-bases a per-serving panel with a real gram serving + a
     plausibility guard (a per-100g panel / a count serving is left alone);
  2. an unverified (no-brand) OFF hit is flagged as an estimate, not exact;
  4. a recipe ingredient with a partial macro override (no calories) keeps it;
  6. fuzzy recipe match never drops a logged ingredient (requires coverage);
  7/8. a quantity edit preserves micro LABELS (nutri_score / nova_group).

Run with: pytest tests/test_food_review_fixes.py -v
"""

import json
import pytest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
import app.lifestyle.models  # noqa: F401
import app.lifestyle.product_models  # noqa: F401
import app.lifestyle.recipe_models  # noqa: F401
from app.lifestyle import product_service as ps
from app.lifestyle import recipe_service as rs
from app.lifestyle.retailer_scrape import _from_json_ld, _grams_from_serving
from app.lifestyle.nutrition_edit import scale_per_100g, update_nutrition
from app.lifestyle.models import NutritionLog
from app.lifestyle.schemas import NutritionLogUpdate


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    with patch.object(ps, "ensure_product_table", lambda: None), \
         patch.object(rs, "ensure_recipe_tables", lambda: None):
        yield s
    s.close()


# ---- Fix 1: scraper re-basing -------------------------------------------------

class TestScraperRebase:
    def test_per_100g_panel_not_inflated(self):
        html = ('<script type="application/ld+json">'
                '{"@type":"Product","name":"Cereal","nutrition":{"@type":"NutritionInformation",'
                '"servingSize":"30g","calories":"376 kcal","proteinContent":"8 g"}}</script>')
        # 376 * (100/30) = 1253 > 900 → treated as already-per-100g, left as-is.
        assert _from_json_ld(html)["macros"]["calories"] == 376.0

    def test_per_serving_panel_rebased(self):
        html = ('<script type="application/ld+json">'
                '{"@type":"Product","name":"Meal","nutrition":{"@type":"NutritionInformation",'
                '"servingSize":"400 g","calories":"520 kcal"}}</script>')
        assert _from_json_ld(html)["macros"]["calories"] == 130.0  # 520/4

    def test_grams_from_serving_ignores_counts(self):
        assert _grams_from_serving("2 biscuits (30g)") == 30.0  # the 30g, not the 2
        assert _grams_from_serving("2 biscuits") is None
        assert _grams_from_serving("1 pot") is None
        assert _grams_from_serving("400 g") == 400.0


# ---- Fix 2: no-brand OFF hit is an estimate ----------------------------------

class TestResolverNoBrand:
    def test_no_brand_hit_flagged_estimate(self, db):
        off_hit = {
            "display_name": "Generic Pasta", "brand": "SomeBrand", "barcode": None,
            "macros": {"calories": 150, "protein_g": 6, "carbs_g": 28, "fat_g": 2, "fibre_g": 1, "sugar_g": 3},
            "serving_size_g": None, "serving_description": None, "micros": None,
        }
        with patch.object(ps, "off_search_by_name", return_value=off_hit):
            res = ps.resolve_nutrition(db, name="chicken pasta", grams=100)  # no brand
        assert res["found"] and res["exact"] is False
        assert res["is_estimate"] is True
        assert res["source"] == "off_search"


# ---- Fixes 4 + 6: recipe macros / coverage -----------------------------------

class TestRecipeFixes:
    def test_partial_macro_override_kept(self, db):
        r = rs.save_recipe(db, name="P", ingredients=[{"name": "egg whites", "grams": 100, "protein_g": 11}])
        rows = rs.expand_recipe_to_logs(db, r)
        assert rows[0].protein_g == 11  # not discarded

    def test_fuzzy_never_drops_logged_ingredient(self, db):
        rs.save_recipe(db, name="ChickenRice", ingredients=[
            {"name": "chicken", "grams": 1}, {"name": "rice", "grams": 1},
        ])
        # An extra 'peas' the recipe lacks → must NOT match (would silently drop peas).
        assert rs.find_recipe(db, ingredient_names=["chicken", "rice", "peas"]) is None
        # The same set still matches.
        assert rs.find_recipe(db, ingredient_names=["rice", "chicken"]) is not None


# ---- Fixes 7/8: edit-path micro labels ---------------------------------------

class TestEditMicroLabels:
    def test_scaler_preserves_labels(self):
        out = scale_per_100g({"salt_g": 0.1, "nutri_score": "b", "nova_group": 4}, 200)
        assert out["salt_g"] == 0.2
        assert out["nutri_score"] == "b"  # not dropped
        assert out["nova_group"] == 4     # not scaled to 8

    def test_quantity_edit_preserves_micro_labels(self, db):
        m = NutritionLog(
            description="yoghurt", meal_type="snack",
            calories=60, protein_g=10, carbs_g=4, fat_g=0, quantity_g=100,
            per_100g_json=json.dumps({
                "calories": 60, "protein_g": 10, "carbs_g": 4, "fat_g": 0,
                "micros": {"salt_g": 0.1, "nutri_score": "a", "nova_group": 1},
            }),
        )
        db.add(m)
        db.commit()
        db.refresh(m)
        out = update_nutrition(db, m.id, NutritionLogUpdate(quantity_g=200))
        assert out.calories == 120.0
        assert out.micros["salt_g"] == 0.2
        assert out.micros["nutri_score"] == "a"  # label preserved
        assert out.micros["nova_group"] == 1     # label not scaled
