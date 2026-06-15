# FILE: tests/test_recipe_memory.py
# Purpose: Unit tests for saved-recipe memory (food overhaul Phase 6).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.db, app.lifestyle.product_models, app.lifestyle.recipe_models, app.lifestyle.recipe_service
# Last-renovated: 2026-06-14
"""
Unit tests for saved-recipe memory (food overhaul Phase 6).

save_recipe stores a named dish + ingredients; find_recipe matches by name,
exact ingredient signature, then fuzzy overlap; expand_recipe_to_logs writes one
row PER ingredient via the shared itemised writer (override macros or scaled
from a linked product). ensure_recipe_tables is neutralised so the live DB is
never touched.

Run with: pytest tests/test_recipe_memory.py -v
"""

import pytest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
import app.lifestyle.models  # noqa: F401
import app.lifestyle.product_models  # noqa: F401
import app.lifestyle.recipe_models  # noqa: F401
from app.lifestyle.product_models import FoodProduct
from app.lifestyle import recipe_service as rs


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    with patch.object(rs, "ensure_recipe_tables", lambda: None):
        yield s
    s.close()


# =========================================================================
# Save / find
# =========================================================================

class TestSaveFind:
    def test_save_and_find_by_name(self, db):
        r = rs.save_recipe(db, name="Beef Chilli", ingredients=[
            {"name": "500g beef mince", "grams": 500, "calories": 1000, "protein_g": 100, "carbs_g": 0, "fat_g": 60},
            {"name": "2 tins tomatoes", "grams": 800, "calories": 240, "protein_g": 12, "carbs_g": 40, "fat_g": 2},
        ])
        assert r.id and len(r.ingredients) == 2
        found = rs.find_recipe(db, name="beef chilli")
        assert found is not None and found.id == r.id

    def test_find_by_contains(self, db):
        rs.save_recipe(db, name="My Overnight Oats", ingredients=[
            {"name": "oats", "grams": 80, "calories": 300, "protein_g": 10, "carbs_g": 50, "fat_g": 6},
        ])
        assert rs.find_recipe(db, name="overnight oats") is not None

    def test_find_by_exact_and_fuzzy_signature(self, db):
        rs.save_recipe(db, name="Chilli", ingredients=[
            {"name": "beef mince", "grams": 500}, {"name": "kidney beans", "grams": 200},
            {"name": "chopped tomatoes", "grams": 400}, {"name": "onion", "grams": 100},
        ])
        # Exact signature (same ingredient name set, any order).
        f = rs.find_recipe(db, ingredient_names=["onion", "beef mince", "chopped tomatoes", "kidney beans"])
        assert f is not None
        # Fuzzy: a SUBSET of the recipe's ingredients (coverage holds) still matches.
        f2 = rs.find_recipe(db, ingredient_names=["beef mince", "kidney beans", "chopped tomatoes"])
        assert f2 is not None
        # But a query with an ingredient the recipe LACKS must NOT match
        # (matching would silently drop that ingredient from the log).
        assert rs.find_recipe(db, ingredient_names=["beef mince", "kidney beans", "chopped tomatoes", "garlic"]) is None

    def test_no_match(self, db):
        assert rs.find_recipe(db, name="nonexistent dish") is None
        assert rs.find_recipe(db, ingredient_names=["unicorn", "stardust"]) is None

    def test_resave_replaces_ingredients(self, db):
        rs.save_recipe(db, name="Dish", ingredients=[{"name": "a", "grams": 1, "calories": 1, "protein_g": 0, "carbs_g": 0, "fat_g": 0}])
        r2 = rs.save_recipe(db, name="Dish", ingredients=[
            {"name": "b", "grams": 1, "calories": 1, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
            {"name": "c", "grams": 1, "calories": 1, "protein_g": 0, "carbs_g": 0, "fat_g": 0},
        ])
        assert len(r2.ingredients) == 2  # replaced, not appended


# =========================================================================
# Expand → one row per ingredient
# =========================================================================

class TestExpand:
    def test_expand_override_macros(self, db):
        r = rs.save_recipe(db, name="Test Meal", ingredients=[
            {"name": "chicken", "grams": 200, "calories": 330, "protein_g": 62, "carbs_g": 0, "fat_g": 7},
            {"name": "rice", "grams": 150, "calories": 195, "protein_g": 4, "carbs_g": 42, "fat_g": 1},
        ])
        rows = rs.expand_recipe_to_logs(db, r)
        assert len(rows) == 2
        assert sorted([row.calories for row in rows]) == [195, 330]
        assert r.times_logged == 1

    def test_expand_product_scaled(self, db):
        p = FoodProduct(name="greek yoghurt", display_name="Greek Yoghurt",
                        calories=60, protein_g=10, carbs_g=4, fat_g=0)
        db.add(p)
        db.commit()
        db.refresh(p)
        r = rs.save_recipe(db, name="Yoghurt Bowl", ingredients=[
            {"name": "Greek Yoghurt", "grams": 200, "food_product_id": p.id},
        ])
        rows = rs.expand_recipe_to_logs(db, r)
        assert len(rows) == 1
        assert rows[0].calories == 120.0  # 60 * (200/100)
