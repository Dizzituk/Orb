# FILE: tests/test_nutrition_edit.py
# Purpose: Unit tests for per-item food-log edit + re-scale (food overhaul Phase 2).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.db, app.lifestyle.models, app.lifestyle.nutrition_edit, app.lifestyle.schemas
# Last-renovated: 2026-06-14
"""
Unit tests for editing a logged food row (food overhaul Phase 2).

Covers the scaler (scale_per_100g) and update_nutrition: quantity edits re-scale
from the stored per-100g basis, explicit macro overrides win and mark the row
verified, a quantity edit with no basis leaves macros alone, and a missing row
returns None. NutritionLog is not encrypted, so no crypto stub is needed.

Run with: pytest tests/test_nutrition_edit.py -v
"""

import json
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
import app.lifestyle.models  # noqa: F401  (registers lifestyle tables for create_all)
from app.lifestyle.models import NutritionLog
from app.lifestyle.schemas import NutritionLogUpdate
from app.lifestyle.nutrition_edit import scale_per_100g, update_nutrition


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _row(db, **kw):
    defaults = dict(description="cooked rice", meal_type="lunch")
    defaults.update(kw)
    e = NutritionLog(**defaults)
    db.add(e)
    db.commit()
    db.refresh(e)
    return e


# =========================================================================
# scale_per_100g (pure)
# =========================================================================

class TestScale:
    def test_basic_scaling(self):
        out = scale_per_100g({"calories": 130, "protein_g": 2.7, "carbs_g": 28}, 200)
        assert out == {"calories": 260.0, "protein_g": 5.4, "carbs_g": 56.0}

    def test_half_portion(self):
        out = scale_per_100g({"calories": 100}, 50)
        assert out == {"calories": 50.0}

    def test_skips_nested_micros_and_nonnumeric(self):
        out = scale_per_100g({"calories": 100, "micros": {"sodium_mg": 10}, "label": "x"}, 100)
        assert out == {"calories": 100.0}

    def test_bad_grams_returns_empty(self):
        assert scale_per_100g({"calories": 100}, None) == {}
        assert scale_per_100g({"calories": 100}, "abc") == {}


# =========================================================================
# update_nutrition
# =========================================================================

class TestUpdate:
    def test_quantity_change_rescales_from_basis(self, db):
        e = _row(
            db, calories=130, protein_g=2.7, carbs_g=28, fat_g=0.3,
            quantity_g=100, source="off_search",
            per_100g_json=json.dumps({"calories": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3}),
        )
        out = update_nutrition(db, e.id, NutritionLogUpdate(quantity_g=200))
        assert out is not None
        assert out.quantity_g == 200
        assert out.calories == 260.0
        assert out.protein_g == 5.4
        assert out.carbs_g == 56.0
        assert out.fat_g == 0.6

    def test_explicit_macro_override_wins_and_marks_verified(self, db):
        e = _row(
            db, calories=130, protein_g=2.7, carbs_g=28, fat_g=0.3,
            quantity_g=100, is_estimate=True, confidence=0.5,
            per_100g_json=json.dumps({"calories": 130, "protein_g": 2.7, "carbs_g": 28, "fat_g": 0.3}),
        )
        # quantity AND an explicit calories override → override wins, not 260.
        out = update_nutrition(db, e.id, NutritionLogUpdate(quantity_g=200, calories=999))
        assert out.calories == 999
        assert out.quantity_g == 200
        assert out.is_estimate is False
        assert out.confidence == 0.99

    def test_quantity_change_without_basis_leaves_macros(self, db):
        e = _row(db, calories=200, protein_g=5, carbs_g=44, fat_g=1)  # no per_100g_json
        out = update_nutrition(db, e.id, NutritionLogUpdate(quantity_g=300))
        assert out.quantity_g == 300
        assert out.calories == 200  # unchanged — nothing to scale from

    def test_set_basis_then_scale(self, db):
        e = _row(db, calories=0, protein_g=0, carbs_g=0, fat_g=0)
        out = update_nutrition(db, e.id, NutritionLogUpdate(
            per_100g={"calories": 160, "protein_g": 6, "carbs_g": 0, "fat_g": 15},
            quantity_g=150,
        ))
        assert out.calories == 240.0
        assert out.protein_g == 9.0
        assert out.fat_g == 22.5
        assert out.per_100g == {"calories": 160, "protein_g": 6, "carbs_g": 0, "fat_g": 15}

    def test_description_and_meal_type_edit(self, db):
        e = _row(db, calories=100, protein_g=1, carbs_g=1, fat_g=1)
        out = update_nutrition(db, e.id, NutritionLogUpdate(
            description="grilled chicken breast", meal_type="dinner",
        ))
        assert out.description == "grilled chicken breast"
        assert out.meal_type == "dinner"

    def test_missing_row_returns_none(self, db):
        assert update_nutrition(db, 99999, NutritionLogUpdate(quantity_g=100)) is None

    def test_edit_recomputes_daily_summary(self, db):
        from app.lifestyle import service
        e = _row(
            db, calories=100, protein_g=2, carbs_g=20, fat_g=1, quantity_g=100,
            per_100g_json=json.dumps({"calories": 100, "protein_g": 2, "carbs_g": 20, "fat_g": 1}),
        )
        update_nutrition(db, e.id, NutritionLogUpdate(quantity_g=250))
        day = service.get_daily_nutrition(db)  # today
        assert day.total_calories == 250.0
