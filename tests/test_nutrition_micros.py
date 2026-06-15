# FILE: tests/test_nutrition_micros.py
# Purpose: Unit tests for micronutrient capture + scaling (food overhaul Phase 3).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.db, app.lifestyle.models, app.lifestyle.product_models, app.lifestyle.product_service
# Last-renovated: 2026-06-14
"""
Unit tests for micronutrient capture (food overhaul Phase 3).

OFF carries sodium/salt/saturates/Nutri-Score/NOVA; the parser used to drop
them. These tests cover: _parse_off_product now extracting them, _scale_micros
scaling numbers but passing labels through, and record_product_use writing the
scaled micros + a full re-scalable per-100g basis (incl. micros) onto the row.

Run with: pytest tests/test_nutrition_micros.py -v
"""

import json
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
import app.lifestyle.models  # noqa: F401  (register nutrition tables)
import app.lifestyle.product_models  # noqa: F401  (register product table)
from app.lifestyle.product_models import FoodProduct
from app.lifestyle.product_service import (
    _parse_off_product, _scale_micros, record_product_use,
)


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


# =========================================================================
# OFF parse
# =========================================================================

class TestParseOff:
    def test_extracts_headline_micros(self):
        payload = {
            "product_name": "0% Greek Yoghurt", "code": "123", "brands": "Aldi",
            "nutriments": {
                "energy-kcal_100g": 60, "proteins_100g": 10, "carbohydrates_100g": 4,
                "fat_100g": 0, "sugars_100g": 4, "fiber_100g": 0,
                "saturated-fat_100g": 0.1, "salt_100g": 0.2, "sodium_100g": 0.08,
            },
            "nutriscore_grade": "a", "nova_group": 1,
        }
        parsed = _parse_off_product(payload)
        assert parsed["macros"]["calories"] == 60
        m = parsed["micros"]
        assert m["saturated_fat_g"] == 0.1
        assert m["salt_g"] == 0.2
        assert m["sodium_g"] == 0.08
        assert m["nutri_score"] == "a"
        assert m["nova_group"] == 1

    def test_micros_none_when_absent(self):
        payload = {
            "product_name": "Mystery", "code": "9",
            "nutriments": {"energy-kcal_100g": 100},
        }
        parsed = _parse_off_product(payload)
        assert parsed["micros"] is None  # nothing carried → flagged absent


# =========================================================================
# Scaling
# =========================================================================

class TestScaleMicros:
    def test_scales_numbers_passes_labels(self):
        out = _scale_micros({"salt_g": 0.5, "saturated_fat_g": 1.0, "nutri_score": "b", "nova_group": 4}, 2.0)
        assert out["salt_g"] == 1.0
        assert out["saturated_fat_g"] == 2.0
        assert out["nutri_score"] == "b"   # label unscaled
        assert out["nova_group"] == 4      # label unscaled

    def test_empty(self):
        assert _scale_micros(None, 2.0) == {}


# =========================================================================
# record_product_use → scaled micros + re-scalable basis on the row
# =========================================================================

class TestRecordProductUse:
    def test_writes_scaled_micros_and_basis(self, db):
        p = FoodProduct(
            name="0% greek yoghurt", display_name="0% Greek Yoghurt",
            calories=60, protein_g=10, carbs_g=4, fat_g=0, fibre_g=0, sugar_g=4,
            micros_json=json.dumps({
                "saturated_fat_g": 0.1, "salt_g": 0.1, "nutri_score": "a", "nova_group": 1,
            }),
        )
        db.add(p)
        db.commit()
        db.refresh(p)

        log = record_product_use(db, p, grams=200)  # factor 2.0

        assert log.calories == 120.0
        assert log.quantity_g == 200
        assert log.food_product_id == p.id

        basis = json.loads(log.per_100g_json)
        assert basis["calories"] == 60
        assert basis["micros"]["nutri_score"] == "a"

        micros = json.loads(log.micros_json)
        assert micros["saturated_fat_g"] == 0.2  # 0.1 * 2
        assert micros["salt_g"] == 0.2
        assert micros["nutri_score"] == "a"      # label not scaled
        assert micros["nova_group"] == 1
