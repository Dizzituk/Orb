# FILE: tests/test_nutrition_resolver.py
# Purpose: Unit tests for the resolve_nutrition ladder + retailer scraper (food overhaul Phase 4).
# Called-by: no static importers found (pytest discovery)
# Depends-on: app.db, app.lifestyle.product_models, app.lifestyle.product_service, app.lifestyle.retailer_scrape
# Last-renovated: 2026-06-14
"""
Unit tests for the nutrition-estimate ladder (food overhaul Phase 4).

The ladder: local library -> OFF (barcode/name) -> retailer scrape -> SIMILAR
same-type item (flagged) -> ask. These tests cover the exact tiers, the
brand-mismatch -> similar_estimate flagging, the not-found -> ask path, and the
retailer JSON-LD parse (per-serving panel re-based to per-100g). OFF network is
mocked; ensure_product_table is neutralised so the live DB is never touched.

Run with: pytest tests/test_nutrition_resolver.py -v
"""

import pytest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base
import app.lifestyle.models  # noqa: F401
import app.lifestyle.product_models  # noqa: F401
from app.lifestyle.product_models import FoodProduct
from app.lifestyle import product_service as ps
from app.lifestyle.retailer_scrape import _from_json_ld


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    # ensure_product_table would create_all on the GLOBAL (live) engine — no-op it.
    with patch.object(ps, "ensure_product_table", lambda: None):
        yield s
    s.close()


def _seed(db, *, display_name, **kw):
    p = FoodProduct(name=ps.normalise_name(display_name), display_name=display_name, **kw)
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


# =========================================================================
# Ladder
# =========================================================================

class TestLadder:
    def test_local_name_exact(self, db):
        _seed(db, display_name="Test Pasta", brand="Aldi",
              calories=120, protein_g=5, carbs_g=20, fat_g=2)
        res = ps.resolve_nutrition(db, name="Test Pasta", grams=200)
        assert res["found"] and res["exact"]
        assert res["source"] == "library_name"
        assert res["scaled"]["calories"] == 240.0  # 120 * 2

    def test_barcode_local_exact(self, db):
        _seed(db, display_name="Bar", barcode="555",
              calories=100, protein_g=1, carbs_g=1, fat_g=1)
        res = ps.resolve_nutrition(db, barcode="555", grams=100)
        assert res["exact"] and res["source"] == "library_barcode"

    def test_similar_flagged_on_brand_mismatch(self, db):
        off_hit = {
            "display_name": "Basil & Tomato Pasta", "brand": "Sainsbury's", "barcode": None,
            "macros": {"calories": 130, "protein_g": 6, "carbs_g": 22, "fat_g": 3, "fibre_g": 1, "sugar_g": 4},
            "serving_size_g": None, "serving_description": None, "micros": {"salt_g": 0.5},
        }
        with patch.object(ps, "off_search_by_name", return_value=off_hit):
            res = ps.resolve_nutrition(db, name="basil tomato topped pasta", brand="Aldi", grams=400)
        assert res["found"] and res["exact"] is False
        assert res["source"] == "similar_estimate"
        assert res["is_estimate"] is True
        assert "similar item" in (res["estimate_note"] or "")
        assert res["scaled"]["calories"] == 520.0  # 130 * 4

    def test_exact_when_brand_matches(self, db):
        off_hit = {
            "display_name": "Aldi Greek Yoghurt", "brand": "Aldi", "barcode": None,
            "macros": {"calories": 60, "protein_g": 10, "carbs_g": 4, "fat_g": 0, "fibre_g": 0, "sugar_g": 4},
            "serving_size_g": None, "serving_description": None, "micros": None,
        }
        with patch.object(ps, "off_search_by_name", return_value=off_hit):
            res = ps.resolve_nutrition(db, name="greek yoghurt", brand="Aldi", grams=100)
        assert res["exact"] is True and res["source"] == "off_search"

    def test_not_found_asks_user(self, db):
        with patch.object(ps, "off_search_by_name", return_value=None), \
             patch.object(ps, "off_lookup_barcode", return_value=None):
            res = ps.resolve_nutrition(db, name="zxqwv nonsense food", grams=100)
        assert res["found"] is False and res["ask_user"] is True
        assert "macros" in (res["message"] or "")

    def test_brand_matches_helper(self):
        assert ps._brand_matches("Aldi", "Aldi Specially Selected") is True
        assert ps._brand_matches("Aldi", "Sainsbury's") is False
        assert ps._brand_matches(None, "Aldi") is False


# =========================================================================
# Retailer JSON-LD scrape
# =========================================================================

class TestScrape:
    def test_jsonld_nutrition_rebased_to_100g(self):
        html = (
            '<html><head><script type="application/ld+json">'
            '{"@type":"Product","name":"Test Ready Meal","brand":{"name":"Tesco"},'
            '"nutrition":{"@type":"NutritionInformation","servingSize":"400 g",'
            '"calories":"520 kcal","proteinContent":"30 g","carbohydrateContent":"60 g",'
            '"fatContent":"18 g","saturatedFatContent":"6 g","saltContent":"2 g"}}'
            '</script></head></html>'
        )
        parsed = _from_json_ld(html)
        assert parsed is not None
        assert parsed["display_name"] == "Test Ready Meal"
        assert parsed["brand"] == "Tesco"
        # 400g serving re-based to 100g (factor 0.25)
        assert parsed["macros"]["calories"] == 130.0
        assert parsed["macros"]["protein_g"] == 7.5
        assert parsed["micros"]["salt_g"] == 0.5  # 2g / 4

    def test_jsonld_none_without_nutrition(self):
        html = '<html><script type="application/ld+json">{"@type":"Product","name":"X"}</script></html>'
        assert _from_json_ld(html) is None


# =========================================================================
# Resolve + log handler path — Bug A regression (live 2026-06-15)
# =========================================================================

class TestResolveAndLog:
    """resolve_nutrition's log path logged the row via record_product_use (which
    COMMITS) and then crashed converting the ORM row with _to_dict ->
    'NutritionLog' object is not iterable. The committed-but-unreported row made
    the model believe logging had failed and re-log it = duplicate diary rows.
    The path must return `logged` as a plain dict, with no crash. The earlier
    Phase-4 tests called resolve_nutrition() directly and never exercised this
    conversion, which is why a green suite still shipped the bug."""

    def test_resolve_and_log_returns_dict_not_crash(self, db):
        from contextlib import contextmanager
        import app.tools.lifestyle_nutrition_tools as lnt

        _seed(db, display_name="Test Pork Loin", brand="Aldi",
              calories=200, protein_g=25, carbs_g=0, fat_g=10)

        @contextmanager
        def _fake_session():
            yield db

        # _resolve_and_log_sync is the body the async handler runs via to_thread;
        # call it directly (same thread) so the test is deterministic.
        with patch.object(lnt, "_db_session", _fake_session):
            res = lnt._resolve_and_log_sync(
                "Test Pork Loin", "Aldi", None, None, 200.0, "meal", True,
            )

        assert res["ok"] is True and res["found"] is True
        assert isinstance(res["logged"], dict)
        assert res["logged"]["calories"] == 400.0  # 200 kcal/100g * 200g
        assert res["logged"]["description"]  # round-tripped via _nutrition_to_out
