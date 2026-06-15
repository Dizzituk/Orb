# FILE: app/lifestyle/product_service.py
# Purpose: Product library service — Job 4 of the memory roadmap (2026-06-10).
# Called-by: app.lifestyle.product_router
# Depends-on: app.lifestyle.models, app.lifestyle.product_models
# Last-renovated: 2026-06-11
"""
Product library service — Job 4 of the memory roadmap (2026-06-10).

Lookup order everywhere: local library first (instant, free), then Open
Food Facts (barcode endpoint or name search), saving whatever is found so
the next encounter is local. This is the "take a picture of an Aldi item /
give a vague description and ASTRA looks it up online and remembers it"
capability.

All network calls are best-effort with short timeouts — a dead OFF API
never blocks logging; the caller just falls back to LLM estimation.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.lifestyle.product_models import FoodProduct, ensure_product_table

logger = logging.getLogger(__name__)

_OFF_PRODUCT_URL = "https://world.openfoodfacts.org/api/v2/product/{barcode}.json"
_OFF_SEARCH_URL = "https://world.openfoodfacts.org/cgi/search.pl"
_OFF_TIMEOUT_S = 6.0
_OFF_HEADERS = {"User-Agent": "ASTRA-personal-assistant/1.0 (single-user; Cornwall UK)"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def normalise_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the dedupe key."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (name or "").lower())
    return " ".join(cleaned.split())


# Micronutrient labels that describe the PRODUCT, not a portion — never scaled.
_MICRO_LABELS = ("nutri_score", "nova_group")


def _safe_load(s: Optional[str]) -> Optional[Dict[str, Any]]:
    """json.loads a text column into a dict, or None. Never raises."""
    if not s:
        return None
    try:
        val = json.loads(s)
        return val if isinstance(val, dict) else None
    except Exception:
        return None


def _scale_micros(micros: Optional[Dict[str, Any]], factor: float) -> Dict[str, Any]:
    """Scale numeric per-100g micros by `factor`; pass labels through unchanged."""
    out: Dict[str, Any] = {}
    for k, v in (micros or {}).items():
        if k in _MICRO_LABELS:
            out[k] = v
        else:
            try:
                out[k] = round(float(v) * factor, 2)
            except (TypeError, ValueError):
                continue
    return out


# =============================================================================
# Local library
# =============================================================================

def get_by_barcode(db: Session, barcode: str) -> Optional[FoodProduct]:
    if not barcode:
        return None
    return db.query(FoodProduct).filter(FoodProduct.barcode == str(barcode).strip()).first()


def search_products(db: Session, query: str, limit: int = 10) -> List[FoodProduct]:
    norm = normalise_name(query)
    if not norm:
        return []
    like = f"%{norm}%"
    return (
        db.query(FoodProduct)
        .filter(
            (FoodProduct.name.like(like))
            | (FoodProduct.display_name.like(f"%{query.strip()}%"))
            | (FoodProduct.brand.like(f"%{query.strip()}%"))
        )
        .order_by(FoodProduct.times_logged.desc())
        .limit(limit)
        .all()
    )


def upsert_product(
    db: Session,
    *,
    display_name: str,
    brand: Optional[str] = None,
    barcode: Optional[str] = None,
    macros: Optional[Dict[str, Any]] = None,
    serving_size_g: Optional[float] = None,
    serving_description: Optional[str] = None,
    micros: Optional[Dict[str, Any]] = None,
    source: str = "manual",
) -> FoodProduct:
    """Create or update a product. Dedupe by barcode first, then name+brand."""
    macros = macros or {}
    norm = normalise_name(display_name)

    product: Optional[FoodProduct] = None
    if barcode:
        product = get_by_barcode(db, barcode)
    if product is None and norm:
        candidates = db.query(FoodProduct).filter(FoodProduct.name == norm).all()
        brand_norm = normalise_name(brand or "")
        for candidate in candidates:
            if normalise_name(candidate.brand or "") == brand_norm:
                product = candidate
                break
        if product is None and candidates:
            product = candidates[0]

    if product is None:
        product = FoodProduct(
            name=norm, display_name=display_name.strip(),
            brand=(brand or None), barcode=(str(barcode).strip() if barcode else None),
            source=source, created_at=_now(),
        )
        db.add(product)

    # Fill/refresh fields — newer lookup data wins.
    for field in ("calories", "protein_g", "carbs_g", "fat_g", "fibre_g", "sugar_g"):
        value = macros.get(field)
        if value is not None:
            setattr(product, field, float(value))
    if micros:
        product.micros_json = json.dumps(micros)
    if barcode and not product.barcode:
        product.barcode = str(barcode).strip()
    if brand and not product.brand:
        product.brand = brand
    if serving_size_g:
        product.serving_size_g = float(serving_size_g)
    if serving_description:
        product.serving_description = serving_description
    product.updated_at = _now()

    db.commit()
    db.refresh(product)
    return product


def record_product_use(
    db: Session,
    product: FoodProduct,
    grams: Optional[float] = None,
    meal_type: str = "meal",
    source: str = "product",
    is_estimate: bool = False,
    confidence: float = 0.9,
    estimate_note: Optional[str] = None,
):
    """Log eating this product: creates a NutritionLog scaled from per-100g
    values, bumps usage stats. Returns the NutritionLog row."""
    from app.lifestyle.models import NutritionLog

    portion = float(grams) if grams else (product.serving_size_g or 100.0)
    factor = portion / 100.0

    def _scaled(value: Optional[float]) -> Optional[float]:
        return round(value * factor, 1) if value is not None else None

    # Re-scalable basis (Phase 2/3): keep the full per-100g picture on the row so
    # a later quantity edit re-scales macros AND micros deterministically.
    per_100g = {
        "calories": product.calories, "protein_g": product.protein_g,
        "carbs_g": product.carbs_g, "fat_g": product.fat_g,
        "fibre_g": product.fibre_g, "sugar_g": product.sugar_g,
    }
    prod_micros = _safe_load(product.micros_json)
    if prod_micros:
        per_100g["micros"] = prod_micros
    scaled_micros = _scale_micros(prod_micros, factor) if prod_micros else None

    log = NutritionLog(
        logged_at=_now(),
        meal_type=meal_type,
        description=f"{product.display_name} ({portion:.0f}g)",
        calories=_scaled(product.calories),
        protein_g=_scaled(product.protein_g),
        carbs_g=_scaled(product.carbs_g),
        fat_g=_scaled(product.fat_g),
        fibre_g=_scaled(product.fibre_g),
        sugar_g=_scaled(product.sugar_g),
        is_estimate=is_estimate,
        confidence=confidence,
        source=source,
        quantity_g=portion,
        per_100g_json=json.dumps(per_100g),
        micros_json=json.dumps(scaled_micros) if scaled_micros else None,
        food_product_id=product.id,
        estimate_note=estimate_note,
    )
    db.add(log)
    product.times_logged = (product.times_logged or 0) + 1
    product.last_logged_at = _now()
    db.commit()
    db.refresh(log)
    return log


# =============================================================================
# Open Food Facts
# =============================================================================

def _http_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        try:
            import httpx
            with httpx.Client(timeout=_OFF_TIMEOUT_S, headers=_OFF_HEADERS) as client:
                response = client.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
        except ImportError:
            import requests
            response = requests.get(url, params=params, timeout=_OFF_TIMEOUT_S, headers=_OFF_HEADERS)
            if response.status_code == 200:
                return response.json()
    except Exception as exc:
        logger.warning("[products] OFF request failed: %s", exc)
    return None


def _parse_off_product(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract our shape from an OFF product dict."""
    if not payload:
        return None
    nutriments = payload.get("nutriments") or {}

    def _per100(key: str) -> Optional[float]:
        value = nutriments.get(f"{key}_100g")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    # Phase 3: headline micronutrients — only those OFF actually carries.
    micros: Dict[str, Any] = {}
    for our_key, off_key in (
        ("saturated_fat_g", "saturated-fat"),
        ("salt_g", "salt"),
        ("sodium_g", "sodium"),
        ("calcium_g", "calcium"),
        ("iron_g", "iron"),
        ("potassium_g", "potassium"),
    ):
        v = _per100(off_key)
        if v is not None:
            micros[our_key] = v
    _grade = payload.get("nutriscore_grade") or payload.get("nutrition_grades")
    if _grade:
        micros["nutri_score"] = str(_grade).lower()
    _nova = payload.get("nova_group")
    if _nova is not None:
        try:
            micros["nova_group"] = int(_nova)
        except (TypeError, ValueError):
            pass

    name = payload.get("product_name") or payload.get("product_name_en")
    if not name:
        return None
    serving_size_g = None
    serving_quantity = payload.get("serving_quantity")
    try:
        if serving_quantity:
            serving_size_g = float(serving_quantity)
    except (TypeError, ValueError):
        pass

    return {
        "display_name": str(name).strip(),
        "brand": (payload.get("brands") or "").split(",")[0].strip() or None,
        "barcode": payload.get("code"),
        "macros": {
            "calories": _per100("energy-kcal"),
            "protein_g": _per100("proteins"),
            "carbs_g": _per100("carbohydrates"),
            "fat_g": _per100("fat"),
            "fibre_g": _per100("fiber"),
            "sugar_g": _per100("sugars"),
        },
        "serving_size_g": serving_size_g,
        "serving_description": payload.get("serving_size"),
        "micros": micros or None,
    }


def off_lookup_barcode(barcode: str) -> Optional[Dict[str, Any]]:
    data = _http_get_json(_OFF_PRODUCT_URL.format(barcode=str(barcode).strip()))
    if not data or data.get("status") in (0, "0"):
        return None
    return _parse_off_product(data.get("product") or {})


def off_search_by_name(name: str, brand: Optional[str] = None) -> Optional[Dict[str, Any]]:
    terms = f"{brand} {name}".strip() if brand else name
    data = _http_get_json(_OFF_SEARCH_URL, params={
        "search_terms": terms,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 6,
        "countries_tags_en": "united-kingdom",
    })
    if not data:
        return None
    for raw in data.get("products") or []:
        parsed = _parse_off_product(raw)
        if parsed and parsed["macros"].get("calories") is not None:
            return parsed
    return None


# =============================================================================
# High-level resolve
# =============================================================================

def resolve_and_save(
    db: Session,
    *,
    name: Optional[str] = None,
    brand: Optional[str] = None,
    barcode: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    The one entry point capture flows should call.
    Local library → Open Food Facts → save → return.
    Returns {"product": FoodProduct, "found_via": str} or None.
    """
    ensure_product_table()

    if barcode:
        local = get_by_barcode(db, barcode)
        if local:
            return {"product": local, "found_via": "library_barcode"}
        parsed = off_lookup_barcode(barcode)
        if parsed:
            product = upsert_product(db, source="off_barcode", **parsed)
            return {"product": product, "found_via": "off_barcode"}

    if name:
        matches = search_products(db, name, limit=1)
        if matches:
            return {"product": matches[0], "found_via": "library_name"}
        parsed = off_search_by_name(name, brand)
        if parsed:
            product = upsert_product(db, source="off_search", **parsed)
            return {"product": product, "found_via": "off_search"}

    return None


def _brand_matches(requested: Optional[str], hit: Optional[str]) -> bool:
    """Loose brand match — any normalised-token overlap."""
    if not requested or not hit:
        return False
    a = set(normalise_name(requested).split())
    b = set(normalise_name(hit).split())
    return bool(a & b)


def _resolution(
    product: FoodProduct, grams: Optional[float], *, exact: bool,
    source: str, estimate_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Scale a resolved product to a portion + provenance for the log tool."""
    portion = float(grams) if grams else (product.serving_size_g or 100.0)
    factor = portion / 100.0

    def sc(v):
        return round(v * factor, 1) if v is not None else None

    per_100g = {
        "calories": product.calories, "protein_g": product.protein_g,
        "carbs_g": product.carbs_g, "fat_g": product.fat_g,
        "fibre_g": product.fibre_g, "sugar_g": product.sugar_g,
    }
    micros = _safe_load(product.micros_json)
    if micros:
        per_100g["micros"] = micros
    confidence = (0.9 if source.startswith("library") else 0.85) if exact else 0.6
    return {
        "found": True,
        "exact": exact,
        "source": source,
        "confidence": confidence,
        "is_estimate": not exact,
        "estimate_note": estimate_note,
        "display_name": product.display_name,
        "brand": product.brand,
        "barcode": product.barcode,
        "food_product_id": product.id,
        "quantity_g": round(portion, 1),
        "scaled": {
            "calories": sc(product.calories), "protein_g": sc(product.protein_g),
            "carbs_g": sc(product.carbs_g), "fat_g": sc(product.fat_g),
            "fibre_g": sc(product.fibre_g), "sugar_g": sc(product.sugar_g),
        },
        "per_100g": per_100g,
        "micros": _scale_micros(micros, factor) if micros else None,
    }


def resolve_nutrition(
    db: Session,
    *,
    name: Optional[str] = None,
    brand: Optional[str] = None,
    barcode: Optional[str] = None,
    grams: Optional[float] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    """Nutrition-estimate LADDER (Phase 4). Exhaust before asking:
      1. exact branded item: local library → OFF (barcode/name);
      2. retailer product-page scrape (a URL the model found via web_search);
      3. SIMILAR same-type item from OFF, FLAGGED as an estimate;
      4. give up → ask the user.
    Never substitutes a different food type. On a hit, carries scaled macros +
    per-100g basis + micros + provenance so the caller logs one rich, re-scalable
    row. Every network hop is best-effort and never blocks.
    """
    ensure_product_table()

    # 1a. Exact by barcode.
    if barcode:
        local = get_by_barcode(db, barcode)
        if local:
            return _resolution(local, grams, exact=True, source="library_barcode")
        parsed = off_lookup_barcode(barcode)
        if parsed:
            product = upsert_product(db, source="off_barcode", **parsed)
            return _resolution(product, grams, exact=True, source="off_barcode")

    # 1b. Exact by name in the local library.
    if name:
        matches = search_products(db, name, limit=1)
        if matches:
            return _resolution(matches[0], grams, exact=True, source="library_name")

    # 2. Retailer product-page scrape.
    if url:
        try:
            from app.lifestyle.retailer_scrape import scrape_retailer_nutrition
            scraped = scrape_retailer_nutrition(url)
        except Exception:
            scraped = None
        if scraped and (scraped.get("macros") or {}).get("calories") is not None:
            product = upsert_product(db, source="retailer_scrape", **scraped)
            return _resolution(product, grams, exact=True, source="retailer_scrape")

    # 3. OFF name search. Exact ONLY when the brand is verified (a matching
    #    brand). With no brand, or a different brand, it's an ESTIMATE — flagged,
    #    never claimed as the exact item.
    if name:
        parsed = off_search_by_name(name, brand)
        if parsed and (parsed.get("macros") or {}).get("calories") is not None:
            hit_brand = parsed.get("brand")
            if brand and _brand_matches(brand, hit_brand):
                product = upsert_product(db, source="off_search", **parsed)
                return _resolution(product, grams, exact=True, source="off_search")
            # Unverified identity (no brand) or a different brand → flagged estimate.
            src = "similar_estimate" if brand else "off_search"
            note = None
            if brand:  # a real substitute of a different brand
                note = "estimated from similar item: " + str(parsed.get("display_name") or "?")
                if hit_brand:
                    note += f" ({hit_brand})"
            product = upsert_product(db, source=src, **parsed)
            return _resolution(product, grams, exact=False, source=src, estimate_note=note)

    # 4. Nothing found — ask the user (never substitute a different food).
    return {
        "found": False,
        "ask_user": True,
        "query": name or barcode or url,
        "message": (
            f"Couldn't find '{name or barcode}' or a close same-type item online. "
            "Ask the user for the macros (calories/protein/carbs/fat), or for a "
            "clearer photo of the nutrition label."
        ),
    }


def product_to_dict(product: FoodProduct) -> Dict[str, Any]:
    return {
        "id": product.id,
        "name": product.display_name,
        "brand": product.brand,
        "barcode": product.barcode,
        "per_100g": {
            "calories": product.calories,
            "protein_g": product.protein_g,
            "carbs_g": product.carbs_g,
            "fat_g": product.fat_g,
            "fibre_g": product.fibre_g,
            "sugar_g": product.sugar_g,
        },
        "serving_size_g": product.serving_size_g,
        "serving_description": product.serving_description,
        "micros": _safe_load(product.micros_json),
        "source": product.source,
        "times_logged": product.times_logged,
        "last_logged_at": product.last_logged_at.isoformat() if product.last_logged_at else None,
    }
