# FILE: app/lifestyle/retailer_scrape.py
# Purpose: Best-effort nutrition scrape of a retailer product page (schema.org JSON-LD first).
# Called-by: app.lifestyle.product_service (resolve_nutrition ladder, tier 3)
# Depends-on: stdlib + httpx/requests
# Last-renovated: 2026-06-14
"""
Retailer product-page nutrition scraper (food overhaul Phase 4).

When the exact branded item isn't in the local library or Open Food Facts, the
resolver can scrape a retailer product page the model found via web_search.
Most UK grocers (Tesco, Sainsbury's, Aldi, Ocado...) embed schema.org
NutritionInformation as JSON-LD, so a deterministic JSON-LD walk gets the panel
without a headless browser. Falls back to __NEXT_DATA__.

Discipline: HONOURS robots.txt for our UA, rate-limits per host, short timeout,
size cap, and is fully best-effort — any failure returns None and the caller
falls through to the similar-item / ask tiers. It never raises.
"""
from __future__ import annotations

import json
import logging
import re
import time
from html import unescape
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
_TIMEOUT_S = 8.0
_MAX_BYTES = 3_000_000
_MIN_GAP_S = 1.5  # per-host courtesy rate-limit

_last_fetch: Dict[str, float] = {}
_robots_cache: Dict[str, Optional[RobotFileParser]] = {}


def _num(value: Any) -> Optional[float]:
    """Leading number out of '250 kcal' / '10.5 g' / '1,234' / 250 -> float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"-?\d[\d,]*\.?\d*", str(value).replace(",", ""))
    try:
        return float(m.group(0)) if m else None
    except (TypeError, ValueError):
        return None


def _robots_ok(url: str) -> bool:
    """True if robots.txt allows our UA to fetch `url`. Fail-open on error
    EXCEPT an explicit disallow."""
    try:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        if host not in _robots_cache:
            rp = RobotFileParser()
            rp.set_url(f"{host}/robots.txt")
            try:
                rp.read()
                _robots_cache[host] = rp
            except Exception:
                _robots_cache[host] = None  # couldn't read → don't block
        rp = _robots_cache[host]
        return rp.can_fetch(_UA, url) if rp else True
    except Exception:
        return True


def _rate_limit(url: str) -> None:
    host = urlparse(url).netloc
    last = _last_fetch.get(host)
    if last is not None:
        wait = _MIN_GAP_S - (time.time() - last)
        if wait > 0:
            time.sleep(min(wait, _MIN_GAP_S))
    _last_fetch[host] = time.time()


def _fetch(url: str) -> Optional[str]:
    headers = {"User-Agent": _UA, "Accept-Language": "en-GB,en;q=0.9"}
    try:
        try:
            import httpx
            with httpx.Client(timeout=_TIMEOUT_S, headers=headers, follow_redirects=True) as client:
                r = client.get(url)
                if r.status_code == 200:
                    return r.text[:_MAX_BYTES]
        except ImportError:
            import requests
            r = requests.get(url, headers=headers, timeout=_TIMEOUT_S)
            if r.status_code == 200:
                return r.text[:_MAX_BYTES]
    except Exception as exc:
        logger.debug("[retailer_scrape] fetch failed for %s: %s", url, exc)
    return None


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _type_is(node: Dict[str, Any], wanted: str) -> bool:
    t = node.get("@type")
    if isinstance(t, list):
        return any(str(x).lower() == wanted for x in t)
    return str(t).lower() == wanted if t else False


# schema.org NutritionInformation field -> our per-100g key (best-effort; values
# are per the page's serving and re-based onto 100g below when a serving is known).
_NUTR_MACROS = {
    "calories": "calories",
    "proteinContent": "protein_g",
    "carbohydrateContent": "carbs_g",
    "fatContent": "fat_g",
    "fiberContent": "fibre_g",
    "sugarContent": "sugar_g",
}
_NUTR_MICROS = {
    "saturatedFatContent": "saturated_fat_g",
    "sodiumContent": "sodium_g",
    "saltContent": "salt_g",
}


def _grams_from_serving(value: Any) -> Optional[float]:
    """Grams from a serving string ONLY when a number is immediately followed by
    g/ml (e.g. '30g', '1/2 pack (200g)' -> 200). Ignores counts like '2 biscuits'
    so we never derive a bogus factor from a leading count."""
    if value is None:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:g|ml)\b", str(value), re.IGNORECASE)
    try:
        return float(m.group(1)) if m else None
    except (TypeError, ValueError):
        return None


def _parse_nutrition_node(nutr: Dict[str, Any], product_name: str, brand: Optional[str]) -> Optional[Dict[str, Any]]:
    macros = {our: _num(nutr.get(k)) for k, our in _NUTR_MACROS.items()}
    if macros.get("calories") is None:
        return None
    micros = {our: _num(nutr.get(k)) for k, our in _NUTR_MICROS.items()}
    micros = {k: v for k, v in micros.items() if v is not None}

    # schema.org doesn't say whether the panel is per-serving or per-100g, and
    # servingSize is free text. Only re-base to per-100g when (a) we can read a
    # real GRAM serving (number + g/ml, not a count like '2 biscuits') AND (b)
    # the re-based energy is physically plausible (<= 900 kcal/100g). If the
    # panel is already per-100g, re-basing would blow past that ceiling, so we
    # keep the values as-is instead of corrupting them.
    serving_size_g = _grams_from_serving(nutr.get("servingSize"))
    if serving_size_g and serving_size_g > 0:
        factor = 100.0 / serving_size_g
        rebased_calories = (macros.get("calories") or 0.0) * factor
        if rebased_calories <= 900.0:
            macros = {k: (round(v * factor, 1) if v is not None else None) for k, v in macros.items()}
            micros = {k: round(v * factor, 2) for k, v in micros.items()}
        # else: panel was already per-100g — leave macros/micros untouched.

    return {
        "display_name": product_name.strip(),
        "brand": brand,
        "macros": macros,
        "micros": micros or None,
        "serving_size_g": serving_size_g,
        "serving_description": str(nutr.get("servingSize")) if nutr.get("servingSize") else None,
    }


def _from_json_ld(html: str) -> Optional[Dict[str, Any]]:
    for raw in re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S | re.I):
        try:
            payload = json.loads(unescape(raw.strip()))
        except Exception:
            continue
        # Prefer a Product node that carries nutrition.
        for node in _walk(payload):
            if not isinstance(node, dict):
                continue
            nutr = node.get("nutrition")
            if isinstance(nutr, dict) and _type_is(nutr, "nutritioninformation"):
                name = node.get("name") or "product"
                brand = node.get("brand")
                if isinstance(brand, dict):
                    brand = brand.get("name")
                parsed = _parse_nutrition_node(nutr, str(name), str(brand) if brand else None)
                if parsed:
                    return parsed
        # Or a bare NutritionInformation node.
        for node in _walk(payload):
            if isinstance(node, dict) and _type_is(node, "nutritioninformation"):
                parsed = _parse_nutrition_node(node, "product", None)
                if parsed:
                    return parsed
    return None


def _from_next_data(html: str) -> Optional[Dict[str, Any]]:
    m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S | re.I)
    if not m:
        return None
    try:
        payload = json.loads(unescape(m.group(1)))
    except Exception:
        return None
    for node in _walk(payload):
        if isinstance(node, dict) and _type_is(node, "nutritioninformation"):
            parsed = _parse_nutrition_node(node, str(node.get("name") or "product"), None)
            if parsed:
                return parsed
    return None


def scrape_retailer_nutrition(url: str) -> Optional[Dict[str, Any]]:
    """Scrape a retailer product page for nutrition. Returns the same shape as
    product_service._parse_off_product (display_name/macros/micros/serving) or
    None. Best-effort and never raises.
    """
    try:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            return None
        if not _robots_ok(url):
            logger.info("[retailer_scrape] robots.txt disallows %s", url)
            return None
        _rate_limit(url)
        html = _fetch(url)
        if not html:
            return None
        return _from_json_ld(html) or _from_next_data(html)
    except Exception as exc:
        logger.debug("[retailer_scrape] scrape failed for %s: %s", url, exc)
        return None


__all__ = ["scrape_retailer_nutrition"]
