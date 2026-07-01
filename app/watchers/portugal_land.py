# FILE: app/watchers/portugal_land.py
# Purpose: Watcher instance — Portugal land prices, EUR/m2, three regions, daily.
# Called-by: app.watchers.framework._ensure_instances_loaded (import registers)
# Depends-on: app.watchers.framework, app.watchers.snippet_source
# Last-renovated: 2026-07-01
"""
Region config comes from ASTRA_LAND_REGIONS (JSON) — the in-code defaults are
PLACEHOLDERS (Central / Coastal / Interior) until Taz supplies exact areas.
Each region records one €/m2 row per day into the watcher ledger.
"""

from __future__ import annotations

import json
import logging
import os

from app.watchers.framework import Reading, WatcherSpec, register_watcher
from app.watchers.snippet_source import observe_via_search

logger = logging.getLogger(__name__)

WATCHER_ID = "portugal_land"

# Placeholder regions — override with ASTRA_LAND_REGIONS in .env.
_DEFAULT_REGIONS = [
    {
        "key": "central",
        "label": "Central Portugal",
        "query": "preço médio terreno Portugal Centro euros por metro quadrado 2026",
    },
    {
        "key": "coastal",
        "label": "Coastal Portugal (Silver Coast)",
        "query": "preço médio terreno Costa de Prata Portugal euros por m2 2026",
    },
    {
        "key": "interior",
        "label": "Interior Portugal",
        "query": "preço médio terreno interior Portugal euros por m2 2026",
    },
]


def _regions() -> list:
    raw = os.getenv("ASTRA_LAND_REGIONS", "").strip()
    if raw:
        try:
            regions = json.loads(raw)
            if isinstance(regions, list) and regions:
                return regions
        except Exception as exc:
            logger.warning("[watchers.portugal_land] bad ASTRA_LAND_REGIONS JSON (%s); using placeholders", exc)
    return _DEFAULT_REGIONS


def _bounds() -> tuple:
    try:
        lo = float(os.getenv("ASTRA_LAND_PRICE_MIN", "0.5"))
    except Exception:
        lo = 0.5
    try:
        hi = float(os.getenv("ASTRA_LAND_PRICE_MAX", "1500"))
    except Exception:
        hi = 1500.0
    return lo, hi


async def _observe_region(key_cfg: dict) -> Reading:
    lo, hi = _bounds()
    return await observe_via_search(
        key=key_cfg["key"],
        query=key_cfg.get("query") or f"preço terreno {key_cfg.get('label', key_cfg['key'])} Portugal €/m2",
        currency="€",
        lo=lo,
        hi=hi,
    )


def _cadence_hours() -> float:
    try:
        return float(os.getenv("ASTRA_LAND_WATCH_CADENCE_HOURS", "24"))
    except Exception:
        return 24.0


register_watcher(WatcherSpec(
    watcher_id=WATCHER_ID,
    title="Portugal land prices",
    unit="eur_per_m2",
    keys=_regions(),
    observe_key=_observe_region,
    tool_name="get_portugal_land_prices",
    tool_description=(
        "Read the Portugal land-price watcher ledger: average EUR per square "
        "metre per tracked region (central/coastal/interior placeholders until "
        "exact areas are configured), one reading per region per day, with "
        "latest values and percentage change over the window. Use when the "
        "user asks how the Portugal land prices are looking, land price "
        "trends, or whether land got cheaper. Set summarise=true for a short "
        "spoken trend summary."
    ),
    cadence_hours=_cadence_hours(),
    currency_symbol="€",
))
