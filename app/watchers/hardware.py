# FILE: app/watchers/hardware.py
# Purpose: Watcher instance — 3090-successor build parts (RTX 5090, TR boards, DDR5, PSUs), GBP, daily.
# Called-by: app.watchers.framework._ensure_instances_loaded (import registers)
# Depends-on: app.watchers.framework, app.watchers.snippet_source
# Last-renovated: 2026-07-01
"""
Used and new prices, one row per item class per day. Item list overridable
via ASTRA_HARDWARE_ITEMS (JSON) — per-class sanity bounds keep cable-money
noise out of GPU medians.
"""

from __future__ import annotations

import json
import logging
import os

from app.watchers.framework import Reading, WatcherSpec, register_watcher
from app.watchers.snippet_source import observe_via_search

logger = logging.getLogger(__name__)

WATCHER_ID = "hardware"

_DEFAULT_ITEMS = [
    {"key": "rtx_5090_new", "label": "RTX 5090 (new)", "query": "RTX 5090 price UK buy", "lo": 1200, "hi": 5000},
    {"key": "rtx_5090_used", "label": "RTX 5090 (used)", "query": "RTX 5090 used price eBay UK", "lo": 800, "hi": 4000},
    {"key": "tr_motherboard_new", "label": "Threadripper motherboard (new)", "query": "TRX50 Threadripper motherboard price UK", "lo": 400, "hi": 1800},
    {"key": "tr_motherboard_used", "label": "Threadripper motherboard (used)", "query": "TRX50 motherboard used price eBay UK", "lo": 250, "hi": 1400},
    {"key": "ddr5_kit_new", "label": "DDR5 128GB kit (new)", "query": "DDR5 128GB kit price UK", "lo": 150, "hi": 1500},
    {"key": "ddr5_kit_used", "label": "DDR5 128GB kit (used)", "query": "DDR5 128GB used price eBay UK", "lo": 100, "hi": 1200},
    {"key": "psu_1600w_new", "label": "1600W PSU (new)", "query": "1600W ATX PSU price UK", "lo": 150, "hi": 900},
    {"key": "psu_1600w_used", "label": "1600W PSU (used)", "query": "1600W PSU used price eBay UK", "lo": 80, "hi": 700},
]


def _items() -> list:
    raw = os.getenv("ASTRA_HARDWARE_ITEMS", "").strip()
    if raw:
        try:
            items = json.loads(raw)
            if isinstance(items, list) and items:
                return items
        except Exception as exc:
            logger.warning("[watchers.hardware] bad ASTRA_HARDWARE_ITEMS JSON (%s); using defaults", exc)
    return _DEFAULT_ITEMS


async def _observe_item(key_cfg: dict) -> Reading:
    return await observe_via_search(
        key=key_cfg["key"],
        query=key_cfg.get("query") or f"{key_cfg.get('label', key_cfg['key'])} price UK",
        currency="£",
        lo=float(key_cfg.get("lo", 0)),
        hi=float(key_cfg.get("hi", 10000)),
    )


def _cadence_hours() -> float:
    try:
        return float(os.getenv("ASTRA_HARDWARE_WATCH_CADENCE_HOURS", "24"))
    except Exception:
        return 24.0


register_watcher(WatcherSpec(
    watcher_id=WATCHER_ID,
    title="Hardware prices (3090-successor build)",
    unit="gbp",
    keys=_items(),
    observe_key=_observe_item,
    tool_name="get_hardware_prices",
    tool_description=(
        "Read the hardware price watcher ledger: RTX 5090, Threadripper-class "
        "motherboards, DDR5 RAM kits and high-wattage PSUs — used and new, in "
        "GBP, one reading per item class per day, with latest values and "
        "percentage change over the window. Use when the user asks about GPU "
        "or PC part prices, whether the 5090 has come down, or the build "
        "budget. Set summarise=true for a short spoken trend summary."
    ),
    cadence_hours=_cadence_hours(),
    currency_symbol="£",
))
