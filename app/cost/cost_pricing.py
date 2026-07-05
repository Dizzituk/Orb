# FILE: app/cost/cost_pricing.py
# Purpose: Model pricing table for ASTRA cost tracking — loaded from data/model_pricing.json.
# Called-by: app.cost.cost_recorder, app.endpoints.cost_dashboard
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-07-04 (Derek phase 1: pricing moved to data file, unknown models price to None)
#
# LANE D EXEMPTION (2026-07-02): the literal model IDs in the pricing data are
# pricing-table KEYS — they LABEL prices for known models, they never SELECT a
# model for routing. An unknown model prices to None (recorded as cost_usd=null
# with a WARN), never a silent 0.00. They are therefore exempt from the
# no-hardcoded-models rule and whitelisted in tests/test_no_hardcoded_models.py.
# Model SELECTION literals remain banned everywhere.
"""
Model pricing table for ASTRA cost tracking.

Rates live in data/model_pricing.json (override the path with
ASTRA_MODEL_PRICING_PATH). USD per 1M tokens, input and output.
Per-model env overrides still apply on top of the file:
ORB_PRICE_{MODEL_KEY}_INPUT / ORB_PRICE_{MODEL_KEY}_OUTPUT.

Unknown model => calculate_cost() returns None (caller records cost_usd=null)
and a WARN is logged once per model per process. Local models are listed in
the data file at 0.0 so their token counts are still recorded.

Usage:
    from app.cost.cost_pricing import calculate_cost, get_model_price

    cost = calculate_cost("claude-opus-4-8", prompt_tokens=5000, completion_tokens=2000)
    # -> USD float, or None if the model is not in the pricing table
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """Per-token pricing for a model (USD per 1M tokens)."""
    input_per_million: float
    output_per_million: float
    note: str = ""


# =========================================================================
# Fallback pricing table (used ONLY if the data file is missing/corrupt)
# =========================================================================
# The canonical table is data/model_pricing.json. This fallback keeps cost
# tracking alive if the data file vanishes; a WARN is logged when it is used.

_FALLBACK_PRICING: Dict[str, ModelPrice] = {
    "claude-opus-4-6": ModelPrice(15.0, 75.0),
    "claude-opus-4-5-20251101": ModelPrice(5.0, 25.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-sonnet-4-5-20250929": ModelPrice(3.0, 15.0),
    "gpt-5.4": ModelPrice(2.50, 15.0),
    "gpt-5.4-pro": ModelPrice(5.0, 30.0),
    "gpt-5.2": ModelPrice(1.75, 14.0),
    "gpt-5.2-pro": ModelPrice(21.0, 168.0),
    "gpt-5-mini": ModelPrice(0.15, 0.60),
    "gpt-4.1": ModelPrice(2.0, 8.0),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60),
    "gemini-3-pro-preview": ModelPrice(2.0, 12.0),
    "gemini-3-flash": ModelPrice(0.50, 3.0),
    "gemini-2.5-pro": ModelPrice(1.25, 5.0),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50),
    "gemini-2.5-flash-lite": ModelPrice(0.10, 0.40),
    "text-embedding-3-small": ModelPrice(0.02, 0.0),
    "gemini-embedding-2-preview": ModelPrice(0.20, 0.0),
}

# Kept as a name for backward compatibility (tests/dashboards may import it).
DEFAULT_PRICING = _FALLBACK_PRICING


def _pricing_file_path() -> Path:
    """Path to the pricing data file (env-overridable)."""
    override = os.getenv("ASTRA_MODEL_PRICING_PATH", "").strip()
    if override:
        return Path(override)
    # app/cost/cost_pricing.py -> repo root -> data/model_pricing.json
    return Path(__file__).resolve().parents[2] / "data" / "model_pricing.json"


def _load_from_file() -> Optional[Dict[str, ModelPrice]]:
    """Load the pricing table from the data file. None on any failure."""
    path = _pricing_file_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        models = raw.get("models")
        if not isinstance(models, dict) or not models:
            logger.warning("[cost_pricing] %s has no usable 'models' block", path)
            return None
        table: Dict[str, ModelPrice] = {}
        for model_id, entry in models.items():
            try:
                table[model_id] = ModelPrice(
                    input_per_million=float(entry["input_per_million"]),
                    output_per_million=float(entry["output_per_million"]),
                    note=str(entry.get("note", "")),
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("[cost_pricing] Skipping bad entry %r: %s", model_id, exc)
        return table or None
    except FileNotFoundError:
        logger.warning("[cost_pricing] Pricing file missing: %s — using built-in fallback", path)
        return None
    except Exception as exc:
        logger.warning("[cost_pricing] Failed to load %s (%s) — using built-in fallback", path, exc)
        return None


def _apply_env_overrides(pricing: Dict[str, ModelPrice]) -> Dict[str, ModelPrice]:
    """
    Apply per-model env overrides on top of the loaded table.

    Env pattern: ORB_PRICE_{MODEL_KEY}_INPUT and ORB_PRICE_{MODEL_KEY}_OUTPUT
    where MODEL_KEY is the model ID uppercased with dots/hyphens replaced by
    underscores, e.g. ORB_PRICE_CLAUDE_OPUS_4_8_INPUT=5.0.
    """
    for model_id in list(pricing.keys()):
        env_key = model_id.upper().replace("-", "_").replace(".", "_")
        input_env = os.getenv(f"ORB_PRICE_{env_key}_INPUT")
        output_env = os.getenv(f"ORB_PRICE_{env_key}_OUTPUT")
        if input_env or output_env:
            try:
                inp = float(input_env) if input_env else pricing[model_id].input_per_million
                out = float(output_env) if output_env else pricing[model_id].output_per_million
                pricing[model_id] = ModelPrice(inp, out, note="env override")
                logger.info("[cost_pricing] Override %s: input=%.2f, output=%.2f", model_id, inp, out)
            except ValueError:
                pass
    return pricing


def _load_pricing() -> Dict[str, ModelPrice]:
    """Load pricing: data file first, built-in fallback second, env on top."""
    table = _load_from_file()
    if table is None:
        table = dict(_FALLBACK_PRICING)
    return _apply_env_overrides(table)


# Module-level pricing cache (loaded once; reload_pricing() clears it)
_pricing: Optional[Dict[str, ModelPrice]] = None
_warned_unknown: Set[str] = set()


def _get_pricing() -> Dict[str, ModelPrice]:
    """Get or initialize the pricing table."""
    global _pricing
    if _pricing is None:
        _pricing = _load_pricing()
    return _pricing


def reload_pricing() -> int:
    """Clear the cache and reload the table. Returns the model count."""
    global _pricing
    _pricing = None
    _warned_unknown.clear()
    return len(_get_pricing())


# =========================================================================
# Public API
# =========================================================================

def get_model_price(model_id: str) -> Optional[ModelPrice]:
    """
    Get pricing for a model.

    Exact match first, then the LONGEST fuzzy substring match (so
    'gpt-5.4-mini-2026-01' matches 'gpt-5.4-mini', not 'gpt-5.4').
    Returns None if no entry matches.
    """
    pricing = _get_pricing()

    if model_id in pricing:
        return pricing[model_id]

    model_lower = model_id.lower().strip()
    best_key: Optional[str] = None
    for key in pricing:
        key_lower = key.lower()
        if key_lower in model_lower or model_lower in key_lower:
            if best_key is None or len(key) > len(best_key):
                best_key = key
    return pricing[best_key] if best_key else None


def calculate_cost(
    model_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> Optional[float]:
    """
    Calculate the cost of an LLM call in USD.

    Returns None if the model is not in the pricing table — the caller must
    record cost_usd=null, never fabricate 0.00. WARNs once per model.
    """
    price = get_model_price(model_id)
    if price is None:
        if model_id not in _warned_unknown:
            _warned_unknown.add(model_id)
            logger.warning(
                "[cost_pricing] No pricing entry for model %r — recording cost=null. "
                "Add it to %s.", model_id, _pricing_file_path(),
            )
        return None

    input_cost = (prompt_tokens / 1_000_000) * price.input_per_million
    output_cost = (completion_tokens / 1_000_000) * price.output_per_million
    return round(input_cost + output_cost, 6)


def get_all_pricing() -> Dict[str, ModelPrice]:
    """Get the full pricing table (for dashboard display)."""
    return dict(_get_pricing())


def get_pricing_source() -> Dict[str, str]:
    """Where the active table came from (for the dashboard/introspection)."""
    path = _pricing_file_path()
    return {
        "path": str(path),
        "source": "file" if path.exists() else "built-in fallback",
        "override_env": "ASTRA_MODEL_PRICING_PATH",
    }


__all__ = [
    "ModelPrice",
    "calculate_cost",
    "get_model_price",
    "get_all_pricing",
    "get_pricing_source",
    "reload_pricing",
    "DEFAULT_PRICING",
]
