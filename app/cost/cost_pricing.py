# FILE: app/overwatcher/cost_pricing.py
"""
Model pricing table for ASTRA cost tracking.

Maps model IDs to per-token costs (USD per 1M tokens, input and output).
Prices are loaded from environment variables with sensible defaults.

Pricing is approximate and should be updated when providers change rates.
All costs are in USD — currency conversion to GBP happens at the budget
enforcement layer (cost_budget.py).

Usage:
    from app.cost.cost_pricing import calculate_cost, get_model_price

    cost = calculate_cost("claude-opus-4-6", prompt_tokens=5000, completion_tokens=2000)
    # Returns cost in USD
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPrice:
    """Per-token pricing for a model (USD per 1M tokens)."""
    input_per_million: float
    output_per_million: float


# =========================================================================
# Default pricing table (USD per 1M tokens)
# =========================================================================
# Updated: February 2026. Adjust as providers change rates.
# These are defaults — override via env vars if needed.

DEFAULT_PRICING: Dict[str, ModelPrice] = {
    # Anthropic (from .env reference: opus $5/$25, sonnet $3/$15)
    "claude-opus-4-6": ModelPrice(input_per_million=15.0, output_per_million=75.0),
    "claude-opus-4-5-20251101": ModelPrice(input_per_million=5.0, output_per_million=25.0),
    "claude-sonnet-4-6": ModelPrice(input_per_million=3.0, output_per_million=15.0),
    "claude-sonnet-4-5-20250929": ModelPrice(input_per_million=3.0, output_per_million=15.0),

    # OpenAI (from .env reference: gpt-5.2 $1.75/$14, gpt-5-mini ~$0.15/$0.60)
    "gpt-5.4": ModelPrice(input_per_million=2.50, output_per_million=15.0),
    "gpt-5.4-pro": ModelPrice(input_per_million=5.0, output_per_million=30.0),
    "gpt-5.2": ModelPrice(input_per_million=1.75, output_per_million=14.0),
    "gpt-5.2-pro": ModelPrice(input_per_million=21.0, output_per_million=168.0),
    "gpt-5-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
    "gpt-4.1": ModelPrice(input_per_million=2.0, output_per_million=8.0),
    "gpt-4.1-mini": ModelPrice(input_per_million=0.40, output_per_million=1.60),

    # Google (from .env reference: flash-lite $0.10/$0.40, pro-preview $2/$12)
    "gemini-3-pro-preview": ModelPrice(input_per_million=2.0, output_per_million=12.0),
    "gemini-3-flash": ModelPrice(input_per_million=0.50, output_per_million=3.0),
    "gemini-2.5-pro": ModelPrice(input_per_million=1.25, output_per_million=5.0),
    "gemini-2.5-flash": ModelPrice(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash-lite": ModelPrice(input_per_million=0.10, output_per_million=0.40),

    # Embeddings
    "text-embedding-3-small": ModelPrice(input_per_million=0.02, output_per_million=0.0),
    "gemini-embedding-2-preview": ModelPrice(input_per_million=0.20, output_per_million=0.0),
}


# =========================================================================
# Env override loading
# =========================================================================

def _load_pricing() -> Dict[str, ModelPrice]:
    """
    Load pricing table with env overrides.

    Env pattern: ORB_PRICE_{MODEL_KEY}_INPUT and ORB_PRICE_{MODEL_KEY}_OUTPUT
    where MODEL_KEY is the model ID uppercased with dots/hyphens replaced by underscores.

    Example:
        ORB_PRICE_CLAUDE_OPUS_4_6_INPUT=15.0
        ORB_PRICE_CLAUDE_OPUS_4_6_OUTPUT=75.0
    """
    pricing = dict(DEFAULT_PRICING)

    for model_id in list(pricing.keys()):
        env_key = model_id.upper().replace("-", "_").replace(".", "_")
        input_env = os.getenv(f"ORB_PRICE_{env_key}_INPUT")
        output_env = os.getenv(f"ORB_PRICE_{env_key}_OUTPUT")

        if input_env or output_env:
            try:
                inp = float(input_env) if input_env else pricing[model_id].input_per_million
                out = float(output_env) if output_env else pricing[model_id].output_per_million
                pricing[model_id] = ModelPrice(input_per_million=inp, output_per_million=out)
                logger.info("[cost_pricing] Override %s: input=%.2f, output=%.2f", model_id, inp, out)
            except ValueError:
                pass

    return pricing


# Module-level pricing (loaded once)
_pricing: Optional[Dict[str, ModelPrice]] = None


def _get_pricing() -> Dict[str, ModelPrice]:
    """Get or initialize the pricing table."""
    global _pricing
    if _pricing is None:
        _pricing = _load_pricing()
    return _pricing


# =========================================================================
# Public API
# =========================================================================

def get_model_price(model_id: str) -> Optional[ModelPrice]:
    """
    Get pricing for a model.

    Returns None if model is not in the pricing table (e.g. local models).
    """
    pricing = _get_pricing()

    # Exact match first
    if model_id in pricing:
        return pricing[model_id]

    # Fuzzy match: strip provider prefixes, try partial match
    model_lower = model_id.lower().strip()
    for key, price in pricing.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return price

    return None


def calculate_cost(
    model_id: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> float:
    """
    Calculate the cost of an LLM call in USD.

    Returns 0.0 if model is not in pricing table (local models, unknown models).
    """
    price = get_model_price(model_id)
    if price is None:
        return 0.0

    input_cost = (prompt_tokens / 1_000_000) * price.input_per_million
    output_cost = (completion_tokens / 1_000_000) * price.output_per_million
    return round(input_cost + output_cost, 6)


def get_all_pricing() -> Dict[str, ModelPrice]:
    """Get the full pricing table (for dashboard display)."""
    return dict(_get_pricing())


__all__ = [
    "ModelPrice",
    "calculate_cost",
    "get_model_price",
    "get_all_pricing",
    "DEFAULT_PRICING",
]
