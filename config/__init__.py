# FILE: config/__init__.py
# Purpose: Configuration package for Orb/ASTRA.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: config.model_ranks
# Last-renovated: 2026-06-11
"""Configuration package for Orb/ASTRA.

Contains:
- model_ranks.py: Model capability rank system (Spec v2.3 §3.1.1)
"""

from config.model_ranks import (
    MODEL_CAPABILITY_RANKS,
    get_capability_rank,
    is_fallback_allowed,
    get_rank_name,
    validate_model_for_stage,
    STAGE_MODEL_CONFIG,
    get_stage_models,
)

__all__ = [
    "MODEL_CAPABILITY_RANKS",
    "get_capability_rank",
    "is_fallback_allowed",
    "get_rank_name",
    "validate_model_for_stage",
    "STAGE_MODEL_CONFIG",
    "get_stage_models",
]
