# FILE: config/__init__.py
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
