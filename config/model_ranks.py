# FILE: config/model_ranks.py
"""Model capability rank system - Single source of truth.

Spec v2.3 §3.1.1: Capability ranks MUST be derived from a configuration
mapping, not string matching.

Rank levels:
  - frontier (3): GPT-5.2 Pro, Claude Opus, Gemini 3 Pro
  - pro (2): GPT-5.2, Claude Sonnet, Gemini 2.5 Pro
  - fast (1): GPT-5 mini, Claude Haiku, Gemini 2 Flash

Rule: Any new model must be added to this mapping before use in high-stakes
stages. Unknown models (rank 0) cannot be used as primaries or fallbacks
for high-stakes.
"""

from __future__ import annotations

from typing import Dict, Tuple

# =============================================================================
# Capability Rank Mapping (Authoritative Source)
# =============================================================================

MODEL_CAPABILITY_RANKS: Dict[str, int] = {
    # -------------------------------------------------------------------------
    # Frontier (rank 3) - Highest capability
    # -------------------------------------------------------------------------
    # OpenAI
    "gpt-5.2-pro": 3,
    "gpt-5.2-pro-2025-12-11": 3,
    # Anthropic
    "claude-opus-4-5-20251101": 3,
    "claude-opus-4-20250514": 3,
    # Google
    "gemini-3-pro-preview": 3,
    "gemini-3-pro": 3,
    
    # -------------------------------------------------------------------------
    # Pro (rank 2) - High capability
    # -------------------------------------------------------------------------
    # OpenAI
    "gpt-5.2": 2,
    "gpt-5.2-chat-latest": 2,
    "gpt-5.2-thinking": 2,
    "gpt-5.1": 2,
    "gpt-5.1-chat-latest": 2,
    "gpt-5": 2,
    "gpt-5-chat-latest": 2,
    # Anthropic
    "claude-sonnet-4-5-20250514": 2,
    "claude-sonnet-4-20250514": 2,
    # Google
    "gemini-2.5-pro": 2,
    "gemini-2.5-pro-preview-05-06": 2,
    
    # -------------------------------------------------------------------------
    # Fast (rank 1) - Lower capability, higher speed
    # -------------------------------------------------------------------------
    # OpenAI
    "gpt-5-mini": 1,
    "gpt-5-nano": 1,
    "gpt-4.1-mini": 1,
    "gpt-4.1-nano": 1,
    # Anthropic
    "claude-haiku-3-5-20241022": 1,
    # Google
    "gemini-2.0-flash": 1,
    "gemini-2-flash": 1,
}


def get_capability_rank(model_id: str) -> int:
    """Returns rank (1-3) or 0 if unknown.
    
    Args:
        model_id: Model identifier string
        
    Returns:
        3 = frontier, 2 = pro, 1 = fast, 0 = unknown
    """
    return MODEL_CAPABILITY_RANKS.get(model_id, 0)


def is_fallback_allowed(
    primary_model: str,
    fallback_model: str,
    is_high_stakes: bool,
) -> bool:
    """Check if fallback is allowed based on capability ranks.
    
    Spec v2.3 §3.2: Fallbacks are only permitted when
    rank(fallback) >= rank(primary) for high-stakes jobs.
    
    Args:
        primary_model: Primary model ID
        fallback_model: Fallback model ID
        is_high_stakes: Whether this is a high-stakes job
        
    Returns:
        True if fallback is allowed
    """
    if not is_high_stakes:
        return True  # Non-high-stakes can fall back to any available model
    
    primary_rank = get_capability_rank(primary_model)
    fallback_rank = get_capability_rank(fallback_model)
    
    # Unknown models (rank 0) cannot be used for high-stakes
    if primary_rank == 0 or fallback_rank == 0:
        return False
    
    return fallback_rank >= primary_rank  # Must be equal or stronger


def get_rank_name(rank: int) -> str:
    """Get human-readable rank name."""
    return {3: "frontier", 2: "pro", 1: "fast", 0: "unknown"}.get(rank, "unknown")


def validate_model_for_stage(
    model_id: str,
    stage_name: str,
    min_rank: int = 1,
) -> Tuple[bool, str]:
    """Validate that a model meets minimum rank for a stage.
    
    Args:
        model_id: Model to validate
        stage_name: Name of the stage (for error messages)
        min_rank: Minimum required rank (default 1)
        
    Returns:
        (is_valid, error_message)
    """
    rank = get_capability_rank(model_id)
    
    if rank == 0:
        return False, f"Unknown model '{model_id}' cannot be used for {stage_name}. Add to MODEL_CAPABILITY_RANKS first."
    
    if rank < min_rank:
        return False, f"Model '{model_id}' (rank {get_rank_name(rank)}) does not meet minimum rank {get_rank_name(min_rank)} for {stage_name}."
    
    return True, ""


# =============================================================================
# Stage Model Assignments (Spec v2.3 §3.3)
# =============================================================================

STAGE_MODEL_CONFIG = {
    # Block 2: Spec Gate
    "spec_gate": {
        "primary": ("openai", "gpt-5.2-pro"),
        "fallback": ("anthropic", "claude-opus-4-5-20251101"),
        "min_rank": 3,  # frontier-only
    },
    # Block 4: Architecture
    "architecture": {
        "primary": ("anthropic", "claude-opus-4-5-20251101"),
        "fallback": ("openai", "gpt-5.2-pro"),
        "min_rank": 3,  # frontier-only
    },
    # Block 5: Critique (frontier-only per spec)
    "critique": {
        "primary": ("google", "gemini-3-pro-preview"),
        "fallback": ("openai", "gpt-5.2-pro"),
        "min_rank": 3,  # frontier-only, HARD FAIL if both unavailable
    },
    # Block 6: Revision
    "revision": {
        "primary": ("anthropic", "claude-opus-4-5-20251101"),
        "fallback": ("openai", "gpt-5.2-pro"),
        "min_rank": 3,  # frontier-only
    },
    # Block 7: Chunk Planning
    "chunk_planning": {
        "primary": ("openai", "gpt-5.2-chat-latest"),
        "fallback": ("anthropic", "claude-sonnet-4-5-20250514"),
        "min_rank": 2,  # pro tier
    },
    # Block 8: Implementation
    "implementation": {
        "primary": ("anthropic", "claude-sonnet-4-5-20250514"),
        "fallback": ("openai", "gpt-5.2-thinking"),
        "min_rank": 2,  # pro tier
    },
    # Block 9: Verification/Overwatcher
    "verification": {
        "primary": ("openai", "gpt-5.2-pro"),
        "fallback": ("google", "gemini-3-pro-preview"),
        "min_rank": 3,  # frontier-only
    },
    # Block 10: Deep Research
    "deep_research": {
        "primary": ("openai", "gpt-5.2-pro"),
        "fallback": ("google", "gemini-3-pro-preview"),
        "min_rank": 3,  # frontier-only, tool-assisted
    },
    # Block 11: Promotion Gate
    "promotion_gate": {
        "primary": ("openai", "gpt-5.2-pro"),
        "fallback": ("google", "gemini-3-pro-preview"),
        "min_rank": 3,  # frontier-only
    },
    # Block 12: Quarantine Report
    "quarantine_report": {
        "primary": ("openai", "gpt-5-mini"),
        "fallback": None,  # No fallback, fast model only
        "min_rank": 1,  # fast tier
    },
}


def get_stage_models(stage_name: str) -> dict:
    """Get model configuration for a stage.
    
    Returns:
        Dict with primary, fallback, min_rank keys
        
    Raises:
        ValueError: If stage not found
    """
    if stage_name not in STAGE_MODEL_CONFIG:
        raise ValueError(f"Unknown stage: {stage_name}")
    return STAGE_MODEL_CONFIG[stage_name]


__all__ = [
    "MODEL_CAPABILITY_RANKS",
    "get_capability_rank",
    "is_fallback_allowed",
    "get_rank_name",
    "validate_model_for_stage",
    "STAGE_MODEL_CONFIG",
    "get_stage_models",
]
