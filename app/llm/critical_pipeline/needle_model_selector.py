# FILE: app/llm/critical_pipeline/needle_model_selector.py
"""
Needle-Based Model Selection — Dynamic model routing per segment.

Phase 3C of Pipeline Evolution.

Selects the architecture generation model based on the estimated cognitive
load (needle count) of each segment. Lower-needle segments use cheaper,
faster models; higher-needle segments use frontier models.

Tier mapping (configurable via env):
    trivial  (1-2 needles) → ARCH_TIER_LOW   (e.g. Sonnet 4.5)
    moderate (3 needles)   → ARCH_TIER_LOW   (e.g. Sonnet 4.5)
    hard     (4 needles)   → ARCH_TIER_HIGH  (frontier-class model)
    must_segment (5+)      → should be segmented, but if reached: ARCH_TIER_HIGH

All model IDs are resolved from stage_models.py / env vars. Zero hardcoded
model strings in this module.

v1.0 (2026-02-10): Initial implementation — Phase 3C.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

NEEDLE_MODEL_SELECTOR_BUILD_ID = "2026-02-10-v1.0-needle-model-selection"
print(f"[NEEDLE_MODEL_SELECTOR_LOADED] BUILD_ID={NEEDLE_MODEL_SELECTOR_BUILD_ID}")


# =============================================================================
# TIER RESOLUTION — All from stage_models / env vars
# =============================================================================

def _get_tier_config(tier_name: str) -> Tuple[str, str]:
    """
    Resolve provider + model for a named tier.

    Tries stage_models first, falls back to env vars.
    Raises RuntimeError if not configured.

    Tier names map to stage_models entries:
        ARCH_TIER_LOW  → lighter model for ≤3 needle segments
        ARCH_TIER_HIGH → frontier model for 4+ needle segments
    """
    try:
        from app.llm.stage_models import get_stage_config
        config = get_stage_config(tier_name)
        return config.provider, config.model
    except (ImportError, Exception) as e:
        logger.debug("[needle_model] stage_models lookup failed for %s: %s", tier_name, e)

    raise RuntimeError(
        f"Model tier '{tier_name}' not configured. "
        f"Set {tier_name}_PROVIDER and {tier_name}_MODEL env vars, "
        f"or add {tier_name} to STAGE_DEFAULTS in stage_models.py."
    )


# =============================================================================
# SELECTOR — Main entry point
# =============================================================================

def select_model_for_needles(
    needle_estimate: int,
    difficulty_tier: Optional[str] = None,
) -> Dict[str, str]:
    """
    Select provider + model based on needle count.

    Args:
        needle_estimate: The needle count (max of blast_radius, concepts, interfaces)
        difficulty_tier: Optional hint ("trivial", "moderate", "hard", "must_segment")

    Returns:
        {"provider": str, "model": str, "tier": str, "reason": str}
    """
    tier = difficulty_tier or _classify_tier(needle_estimate)

    if tier in ("trivial", "moderate"):
        stage_name = "ARCH_TIER_LOW"
    else:
        stage_name = "ARCH_TIER_HIGH"

    try:
        provider, model = _get_tier_config(stage_name)
    except RuntimeError:
        # If tier config not available, fall through to default pipeline model
        logger.warning(
            "[needle_model] Tier %s not configured — using default CRITICAL_PIPELINE model",
            stage_name,
        )
        try:
            from app.llm.stage_models import get_stage_config
            config = get_stage_config("CRITICAL_PIPELINE")
            provider, model = config.provider, config.model
        except Exception:
            raise RuntimeError(
                "Neither ARCH_TIER_LOW/HIGH nor CRITICAL_PIPELINE configured. "
                "Check stage_models.py and env vars."
            )

    reason = (
        f"Needle estimate {needle_estimate} ({tier}) → "
        f"{stage_name} → {provider}/{model}"
    )
    logger.info("[needle_model] %s", reason)

    return {
        "provider": provider,
        "model": model,
        "tier": stage_name,
        "reason": reason,
    }


def _classify_tier(needle_estimate: int) -> str:
    """Classify needle count into difficulty tier."""
    if needle_estimate <= 2:
        return "trivial"
    if needle_estimate <= 3:
        return "moderate"
    if needle_estimate <= 4:
        return "hard"
    return "must_segment"


# =============================================================================
# SEGMENT-CONTEXT AWARE SELECTION
# =============================================================================

def select_model_for_segment(
    segment_context: Optional[Dict[str, Any]] = None,
    grounding_data: Optional[Dict[str, Any]] = None,
    default_config: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Select model for a specific segment, using all available context.

    Priority:
    1. Segment-level needle info (if segments carry per-segment needle data)
    2. Job-level needle estimate from grounding_data (divided by segment count)
    3. Heuristic from segment file count + interface contract complexity
    4. Default pipeline model (no needle data available)

    Args:
        segment_context: The segment's context dict (from build_segment_context)
        grounding_data: The job's grounding_data (contains needle_estimate)
        default_config: Fallback {"provider": ..., "model": ...}

    Returns:
        {"provider": str, "model": str, "tier": str, "reason": str}
    """
    # Try to extract needle estimate
    _needle_est = None
    _difficulty = None

    # Source 1: grounding_data.needle_estimate (job-level)
    if grounding_data:
        ne = grounding_data.get("needle_estimate")
        if ne and isinstance(ne, dict):
            _needle_est = ne.get("needle_estimate", 0)
            _difficulty = ne.get("difficulty_tier")

            # If multi-segment, estimate per-segment needles
            seg_data = grounding_data.get("segmentation", {})
            total_segs = seg_data.get("total_segments", 1)
            if total_segs > 1 and _needle_est:
                # Per-segment estimate: divide total, round up, min 1
                _per_seg = max(1, -(-_needle_est // total_segs))
                logger.debug(
                    "[needle_model] Job needles=%d / %d segments → ~%d per segment",
                    _needle_est, total_segs, _per_seg,
                )
                _needle_est = _per_seg
                _difficulty = _classify_tier(_per_seg)

    # Source 2: Heuristic from segment context
    if _needle_est is None and segment_context:
        file_count = len(segment_context.get("file_scope", []))
        has_contract = bool(segment_context.get("interface_contract", "").strip())
        deps = len(segment_context.get("dependencies", []))

        # Rough heuristic: files + contract complexity
        _needle_est = max(1, file_count // 3)
        if has_contract:
            _needle_est = max(_needle_est, 2)
        if deps >= 2:
            _needle_est = max(_needle_est, 3)

        _difficulty = _classify_tier(_needle_est)
        logger.debug(
            "[needle_model] Heuristic: %d files, contract=%s, deps=%d → needles=%d",
            file_count, has_contract, deps, _needle_est,
        )

    # Source 3: No needle data — use default
    if _needle_est is None:
        if default_config:
            return {
                "provider": default_config["provider"],
                "model": default_config["model"],
                "tier": "default",
                "reason": "No needle data available — using default pipeline model",
            }
        # Last resort
        try:
            return select_model_for_needles(3, "moderate")  # conservative default
        except RuntimeError:
            raise

    return select_model_for_needles(_needle_est, _difficulty)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "select_model_for_needles",
    "select_model_for_segment",
    "NEEDLE_MODEL_SELECTOR_BUILD_ID",
]
