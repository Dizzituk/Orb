# FILE: app/llm/translation_routing.py
"""
Translation layer routing helpers for stream router.

Provides:
- Translation layer invocation
- Intent to routing info mapping
- Canonical intent handlers

v1.2 (2026-01): Uses centralized stage_models for dynamic provider/model config
v1.1 (2026-01): FIXED - Models now read from env vars at RUNTIME (not hardcoded)
v1.0 (2026-01): Extracted from stream_router.py
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CENTRALIZED MODEL CONFIG (v1.2)
# =============================================================================

from .stage_models import (
    get_spec_gate_config,
    get_weaver_config,
    get_critical_pipeline_config,
    get_overwatcher_config,
    StageConfig,
)


# =============================================================================
# TRANSLATION LAYER IMPORT
# =============================================================================

try:
    from app.translation import (
        translate_message_sync,
        TranslationMode,
        CanonicalIntent,
        LatencyTier,
        TranslationResult,
    )
    TRANSLATION_LAYER_AVAILABLE = True
except ImportError:
    TRANSLATION_LAYER_AVAILABLE = False
    TranslationMode = None
    CanonicalIntent = None
    LatencyTier = None
    TranslationResult = None
    logging.warning("[translation_routing] Translation layer not available")

# Import archmap config
try:
    from app.llm.local_tools.archmap_helpers import ARCHMAP_PROVIDER, ARCHMAP_MODEL
except ImportError:
    ARCHMAP_PROVIDER = "openai"
    ARCHMAP_MODEL = "gpt-4.1"


# =============================================================================
# TRANSLATION LAYER ROUTING
# =============================================================================

def route_via_translation_layer(
    message: str,
    user_id: str = "default",
    conversation_id: Optional[str] = None,
) -> Optional["TranslationResult"]:
    """
    Route message through translation layer.
    Returns TranslationResult or None if layer unavailable.
    """
    if not TRANSLATION_LAYER_AVAILABLE:
        return None
    
    try:
        result = translate_message_sync(
            text=message,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        logger.debug(
            f"[translation] Mode={result.mode.value}, Intent={result.resolved_intent.value}, "
            f"Execute={result.should_execute}, Tier={result.latency_tier.value}"
        )
        return result
    except Exception as e:
        logger.warning(f"[translation] Layer failed, falling back to legacy: {e}")
        return None


def intent_to_routing_info(intent: "CanonicalIntent") -> Optional[dict]:
    """
    Map canonical intent to routing information.
    
    v1.2: Uses centralized stage_models for all provider/model lookups.
    """
    if not TRANSLATION_LAYER_AVAILABLE or CanonicalIntent is None:
        return None
    
    # v1.2: Get configs from centralized stage_models
    spec_gate = get_spec_gate_config()
    weaver = get_weaver_config()
    critical = get_critical_pipeline_config()
    overwatcher = get_overwatcher_config()
    
    # Log resolved models for audit
    logger.debug(f"[translation_routing] SpecGate: {spec_gate}")
    logger.debug(f"[translation_routing] Weaver: {weaver}")
    logger.debug(f"[translation_routing] CriticalPipeline: {critical}")
    logger.debug(f"[translation_routing] Overwatcher: {overwatcher}")
    
    mapping = {
        CanonicalIntent.ARCHITECTURE_MAP_WITH_FILES: {
            "type": "local.architecture_map",
            "provider": ARCHMAP_PROVIDER,
            "model": ARCHMAP_MODEL,
            "reason": "Translation layer: CREATE ARCHITECTURE MAP (full)",
        },
        CanonicalIntent.ARCHITECTURE_MAP_STRUCTURE_ONLY: {
            "type": "local.architecture_map_structure",
            "provider": ARCHMAP_PROVIDER,
            "model": ARCHMAP_MODEL,
            "reason": "Translation layer: Create architecture map (structure only)",
        },
        CanonicalIntent.ARCHITECTURE_UPDATE_ATLAS_ONLY: {
            "type": "local.update_architecture",
            "provider": "local",
            "model": "architecture_scanner",
            "reason": "Translation layer: UPDATE ARCHITECTURE",
        },
        CanonicalIntent.START_SANDBOX_ZOMBIE_SELF: {
            "type": "local.sandbox",
            "provider": "local",
            "model": "sandbox_manager",
            "reason": "Translation layer: START SANDBOX ZOMBIE",
        },
        CanonicalIntent.SCAN_SANDBOX_STRUCTURE: {
            "type": "local.sandbox_structure",
            "provider": "local",
            "model": "sandbox_structure_scanner",
            "reason": "Translation layer: SCAN SANDBOX STRUCTURE",
        },
        # v1.2: Dynamic config from stage_models
        CanonicalIntent.RUN_CRITICAL_PIPELINE_FOR_JOB: {
            "type": "high_stakes.critical_pipeline",
            "provider": critical.provider,
            "model": critical.model,
            "reason": (
                "Translation layer: RUN CRITICAL PIPELINE "
                f"({critical.provider}/{critical.model})"
            ),
        },
        CanonicalIntent.WEAVER_BUILD_SPEC: {
            "type": "local.weaver",
            "provider": weaver.provider,
            "model": weaver.model,
            "reason": (
                "Translation layer: WEAVER BUILD SPEC "
                f"({weaver.provider}/{weaver.model})"
            ),
        },
        CanonicalIntent.SEND_TO_SPEC_GATE: {
            "type": "local.spec_gate",
            "provider": spec_gate.provider,
            "model": spec_gate.model,
            "reason": (
                "Translation layer: SEND TO SPEC GATE "
                f"({spec_gate.provider}/{spec_gate.model})"
            ),
        },
        CanonicalIntent.OVERWATCHER_EXECUTE_CHANGES: {
            "type": "local.overwatcher",
            "provider": overwatcher.provider,
            "model": overwatcher.model,
            "reason": (
                "Translation layer: OVERWATCHER EXECUTE "
                f"({overwatcher.provider}/{overwatcher.model})"
            ),
        },
    }
    return mapping.get(intent, None)


# =============================================================================
# LEGACY COMPATIBILITY - Re-export from stage_models
# =============================================================================

def _get_spec_gate_config():
    """Legacy compatibility wrapper."""
    cfg = get_spec_gate_config()
    return cfg.provider, cfg.model


def _get_critical_pipeline_config():
    """Legacy compatibility wrapper."""
    cfg = get_critical_pipeline_config()
    return cfg.provider, cfg.model


def _get_overwatcher_config():
    """Legacy compatibility wrapper."""
    cfg = get_overwatcher_config()
    return cfg.provider, cfg.model


def _get_weaver_config():
    """Legacy compatibility wrapper."""
    cfg = get_weaver_config()
    return cfg.provider, cfg.model


__all__ = [
    "TRANSLATION_LAYER_AVAILABLE",
    "TranslationMode",
    "CanonicalIntent", 
    "LatencyTier",
    "TranslationResult",
    "route_via_translation_layer",
    "intent_to_routing_info",
    # Legacy compatibility
    "_get_spec_gate_config",
    "_get_critical_pipeline_config",
    "_get_overwatcher_config",
    "_get_weaver_config",
]
