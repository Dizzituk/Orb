# FILE: app/llm/stage_models.py
# Purpose: Centralized stage-based model configuration.
# Called-by: app.llm._spec_gate_stream_utils_2, app.llm.critical_pipeline.config, app.llm.critical_pipeline.needle_model_selector, app.llm.local_tools.zobie.streams.archmap_full (+13 more)
# Depends-on: app.llm.frontier_models
# Last-renovated: 2026-06-11
"""
Centralized stage-based model configuration.

Each ASTRA pipeline stage can have its own provider/model configured via env vars.
No assumptions about which models belong to which providers.

Pattern:
    {STAGE}_PROVIDER  - the provider to use (openai, anthropic, google)
    {STAGE}_MODEL     - the model ID for that provider

Optional per-stage settings:
    {STAGE}_MAX_OUTPUT_TOKENS  - token limit for this stage
    {STAGE}_TIMEOUT_SECONDS    - timeout for this stage

Example .env:
    SPEC_GATE_PROVIDER=google
    SPEC_GATE_MODEL=gemini-2.0-flash
    SPEC_GATE_MAX_OUTPUT_TOKENS=4000
    
    WEAVER_PROVIDER=openai
    WEAVER_MODEL=gpt-4.1-mini
    
    CRITICAL_PIPELINE_PROVIDER=anthropic
    CRITICAL_PIPELINE_MODEL=claude-sonnet-4-5-20250929

v1.2 (2026-04-18): Frontier alias resolution + per-stage reasoning defaults.
    STAGE_DEFAULTS now references aliases like ``openai:frontier-reasoning``
    (see app.llm.frontier_models) so the "latest good model" lives in ONE
    file. get_stage_config() resolves aliases at call time and attaches the
    stage's reasoning config automatically. Concrete env-var overrides still
    work unchanged.
v1.1 (2026-01): Added all pipeline stages, token limits, timeouts
v1.0 (2026-01): Initial implementation
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Any

from app.llm.frontier_models import (
    resolve_model_alias,
    get_reasoning_for_stage,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STAGE CONFIGURATION
# =============================================================================

@dataclass
class StageConfig:
    """Configuration for a pipeline stage."""
    provider: str
    model: str
    stage_name: str
    max_output_tokens: int = 4000
    timeout_seconds: int = 60
    # v1.2: Reasoning / thinking config attached per stage.
    # Shape: ``{"effort": "low"|"medium"|"high"|"xhigh"|"max"}`` or None.
    # Resolved from app.llm.frontier_models.STAGE_REASONING.
    reasoning: Optional[Dict[str, str]] = field(default=None)

    def __str__(self) -> str:
        r = f" reasoning={self.reasoning}" if self.reasoning else ""
        return f"{self.stage_name}: {self.provider}/{self.model}{r}"

    def to_dict(self) -> dict:
        return {
            "stage": self.stage_name,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "timeout_seconds": self.timeout_seconds,
            "reasoning": self.reasoning,
        }


# =============================================================================
# STAGE DEFAULTS
# =============================================================================
# Default (provider, model, max_tokens, timeout) per stage
# Used when env vars not set - intentionally cheap/fast for safety

# v1.2 (2026-04-18): Replaced hard-coded model IDs with frontier aliases.
# Single source of truth for "latest good model" now lives in
# app.llm.frontier_models.FRONTIER_ALIASES. When new models drop, update
# that one file; every stage picks up the new model automatically.
#
# HARD RULE: no stage points to a Pro-tier OpenAI model (gpt-*-pro).
# The reasoning parameter gets us most of the quality lift at a fraction
# of the cost.
STAGE_DEFAULTS: Dict[str, Tuple[str, str, int, int]] = {
    # Core pipeline stages
    "WEAVER":             ("openai",    "openai:frontier-reasoning",         8000,  60),
    "SPEC_GATE":          ("anthropic", "anthropic:frontier-opus-thinking",  8000,  120),
    "PLANNER":            ("openai",    "openai:frontier-reasoning",         60000, 120),
    "ARCHITECTURE":       ("anthropic", "anthropic:frontier-opus-thinking",  60000, 300),
    "CRITIQUE":           ("anthropic", "anthropic:frontier-opus-thinking",  60000, 180),
    "REVISION":           ("anthropic", "anthropic:frontier-opus-thinking",  60000, 300),
    "IMPLEMENTER":        ("openai",    "openai:frontier-reasoning",         60000, 300),
    "OVERWATCHER":        ("anthropic", "anthropic:frontier-verifier",      4000,  120),
    "CRITICAL_PIPELINE":  ("openai",    "openai:frontier-reasoning",         60000, 600),

    # Supervisor (Phase 2A — interface contracts between segments)
    "CRITICAL_SUPERVISOR": ("anthropic", "anthropic:frontier-opus-thinking", 8000, 120),
    "COHESION_CHECK":      ("anthropic", "anthropic:frontier-sonnet",        4000, 120),
    "NEEDLE_CLASSIFIER":   ("google",    "google:frontier-flash-lite",       500,  30),
    "SMART_SEGMENTATION":  ("google",    "google:frontier-flash",            2000, 45),
    # Phase 3C: Needle-based model tiers for architecture generation
    "ARCH_TIER_LOW":       ("google",    "google:frontier-flash",            60000, 600),
    "ARCH_TIER_HIGH":      ("anthropic", "anthropic:frontier-opus-thinking", 60000, 600),
    # Phase 3D: Cohesion autofix micro-patch
    "COHESION_MICRO_PATCH": ("google",   "google:frontier-flash",            8000, 60),
    # Phase 4A: Post-write verification
    "JOB_CHECKER":         ("anthropic", "anthropic:frontier-sonnet",        1500, 45),
    # Phase 4C: Weaver conversation compaction
    "WEAVER_COMPACTION":   ("google",    "google:frontier-flash",            1500, 45),

    # v1.3 (2026-04-18): Always-on spec reviewer. 16k output budget so
    # detailed findings + rationale fit. Timeout generous because
    # reasoning + 200KB source context is slow.
    # JOB 12 (2026-06-10): now resolves through the verifier-family alias
    # (claude-fable-5 by default; ASTRA_VERIFIER_FAMILY_MODEL overrides).
    "SPEC_REVIEW":         ("anthropic", "anthropic:frontier-verifier", 16000, 600),

    # JOB 11-12 (2026-06-10): the agentic verifier final checkout. Gated
    # behind ASTRA_AGENTIC_VERIFIER=1; the most expensive stage when on.
    "VERIFIER_AGENT":      ("anthropic", "anthropic:frontier-verifier", 16000, 900),

    # Support stages
    "CHAT":               ("google",    "google:frontier-flash",             4000,  30),
    "ARCHMAP":            ("anthropic", "anthropic:frontier-opus-thinking",  60000, 300),
    "SUMMARIZER":         ("google",    "google:frontier-flash-lite",        2000,  60),
    "CLASSIFIER":         ("google",    "google:frontier-flash-lite",        500,   10),

    # Legacy/aliases
    "OVERWATCH":          ("google",    "google:frontier-flash",             1200,  60),
}


# =============================================================================
# LEGACY ENV VAR MAPPING
# =============================================================================
# Maps old env var patterns to new stage names for backwards compatibility
# Format: "STAGE": [("legacy_model_var", "legacy_provider_extraction_hint"), ...]

LEGACY_MODEL_VARS: Dict[str, List[Tuple[str, str]]] = {
    "SPEC_GATE": [
        ("OPENAI_SPEC_GATE_MODEL", "openai"),      # Old: OPENAI_SPEC_GATE_MODEL=gpt-4.1
        ("SPEC_GATE_MODEL", None),                  # Partial migration
    ],
    "CRITIQUE": [
        ("GEMINI_CRITIC_MODEL", "google"),          # Old: GEMINI_CRITIC_MODEL=gemini-2.0-flash
    ],
    "PLANNER": [
        ("ASTRA_PLANNER_MODEL", None),              # Old: ASTRA_PLANNER_MODEL=gemini-2.0-flash
    ],
    "IMPLEMENTER": [
        ("ASTRA_IMPLEMENTER_MODEL", None),          # Old: ASTRA_IMPLEMENTER_MODEL=claude-sonnet-4-5
    ],
    "OVERWATCHER": [
        ("ASTRA_OVERWATCH_MODEL", None),            # Old: ASTRA_OVERWATCH_MODEL=gpt-5.2-pro
    ],
    "ARCHMAP": [
        ("ORB_ZOBIE_ARCHMAP_MODEL", None),          # Old: ORB_ZOBIE_ARCHMAP_MODEL=claude-opus-4-5
    ],
}

LEGACY_TOKEN_VARS: Dict[str, List[str]] = {
    "ARCHITECTURE": ["OPUS_DRAFT_MAX_TOKENS"],
    "REVISION": ["OPUS_REVISION_MAX_TOKENS"],
    "CRITIQUE": ["GEMINI_CRITIC_MAX_TOKENS"],
    "SPEC_GATE": ["SPEC_GATE_MAX_OUTPUT_TOKENS_DEFAULT"],
    "OVERWATCHER": ["ASTRA_OVERWATCH_MAX_OUTPUT_DEFAULT"],
}


# =============================================================================
# CORE LOOKUP FUNCTION
# =============================================================================

def get_stage_config(stage: str) -> StageConfig:
    """
    Get provider/model configuration for a pipeline stage.
    
    Reads from environment variables:
        {STAGE}_PROVIDER          - provider name (openai, anthropic, google)
        {STAGE}_MODEL             - model ID
        {STAGE}_MAX_OUTPUT_TOKENS - token limit (optional)
        {STAGE}_TIMEOUT_SECONDS   - timeout (optional)
    
    Falls back to legacy env vars, then STAGE_DEFAULTS if not set.
    
    Args:
        stage: Stage name (e.g., "SPEC_GATE", "WEAVER", "CRITICAL_PIPELINE")
    
    Returns:
        StageConfig with provider, model, max_output_tokens, timeout_seconds
    
    Example:
        >>> config = get_stage_config("SPEC_GATE")
        >>> config.provider
        'google'
        >>> config.model
        'gemini-2.0-flash'
        >>> config.max_output_tokens
        4000
    """
    stage_upper = stage.upper().replace("-", "_").replace(" ", "_")
    
    # Get defaults for this stage
    defaults = STAGE_DEFAULTS.get(stage_upper)
    if defaults:
        default_provider, default_model, default_tokens, default_timeout = defaults
    else:
        # Ultimate fallback (v3.2: google instead of openai)
        default_provider = "google"
        default_model = "gemini-2.5-flash"
        default_tokens = 4000
        default_timeout = 60
    
    # ==========================================================================
    # PROVIDER lookup (new pattern first, then infer from legacy)
    # ==========================================================================
    provider = os.getenv(f"{stage_upper}_PROVIDER", "").strip()
    
    # ==========================================================================
    # MODEL lookup (new pattern first, then legacy vars)
    # ==========================================================================
    model = os.getenv(f"{stage_upper}_MODEL", "").strip()
    
    # If new-style MODEL not set, check legacy vars
    if not model and stage_upper in LEGACY_MODEL_VARS:
        for legacy_var, legacy_provider_hint in LEGACY_MODEL_VARS[stage_upper]:
            legacy_value = os.getenv(legacy_var, "").strip()
            if legacy_value:
                model = legacy_value
                # If provider not explicitly set, try to infer from legacy var name
                if not provider and legacy_provider_hint:
                    provider = legacy_provider_hint
                logger.debug(f"[stage_models] {stage_upper}: Using legacy var {legacy_var}={legacy_value}")
                break
    
    # If model set but provider not, try to infer provider from model name
    if model and not provider:
        provider = _infer_provider_from_model(model)
    
    # Apply defaults if still not set
    if not provider:
        provider = default_provider
    if not model:
        model = default_model
    
    # ==========================================================================
    # TOKEN LIMIT lookup (new pattern first, then legacy vars)
    # ==========================================================================
    max_tokens_str = os.getenv(f"{stage_upper}_MAX_OUTPUT_TOKENS", "").strip()
    
    # If new-style not set, check legacy vars
    if not max_tokens_str and stage_upper in LEGACY_TOKEN_VARS:
        for legacy_var in LEGACY_TOKEN_VARS[stage_upper]:
            legacy_value = os.getenv(legacy_var, "").strip()
            if legacy_value:
                max_tokens_str = legacy_value
                logger.debug(f"[stage_models] {stage_upper}: Using legacy token var {legacy_var}={legacy_value}")
                break
    
    # ==========================================================================
    # TIMEOUT lookup
    # ==========================================================================
    timeout_str = os.getenv(f"{stage_upper}_TIMEOUT_SECONDS", "").strip()
    
    # Parse numeric values
    try:
        max_tokens = int(max_tokens_str) if max_tokens_str else default_tokens
    except ValueError:
        max_tokens = default_tokens
    
    try:
        timeout = int(timeout_str) if timeout_str else default_timeout
    except ValueError:
        timeout = default_timeout
    
    # v1.2 (2026-04-18): Resolve frontier aliases to concrete model IDs.
    # Aliases from app.llm.frontier_models (e.g. "openai:frontier-reasoning"
    # -> "gpt-5.4") pass through here. Concrete model names are returned
    # unchanged, so env-var overrides that hardcode models still work.
    model = resolve_model_alias(model)

    # v1.2: Attach per-stage reasoning config. None means "no reasoning"
    # (light stages like classifier/summariser). High-stakes stages get
    # reasoning=high by default per Taz's directive.
    reasoning = get_reasoning_for_stage(stage_upper)

    config = StageConfig(
        provider=provider,
        model=model,
        stage_name=stage_upper,
        max_output_tokens=max_tokens,
        timeout_seconds=timeout,
        reasoning=reasoning,
    )
    
    logger.debug(f"[stage_models] {config}")
    
    return config


def _infer_provider_from_model(model: str) -> str:
    """
    Infer provider from model name when provider not explicitly set.
    
    This is a best-effort heuristic for backwards compatibility.
    """
    model_lower = model.lower()
    
    if "claude" in model_lower or "opus" in model_lower or "sonnet" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower:
        return "google"
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower:
        return "openai"
    else:
        # Unknown model, default to google (v3.2)
        return "google"


def get_stage_provider(stage: str) -> str:
    """Get just the provider for a stage."""
    return get_stage_config(stage).provider


def get_stage_model(stage: str) -> str:
    """Get just the model for a stage."""
    return get_stage_config(stage).model


def get_stage_max_tokens(stage: str) -> int:
    """Get max output tokens for a stage."""
    return get_stage_config(stage).max_output_tokens


def get_stage_timeout(stage: str) -> int:
    """Get timeout seconds for a stage."""
    return get_stage_config(stage).timeout_seconds


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON STAGES
# =============================================================================

def get_weaver_config() -> StageConfig:
    """Get Weaver configuration."""
    return get_stage_config("WEAVER")


def get_spec_gate_config() -> StageConfig:
    """Get Spec Gate configuration."""
    return get_stage_config("SPEC_GATE")


def get_planner_config() -> StageConfig:
    """Get Planner configuration."""
    return get_stage_config("PLANNER")


def get_architecture_config() -> StageConfig:
    """Get Architecture configuration."""
    return get_stage_config("ARCHITECTURE")


def get_critique_config() -> StageConfig:
    """Get Critique configuration."""
    return get_stage_config("CRITIQUE")


def get_revision_config() -> StageConfig:
    """Get Revision configuration."""
    return get_stage_config("REVISION")


def get_implementer_config() -> StageConfig:
    """Get Implementer configuration."""
    return get_stage_config("IMPLEMENTER")


def get_overwatcher_config() -> StageConfig:
    """Get Overwatcher configuration."""
    return get_stage_config("OVERWATCHER")


def get_critical_pipeline_config() -> StageConfig:
    """Get Critical Pipeline configuration."""
    return get_stage_config("CRITICAL_PIPELINE")


def get_chat_config() -> StageConfig:
    """Get Chat configuration."""
    return get_stage_config("CHAT")


def get_archmap_config() -> StageConfig:
    """Get Archmap configuration."""
    return get_stage_config("ARCHMAP")


def get_summarizer_config() -> StageConfig:
    """Get Summarizer configuration."""
    return get_stage_config("SUMMARIZER")


def get_classifier_config() -> StageConfig:
    """Get Classifier configuration."""
    return get_stage_config("CLASSIFIER")


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def get_spec_gate_provider() -> str:
    """Legacy: Get Spec Gate provider."""
    return get_spec_gate_config().provider


def get_spec_gate_model() -> str:
    """Legacy: Get Spec Gate model."""
    return get_spec_gate_config().model


# =============================================================================
# AUDIT / DEBUG HELPERS
# =============================================================================

def get_all_stage_configs() -> Dict[str, StageConfig]:
    """Get configurations for all known stages. Useful for debugging."""
    return {
        stage: get_stage_config(stage)
        for stage in STAGE_DEFAULTS.keys()
    }


def print_stage_config_audit():
    """Print all stage configurations for debugging."""
    print("\n" + "=" * 80)
    print("STAGE MODEL CONFIGURATION AUDIT")
    print("=" * 80)
    
    for stage in sorted(STAGE_DEFAULTS.keys()):
        config = get_stage_config(stage)
        defaults = STAGE_DEFAULTS[stage]
        
        env_provider = os.getenv(f"{stage}_PROVIDER", "<not set>")
        env_model = os.getenv(f"{stage}_MODEL", "<not set>")
        env_tokens = os.getenv(f"{stage}_MAX_OUTPUT_TOKENS", "<not set>")
        
        print(f"\n{stage}:")
        print(f"  Resolved: {config.provider}/{config.model} (tokens={config.max_output_tokens}, timeout={config.timeout_seconds}s)")
        print(f"  Env vars: PROVIDER={env_provider}, MODEL={env_model}, TOKENS={env_tokens}")
        print(f"  Defaults: {defaults[0]}/{defaults[1]} (tokens={defaults[2]}, timeout={defaults[3]}s)")
    
    print("\n" + "=" * 80)


def get_env_audit() -> Dict[str, str]:
    """Get current env var values for all stage configs."""
    result = {}
    for stage in STAGE_DEFAULTS.keys():
        result[f"{stage}_PROVIDER"] = os.getenv(f"{stage}_PROVIDER", "<not set>")
        result[f"{stage}_MODEL"] = os.getenv(f"{stage}_MODEL", "<not set>")
        result[f"{stage}_MAX_OUTPUT_TOKENS"] = os.getenv(f"{stage}_MAX_OUTPUT_TOKENS", "<not set>")
    return result


def get_stage_summary() -> List[dict]:
    """Get summary of all stages for logging/display."""
    return [
        get_stage_config(stage).to_dict()
        for stage in sorted(STAGE_DEFAULTS.keys())
    ]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core
    "StageConfig",
    "get_stage_config",
    "get_stage_provider",
    "get_stage_model",
    "get_stage_max_tokens",
    "get_stage_timeout",
    "STAGE_DEFAULTS",
    
    # Stage-specific getters
    "get_weaver_config",
    "get_spec_gate_config",
    "get_planner_config",
    "get_architecture_config",
    "get_critique_config",
    "get_revision_config",
    "get_implementer_config",
    "get_overwatcher_config",
    "get_critical_pipeline_config",
    "get_chat_config",
    "get_archmap_config",
    "get_summarizer_config",
    "get_classifier_config",
    
    # Legacy compatibility
    "get_spec_gate_provider",
    "get_spec_gate_model",
    
    # Debug
    "get_all_stage_configs",
    "print_stage_config_audit",
    "get_env_audit",
    "get_stage_summary",
]
