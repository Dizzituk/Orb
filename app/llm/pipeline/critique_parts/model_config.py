# FILE: app/llm/pipeline/critique_parts/model_config.py
"""Model and provider configuration resolution for the critique pipeline."""

import logging
import os

logger = logging.getLogger(__name__)

# Stage models (env-driven model resolution)
try:
    from app.llm.stage_models import get_critique_config
    _STAGE_MODELS_AVAILABLE = True
except ImportError:
    _STAGE_MODELS_AVAILABLE = False


def _get_critique_model_config() -> tuple[str, str, int]:
    """Get critique provider/model from stage_models or env vars AT RUNTIME.
    
    Returns: (provider, model, max_tokens)
    """
    if _STAGE_MODELS_AVAILABLE:
        try:
            cfg = get_critique_config()
            return cfg.provider, cfg.model, cfg.max_output_tokens
        except Exception:
            pass
    
    # Fallback to legacy env vars
    provider = os.getenv("CRITIQUE_PROVIDER", "google")
    model = os.getenv("CRITIQUE_MODEL") or os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.0-flash")
    max_tokens = int(os.getenv("CRITIQUE_MAX_OUTPUT_TOKENS") or os.getenv("GEMINI_CRITIC_MAX_TOKENS", "60000"))
    return provider, model, max_tokens


# Legacy exports (for backward compatibility)
GEMINI_CRITIC_MODEL = os.getenv("CRITIQUE_MODEL") or os.getenv("GEMINI_CRITIC_MODEL", "gemini-2.0-flash")
GEMINI_CRITIC_MAX_TOKENS = int(os.getenv("CRITIQUE_MAX_OUTPUT_TOKENS") or os.getenv("GEMINI_CRITIC_MAX_TOKENS", "60000"))
