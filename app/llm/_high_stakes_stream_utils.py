from __future__ import annotations
import os


def _get_spec_gate_config() -> tuple[str, str]:
    """
    Get Spec Gate provider and model from env vars AT RUNTIME.
    
    v3.4: Ensures env vars are respected without hardcoded overrides.
    
    Precedence for model:
    1. OPENAI_SPEC_GATE_MODEL (explicit spec gate override)
    2. OPENAI_DEFAULT_MODEL (general default)
    3. "gpt-4.1-mini" (hardcoded fallback only if both env vars unset)
    
    Returns: (provider, model)
    """
    provider = os.getenv("SPEC_GATE_PROVIDER", "openai")
    
    # v3.4: Explicit precedence - OPENAI_SPEC_GATE_MODEL wins over everything
    model = os.getenv("OPENAI_SPEC_GATE_MODEL")
    if not model:
        model = os.getenv("OPENAI_DEFAULT_MODEL")
    if not model:
        model = "gpt-4.1-mini"  # Last resort hardcoded default
    
    return provider, model
