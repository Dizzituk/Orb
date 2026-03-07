# FILE: app/agentic_pipeline/config.py
"""
Agentic Pipeline Configuration.

Feature flags and settings for the pipeline redesign.
Controls which pipeline path is active (existing vs agentic)
and whether comparison mode is enabled.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flags (environment variable overrides)
# ---------------------------------------------------------------------------

# Master switch: enable the agentic pipeline as the primary path
# When False, the existing per-segment pipeline runs as normal.
# When True, the agentic pipeline replaces the segment loop.
AGENTIC_PIPELINE_ENABLED = os.getenv("ASTRA_AGENTIC_PIPELINE", "false").lower() == "true"

# Comparison mode: run agentic pipeline alongside existing pipeline
# and save comparison reports. Does NOT replace the existing pipeline.
COMPARISON_MODE_ENABLED = os.getenv("ASTRA_COMPARISON_MODE", "false").lower() == "true"

# Extraction-only mode: skip LLM fallback in step_process_task.py
# When True, DIRECT_EXTRACTION is the only code placement path.
# When False, the LLM implementer fallback is still available.
EXTRACTION_ONLY = os.getenv("ASTRA_EXTRACTION_ONLY", "false").lower() == "true"

# Provider/model defaults for the agentic loop (Stage 1)
AGENTIC_LOOP_PROVIDER = os.getenv("ASTRA_AGENTIC_PROVIDER", "openai")
AGENTIC_LOOP_MODEL = os.getenv("ASTRA_AGENTIC_MODEL", "gpt-5.4")

# Provider/model defaults for phase checkout (Stage 3)
CHECKOUT_PROVIDER = os.getenv("ASTRA_CHECKOUT_PROVIDER", "anthropic")
CHECKOUT_MODEL = os.getenv("ASTRA_CHECKOUT_MODEL", "claude-sonnet-4-6")


def log_config() -> None:
    """Log current configuration at startup."""
    logger.info(
        "[agentic_config] AGENTIC_PIPELINE_ENABLED=%s, COMPARISON_MODE=%s, "
        "EXTRACTION_ONLY=%s, loop=%s/%s, checkout=%s/%s",
        AGENTIC_PIPELINE_ENABLED, COMPARISON_MODE_ENABLED, EXTRACTION_ONLY,
        AGENTIC_LOOP_PROVIDER, AGENTIC_LOOP_MODEL,
        CHECKOUT_PROVIDER, CHECKOUT_MODEL,
    )
