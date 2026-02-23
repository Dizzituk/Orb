# FILE: app/llm/prompt_tiers.py
"""
Tiered system prompts for cost optimisation.

Different pipeline stages need different levels of system prompt detail.
Expensive context tokens should only be spent when the model needs them.

Tiers:
    MINIMAL  — Classifier, summarizer, chat: just role + output format
    STANDARD — Sonnet stages (implementer, enrichment): role + key constraints
    FULL     — Opus stages (spec gate, overwatcher): full governance policy

Usage:
    from app.llm.prompt_tiers import get_system_prompt, PromptTier

    prompt = get_system_prompt(PromptTier.STANDARD, stage="implementer")
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PromptTier(str, Enum):
    """System prompt detail level."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    FULL = "full"


# =========================================================================
# Tier definitions
# =========================================================================

MINIMAL_SYSTEM_PROMPT = """You are ASTRA, an AI assistant for software development.
Respond in the requested format. Be precise and concise."""


STANDARD_SYSTEM_PROMPT = """You are ASTRA, an AI assistant for software development.

Key constraints:
- File size limit: 20KB target, 30KB maximum per file
- Split large files into cooperating modules
- Use existing patterns and imports from the codebase
- Never guess file paths or function signatures — use only what's in evidence
- Python code must pass ast.parse() without errors
- All imports must reference real modules

Respond precisely in the requested format."""


FULL_SYSTEM_PROMPT = """You are ASTRA, an AI orchestration system for autonomous software development.

GOVERNANCE RULES (non-negotiable):
1. EVIDENCE-FIRST: Every claim must be grounded in filesystem evidence.
   Never guess file paths, function signatures, or module contents.
2. FILE SIZE DISCIPLINE: 20KB target, 30KB maximum. Split into modules.
3. NO DESTRUCTIVE ACTIONS without explicit user confirmation.
4. IMPORT INTEGRITY: All imports must reference real, existing modules.
5. SANDBOX ONLY: Generated code runs in sandbox — never on host.
6. QUALITY OVER SPEED: Get it right, don't get it fast.

ARCHITECTURAL CONSTRAINTS:
- Python code must pass ast.parse() without syntax errors
- Function signatures must match their architecture specification
- Cross-segment imports must be validated against the export map
- No duplicate public function definitions across segments
- Use existing patterns from the codebase, don't invent new ones

When unsure, ask for clarification rather than guessing.
Respond precisely in the requested format."""


# =========================================================================
# Stage → Tier mapping
# =========================================================================

STAGE_TIER_MAP = {
    # Minimal: cheap models, simple tasks
    "classifier": PromptTier.MINIMAL,
    "summarizer": PromptTier.MINIMAL,
    "chat": PromptTier.MINIMAL,
    "embedding": PromptTier.MINIMAL,

    # Standard: mid-tier models, structured work
    "implementer": PromptTier.STANDARD,
    "segment_enrichment": PromptTier.STANDARD,
    "coherence_guardian": PromptTier.STANDARD,
    "critique": PromptTier.STANDARD,
    "revision": PromptTier.STANDARD,
    "phase_checkout": PromptTier.STANDARD,
    "final_checkout": PromptTier.STANDARD,
    "weaver": PromptTier.STANDARD,

    # Full: expensive models, decision gates
    "spec_gate": PromptTier.FULL,
    "overwatcher": PromptTier.FULL,
    "architecture": PromptTier.FULL,
    "planner": PromptTier.FULL,
    "archmap": PromptTier.FULL,
}

TIER_PROMPTS = {
    PromptTier.MINIMAL: MINIMAL_SYSTEM_PROMPT,
    PromptTier.STANDARD: STANDARD_SYSTEM_PROMPT,
    PromptTier.FULL: FULL_SYSTEM_PROMPT,
}


# =========================================================================
# Public API
# =========================================================================

def get_prompt_tier(stage: str) -> PromptTier:
    """Get the prompt tier for a pipeline stage."""
    return STAGE_TIER_MAP.get(stage, PromptTier.STANDARD)


def get_system_prompt(
    tier: Optional[PromptTier] = None,
    stage: Optional[str] = None,
) -> str:
    """
    Get the system prompt for a given tier or stage.

    If tier is provided, uses it directly.
    If stage is provided, looks up the tier.
    If neither, returns STANDARD.
    """
    if tier is None and stage is not None:
        tier = get_prompt_tier(stage)
    elif tier is None:
        tier = PromptTier.STANDARD

    return TIER_PROMPTS.get(tier, STANDARD_SYSTEM_PROMPT)


def estimate_prompt_tokens(tier: PromptTier) -> int:
    """Rough token estimate for a prompt tier (for budgeting)."""
    estimates = {
        PromptTier.MINIMAL: 30,
        PromptTier.STANDARD: 120,
        PromptTier.FULL: 250,
    }
    return estimates.get(tier, 120)


__all__ = [
    "PromptTier",
    "get_prompt_tier",
    "get_system_prompt",
    "estimate_prompt_tokens",
    "MINIMAL_SYSTEM_PROMPT",
    "STANDARD_SYSTEM_PROMPT",
    "FULL_SYSTEM_PROMPT",
]
