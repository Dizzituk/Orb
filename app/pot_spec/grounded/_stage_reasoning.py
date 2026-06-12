# FILE: app/pot_spec/grounded/_stage_reasoning.py
# Purpose: SpecGate stage reasoning + timeout resolution (env-overridable).
# Called-by: _simple_create_evidence, _simple_create_review, _simple_create_utils_17
# Depends-on: app.llm.frontier_models (soft)
# Last-renovated: 2026-06-11
"""
SpecGate stage reasoning/timeout helpers.

Why (2026-06-11): the simple_create call sites hardcoded
reasoning={"effort": "high"} (v2.7), which silently ignored the
per-stage env override SPEC_GATE_REASONING_EFFORT added the same day
(SpecGate moved to Claude Fable 5 at xhigh). These helpers make every
SpecGate LLM call resolve effort and timeout from the live env/stage
config, with the old v2.7 values as hard fallbacks.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

_FALLBACK_REASONING: Dict[str, str] = {"effort": "high"}
_FALLBACK_TIMEOUT_SECONDS = 300


def spec_gate_reasoning() -> Optional[Dict[str, str]]:
    """Effort config for SpecGate LLM calls.

    Resolves via frontier_models.get_reasoning_for_stage("SPEC_GATE"),
    which honours SPEC_GATE_REASONING_EFFORT from .env. Falls back to
    the historical {"effort": "high"} if resolution fails.
    """
    try:
        from app.llm.frontier_models import get_reasoning_for_stage
        return get_reasoning_for_stage("SPEC_GATE") or dict(_FALLBACK_REASONING)
    except Exception:
        return dict(_FALLBACK_REASONING)


def spec_gate_timeout_seconds() -> int:
    """Per-call timeout for SpecGate LLM calls.

    Reads SPEC_GATE_TIMEOUT_SECONDS (600 in .env as of 2026-06-11 for
    Fable 5 xhigh, which can run several minutes). Falls back to the
    historical 300s floor.
    """
    try:
        raw = os.getenv("SPEC_GATE_TIMEOUT_SECONDS", "").strip()
        if raw:
            val = int(raw)
            if val > 0:
                return val
    except Exception:
        pass
    return _FALLBACK_TIMEOUT_SECONDS


__all__ = ["spec_gate_reasoning", "spec_gate_timeout_seconds"]
