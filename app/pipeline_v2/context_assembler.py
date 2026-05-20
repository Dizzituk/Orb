# FILE: app/pipeline_v2/context_assembler.py
"""
ASTRA v2.2 Context Assembler — Layer 3 of Piece 5 (context threading).

Given a stage name, returns the slice of the decision ledger + prior stage
summaries the stage actually needs, rather than a naive history dump.

Usage from any stage:

    from app.pipeline_v2.context_assembler import assemble_context
    ctx = assemble_context(job_dir=job_dir, stage="agentic_builder")
    prompt = f"{ctx}\n\n{task_specific_prompt}"

The output is a markdown block safe to inject directly into a system or
user prompt. Empty string if the ledger and summaries are both absent.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from app.pipeline_v2.ledger import (
    DecisionLedger,
    load_ledger,
    ledger_slice_by_constraint,
)
from app.pipeline_v2.ledger_format import format_decisions_only
from app.pipeline_v2.stage_summaries import (
    read_summaries,
    read_summaries_for_stages,
    format_summaries_for_prompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Stage → concerns + upstream-stages map
# =============================================================================
# `concerns` drive ledger retrieval (matched against decision entries'
# `constrains` field). `upstream` lists the stages whose summaries this
# stage should read.
#
# "all" in concerns means "every decision applies." Builders and planners
# need full context; scaffolders and verifiers need narrower slices.

_STAGE_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "weaver": {
        "concerns": ["all"],
        "upstream": [],
    },
    "spec_gate": {
        "concerns": ["all"],
        "upstream": ["weaver"],
    },
    "planner": {
        "concerns": ["all"],
        "upstream": ["spec_gate", "weaver"],
    },
    "scaffold_engine": {
        "concerns": ["file layout", "paths", "structure", "build artefacts"],
        "upstream": ["planner", "spec_gate"],
    },
    "agentic_builder": {
        "concerns": ["all"],
        "upstream": ["scaffold_engine", "planner", "spec_gate"],
    },
    "verifier": {
        "concerns": ["verification", "acceptance", "build artefacts"],
        "upstream": ["agentic_builder", "scaffold_engine"],
    },
    "bvl": {
        "concerns": ["verification", "acceptance", "build artefacts"],
        "upstream": ["agentic_builder", "scaffold_engine"],
    },
}


# =============================================================================
# Public API
# =============================================================================

def assemble_context(
    job_dir: str,
    stage: str,
    concerns: Optional[List[str]] = None,
    upstream_stages: Optional[List[str]] = None,
    max_chars: int = 16000,
) -> str:
    """Build the stage-specific context block.

    Args:
        job_dir: The build's job directory (where decision_ledger.json and
            stage_summaries.md live).
        stage: Name of the stage requesting context. Determines default
            concerns and upstream-stage lookups.
        concerns: Override the default concern list for this stage.
        upstream_stages: Override the default upstream-stages list.
        max_chars: Cap on total output length. Decisions are prioritised
            over summaries if the cap would be exceeded.

    Returns markdown ready to inject into a prompt, or "" if nothing is
    available.
    """
    profile = _STAGE_PROFILES.get(stage.lower(), {"concerns": ["all"], "upstream": []})
    effective_concerns = concerns if concerns is not None else profile["concerns"]
    effective_upstream = upstream_stages if upstream_stages is not None else profile["upstream"]

    # ---- Decisions
    decisions_md = _format_decisions_for_stage(job_dir, effective_concerns)

    # ---- Stage summaries
    summaries_md = _format_summaries_for_stage(job_dir, effective_upstream)

    # ---- Combine with budget
    return _combine_with_budget(decisions_md, summaries_md, max_chars)


# =============================================================================
# Internal helpers
# =============================================================================

def _format_decisions_for_stage(
    job_dir: str,
    concerns: List[str],
) -> str:
    ledger = load_ledger(job_dir)
    if ledger is None or ledger.entry_count == 0:
        return ""

    # "all" concerns -> full decisions list
    if "all" in [c.lower() for c in concerns]:
        return format_decisions_only(ledger)

    # Otherwise filter by constrains intersection
    decisions = ledger_slice_by_constraint(ledger, concerns)
    if not decisions:
        # Fall back to the full list rather than returning nothing; a stage
        # that declared narrow concerns but finds no matches is usually
        # better served by seeing all decisions than by seeing none.
        return format_decisions_only(ledger)

    # Inline-render this filtered slice using the same format
    lines = ["### Prior Decisions (binding)\n"]
    for e in decisions:
        header = f"- [{e.entry_id}] **{e.key or e.summary}**"
        if e.value is not None:
            header += f" = `{e.value}`"
        lines.append(header)
        if e.rationale:
            lines.append(f"  - _Why_: {e.rationale}")
        if e.constrains:
            lines.append(f"  - _Binds_: {', '.join(e.constrains)}")
    return "\n".join(lines)


def _format_summaries_for_stage(
    job_dir: str,
    upstream_stages: List[str],
) -> str:
    if not upstream_stages:
        return ""
    summaries = read_summaries_for_stages(job_dir, upstream_stages)
    if not summaries:
        return ""
    return format_summaries_for_prompt(summaries)


def _combine_with_budget(
    decisions_md: str,
    summaries_md: str,
    max_chars: int,
) -> str:
    """Join decisions + summaries, capped at max_chars. Decisions win."""
    if not decisions_md and not summaries_md:
        return ""

    # Decisions are the highest-priority signal — always kept in full.
    if len(decisions_md) >= max_chars:
        logger.debug(
            "[context_assembler] Decisions alone exceed budget (%d/%d), "
            "truncating", len(decisions_md), max_chars,
        )
        return decisions_md[:max_chars]

    remaining = max_chars - len(decisions_md) - 4  # separator
    if not summaries_md:
        return decisions_md

    if len(summaries_md) > remaining:
        summaries_md = summaries_md[:remaining] + "\n_(summaries truncated)_"

    if decisions_md and summaries_md:
        return f"{decisions_md}\n\n{summaries_md}"
    return decisions_md or summaries_md


# =============================================================================
# Diagnostics
# =============================================================================

def describe_stage_profile(stage: str) -> Dict[str, List[str]]:
    """Return the concerns + upstream config for a stage (for logging/debug)."""
    return dict(_STAGE_PROFILES.get(
        stage.lower(),
        {"concerns": ["all"], "upstream": []},
    ))


def list_known_stages() -> List[str]:
    """Return all stages the assembler knows profiles for."""
    return list(_STAGE_PROFILES.keys())
