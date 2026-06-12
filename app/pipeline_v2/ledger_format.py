# FILE: app/pipeline_v2/ledger_format.py
# Purpose: ASTRA v2.2 Decision Ledger — prompt formatting.
# Called-by: app.pipeline_v2.context_assembler
# Depends-on: app.pipeline_v2.ledger
# Last-renovated: 2026-06-11
"""
ASTRA v2.2 Decision Ledger — prompt formatting.

Separated from ledger.py to keep the core data/API layer small and the
presentation layer independently iterable. Stages use these helpers to
turn a ledger slice into a markdown block for LLM prompt injection.

Typical usage:
    from app.pipeline_v2.ledger import load_ledger, ledger_slice_by_constraint
    from app.pipeline_v2.ledger_format import (
        format_ledger_for_prompt,
        format_decisions_only,
    )
"""

from __future__ import annotations

from typing import List, Optional

from app.pipeline_v2.ledger import (
    DecisionLedger,
    LedgerEntry,
    ledger_slice,
    ledger_slice_by_constraint,
    load_file_evidence,
)


# =============================================================================
# Full slice — decisions, constraints, facts, file reads, flags
# =============================================================================

def format_ledger_for_prompt(
    ledger: DecisionLedger,
    segment_id: str,
    job_dir: str,
    include_file_content: bool = True,
    max_content_chars: int = 120_000,
    include_types: Optional[List[str]] = None,
) -> str:
    """Format a ledger slice as markdown for LLM prompt injection.

    Surfaces rationale, assumptions and `constrains` on decision entries —
    the three fields that drive cross-stage consistency under v2.2.
    """
    entries = ledger_slice(
        ledger, segment_id,
        include_types=include_types,
        exclude_corrected=True,
    )
    if not entries:
        return ""

    parts: List[str] = []
    parts.append("### Decision Ledger (GROUND TRUTH)\n")
    parts.append(
        "The following decisions and evidence have been recorded by earlier "
        "pipeline stages. These are binding. Do NOT invent, guess, or "
        "approximate values that appear here. If you must overturn a prior "
        "decision, emit a new ledger entry with the same `key`, your new "
        "`value`, and `supersedes` pointing at the old entry id — along "
        "with a rationale.\n"
    )

    decisions   = [e for e in entries if e.type == "decision"]
    constraints = [e for e in entries if e.type == "constraint"]
    facts       = [e for e in entries if e.type == "codebase_fact"]
    file_reads  = [e for e in entries if e.type == "file_read"]
    flags       = [e for e in entries if e.type == "flag"]

    _append_decisions_section(parts, decisions)
    _append_constraints_section(parts, constraints)
    _append_facts_section(parts, facts)
    _append_file_reads_section(
        parts, file_reads, job_dir,
        include_file_content=include_file_content,
        max_content_chars=max_content_chars,
    )
    _append_flags_section(parts, flags)

    return "\n".join(parts)


def format_decisions_only(
    ledger: DecisionLedger,
    stage_concerns: Optional[List[str]] = None,
) -> str:
    """Compact decisions-only view for stages that don't need file content.

    If `stage_concerns` is given, filters to decisions whose `constrains`
    intersect the stage's concerns. Otherwise returns all active decisions.
    """
    if stage_concerns:
        decisions = ledger_slice_by_constraint(ledger, stage_concerns)
    else:
        decisions = [
            e for e in ledger.entries
            if e.type == "decision" and e.status != "corrected"
        ]

    if not decisions:
        return ""

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


# =============================================================================
# Section helpers
# =============================================================================

def _append_decisions_section(
    parts: List[str],
    decisions: List[LedgerEntry],
) -> None:
    if not decisions:
        return
    parts.append("#### Architectural Decisions\n")
    parts.append("These bind downstream stages. Match them exactly.\n")
    for e in decisions:
        header = f"- **[{e.entry_id}] {e.key or e.summary}**"
        if e.value is not None:
            header += f" = `{e.value}`"
        parts.append(header)
        if e.rationale:
            parts.append(f"  - _Rationale_: {e.rationale}")
        if e.assumptions:
            parts.append(f"  - _Assumes_: {', '.join(e.assumptions)}")
        if e.constrains:
            parts.append(f"  - _Constrains_: {', '.join(e.constrains)}")
    parts.append("")


def _append_constraints_section(
    parts: List[str],
    constraints: List[LedgerEntry],
) -> None:
    if not constraints:
        return
    parts.append("#### Constraints\n")
    for e in constraints:
        parts.append(f"- [{e.entry_id}] {e.summary}")
    parts.append("")


def _append_facts_section(
    parts: List[str],
    facts: List[LedgerEntry],
) -> None:
    if not facts:
        return
    parts.append("#### Codebase Facts\n")
    for e in facts:
        parts.append(f"- {e.summary}")
        if e.value:
            parts.append(f"  \u2192 `{e.value}`")
    parts.append("")


def _append_file_reads_section(
    parts: List[str],
    file_reads: List[LedgerEntry],
    job_dir: str,
    include_file_content: bool,
    max_content_chars: int,
) -> None:
    if not file_reads:
        return
    parts.append("#### Source Files\n")
    content_chars_used = 0
    for e in file_reads:
        if (
            include_file_content
            and e.content_ref
            and content_chars_used < max_content_chars
        ):
            content = load_file_evidence(
                job_dir, e.content_ref,
                max_chars=max_content_chars - content_chars_used,
            )
            if content:
                parts.append(
                    f"**`{e.path}`** ({e.content_size or len(content):,} chars)"
                )
                parts.append(f"```\n{content}\n```\n")
                content_chars_used += len(content)
                continue
        parts.append(f"- `{e.path}` ({e.summary})")
    parts.append("")


def _append_flags_section(
    parts: List[str],
    flags: List[LedgerEntry],
) -> None:
    if not flags:
        return
    parts.append("#### \u26a0\ufe0f Flags from Review\n")
    for e in flags:
        parts.append(f"- [{e.entry_id}] {e.summary}")
    parts.append("")
