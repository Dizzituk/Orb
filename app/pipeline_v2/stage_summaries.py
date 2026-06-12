# FILE: app/pipeline_v2/stage_summaries.py
# Purpose: ASTRA v2.2 Stage Summaries — Layer 2 of Piece 5 (context threading).
# Called-by: app.pipeline_v2.context_assembler, app.pipeline_v2.orchestrator
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
ASTRA v2.2 Stage Summaries — Layer 2 of Piece 5 (context threading).

Each pipeline stage, on completion, writes a short summary (what it did,
what the next stage needs to know) to an append-only markdown file per
build. Downstream stages read these summaries via the context assembler
instead of raw conversation history, which stops late stages from
drowning in irrelevant context from early stages.

File layout:
    <job_dir>/stage_summaries.md

Format (per entry):

    ## <stage> — <ISO timestamp>
    **Status:** passed  **Duration:** 12.3s
    **What happened:**
    <body>

    **For the next stage:**
    <handover_notes>

    ---

Programmatic reads parse on the `---` separator. Writes are append-only;
we never rewrite prior entries.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

logger = logging.getLogger(__name__)

SUMMARIES_FILENAME = "stage_summaries.md"
_ENTRY_SEPARATOR = "\n---\n"


@dataclass
class StageSummary:
    """One stage's handover record."""
    stage: str
    timestamp: str
    status: str
    duration_s: Optional[float]
    body: str
    handover_notes: str


# =============================================================================
# WRITE
# =============================================================================

def write_summary(
    job_dir: str,
    stage: str,
    body: str,
    handover_notes: str = "",
    duration_s: Optional[float] = None,
    status: str = "passed",
) -> str:
    """Append a stage summary to <job_dir>/stage_summaries.md.

    Returns the absolute path written.
    """
    os.makedirs(job_dir, exist_ok=True)
    path = os.path.join(job_dir, SUMMARIES_FILENAME)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    dur_txt = f"{duration_s:.1f}s" if duration_s is not None else "—"
    entry_lines = [
        f"## {stage} — {ts}",
        f"**Status:** {status}  **Duration:** {dur_txt}",
        "",
        "**What happened:**",
        (body or "").strip() or "_(no body)_",
    ]
    if handover_notes.strip():
        entry_lines += [
            "",
            "**For the next stage:**",
            handover_notes.strip(),
        ]
    entry_lines.append("")  # trailing blank before separator
    entry = "\n".join(entry_lines)

    with open(path, "a", encoding="utf-8") as f:
        f.write(entry + _ENTRY_SEPARATOR)

    logger.info(
        "[stage_summaries] Wrote summary: %s (%s, %s)", stage, status, dur_txt,
    )
    return path


# =============================================================================
# READ
# =============================================================================

def read_summaries(job_dir: str) -> List[StageSummary]:
    """Parse <job_dir>/stage_summaries.md into a list of StageSummary.

    Returns [] if the file does not exist or is empty.
    """
    path = os.path.join(job_dir, SUMMARIES_FILENAME)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        logger.warning("[stage_summaries] Read failed: %s", e)
        return []

    blocks = [b.strip() for b in text.split(_ENTRY_SEPARATOR) if b.strip()]
    summaries: List[StageSummary] = []
    for block in blocks:
        parsed = _parse_block(block)
        if parsed is not None:
            summaries.append(parsed)
    return summaries


def read_summaries_for_stages(
    job_dir: str,
    stage_names: List[str],
) -> List[StageSummary]:
    """Return summaries filtered to a given set of stage names, preserving order."""
    all_summaries = read_summaries(job_dir)
    wanted = {s.lower() for s in stage_names}
    return [s for s in all_summaries if s.stage.lower() in wanted]


# =============================================================================
# PARSING — internal
# =============================================================================

_HEADER_RE = re.compile(r"^##\s+(?P<stage>[^\u2014]+?)\s+\u2014\s+(?P<ts>\S+)")
_META_RE = re.compile(
    r"\*\*Status:\*\*\s+(?P<status>\S+)\s+\*\*Duration:\*\*\s+(?P<dur>\S+)"
)


def _parse_block(block: str) -> Optional[StageSummary]:
    lines = block.splitlines()
    if not lines:
        return None

    header = _HEADER_RE.match(lines[0])
    if not header:
        return None
    stage = header.group("stage").strip()
    ts = header.group("ts").strip()

    status = "unknown"
    duration_s: Optional[float] = None
    for line in lines[1:4]:
        m = _META_RE.search(line)
        if m:
            status = m.group("status")
            dur_txt = m.group("dur")
            if dur_txt.endswith("s"):
                try:
                    duration_s = float(dur_txt[:-1])
                except ValueError:
                    duration_s = None
            break

    body, handover = _split_body_and_handover(lines)

    return StageSummary(
        stage=stage,
        timestamp=ts,
        status=status,
        duration_s=duration_s,
        body=body.strip(),
        handover_notes=handover.strip(),
    )


def _split_body_and_handover(lines: List[str]) -> tuple[str, str]:
    """Split the body (after 'What happened:') and handover section."""
    body_lines: List[str] = []
    handover_lines: List[str] = []
    mode = "preamble"
    for line in lines:
        if line.startswith("**What happened:**"):
            mode = "body"
            continue
        if line.startswith("**For the next stage:**"):
            mode = "handover"
            continue
        if mode == "body":
            body_lines.append(line)
        elif mode == "handover":
            handover_lines.append(line)
    return "\n".join(body_lines), "\n".join(handover_lines)


# =============================================================================
# FORMAT FOR PROMPT
# =============================================================================

def format_summaries_for_prompt(summaries: List[StageSummary]) -> str:
    """Render a list of summaries as a compact markdown block for LLM prompts."""
    if not summaries:
        return ""

    parts = ["### Prior Stage Summaries\n"]
    parts.append(
        "These are handover notes from earlier pipeline stages. Use them "
        "to understand what has already been done and what was flagged "
        "for attention.\n"
    )
    for s in summaries:
        dur = f"{s.duration_s:.1f}s" if s.duration_s is not None else "-"
        parts.append(f"#### {s.stage} (status: {s.status}, duration: {dur})")
        if s.body:
            parts.append(s.body)
        if s.handover_notes:
            parts.append(f"\n_Handover:_ {s.handover_notes}")
        parts.append("")
    return "\n".join(parts)
