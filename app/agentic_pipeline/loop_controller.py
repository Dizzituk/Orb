# FILE: app/agentic_pipeline/loop_controller.py
# Purpose: Agentic Loop Controller — Turn Management Engine.
# Called-by: app.agentic_pipeline.pipeline, app.agentic_pipeline.segment_printer, app.pipeline_v2.stages.architect, app.pipeline_v2.stages.builder
# Depends-on: app.agentic_pipeline, app.agentic_pipeline.checks
# Last-renovated: 2026-06-11
"""
Agentic Loop Controller — Turn Management Engine.

Turn 1: Model generates all segment architectures
Turn 2: Deterministic check #1 (tool call, no LLM)
Turn 3: Model reviews + fixes
Turn 4: Deterministic check #2
Turn 5: Final sign-off or exit

Max 3 check cycles then exit.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.agentic_pipeline.loop_parser import parse_agentic_output, ParseResult
from app.agentic_pipeline.checks.check_runner import (
    CheckReport, format_check_report_for_prompt, run_deterministic_checks,
)

logger = logging.getLogger(__name__)

MAX_CHECK_CYCLES = 3

SEGMENT_DELIMITER_INSTRUCTION = """
## OUTPUT FORMAT (MANDATORY)

For each segment, wrap the architecture document in delimiters:

=== SEGMENT: <segment_id> ===
[COMPLETE ARCHITECTURE DOCUMENT WITH ALL CODE BLOCKS]
=== END SEGMENT: <segment_id> ===

Each architecture document must contain:
1. File Inventory table (New Files / Modified Files)
2. Complete, production-ready code blocks for EVERY file
3. File path comments at the top of each code block (// FILE: path/to/file.ext)

Code blocks must be directly extractable. No pseudocode. No placeholders.

If a segment needs NO changes:
=== SEGMENT: <segment_id> ===
## NO_CHANGES_NEEDED
Reason: <one-line explanation>
=== END SEGMENT: <segment_id> ===
"""


@dataclass
class LoopTurn:
    turn_number: int
    role: str
    content_summary: str = ""
    duration_seconds: float = 0.0
    token_count: int = 0


@dataclass
class LoopResult:
    arch_docs: Dict[str, str] = field(default_factory=dict)
    parse_result: Optional[ParseResult] = None
    final_check: Optional[CheckReport] = None
    turns: List[LoopTurn] = field(default_factory=list)
    total_llm_calls: int = 0
    total_check_cycles: int = 0
    converged: bool = False
    exit_reason: str = ""
    confidence_score: float = 0.0

    @property
    def success(self) -> bool:
        return bool(self.arch_docs) and (
            self.converged or not (self.final_check and self.final_check.has_blockers)
        )


_SYSTEM_PROMPT = """You are ASTRA's Architecture Completer.

You fill numbered [LLM_FILL_N] markers in deterministic templates.
You output ONLY fills — not the templates themselves.

OUTPUT FORMAT — MANDATORY:

For EVERY segment and EVERY numbered marker, output:

=== FILL: <segment_id> / <fill_number> ===
<your content>
=== END FILL ===

FILL RULES:

[LLM_FILL_1] = 1-2 paragraph architecture vision.

[LLM_FILL_2] = COMPLETE code for EVERY file in the segment.
  For each file, write a ### heading with the file path, then a fenced code block.
  The FIRST LINE of every code block MUST be a FILE comment:
    # FILE: app/example.py       (Python)
    // FILE: src/Example.tsx      (TypeScript/JS)
  Code must be COMPLETE. No placeholders. No TODOs. No prose inside code blocks.
  For MODIFY files: output the FULL updated file, not a diff.

[LLM_FILL_3] = Brief notes (optional, can be empty).

EXAMPLE (correct [LLM_FILL_2]):

### `app/debug/models.py`

```python
# FILE: app/debug/models.py
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.db import Base

class DebugProject(Base):
    __tablename__ = "debug_projects"
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, server_default=func.now())
```

### `app/debug/crud.py`

```python
# FILE: app/debug/crud.py
from sqlalchemy.orm import Session
from .models import DebugProject

def create_project(db: Session, name: str) -> DebugProject:
    project = DebugProject(name=name)
    db.add(project)
    db.commit()
    return project
```

CRITICAL:
- You MUST output ALL fills for ALL segments. Skipping fills = build failure.
- Every file MUST have its own fenced code block with # FILE: or // FILE: first line.
- Do NOT repeat template content. Output ONLY fills.
"""


def build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def number_fill_markers(template: str, segment_id: str) -> tuple:
    """Replace [LLM_FILL] with numbered [LLM_FILL_1], [LLM_FILL_2], etc.

    Returns (numbered_template, fill_count).
    """
    count = 0
    result = template
    while "[LLM_FILL]" in result:
        count += 1
        result = result.replace("[LLM_FILL]", f"[LLM_FILL_{count}]", 1)
    return result, count


def parse_fills(output: str) -> dict:
    """Parse === FILL: seg_id / N === blocks from model output.

    Returns dict of {(segment_id, fill_number): content}.
    """
    import re
    fills = {}
    pattern = re.compile(
        r'=== FILL:\s*([^/]+?)\s*/\s*(\d+)\s*===\n(.*?)\n=== END FILL ===',
        re.DOTALL,
    )
    for m in pattern.finditer(output):
        seg_id = m.group(1).strip()
        fill_num = int(m.group(2).strip())
        content = m.group(3).strip()
        fills[(seg_id, fill_num)] = content
    return fills


def splice_fills_into_templates(
    templates: dict, fills: dict,
) -> dict:
    """Replace [LLM_FILL_N] markers with Opus's fill content.

    Args:
        templates: {segment_id: numbered_template_string}
        fills: {(segment_id, fill_number): content}

    Returns:
        {segment_id: complete_arch_doc}
    """
    arch_docs = {}
    for seg_id, template in templates.items():
        doc = template
        fill_num = 1
        while f"[LLM_FILL_{fill_num}]" in doc:
            fill_content = fills.get((seg_id, fill_num), "")
            if not fill_content:
                # Try fuzzy match — segment_id might be truncated in output
                for (fid, fnum), fcontent in fills.items():
                    if fnum == fill_num and (fid in seg_id or seg_id in fid):
                        fill_content = fcontent
                        break
            doc = doc.replace(f"[LLM_FILL_{fill_num}]", fill_content, 1)
            fill_num += 1
        arch_docs[seg_id] = doc
    return arch_docs


def _extract_content(response: Any) -> str:
    """Extract text content from an LLM response object."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        if "content" in response:
            c = response["content"]
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                return "".join(
                    item.get("text", "") for item in c
                    if isinstance(item, dict) and item.get("type") == "text"
                )
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
    for attr in ("text", "content", "output"):
        val = getattr(response, attr, None)
        if isinstance(val, str):
            return val
    return str(response)


async def run_agentic_loop(
    context: str,
    batch_segment_ids: List[str],
    preflight_result: Any,
    segment_file_scopes: Dict[str, List[str]],
    llm_call_fn: Callable,
    provider_id: str = "openai",
    model_id: str = "gpt-5.4",
    on_progress: Optional[Callable[[str], None]] = None,
    numbered_templates: Optional[Dict[str, str]] = None,
) -> LoopResult:
    """Run the agentic architecture loop for one batch."""
    result = LoopResult()
    messages: List[Dict[str, str]] = []

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info("[agentic_loop] %s", msg)

    system_prompt = build_system_prompt()

    # --- Turn 1: Generate ---
    _progress(f"Turn 1: Generating architectures for {len(batch_segment_ids)} segments...")
    t0 = time.time()

    # Count total fills and build a checklist per segment
    fill_checklist_parts = []
    total_fills = 0
    for sid in batch_segment_ids:
        tmpl = (numbered_templates or {}).get(sid, "")
        seg_fill_count = tmpl.count("[LLM_FILL_")
        total_fills += seg_fill_count
        fill_checklist_parts.append(f"  {sid}: fills 1..{seg_fill_count}")
    fill_checklist = "\n".join(fill_checklist_parts)

    generation_prompt = (
        f"Fill ALL {total_fills} numbered [LLM_FILL_N] markers.\n\n"
        f"CHECKLIST — you must output every one of these:\n{fill_checklist}\n\n"
        f"Use === FILL: <segment_id> / <N> === delimiters for each fill.\n"
        f"[LLM_FILL_2] is the most important: it must contain a fenced code block\n"
        f"for EVERY file in the segment, each starting with # FILE: or // FILE:.\n\n"
        f"{context}"
    )
    messages.append({"role": "user", "content": generation_prompt})

    try:
        gen_response = await llm_call_fn(
            provider_id=provider_id, model_id=model_id,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            max_tokens=128_000, timeout_seconds=600,
        )
    except Exception as e:
        result.exit_reason = f"LLM call failed on Turn 1: {e}"
        _progress(result.exit_reason)
        return result

    gen_content = _extract_content(gen_response)
    gen_duration = time.time() - t0
    result.total_llm_calls += 1
    result.turns.append(LoopTurn(1, "model", f"Generated {len(gen_content)} chars", gen_duration))
    _progress(f"Turn 1 complete: {len(gen_content)} chars in {gen_duration:.1f}s")

    # --- Splice fills into templates ---
    fills = parse_fills(gen_content)
    _progress(f"Parsed {len(fills)} fills from output")

    if numbered_templates and fills:
        arch_docs = splice_fills_into_templates(numbered_templates, fills)
        _progress(f"Spliced fills into {len(arch_docs)} templates")
    else:
        # Fallback: parse as segment-delimited output (old format)
        parsed = parse_agentic_output(gen_content, batch_segment_ids)
        arch_docs = parsed.get_all_arch_docs()
        _progress(f"Fallback parse: {len(arch_docs)} arch docs")

    # Create a ParseResult-compatible object for confidence scoring
    parsed = parse_agentic_output(gen_content, batch_segment_ids) if not (numbered_templates and fills) else None

    # Store a COMPACT summary in messages instead of the full raw output.
    # This prevents the fix call from re-sending 169K chars of Turn 1.
    _compact_summary = f"[Turn 1 produced {len(arch_docs)} architecture docs, {len(fills)} fills parsed and spliced.]"
    messages.append({"role": "assistant", "content": _compact_summary})

    # --- Check-Fix Loop ---
    current_output = gen_content

    for cycle in range(1, MAX_CHECK_CYCLES + 1):
        result.total_check_cycles = cycle

        _progress(f"Check cycle {cycle}: Running deterministic checks...")
        t1 = time.time()

        # On subsequent cycles, re-parse fills from the fix output
        if cycle > 1:
            fills = parse_fills(current_output)
            if numbered_templates and fills:
                arch_docs = splice_fills_into_templates(numbered_templates, fills)
            else:
                parsed_out = parse_agentic_output(current_output, batch_segment_ids)
                arch_docs = parsed_out.get_all_arch_docs()
        check_report = run_deterministic_checks(
            arch_docs=arch_docs, preflight=preflight_result,
            segment_file_scopes=segment_file_scopes, cycle_number=cycle,
        )
        result.final_check = check_report
        check_duration = time.time() - t1

        result.turns.append(LoopTurn(
            len(result.turns) + 1, "check",
            f"Cycle {cycle}: {len(check_report.hard_blocks)} blocks, {len(check_report.warnings)} warnings",
            check_duration,
        ))

        _progress(f"Check cycle {cycle}: {len(check_report.hard_blocks)} HARD_BLOCK, "
                   f"{len(check_report.warnings)} WARNING, {len(check_report.infos)} INFO")

        # All clear
        if check_report.passed:
            result.arch_docs = arch_docs
            result.converged = True
            result.exit_reason = f"Converged after {cycle} check cycle(s)"
            result.confidence_score = _compute_loop_confidence(check_report, parsed, cycle)
            _progress(result.exit_reason)
            return result

        # Last cycle
        if cycle >= MAX_CHECK_CYCLES:
            result.arch_docs = arch_docs
            result.exit_reason = f"Exiting after {MAX_CHECK_CYCLES} cycles with {len(check_report.hard_blocks)} blockers"
            result.confidence_score = _compute_loop_confidence(check_report, parsed, cycle)
            _progress(result.exit_reason)
            return result

        # Feed back for fixes
        _progress(f"Turn {len(result.turns) + 1}: Requesting fixes...")
        t2 = time.time()

        check_prompt = format_check_report_for_prompt(check_report)
        # Include only the affected arch docs in the fix prompt, not all of them
        _affected_segs = set()
        for item in (check_report.hard_blocks + check_report.warnings):
            seg = getattr(item, "segment_id", "") or ""
            if seg:
                _affected_segs.add(seg)
        _fix_context_parts = []
        for seg_id in _affected_segs:
            doc = arch_docs.get(seg_id, "")
            if doc:
                _fix_context_parts.append(f"=== SEGMENT: {seg_id} ===\n{doc}\n=== END SEGMENT ===\n")
        _fix_context = "\n".join(_fix_context_parts) if _fix_context_parts else "(no affected segments found)"
        fix_instruction = (
            f"{check_prompt}\n\n"
            f"Here are the affected architecture docs:\n\n{_fix_context}\n\n"
            f"Fix ALL hard blocks. Output ONLY the corrected fills using === FILL: seg_id / N === delimiters."
        )
        messages.append({"role": "user", "content": fix_instruction})

        try:
            fix_response = await llm_call_fn(
                provider_id=provider_id, model_id=model_id,
                messages=[{"role": "system", "content": system_prompt}] + messages,
                max_tokens=128_000, timeout_seconds=600,
            )
        except Exception as e:
            result.arch_docs = arch_docs
            result.exit_reason = f"LLM call failed on fix cycle {cycle}: {e}"
            _progress(result.exit_reason)
            return result

        fix_content = _extract_content(fix_response)
        fix_duration = time.time() - t2
        result.total_llm_calls += 1
        result.turns.append(LoopTurn(
            len(result.turns) + 1, "model",
            f"Fix cycle {cycle}: {len(fix_content)} chars", fix_duration,
        ))
        messages.append({"role": "assistant", "content": fix_content})
        current_output = fix_content
        _progress(f"Fix cycle {cycle} complete: {len(fix_content)} chars in {fix_duration:.1f}s")

    # Fallthrough
    result.arch_docs = parse_agentic_output(current_output, batch_segment_ids).get_all_arch_docs()
    result.exit_reason = "Loop completed (fallthrough)"
    return result


def _compute_loop_confidence(check_report: CheckReport, parsed: Optional[ParseResult], cycles_used: int) -> float:
    score = 0.0
    if parsed and parsed.segment_count > 0:
        score += max(0.0, 0.3 - len(parsed.parse_errors) * 0.05)
    elif parsed is None:
        score += 0.3  # Fills-only path — no parse errors possible
    if not check_report.has_blockers:
        score += 0.4
        score += max(0.0, 0.1 - len(check_report.warnings) * 0.02)
    elif check_report.results:
        score += 0.2 * (1.0 - len(check_report.hard_blocks) / len(check_report.results))
    if cycles_used <= 1:
        score += 0.2
    elif cycles_used == 2:
        score += 0.1
    return min(score, 1.0)
