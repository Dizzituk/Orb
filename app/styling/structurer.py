# FILE: app/styling/structurer.py
# Purpose: Skill-guided structuring pass for styled file creators.
# Called-by: app.debug.executors.styled_files
# Depends-on: app.llm.router, app.llm.schemas
# Last-renovated: 2026-06-11
"""
Skill-guided structuring pass for styled file creators.

Given a skill playbook and a free-form brief (what the user wants),
this module calls a reasoning-tier model to produce a structured
content list (or sheet list) that honours the skill's rules. That
structured output is then handed to the existing docx/pdf/xlsx/html
builders for rendering.

This is the "reasoning step" in the two-stage pipeline:
  Stage 1 (here)  : skill + brief + source_material -> JSON plan
  Stage 2 (builders): JSON plan -> styled file on disk

The model is routed via app.llm.router.call_llm_async so the existing
provider/model selection, fallbacks, and audit trail all apply. We ask
for JobType.TEXT_HEAVY which maps to the reasoning tier (full GPT-5.4
equivalent), because getting structure wrong is expensive and model
cost is trivial by comparison.

Public API:
    structure_document(skill, brief, title, source_material=None) -> list[dict]
    structure_spreadsheet(skill, brief, source_material=None) -> list[dict]

Both return plain Python objects matching the content schema used by
app/styling/{docx,pdf,xlsx,html}_builder.py.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# PROMPT SCAFFOLDING
# ──────────────────────────────────────────────────────────────────────

_DOC_BLOCK_SCHEMA = """\
Produce a JSON list of content blocks. Each block is one of:
  { "type": "heading", "level": 1 | 2 | 3, "text": "..." }
  { "type": "paragraph", "text": "..." }
  { "type": "list", "items": ["...", "..."], "ordered": false }
  { "type": "table", "headers": ["..."], "rows": [["..."], ["..."]] }
  { "type": "rule" }
  { "type": "spacer" }

Order in the list determines render order. Tables should be used sparingly
in narrative documents - prefer prose unless tabular data is genuinely
the clearest form. Lists should be used only for genuine lists (steps,
items, options), not for prose broken into fragments.
"""

_XLSX_SHEET_SCHEMA = """\
Produce a JSON list of sheet objects. Each sheet is:
  {
    "name": "Tab name (max 31 chars)",
    "headers": ["col1", "col2", ...],
    "rows": [["val1", "val2", ...], ...],
    "freeze_header": true,
    "auto_filter": true,
    "column_widths": [12, 30, 18]    // optional
  }

Header row MUST be row 1 - no decorative title rows above it, no
instruction rows between headers and data. Summaries (totals/averages)
go on a separate Dashboard sheet, never as a trailing row inside a data
sheet (that would break auto-filter).
"""


_DOC_SYSTEM_PROMPT = """\
You are a document structurer. Your job is to take a user's brief and
produce a structured JSON plan for a document that obeys a specific
stylistic playbook (the "skill").

You are NOT writing the document to be rendered as prose inside the chat.
You are producing a JSON array of content blocks that a rendering engine
will turn into a .docx or .pdf. Your output will be parsed as JSON.

Rules for your output:

1. Output ONLY valid JSON - a single top-level array of block objects.
   No preamble. No commentary. No code fences. Just the JSON array.
2. Every block must conform to the schema given below.
3. You MUST follow the skill playbook. The playbook tells you the
   document's register, what it must contain, what it must NOT contain,
   and how it should be toned. Violations of the playbook are defects,
   not style choices.
4. If source material is provided, weave it into the document where it
   fits the playbook. Do NOT invent facts not present in the source.
5. If the brief is ambiguous, err on the side of the playbook's
   conservative choices - better a plainer document than one that
   violates the register.

After these instructions you will be given:
  (a) the skill playbook (markdown), and
  (b) the user's brief and any source material.

Produce the JSON plan.
"""


_XLSX_SYSTEM_PROMPT = """\
You are a spreadsheet structurer. Your job is to take a user's brief
and produce a structured JSON plan for an Excel workbook that obeys a
specific stylistic playbook (the "skill").

Output ONLY valid JSON - a single top-level array of sheet objects.
No preamble. No commentary. No code fences.

Rules:
1. Every sheet object must conform to the schema.
2. Follow the skill playbook strictly - it defines multi-sheet
   conventions, header rules, and what NOT to do.
3. If source data is provided as a list of records, transform it into
   the sheet format specified by the playbook. Do not invent data.
4. When source data has more rows than are practical to inline (> 500),
   include a representative sample and note the truncation in a
   "Readme" sheet.
"""


# ──────────────────────────────────────────────────────────────────────
# JSON EXTRACTION
# ──────────────────────────────────────────────────────────────────────

def _extract_json(raw: str) -> Optional[Any]:
    """Pull a JSON value out of a model response, tolerant of fences."""
    if not raw:
        return None
    s = raw.strip()
    # Strip code fences.
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    # Find the outermost [...] or {...}.
    for opener, closer in (("[", "]"), ("{", "}")):
        start = s.find(opener)
        end = s.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidate = s[start:end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINTS
# ──────────────────────────────────────────────────────────────────────

def _format_source_material(source: Any) -> str:
    """Serialise source material into a compact, model-readable string."""
    if source is None:
        return ""
    if isinstance(source, str):
        return source.strip()
    try:
        return json.dumps(source, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(source)


async def _call_structurer(
    system_prompt: str,
    skill_body: str,
    user_block: str,
    job_type_str: str = "text.heavy",
) -> str:
    """One call into ASTRA's LLM router. Returns raw response text."""
    from app.llm.router import call_llm_async
    from app.llm.schemas import LLMTask, JobType

    # Map string to enum; default to TEXT_HEAVY (reasoning tier).
    try:
        job_type = JobType(job_type_str)
    except ValueError:
        job_type = JobType.TEXT_HEAVY

    task = LLMTask(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"SKILL PLAYBOOK:\n\n{skill_body}\n\n---\n\n{user_block}"},
        ],
        job_type=job_type,
    )
    result = await call_llm_async(task)
    return (result.content or "").strip()


async def structure_document_async(
    skill: Dict[str, Any],
    brief: str,
    title: str = "",
    source_material: Any = None,
    job_type_str: str = "text.heavy",
) -> List[Dict[str, Any]]:
    """Produce a document block list from a skill + brief.

    Returns an empty list if the model's output cannot be parsed as the
    expected schema (caller should fall back to a minimal safe layout).
    """
    source_block = _format_source_material(source_material)
    user_block_parts = [f"TITLE: {title or '(untitled)'}", "", f"BRIEF:\n{brief.strip()}"]
    if source_block:
        user_block_parts += ["", f"SOURCE MATERIAL:\n{source_block}"]
    user_block_parts += ["", "BLOCK SCHEMA:", _DOC_BLOCK_SCHEMA, "", "Return ONLY the JSON array of blocks."]
    user_block = "\n".join(user_block_parts)

    raw = await _call_structurer(_DOC_SYSTEM_PROMPT, skill["body"], user_block, job_type_str)
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        logger.warning("[structurer] Model returned non-list; raw=%r", raw[:300])
        return []
    # Trust-but-verify: keep only well-shaped blocks.
    clean: List[Dict[str, Any]] = []
    for b in parsed:
        if isinstance(b, dict) and isinstance(b.get("type"), str):
            clean.append(b)
    return clean


async def structure_spreadsheet_async(
    skill: Dict[str, Any],
    brief: str,
    source_material: Any = None,
    job_type_str: str = "text.heavy",
) -> List[Dict[str, Any]]:
    """Produce a sheet list from a skill + brief.

    Returns an empty list on parse failure.
    """
    source_block = _format_source_material(source_material)
    user_block_parts = [f"BRIEF:\n{brief.strip()}"]
    if source_block:
        user_block_parts += ["", f"SOURCE DATA:\n{source_block}"]
    user_block_parts += ["", "SHEET SCHEMA:", _XLSX_SHEET_SCHEMA, "", "Return ONLY the JSON array of sheet objects."]
    user_block = "\n".join(user_block_parts)

    raw = await _call_structurer(_XLSX_SYSTEM_PROMPT, skill["body"], user_block, job_type_str)
    parsed = _extract_json(raw)
    if not isinstance(parsed, list):
        logger.warning("[structurer] Model returned non-list for xlsx; raw=%r", raw[:300])
        return []
    clean: List[Dict[str, Any]] = []
    for sh in parsed:
        if isinstance(sh, dict) and sh.get("name"):
            clean.append(sh)
    return clean


# ──────────────────────────────────────────────────────────────────────
# SYNC WRAPPERS (for callers that aren't async)
# ──────────────────────────────────────────────────────────────────────

def structure_document(
    skill: Dict[str, Any],
    brief: str,
    title: str = "",
    source_material: Any = None,
    job_type_str: str = "text.heavy",
) -> List[Dict[str, Any]]:
    """Sync wrapper around structure_document_async."""
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(structure_document_async(skill, brief, title, source_material, job_type_str))
    raise RuntimeError("structure_document() called inside an event loop; use structure_document_async()")


def structure_spreadsheet(
    skill: Dict[str, Any],
    brief: str,
    source_material: Any = None,
    job_type_str: str = "text.heavy",
) -> List[Dict[str, Any]]:
    """Sync wrapper around structure_spreadsheet_async."""
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(structure_spreadsheet_async(skill, brief, source_material, job_type_str))
    raise RuntimeError("structure_spreadsheet() called inside an event loop; use structure_spreadsheet_async()")
