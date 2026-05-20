# FILE: app/debug/executors/styled_files.py
"""
Styled file creation executors: create_docx, create_pdf, create_xlsx, create_html_report.

Two calling modes:

  1. STRUCTURED (legacy): caller passes a fully-formed `content` list
     (or `sheets` list for xlsx). Blocks are rendered as-is via the
     builders. Back-compatible with every previous caller.

  2. BRIEF (v0.18.0): caller passes a natural-language `brief` (plus
     optional `source_material`). The executor resolves a skill
     playbook (explicit `skill` param or auto-detected from filename),
     runs a skill-guided structuring pass through the reasoning-tier
     LLM (GPT-5.4 / Opus class), and renders the returned blocks. This
     is the "write me a formal letter about X" path.

Theme defaults to 'auto' which inspects the filename for keywords like
'legal' or 'evidence' and picks the minimal theme; everything else gets
the default styled theme. Same path safety rules as user_files: must
write inside an allowed user folder.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.debug.executors.file_ops import _validate_path_for_write
from app.styling.themes import pick_theme
from app.styling.skill_loader import resolve_skill
from app.styling.structurer import (
    structure_document_async,
    structure_spreadsheet_async,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# SHARED HELPERS
# ----------------------------------------------------------------------

def _coerce_source_material(raw: Any) -> Any:
    """Accept dict/list/str/None. JSON strings are parsed."""
    if raw is None or isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s
    return raw


async def _structure_doc_if_brief(
    path: str,
    title: str,
    params: Dict[str, Any],
    existing_content: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """If the caller provided a brief (and no content), structure it.

    Returns the final block list to render. Falls back to a minimal
    safe layout if structuring fails, rather than producing a broken
    document.
    """
    brief = (params.get("brief") or "").strip()
    if existing_content:
        # Caller already provided structured blocks; respect them.
        return existing_content
    if not brief:
        # No content and no brief: nothing to structure.
        return []

    skill_name = (params.get("skill") or "").strip() or None
    skill = resolve_skill(skill_name, filename_or_path=f"{title} {path}", kind="doc")
    if not skill:
        logger.info("[styled_files] No skill resolved; falling back to brief-as-paragraph")
        return [
            {"type": "heading", "level": 1, "text": title or "Document"},
            {"type": "paragraph", "text": brief},
        ]

    logger.info("[styled_files] Structuring doc via skill=%s", skill["id"])
    source = _coerce_source_material(params.get("source_material"))
    try:
        blocks = await structure_document_async(
            skill=skill,
            brief=brief,
            title=title,
            source_material=source,
            job_type_str=(params.get("model_tier") or "text.heavy"),
        )
    except Exception as e:
        logger.error("[styled_files] structure_document failed: %s", e)
        blocks = []

    if not blocks:
        # Safe fallback so we never write an empty file.
        return [
            {"type": "heading", "level": 1, "text": title or "Document"},
            {"type": "paragraph", "text": brief},
            {"type": "rule"},
            {"type": "paragraph", "text": f"(Structuring via skill '{skill['id']}' returned no content; raw brief preserved above.)"},
        ]
    return blocks


async def _structure_xlsx_if_brief(
    path: str,
    title: str,
    params: Dict[str, Any],
    existing_sheets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Same idea for spreadsheets."""
    brief = (params.get("brief") or "").strip()
    if existing_sheets:
        return existing_sheets
    if not brief:
        return []

    skill_name = (params.get("skill") or "").strip() or None
    skill = resolve_skill(skill_name, filename_or_path=f"{title} {path}", kind="xlsx")
    if not skill:
        return [{"name": "Sheet1", "headers": ["Note"], "rows": [[brief]]}]

    logger.info("[styled_files] Structuring xlsx via skill=%s", skill["id"])
    source = _coerce_source_material(params.get("source_material"))
    try:
        sheets = await structure_spreadsheet_async(
            skill=skill,
            brief=brief,
            source_material=source,
            job_type_str=(params.get("model_tier") or "text.heavy"),
        )
    except Exception as e:
        logger.error("[styled_files] structure_spreadsheet failed: %s", e)
        sheets = []

    if not sheets:
        return [{"name": "Sheet1", "headers": ["Note"], "rows": [[brief]]}]
    return sheets


def _coerce_content(raw: Any) -> List[Dict[str, Any]]:
    """Accept either a list of dicts or a JSON string thereof."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


# =============================================================================
# DOCX
# =============================================================================

async def execute_create_docx(params: Dict[str, Any]) -> str:
    """Create a styled .docx file.

    Two modes:
      - content = [...blocks...]    -> render as-is (legacy)
      - brief = "natural language" -> structure via skill, then render
    """
    path = (params.get("path") or "").strip()
    title = (params.get("title") or "").strip() or "Untitled"
    subtitle = (params.get("subtitle") or "").strip() or None
    author = (params.get("author") or "").strip() or None
    theme_name = (params.get("theme") or "auto").strip()
    content = _coerce_content(params.get("content"))

    ok, err = _validate_path_for_write(path)
    if not ok:
        return err

    if not path.lower().endswith(".docx"):
        path = path + ".docx"

    # Skill-guided structuring when a brief is provided and no explicit
    # content blocks were given.
    content = await _structure_doc_if_brief(path, title, params, content)
    if not content:
        return "Error: either 'content' (structured blocks) or 'brief' (natural language) must be provided."

    theme = pick_theme(f"{title} {path}", theme_name)

    try:
        from app.styling.docx_builder import build_docx
        result = build_docx(path, title, content, theme, subtitle=subtitle, author=author)
    except Exception as e:
        logger.error("[executors.styled_files] create_docx failed: %s", e)
        return f"DOCX creation failed: {e}"

    return (
        f"Created DOCX: {result['path']} "
        f"({result['size_bytes']} bytes, {result['blocks_rendered']} blocks, "
        f"theme={result['theme']})."
    )


# =============================================================================
# PDF
# =============================================================================

async def execute_create_pdf(params: Dict[str, Any]) -> str:
    """Create a styled .pdf file.

    Two modes:
      - content = [...blocks...]    -> render as-is (legacy)
      - brief = "natural language" -> structure via skill, then render
    """
    path = (params.get("path") or "").strip()
    title = (params.get("title") or "").strip() or "Untitled"
    subtitle = (params.get("subtitle") or "").strip() or None
    author = (params.get("author") or "").strip() or None
    theme_name = (params.get("theme") or "auto").strip()
    content = _coerce_content(params.get("content"))

    ok, err = _validate_path_for_write(path)
    if not ok:
        return err

    if not path.lower().endswith(".pdf"):
        path = path + ".pdf"

    content = await _structure_doc_if_brief(path, title, params, content)
    if not content:
        return "Error: either 'content' (structured blocks) or 'brief' (natural language) must be provided."

    theme = pick_theme(f"{title} {path}", theme_name)

    try:
        from app.styling.pdf_builder import build_pdf
        result = build_pdf(path, title, content, theme, subtitle=subtitle, author=author)
    except Exception as e:
        logger.error("[executors.styled_files] create_pdf failed: %s", e)
        return f"PDF creation failed: {e}"

    return (
        f"Created PDF: {result['path']} "
        f"({result['size_bytes']} bytes, {result['blocks_rendered']} blocks, "
        f"theme={result['theme']})."
    )


# =============================================================================
# XLSX
# =============================================================================

async def execute_create_xlsx(params: Dict[str, Any]) -> str:
    """Create a styled .xlsx workbook."""
    path = (params.get("path") or "").strip()
    title = (params.get("title") or "").strip() or "Workbook"
    theme_name = (params.get("theme") or "auto").strip()

    sheets_raw = params.get("sheets")
    if isinstance(sheets_raw, str):
        try:
            sheets = json.loads(sheets_raw)
        except Exception:
            sheets = []
    elif isinstance(sheets_raw, list):
        sheets = sheets_raw
    else:
        sheets = []

    if not isinstance(sheets, list):
        sheets = []

    ok, err = _validate_path_for_write(path)
    if not ok:
        return err

    # Skill-guided structuring when a brief is provided and no sheets
    # were passed.
    sheets = await _structure_xlsx_if_brief(path, title, params, sheets)

    if not sheets:
        return "Error: either 'sheets' (list of sheet objects) or 'brief' (natural language) must be provided."

    if not path.lower().endswith(".xlsx"):
        path = path + ".xlsx"

    theme = pick_theme(f"{title} {path}", theme_name)

    try:
        from app.styling.xlsx_builder import build_xlsx
        result = build_xlsx(path, sheets, theme, title=title)
    except Exception as e:
        logger.error("[executors.styled_files] create_xlsx failed: %s", e)
        return f"XLSX creation failed: {e}"

    return (
        f"Created XLSX: {result['path']} "
        f"({result['size_bytes']} bytes, {result['sheets_rendered']} sheets, "
        f"theme={result['theme']})."
    )


# =============================================================================
# HTML
# =============================================================================

async def execute_create_html_report(params: Dict[str, Any]) -> str:
    """Create a styled single-file HTML report.

    Two modes:
      - content = [...blocks...]    -> render as-is (legacy)
      - brief = "natural language" -> structure via skill, then render
    """
    path = (params.get("path") or "").strip()
    title = (params.get("title") or "").strip() or "Untitled"
    subtitle = (params.get("subtitle") or "").strip() or None
    author = (params.get("author") or "").strip() or None
    theme_name = (params.get("theme") or "auto").strip()
    content = _coerce_content(params.get("content"))

    ok, err = _validate_path_for_write(path)
    if not ok:
        return err

    if not path.lower().endswith((".html", ".htm")):
        path = path + ".html"

    content = await _structure_doc_if_brief(path, title, params, content)
    if not content:
        return "Error: either 'content' (structured blocks) or 'brief' (natural language) must be provided."

    theme = pick_theme(f"{title} {path}", theme_name)

    try:
        from app.styling.html_builder import build_html
        result = build_html(path, title, content, theme, subtitle=subtitle, author=author)
    except Exception as e:
        logger.error("[executors.styled_files] create_html_report failed: %s", e)
        return f"HTML creation failed: {e}"

    return (
        f"Created HTML report: {result['path']} "
        f"({result['size_bytes']} bytes, {result['blocks_rendered']} blocks, "
        f"theme={result['theme']})."
    )
