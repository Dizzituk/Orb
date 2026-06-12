# FILE: scripts/legal_case/_rebuild_docx.py
# Purpose: Rebuild the narrative evidence document with structured headings,
# Called-by: scripts.legal_case.legal_case_rebuild
# Depends-on: app.styling.docx_builder, app.styling.themes, scripts.legal_case._exhibit_index, scripts.legal_case._narrative_parser
# Last-renovated: 2026-06-11
"""
Rebuild the narrative evidence document with structured headings,
astra_minimal styling, and an appended exhibit index table.

The prose content is taken verbatim from the existing narrative txt
(work_legal_management_log_content.txt) to preserve the factual record
exactly. We only add:

  1. Structured headings (level 1 for section themes, level 3 for
     dated entries) so the document reads as a proper evidence log
     rather than a wall of text.
  2. A rule + "Exhibit Index" section at the end containing a table
     that maps every spreadsheet day to its supporting screenshot.
  3. A short closing drafting note.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from scripts.legal_case._narrative_parser import parse_narrative
from scripts.legal_case._exhibit_index import EXHIBIT_HEADERS, build_exhibit_rows

logger = logging.getLogger(__name__)


def rebuild(
    narrative_txt_path: Path,
    reconciliation_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Build the styled docx. Returns the build_docx result dict."""
    # 1. Narrative body (preserves the existing prose verbatim).
    blocks: List[Dict[str, Any]] = parse_narrative(narrative_txt_path, skip_title=True)

    # 2. Exhibit index section.
    exhibit_rows, _mapping = build_exhibit_rows(reconciliation_path)
    blocks.append({"type": "spacer"})
    blocks.append({"type": "rule"})
    blocks.append({"type": "heading", "level": 1, "text": "Exhibit Index"})
    blocks.append({
        "type": "paragraph",
        "text": (
            "The following table maps each working day for which operational "
            "data is recorded in the companion spreadsheet to the screenshot "
            "evidence that corroborates it. Filenames refer to the Work Screen "
            "shots folder, organised by month."
        ),
    })
    blocks.append({
        "type": "table",
        "headers": EXHIBIT_HEADERS,
        "rows": exhibit_rows,
    })

    # 3. Closing drafting note (rewrites the one the original already had
    #    so it appears AFTER the exhibit index rather than mid-document).
    blocks.append({"type": "spacer"})
    blocks.append({"type": "heading", "level": 1, "text": "Notes on this document"})
    blocks.append({
        "type": "paragraph",
        "text": (
            "This document is a narrative evidence summary. It is not a sworn "
            "witness statement and it does not contain legal conclusions. "
            "Supporting exhibits — screenshots, photographs, videos, payment "
            "records, messages, and the companion spreadsheet (delivery_log_"
            "cleaned.xlsx) — are listed in the Exhibit Index above. Each "
            "entry in the body that refers to a specific day should be "
            "cross-referenced against the Exhibit Index and the corresponding "
            "row in the spreadsheet."
        ),
    })
    blocks.append({
        "type": "paragraph",
        "text": (
            "Where a visible field on a screenshot (for example the rounded "
            "\"Duration\" figure on the end-of-tour screen) differs slightly "
            "from the corresponding cell in the spreadsheet, this is because "
            "the spreadsheet computes total delivery time from clock-in and "
            "clock-out timestamps, while the screenshot displays a rounded "
            "duration. Such differences are expected and do not indicate a "
            "conflict in the evidence."
        ),
    })

    from app.styling.themes import pick_theme
    from app.styling.docx_builder import build_docx

    # Keyword-based auto theme pick will choose astra_minimal because the
    # path/filename contains 'legal' and 'evidence'.
    theme = pick_theme(f"legal evidence {output_path.name}", "auto")
    result = build_docx(
        output_path=str(output_path),
        title="Work Legal Management and Conditions Log",
        content=blocks,
        theme=theme,
        subtitle="Narrative evidence record with exhibit index",
        author=None,
    )
    logger.info(
        "[rebuild_docx] Wrote %s (%d blocks, theme=%s)",
        output_path, result.get("blocks_rendered"), result.get("theme"),
    )
    return result
