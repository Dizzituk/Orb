# FILE: scripts/legal_case/legal_case_rebuild.py
"""
Legal case artefact rebuilder — orchestration entry point.

Runs after legal_case_extractor and legal_case_reconcile have already
produced screenshots_ocr.json and screenshots_reconciliation.json.

Rebuilds two artefacts to a shippable standard:

  1. delivery_log_cleaned.xlsx    — Daily Log with Evidence Ref column,
                                     Expenses, and Dashboard. Styled via
                                     ASTRA's xlsx_builder using the
                                     astra_minimal theme.

  2. work_legal_management_log.docx — Narrative evidence record with
                                       proper heading hierarchy, an
                                       appended Exhibit Index table, and
                                       a closing drafting note.

Both artefacts use the `legal_evidence_bundle` archetype's formatting
conventions (see app/styling/archetypes/legal_evidence_bundle.md).

Before overwriting, previous versions of each output file are moved
into an _archive/ subfolder with a timestamp suffix. No destructive
deletes, per ASTRA's hard rules.

Usage:
    python -m scripts.legal_case.legal_case_rebuild
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict

from scripts.legal_case._bootstrap import bootstrap_astra
from scripts.legal_case._rebuild_docx import rebuild as rebuild_docx
from scripts.legal_case._rebuild_xlsx import rebuild as rebuild_xlsx

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

LEGAL_ROOT = Path(r"C:\Users\dizzi\OneDrive\Documents\Work Legal")
ARCHIVE_DIR = LEGAL_ROOT / "_archive"
NARRATIVE_TXT = LEGAL_ROOT / "work_legal_management_log_content.txt"
RECONCILIATION = LEGAL_ROOT / "screenshots_reconciliation.json"
SOURCE_XLSX = LEGAL_ROOT / "delivery_log.xlsx"
OUTPUT_XLSX = LEGAL_ROOT / "delivery_log_cleaned.xlsx"
OUTPUT_DOCX = LEGAL_ROOT / "work_legal_management_log.docx"


def _archive_existing(path: Path) -> None:
    """Move an existing file to _archive with a timestamp suffix."""
    if not path.exists():
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    dest = ARCHIVE_DIR / f"{path.stem}.{ts}{path.suffix}"
    shutil.move(str(path), str(dest))
    logger.info("[rebuild] Archived %s -> %s", path.name, dest.name)


def run() -> Dict[str, Dict]:
    """Orchestrate the full rebuild. Returns a dict of artefact results."""
    bootstrap_astra()

    # Preconditions.
    for required in (NARRATIVE_TXT, RECONCILIATION, SOURCE_XLSX):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    # Archive previous outputs (non-destructive).
    _archive_existing(OUTPUT_XLSX)
    _archive_existing(OUTPUT_DOCX)

    # 1. Spreadsheet.
    xlsx_result = rebuild_xlsx(
        source_xlsx=SOURCE_XLSX,
        reconciliation_path=RECONCILIATION,
        output_path=OUTPUT_XLSX,
    )

    # 2. Narrative document.
    docx_result = rebuild_docx(
        narrative_txt_path=NARRATIVE_TXT,
        reconciliation_path=RECONCILIATION,
        output_path=OUTPUT_DOCX,
    )

    logger.info("[rebuild] Done.")
    logger.info("[rebuild]   XLSX: %s (%d bytes)", xlsx_result["path"], xlsx_result["size_bytes"])
    logger.info("[rebuild]   DOCX: %s (%d bytes)", docx_result["path"], docx_result["size_bytes"])
    return {"xlsx": xlsx_result, "docx": docx_result}


if __name__ == "__main__":
    run()
