# FILE: scripts/legal_case/_exhibit_index.py
# Purpose: Build the exhibit index table rows from the reconciliation JSON.
# Called-by: scripts.legal_case._rebuild_docx
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Build the exhibit index table rows from the reconciliation JSON.

The exhibit index maps each dated day in the Daily Log to the screenshot
filename(s) that corroborate it. This is the cross-reference table that
a solicitor uses to find the underlying evidence for any claim in the
narrative or any row in the spreadsheet.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _format_date_human(iso: str) -> str:
    """2025-11-18 -> 18 Nov 2025."""
    import datetime as _dt
    try:
        d = _dt.date.fromisoformat(iso)
    except Exception:
        return iso
    return d.strftime("%d %b %Y")


def build_exhibit_rows(reconciliation_path: Path) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
    """Return (table_rows, mapping) where:
      - table_rows is the body for the exhibit index table (no header row)
      - mapping is a list of dicts per-day, for use by the spreadsheet
        rebuilder to populate the Evidence Ref column.
    """
    data = json.loads(reconciliation_path.read_text(encoding="utf-8"))
    rows: List[List[str]] = []
    mapping: List[Dict[str, Any]] = []

    for i, m in enumerate(data.get("matches", []), start=1):
        ref = f"E{i:03d}"
        filenames = m.get("screenshot_filenames", []) or []
        # Compact: strip the "Screenshot_" prefix and ".png" suffix, keep just
        # the YYYYMMDD-HHMMSS stamp which is unique and sortable, and shorter.
        compact = []
        for fn in filenames:
            base = fn
            if base.startswith("Screenshot_"):
                base = base[len("Screenshot_"):]
            for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            # Trim numeric suffixes like (1) that some duplicates carry.
            if base.endswith("(1)"):
                base = base[:-3]
            compact.append(base)
        fn_display = ", ".join(compact[:2])
        if len(compact) > 2:
            fn_display += f" (+{len(compact) - 2} more)"

        date_human = _format_date_human(m["date"])
        parcels = m.get("sheet_parcels")
        stops = m.get("sheet_stops")
        hrs = m.get("sheet_delivery_hours")

        # A compact factual description, no editorialising.
        parts: List[str] = []
        if stops is not None:
            parts.append(f"{stops} stops assigned")
        if parcels is not None:
            parts.append(f"{parcels} parcels delivered")
        if m.get("sheet_failed"):
            parts.append(f"{m['sheet_failed']} failed")
        if hrs is not None:
            try:
                parts.append(f"{float(hrs):.2f}h delivery time")
            except Exception:
                pass
        description = "; ".join(parts) if parts else "Finished-day summary screenshot"

        # Fold the day-of-week into the description so we keep the info
        # but free up a column for the content that actually needs space.
        day = m.get("day") or ""
        if day and description:
            description = f"{day}. {description}"
        elif day:
            description = day
        rows.append([
            ref,
            date_human,
            fn_display,
            description,
        ])
        mapping.append({
            "date": m["date"],
            "ref": ref,
            "filenames": filenames,
        })

    # Dates without evidence get their own rows at the end.
    for iso in data.get("dates_without_evidence", []):
                rows.append([
            "",  # no exhibit ID - there's no exhibit
            _format_date_human(iso),
            "(none)",
            "Spreadsheet row present but no image evidence located",
        ])

    return rows, mapping


EXHIBIT_HEADERS = ["Ref", "Date", "Screenshot(s)", "Description"]
