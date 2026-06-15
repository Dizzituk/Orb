# FILE: app/documents/csv_md_convert.py
# Purpose: csv ⇄ sheet snapshot and md ⇄ doc snapshot — the trivial direct
#          conversion paths (stdlib only).
# Called-by: app.documents.router
# Depends-on: stdlib (csv, re)
# Last-renovated: 2026-06-12
"""
csv/md conversion.

csv -> one-sheet workbook snapshot (numbers detected, everything else
string); save writes plain comma CSV of VALUES (formulas evaluate in the
editor; their last computed value is what lands in the file — documented).

md -> doc snapshot: #/##/### headings, -/* bullets, "1." numbered lines,
**bold** and *italic* inline. Save inverts the same rules. Anything fancier
(tables, code fences, links) survives as literal text — documented loss.
"""
from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path

from app.documents.docx_convert import (
    BODY_SIZE,  # noqa: F401  (kept for symmetry/reference)
    BULLET_PREFIX,
    H1_SIZE,
    H2_SIZE,
    H3_SIZE,
)

logger = logging.getLogger(__name__)

_TRUE = 1


# ── csv ────────────────────────────────────────────────────────────────────

def csv_to_snapshot(path: str) -> dict:
    raw = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except Exception:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(raw), dialect))

    cell_data: dict = {}
    max_col = 0
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if value == "":
                continue
            entry: dict = {"v": value, "t": 1}
            try:
                number = float(value)
                if value.strip() and not value.strip().startswith("+"):
                    entry = {"v": number, "t": 2}
                    if number.is_integer() and "." not in value and "e" not in value.lower():
                        entry["v"] = int(number)
            except ValueError:
                pass
            cell_data.setdefault(str(r), {})[str(c)] = entry
            max_col = max(max_col, c)

    sheet_id = "sheet-01"
    return {
        "id": f"wb-{Path(path).stem[:40]}",
        "name": Path(path).name,
        "appVersion": "0.25.0",
        "locale": "enUS",
        "sheetOrder": [sheet_id],
        "sheets": {sheet_id: {
            "id": sheet_id,
            "name": Path(path).stem[:31] or "Sheet1",
            "rowCount": max(len(rows) + 50, 100),
            "columnCount": max(max_col + 5, 26),
            "cellData": cell_data,
            "mergeData": [],
        }},
        "styles": {},
    }


def snapshot_to_csv(snapshot: dict, path: str) -> None:
    """First sheet's VALUES -> comma CSV (formulas write their last value)."""
    sheets = snapshot.get("sheets") or {}
    order = snapshot.get("sheetOrder") or list(sheets.keys())
    if not order:
        Path(path).write_text("", encoding="utf-8")
        return
    sheet = sheets.get(order[0]) or {}
    cell_data = sheet.get("cellData") or {}
    if len(order) > 1:
        logger.warning("[documents] csv save keeps only the first sheet (%s)", order[0])

    max_row = max((int(r) for r in cell_data.keys()), default=-1)
    out_rows = []
    for r in range(max_row + 1):
        columns = cell_data.get(str(r)) or {}
        max_col = max((int(c) for c in columns.keys()), default=-1)
        row = []
        for c in range(max_col + 1):
            entry = columns.get(str(c)) or {}
            value = entry.get("v")
            row.append("" if value is None else value)
        out_rows.append(row)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(out_rows)


# ── markdown ───────────────────────────────────────────────────────────────

_INLINE = re.compile(r"(\*\*([^*]+)\*\*)|(\*([^*]+)\*)")


def md_to_snapshot(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    stream_parts: list[str] = []
    text_runs: list[dict] = []
    paragraphs: list[dict] = []
    cursor = 0

    for line in text.splitlines():
        heading = 0
        body = line
        m = re.match(r"^(#{1,3})\s+", line)
        if m:
            heading = len(m.group(1))
            body = line[m.end():]
        elif re.match(r"^\s*[-*]\s+", line):
            body = BULLET_PREFIX + re.sub(r"^\s*[-*]\s+", "", line)
        # numbered lines pass through as-is ("1. " already reads correctly)

        # inline **bold** / *italic* -> runs (markers stripped)
        plain_text = ""
        runs_local: list[tuple[int, int, str]] = []
        pos = 0
        for match in _INLINE.finditer(body):
            plain_text += body[pos:match.start()]
            inner = match.group(2) if match.group(2) is not None else match.group(4)
            kind = "bold" if match.group(2) is not None else "italic"
            runs_local.append((len(plain_text), len(plain_text) + len(inner), kind))
            plain_text += inner
            pos = match.end()
        plain_text += body[pos:]

        stream_parts.append(plain_text)
        if heading:
            size = {1: H1_SIZE, 2: H2_SIZE, 3: H3_SIZE}[heading]
            if plain_text.strip():
                text_runs.append({"st": cursor, "ed": cursor + len(plain_text),
                                  "ts": {"bl": _TRUE, "fs": size}})
        else:
            for st, ed, kind in runs_local:
                ts = {"bl": _TRUE} if kind == "bold" else {"it": _TRUE}
                text_runs.append({"st": cursor + st, "ed": cursor + ed, "ts": ts})
        cursor += len(plain_text)

        stream_parts.append("\r")
        paragraphs.append({"startIndex": cursor})
        cursor += 1

    stream_parts.append("\n")
    data_stream = "".join(stream_parts)
    return {
        "id": f"doc-{Path(path).stem[:40]}",
        "body": {
            "dataStream": data_stream,
            "textRuns": text_runs,
            "paragraphs": paragraphs,
            "sectionBreaks": [{"startIndex": max(len(data_stream) - 1, 0)}],
        },
        "documentStyle": {"pageSize": {"width": 595, "height": 842}},
    }


def snapshot_to_md(snapshot: dict, path: str) -> None:
    """Doc snapshot -> markdown using the same heading/bullet conventions."""
    body = snapshot.get("body") or {}
    stream: str = body.get("dataStream") or ""
    text_runs = body.get("textRuns") or []

    lines: list[str] = []
    cursor = 0
    for para_meta in body.get("paragraphs") or []:
        end = int(para_meta.get("startIndex", cursor))
        text = stream[cursor:end]
        runs = [r for r in text_runs
                if int(r.get("st", 0)) < end and int(r.get("ed", 0)) > cursor]

        sizes = [int((r.get("ts") or {}).get("fs") or 0) for r in runs]
        all_bold = bool(runs) and all((r.get("ts") or {}).get("bl") == _TRUE for r in runs)
        if text.strip() and all_bold and sizes:
            biggest = max(sizes)
            level = 1 if biggest >= H1_SIZE else 2 if biggest >= H2_SIZE \
                else 3 if biggest >= H3_SIZE else 0
            if level:
                lines.append("#" * level + " " + text.strip())
                cursor = end + 1
                continue

        if text.startswith(BULLET_PREFIX):
            lines.append("- " + text[len(BULLET_PREFIX):])
            cursor = end + 1
            continue

        # inline bold/italic back to markers
        out = ""
        pos = cursor
        for run_meta in sorted(runs, key=lambda r: int(r.get("st", 0))):
            st = max(int(run_meta.get("st", 0)), cursor)
            ed = min(int(run_meta.get("ed", 0)), end)
            ts = run_meta.get("ts") or {}
            if st > pos:
                out += stream[pos:st]
            piece = stream[st:ed]
            if ts.get("bl") == _TRUE and "fs" not in ts:
                out += f"**{piece}**"
            elif ts.get("it") == _TRUE:
                out += f"*{piece}*"
            else:
                out += piece
            pos = ed
        out += stream[pos:end]
        lines.append(out)
        cursor = end + 1

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
