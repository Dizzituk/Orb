# FILE: scripts/legal_case/_narrative_parser.py
"""
Parse the existing narrative document content (plain-text form) into
structured blocks that ASTRA's docx builder can render.

The existing txt version of the narrative is treated as source of truth
for prose — we are not re-generating the text, only giving it structure,
styling, and an appended exhibit index.

Output shape matches the content-block schema expected by
app/styling/docx_builder.py:
  [
    {"type": "heading", "level": 1, "text": "..."},
    {"type": "paragraph", "text": "..."},
    {"type": "rule"} / {"type": "spacer"},
    {"type": "table", "headers": [...], "rows": [[...], ...]},
    ...
  ]
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

# Top-level sections in the existing narrative. Anything matching one of
# these lines becomes a level-1 heading.
_KNOWN_SECTIONS = {
    "Lorry Delays / Depot Infrastructure",
    "Manager Statements",
    "Other Drivers / Workplace Culture",
    "Ming's Behaviour",
    "Safety / Health Issues",
    "Structural Observations",
    "Drafting note",
}

# Date-prefixed line used as a sub-heading inside each section.
_DATE_ENTRY_RE = re.compile(
    r"^\d{1,2}\s+[A-Za-z]+\s+\d{4}"       # "19 March 2026"
    r"|^[A-Z][a-z]+\s+[A-Z][a-z]+\s+\d{4}" # "Early April 2026"
    r"|^Around\s+\d"                       # "Around 24 March 2026"
    r"|^Mid-[A-Z][a-z]+\s+\d{4}"           # "Mid-April 2026"
    r"|^\d{1,2}\s+or\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}"  # "13 or 14 April 2026"
    r"|^Ongoing\s*$"                       # literal "Ongoing"
)

# Lines known to be intro / purpose at the top — treated specially.
_INTRO_HEADERS = {"Purpose", "Important context", "Recurring infrastructure issue", "Cold Rose", "Coverack", "Team lift parcels", "Driver health", "10 April 2026 route comparison", "Carryover compounding", "Pressure to switch areas", "Customer relationships", "Self-management of impossible volume", "Fuel cost pressure"}


def parse_narrative(txt_path: Path, skip_title: bool = False) -> List[Dict[str, Any]]:
    """Read the narrative txt and return a list of doc blocks.

    When skip_title is True, the first paragraph (which is normally
    the document's own title line) is dropped so it doesn't collide
    with a title the renderer adds separately via build_docx(title=...).
    """
    raw = txt_path.read_text(encoding="utf-8")
    # Split into non-empty paragraph groups.
    paragraphs: List[str] = []
    buf: List[str] = []
    for line in raw.splitlines():
        if line.strip():
            buf.append(line.rstrip())
        else:
            if buf:
                paragraphs.append("\n".join(buf))
                buf = []
    if buf:
        paragraphs.append("\n".join(buf))

    blocks: List[Dict[str, Any]] = []
    first_line_done = False

    for para in paragraphs:
        first_line = para.splitlines()[0].strip()

        # Top of document: the first paragraph is the title line.
        if not first_line_done:
            first_line_done = True
            if not skip_title:
                blocks.append({"type": "heading", "level": 1, "text": first_line})
                rest = "\n".join(para.splitlines()[1:]).strip()
                if rest:
                    blocks.append({"type": "paragraph", "text": rest})
            continue

        # Major top-level section headings.
        if first_line in _KNOWN_SECTIONS:
            blocks.append({"type": "spacer"})
            blocks.append({"type": "heading", "level": 1, "text": first_line})
            rest = "\n".join(para.splitlines()[1:]).strip()
            if rest:
                blocks.append({"type": "paragraph", "text": rest})
            continue

        # Intro sub-headings (Purpose / Important context / etc.).
        if first_line in _INTRO_HEADERS:
            blocks.append({"type": "heading", "level": 2, "text": first_line})
            rest = "\n".join(para.splitlines()[1:]).strip()
            if rest:
                blocks.append({"type": "paragraph", "text": rest})
            continue

        # Date-prefixed entries become level-3 headings.
        if _DATE_ENTRY_RE.match(first_line):
            blocks.append({"type": "heading", "level": 3, "text": first_line})
            rest = "\n".join(para.splitlines()[1:]).strip()
            if rest:
                blocks.append({"type": "paragraph", "text": rest})
            continue

        # Fallback: treat the whole paragraph as prose.
        blocks.append({"type": "paragraph", "text": para})

    return blocks
