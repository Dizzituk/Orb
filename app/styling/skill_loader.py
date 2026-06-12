# FILE: app/styling/skill_loader.py
# Purpose: Skill resolver for styled file creators.
# Called-by: app.debug.executors.styled_files
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Skill resolver for styled file creators.

A "skill" is a markdown playbook under app/styling/skills/ that tells the
structuring step how a particular class of document should be shaped,
toned, and formatted. The resolver handles three jobs:

  1. Explicit resolution: skill="formal_document" loads formal_document.md.
  2. Auto-detection: infer a skill from a filename/title when the caller
     does not specify one (legal keywords -> formal_document, etc.).
  3. Safe fallback: if nothing matches, return None and let the caller
     render blocks as-is (old behaviour).

The loader is deliberately tiny: one function, a small keyword map, and
a filesystem read. No caching yet — skill files are small and rare, and
a disk hit per call is not worth complicating this module over.

Public API:
    resolve_skill(skill_name=None, filename_or_path="") -> Optional[SkillDoc]
    list_skills() -> list[str]

Where SkillDoc is a plain dict: {"id": str, "body": str, "path": Path}.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

SKILLS_DIR: Path = Path(__file__).parent / "skills"

# Filename keywords that steer auto-detection toward formal register.
# Used only when the caller passes neither a skill name nor one already
# set by a higher-level classifier.
_FORMAL_KEYWORDS = (
    "legal", "evidence", "solicitor", "lawyer", "court", "tribunal",
    "letter", "formal", "statement", "affidavit", "witness",
    "complaint", "grievance", "claim", "case", "council", "hmrc",
    "dispute", "appeal", "subject_access", "leigh_day",
)

# Keywords that steer toward the spreadsheet-specific skill. Only
# consulted for xlsx callers.
_DATA_XLSX_KEYWORDS = (
    "log", "tracker", "ledger", "register", "manifest", "inventory",
    "schedule", "rota", "expenses", "evidence", "records",
)


def list_skills() -> List[str]:
    """Return all available skill IDs (markdown filename stems)."""
    if not SKILLS_DIR.exists():
        return []
    return sorted(
        p.stem
        for p in SKILLS_DIR.glob("*.md")
        if p.stem.lower() != "readme"
    )


def _load(skill_id: str) -> Optional[Dict]:
    """Load a specific skill by id, or None if the file does not exist."""
    path = SKILLS_DIR / f"{skill_id}.md"
    if not path.exists() or not path.is_file():
        return None
    try:
        body = path.read_text(encoding="utf-8")
    except Exception:
        return None
    return {"id": skill_id, "body": body, "path": path}


def _autodetect_id(filename_or_path: str, kind: str) -> str:
    """Choose a skill id based on filename keywords and artefact kind.

    kind: one of 'doc' (docx/pdf/html) or 'xlsx'.
    """
    hay = (filename_or_path or "").lower()

    if kind == "xlsx":
        # Spreadsheets default to data_spreadsheet if any data-ish
        # keyword appears; otherwise still data_spreadsheet (it's the
        # only xlsx skill at the moment and is the sensible default).
        for kw in _DATA_XLSX_KEYWORDS:
            if kw in hay:
                return "data_spreadsheet"
        return "data_spreadsheet"

    # Documents - check for the more specific evidence-bundle keywords
    # before falling back to the general formal_document.
    if "evidence" in hay and any(k in hay for k in ("bundle", "record", "log", "case", "leigh_day", "solicitor")):
        # Only return it if the file exists (user may not have installed it).
        if _load("legal_evidence_bundle") is not None:
            return "legal_evidence_bundle"

    for kw in _FORMAL_KEYWORDS:
        if kw in hay:
            return "formal_document"
    return "casual_document"


def resolve_skill(
    skill_name: Optional[str] = None,
    filename_or_path: str = "",
    kind: str = "doc",
) -> Optional[Dict]:
    """Resolve the skill to use for this creation call.

    Args:
        skill_name: explicit skill id from the caller. Wins if provided
            and the file exists.
        filename_or_path: filename or path of the artefact being created,
            used for keyword-based auto-detection.
        kind: 'doc' for docx/pdf/html, 'xlsx' for spreadsheets.

    Returns:
        A dict {"id", "body", "path"} if a skill was resolved, else None.
    """
    if skill_name:
        loaded = _load(skill_name.strip())
        if loaded is not None:
            return loaded
        # Fall through to auto-detection if the requested skill file
        # does not exist — better than silently rendering without a
        # skill the caller expected.

    auto_id = _autodetect_id(filename_or_path, kind)
    return _load(auto_id)
