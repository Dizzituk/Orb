# FILE: app/memory/_working_set_bootstrap.py
"""
Reference-driven bootstrap for the working set.

When the user says "the HTML file" or "the dashboard" and we have a
canonical folder for the project, this module looks for an
unambiguous match on disk and registers it into the working set.
Runs at the start of every chat turn, BEFORE prompt building, so the
context block has fresh data to inject.

Strict rules per Taz's design (never confidently guess):
- Only fires on explicit reference phrases (regex on the user message)
- Only registers if there is EXACTLY ONE candidate in the canonical
  folder.  Zero candidates = skip silently.  Multiple = skip silently
  (let the model ask the user which one).
- Never crawls outside the canonical folder.
- Never auto-loads files just because a content keyword appeared
  (e.g. "mileage" does NOT trigger a delivery_log.xlsx load).

Project canonical folders are inferred from the project name on first
use and cached on the working set.  E.g. project name "Work" maps to
OneDrive/Documents/Work/.

v1.0 (2026-05-24): Initial implementation.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

from app.memory import working_set

logger = logging.getLogger(__name__)


# =============================================================================
# REFERENCE PATTERN -> CANDIDATE FILE GLOBS
# =============================================================================
# Each (pattern, globs) entry means: if the user message matches the
# pattern, look in the canonical folder for files matching any of the
# globs.  Order matters — earlier patterns win on ties.
# Patterns are case-insensitive and use word boundaries so "the dashboard"
# doesn't match "this dashboard" or "car dashboard".

_REFERENCE_PATTERNS: List[Tuple[re.Pattern, List[str]]] = [
    # "the HTML file" / "the HTML" / "the html"
    (
        re.compile(r"\bthe\s+html(?:\s+file)?\b", re.IGNORECASE),
        ["*.html", "*.htm"],
    ),
    # "the dashboard" / "the dashboard.html"
    (
        re.compile(r"\bthe\s+dashboard(?:\.[a-z]+)?\b", re.IGNORECASE),
        ["dashboard.*", "*dashboard*.html", "*dashboard*.htm"],
    ),
    # "the spreadsheet" / "the excel" / "the xlsx"
    (
        re.compile(r"\bthe\s+(?:spreadsheet|excel|xlsx)\b", re.IGNORECASE),
        ["*.xlsx", "*.xls", "*.csv"],
    ),
    # "the log" / "the log file"
    (
        re.compile(r"\bthe\s+log(?:\s+file)?\b", re.IGNORECASE),
        ["*log*.txt", "*log*.html", "*log*.md", "incidents*.txt"],
    ),
    # "the doc" / "the document" / "the word doc"
    (
        re.compile(r"\bthe\s+(?:word\s+)?doc(?:ument)?\b", re.IGNORECASE),
        ["*.docx", "*.doc", "*.md"],
    ),
    # "the pdf"
    (
        re.compile(r"\bthe\s+pdf\b", re.IGNORECASE),
        ["*.pdf"],
    ),
    # "the work folder" / "in the work folder"
    # (we don't auto-pick a file here, but it's a clue to set the
    # canonical folder if not already set)
]


# =============================================================================
# CANONICAL FOLDER INFERENCE
# =============================================================================

# OneDrive Documents is the canonical root for project folders.
_ONEDRIVE_DOCS = r"C:\Users\dizzi\OneDrive\Documents"


def _infer_canonical_folder(project_name: str) -> Optional[str]:
    """Map a project name to a canonical folder under OneDrive/Documents.
    Returns the folder path if it exists on disk, else None.

    Example: project "Work" -> C:/Users/dizzi/OneDrive/Documents/Work
    """
    if not project_name:
        return None
    # Try the exact-name folder first, then a few normalised variants.
    candidates = [
        project_name,
        project_name.title(),
        project_name.lower(),
        project_name.replace(" ", "_"),
        project_name.replace("_", " "),
    ]
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        folder = os.path.join(_ONEDRIVE_DOCS, cand)
        if os.path.isdir(folder):
            return os.path.normpath(folder)
    return None


def _ensure_canonical_folder(project_id: int, project_name: Optional[str]) -> Optional[str]:
    """Make sure the working set knows this project's canonical folder.
    Returns the folder path or None if it can't be determined."""
    ws = working_set.get_or_create(project_id)
    if ws.canonical_folder and os.path.isdir(ws.canonical_folder):
        return ws.canonical_folder
    # Try to infer from project name
    if project_name:
        folder = _infer_canonical_folder(project_name)
        if folder:
            ws.canonical_folder = folder
            logger.info(
                "[working_set_bootstrap] Canonical folder for project=%d (%r) set to %s",
                project_id, project_name, folder,
            )
            # Persist immediately so we don't re-infer on every turn
            try:
                from app.memory.working_set import _save
                _save(ws)
            except Exception:
                pass
            return folder
    return None


# =============================================================================
# CANDIDATE MATCHING (strict: exactly-one or skip)
# =============================================================================

def _find_candidates(folder: str, globs: List[str]) -> List[str]:
    """Return absolute paths matching any of the globs in the folder.
    Non-recursive — we don't crawl subfolders.  Files starting with
    a dot are ignored (don't surface .DS_Store, .git, etc)."""
    if not folder or not os.path.isdir(folder):
        return []
    matches: set = set()
    folder_p = Path(folder)
    for pattern in globs:
        try:
            for p in folder_p.glob(pattern):
                if p.is_file() and not p.name.startswith("."):
                    matches.add(str(p.resolve()))
        except OSError:
            continue
    return sorted(matches)


# =============================================================================
# PUBLIC ENTRY
# =============================================================================

def bootstrap_from_message(
    project_id: int,
    project_name: Optional[str],
    message: str,
) -> int:
    """Inspect the user's message for explicit file references; if any
    resolve to exactly one file in the canonical folder, register them
    into the working set.

    Returns the number of files registered this turn (for logging).
    Errors are caught and logged but never raised — bootstrap must
    never break a chat turn.
    """
    if not project_id or not message:
        return 0
    try:
        folder = _ensure_canonical_folder(project_id, project_name)
        if not folder:
            return 0
        registered = 0
        seen_paths: set = set()
        for pattern, globs in _REFERENCE_PATTERNS:
            if not pattern.search(message):
                continue
            candidates = _find_candidates(folder, globs)
            if len(candidates) != 1:
                # Zero candidates: nothing to do.
                # Multiple candidates: silent skip (model will need to
                # ask the user which one — see "never confidently
                # guess" rule).  We deliberately do NOT pick one.
                if len(candidates) > 1:
                    logger.info(
                        "[working_set_bootstrap] Ambiguous reference "
                        "(%d matches in %s) — skipping auto-register",
                        len(candidates), folder,
                    )
                continue
            path = candidates[0]
            if path in seen_paths:
                continue
            seen_paths.add(path)
            working_set.register_file(
                project_id=project_id,
                path=path,
                action="read",  # bootstrap is read-equivalent
                model=working_set.get_current_model(),
            )
            registered += 1
            logger.info(
                "[working_set_bootstrap] Auto-registered '%s' (matched %r)",
                path, pattern.pattern,
            )
        return registered
    except Exception as e:
        logger.warning("[working_set_bootstrap] Failed: %s", e)
        return 0
