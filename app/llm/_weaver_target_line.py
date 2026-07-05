# FILE: app/llm/_weaver_target_line.py
# Purpose: live8 — deterministic Target-line upgrade: a stated absolute path outranks a verbal chain.
# Called-by: app.llm.weaver_stream
# Depends-on: app.llm.greenfield_autoscope
# Last-renovated: 2026-07-04
"""The Target folder/location line is LOAD-BEARING for greenfield jobs, and
the model cannot be trusted alone with it: at 23:23 on 2026-07-04 the weave
restated the verbal chain "Documents/Games/Tazza's Tetris" even though
Astra's own previous reply had named the exact real path
C:\\Users\\dizzi\\OneDrive\\Documents\\Games\\Tazza's Tetris. The verbal form
happens to resolve on this machine (registry User Shell Folders), but it is
one renamed user folder or registry hiccup away from re-minting a phantom
target (the live5 incident). This post-pass fixes the FORM deterministically:

    if the woven Target line is a relative/verbal chain AND the conversation
    (user or assistant) states exactly one absolute path whose trailing
    segments match that chain, rewrite the line to the absolute path.

Never changes WHICH folder is targeted — only upgrades a verbal citation of
the same place to the precise form. Ambiguity (two different matching
absolute paths) leaves the line untouched; the autoscope's registry
resolution remains the fallback.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_TARGET_LINE_RE = re.compile(
    r"^(?P<prefix>\s*\*{0,2}Target folder/location\*{0,2}\s*:\s*)(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_ABS_ANCHOR_RE = re.compile(r"^[A-Za-z]:[\\/]")


def _norm_seg(seg: str) -> str:
    """Segment form for comparison: straight apostrophes, casefold, and the
    'my Documents' -> 'documents' user phrasing collapsed."""
    s = seg.replace("’", "'").replace("‘", "'").strip().casefold()
    if s.startswith("my ") and len(s) > 3:
        s = s[3:]
    return s


def _collect_abs_candidates(messages: Any) -> List[str]:
    """Absolute path candidates named anywhere in the conversation, using the
    same tokenizer + cleaner as the autoscope (single source of truth)."""
    from app.llm.greenfield_autoscope import _WIN_PATH_RE, _clean_candidate

    out: List[str] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "")
        if not content:
            continue
        for raw in _WIN_PATH_RE.findall(content):
            cand = _clean_candidate(raw)
            if not cand or not _ABS_ANCHOR_RE.match(cand):
                continue
            norm = os.path.normpath(cand).replace("\\", "/")
            # normpath resolves traversal; anything that still escapes or is a
            # bare drive root / absurdly long is not a project folder citation.
            if ".." in norm.split("/"):
                continue
            if len(norm) < 4 or norm[1] != ":" or "/" not in norm[3:]:
                continue
            if len(norm) > 240:
                continue
            out.append(norm)
    return out


def upgrade_target_line(job_description: str, messages: Any) -> Tuple[str, Optional[str]]:
    """Rewrite a verbal Target line to the conversation's stated absolute path.

    Returns (possibly-rewritten text, the absolute path used or None).
    Deterministic and conservative: ALL verbal segments must match a
    contiguous run of the stated path's segments, and exactly ONE distinct
    root may result — otherwise no change.
    """
    if not job_description:
        return job_description, None

    m = _TARGET_LINE_RE.search(job_description)
    if not m:
        return job_description, None

    value = m.group("value").strip()
    if _ABS_ANCHOR_RE.match(value):
        return job_description, None  # already the precise form

    # Verbal chain -> comparison segments (same arrow/spaced-slash tolerance
    # as the autoscope's line sanitiser).
    sane = value.replace("→", "/").replace("›", "/").replace(" > ", "/")
    sane = re.sub(r"\s+/\s+", "/", sane)
    chain = [_norm_seg(s) for s in re.split(r"[\\/]", sane) if s.strip()]
    if not chain:
        return job_description, None

    matches: Dict[str, str] = {}  # casefolded root -> last-seen spelling
    for cand in _collect_abs_candidates(messages):
        parts = [p for p in cand.split("/") if p]
        segs = [_norm_seg(p) for p in parts]
        # The chain must match a CONTIGUOUS run of segments below the drive
        # anchor (segs[0] is 'c:'). Segments after the run — file names,
        # prose the tempered path regex swallowed ("...\\spec.md` now") —
        # are truncated: the run's end IS the folder the message named.
        for start in range(1, len(segs) - len(chain) + 1):
            if segs[start:start + len(chain)] == chain:
                root = "/".join(parts[: start + len(chain)])
                matches[root.casefold()] = root
                break

    if len(matches) != 1:
        if len(matches) > 1:
            logger.info(
                "[weaver_target_line] %d distinct absolute paths match %r — "
                "ambiguous, leaving the verbal line for autoscope resolution",
                len(matches), value,
            )
        return job_description, None

    resolved = next(iter(matches.values())).replace("/", "\\")
    new_text = (
        job_description[: m.start("value")] + resolved + job_description[m.end("value"):]
    )
    logger.info(
        "[weaver_target_line] Upgraded Target line %r -> %r (stated in conversation)",
        value, resolved,
    )
    return new_text, resolved


__all__ = ["upgrade_target_line"]
