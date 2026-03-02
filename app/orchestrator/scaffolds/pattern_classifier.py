# FILE: app/orchestrator/scaffolds/pattern_classifier.py
"""
Pattern Classifier — detects which scaffold template to apply.

Analyses the file path, architecture description, exports, and
props to classify each file into a known component pattern.

v1.0 (2026-03-01): Classifies React TSX component types.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Result of pattern classification for a single file."""
    file_path: str
    pattern: str          # "view" | "grid" | "detail" | "data" | "css" | "unknown"
    confidence: float     # 0.0 - 1.0
    signals: List[str]    # Reasons for classification


# ─── Classification signals ──────────────────────────────────────────

# Note: Use re.MULTILINE so $ anchors match end-of-line in combined text
_VIEW_SIGNALS = [
    (re.compile(r"View\.tsx$", re.I | re.M), 0.4, "filename ends with View.tsx"),
    (re.compile(r"(?:tab|tabs|navigation|route|routing)", re.I), 0.15, "mentions tabs/navigation"),
    (re.compile(r"useState", re.I), 0.1, "uses state management"),
    (re.compile(r"(?:container|dashboard|main|page)", re.I), 0.1, "container/dashboard pattern"),
]

_GRID_SIGNALS = [
    (re.compile(r"Grid\.tsx$", re.I | re.M), 0.4, "filename ends with Grid.tsx"),
    (re.compile(r"List\.tsx$", re.I | re.M), 0.3, "filename ends with List.tsx"),
    (re.compile(r"(?:\.map\(|cards?|grid|gallery|collection)", re.I), 0.15, "data mapping/grid pattern"),
    (re.compile(r"(?:onClick|onSelect|onCourseSelect)", re.I), 0.1, "item selection callback"),
]

_DETAIL_SIGNALS = [
    (re.compile(r"Detail\.tsx$", re.I | re.M), 0.4, "filename ends with Detail.tsx"),
    (re.compile(r"(?:single|detail|info|profile|page)", re.I), 0.15, "detail/single-item pattern"),
    (re.compile(r"(?:back|goBack|onBack|return)", re.I), 0.15, "back navigation"),
    (re.compile(r"(?:sections?|layout|heading)", re.I), 0.1, "section layout"),
]

_DATA_SIGNALS = [
    (re.compile(r"data\.ts$", re.I | re.M), 0.4, "filename ends with data.ts"),
    (re.compile(r"(?:interface|type\s+\w+\s*=)", re.I), 0.15, "TypeScript interfaces/types"),
    (re.compile(r"(?:mock|dummy|sample|DUMMY_)", re.I), 0.15, "mock/dummy data"),
    (re.compile(r"(?:export\s+const|export\s+interface)", re.I), 0.1, "data exports"),
]

_CSS_SIGNALS = [
    (re.compile(r"\.css$", re.I | re.M), 0.5, "CSS file extension"),
    (re.compile(r"(?:var\(--|grid|flex|responsive)", re.I), 0.15, "CSS patterns"),
]

_ALL_PATTERNS = {
    "view": _VIEW_SIGNALS,
    "grid": _GRID_SIGNALS,
    "detail": _DETAIL_SIGNALS,
    "data": _DATA_SIGNALS,
    "css": _CSS_SIGNALS,
}


def classify_file_pattern(
    file_path: str,
    architecture_text: str = "",
    exports: Optional[List[str]] = None,
    design_notes: str = "",
) -> PatternMatch:
    """Classify a file into a component pattern.

    Scores each pattern by checking the file path, architecture
    description, export list, and design notes against known signals.

    Args:
        file_path: Target file path.
        architecture_text: Architecture doc section for this file.
        exports: Exported symbol names.
        design_notes: Additional design notes.

    Returns:
        PatternMatch with the best-matching pattern.
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Quick exit for non-frontend files
    if ext not in (".tsx", ".ts", ".jsx", ".js", ".css"):
        return PatternMatch(file_path, "unknown", 0.0, ["not a frontend file"])

    # Combine all text for signal matching
    combined = f"{file_path}\n{architecture_text}\n{design_notes}"
    if exports:
        combined += "\n" + " ".join(exports)

    scores: Dict[str, tuple[float, List[str]]] = {}

    for pattern_name, signals in _ALL_PATTERNS.items():
        score = 0.0
        matched_signals: List[str] = []

        for regex, weight, reason in signals:
            if regex.search(combined):
                score += weight
                matched_signals.append(reason)

        scores[pattern_name] = (score, matched_signals)

    # Find the best match
    best_pattern = "unknown"
    best_score = 0.0
    best_signals: List[str] = []

    for pattern_name, (score, signals) in scores.items():
        if score > best_score:
            best_pattern = pattern_name
            best_score = score
            best_signals = signals

    # Minimum confidence threshold
    if best_score < 0.3:
        return PatternMatch(file_path, "unknown", best_score, best_signals)

    logger.info(
        "[scaffold] Classified %s as '%s' (%.2f): %s",
        os.path.basename(file_path), best_pattern, best_score,
        ", ".join(best_signals),
    )

    return PatternMatch(
        file_path=file_path,
        pattern=best_pattern,
        confidence=best_score,
        signals=best_signals,
    )
