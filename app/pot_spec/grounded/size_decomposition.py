# FILE: app/pot_spec/grounded/size_decomposition.py
"""
Size Analyzer — Decomposition Suggestion Engine.

Identifies oversized source blocks and generates sub-file suggestions
to bring them under the 400-line / 15KB cap. Used by size_analyzer.py.

v1.0 (2026-02-14): Extracted from size_analyzer.py for module cap compliance.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from .size_models import AVG_CHARS_PER_LINE, MAX_FILE_LINES, SourceBlock

logger = logging.getLogger(__name__)


def estimate_from_source_blocks(
    blocks: List[SourceBlock],
    total_source_lines: int,
    output_file_count: int,
) -> List[Dict[str, Any]]:
    """
    Identify source blocks that would produce oversized output files.

    Returns list of oversized block descriptors with extractable closures.
    """
    oversized: List[Dict[str, Any]] = []

    for block in blocks:
        if block.line_count > MAX_FILE_LINES:
            extractable_closures = [
                nb for nb in block.nested_blocks
                if nb.line_count >= 50
            ]
            extractable_lines = sum(nb.line_count for nb in extractable_closures)
            remaining_lines = block.line_count - extractable_lines

            oversized.append({
                "block_name": block.name,
                "block_kind": block.kind,
                "start_line": block.start_line,
                "end_line": block.end_line,
                "total_lines": block.line_count,
                "remaining_after_extraction": remaining_lines,
                "extractable_closures": [
                    {
                        "name": nb.name,
                        "kind": nb.kind,
                        "lines": nb.line_count,
                        "start": nb.start_line,
                        "end": nb.end_line,
                    }
                    for nb in extractable_closures
                ],
            })

    return oversized


def suggest_decomposition_for_block(
    block_info: Dict[str, Any],
    source_rel_path: str,
    output_package_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate decomposition sub-file suggestions for an oversized block.

    Groups extractable closures into files under MAX_FILE_LINES, then
    post-validates that no suggestion itself exceeds the cap.
    """
    suggestions: List[Dict[str, Any]] = []
    closures = block_info.get("extractable_closures", [])
    remaining = block_info["total_lines"]

    if not output_package_dir:
        output_package_dir = ""

    current_group: List[Dict] = []
    current_lines = 0

    for closure in sorted(closures, key=lambda c: c["lines"], reverse=True):
        if current_lines + closure["lines"] > MAX_FILE_LINES and current_group:
            _emit_suggestion(suggestions, current_group, current_lines,
                             output_package_dir, len(suggestions))
            current_group = []
            current_lines = 0

        current_group.append(closure)
        current_lines += closure["lines"]
        remaining -= closure["lines"]

    if current_group:
        _emit_suggestion(suggestions, current_group, current_lines,
                         output_package_dir, len(suggestions))

    if remaining > MAX_FILE_LINES:
        logger.warning(
            "[size_decomp] Block '%s' still ~%d lines after extraction (cap=%d).",
            block_info["block_name"], remaining, MAX_FILE_LINES,
        )

    # Post-validate: split any suggestion that itself exceeds the cap
    validated: List[Dict[str, Any]] = []
    for suggestion in suggestions:
        est_lines = suggestion.get("estimated_lines", 0)
        if est_lines > MAX_FILE_LINES:
            n_splits = (est_lines + MAX_FILE_LINES - 1) // MAX_FILE_LINES
            lines_per_split = (est_lines + n_splits - 1) // n_splits
            base_path = suggestion["rel_path"]
            base_stem = os.path.splitext(os.path.basename(base_path))[0]
            base_dir = os.path.dirname(base_path)
            closures_list = suggestion.get("source_closures", [])

            logger.info(
                "[size_decomp] Suggestion '%s' at ~%d lines — splitting into %d",
                base_path, est_lines, n_splits,
            )

            for i in range(n_splits):
                suffix = f"_part{i+1}" if n_splits > 1 else ""
                split_path = os.path.join(
                    base_dir, f"{base_stem}{suffix}.py"
                ).replace("\\", "/")
                split_lines = min(lines_per_split, est_lines - (i * lines_per_split))
                validated.append({
                    "rel_path": split_path,
                    "estimated_lines": split_lines + 30,
                    "estimated_kb": round((split_lines + 30) * AVG_CHARS_PER_LINE / 1024, 1),
                    "source_closures": closures_list if i == 0 else [],
                    "description": f"Split {i+1}/{n_splits} of {base_stem}",
                })
        else:
            validated.append(suggestion)

    return validated


def _emit_suggestion(
    suggestions: List[Dict[str, Any]],
    closures: List[Dict],
    total_lines: int,
    package_dir: str,
    index: int,
) -> None:
    """Create a sub-file suggestion from a group of closures."""
    primary = closures[0]
    name = primary["name"].lstrip("_")
    if name.startswith("run_"):
        name = name[4:]

    file_name = f"{name}.py"
    rel_path = os.path.join(package_dir, file_name).replace("\\", "/") if package_dir else file_name

    suggestions.append({
        "rel_path": rel_path,
        "estimated_lines": total_lines + 30,
        "estimated_kb": round((total_lines + 30) * AVG_CHARS_PER_LINE / 1024, 1),
        "source_closures": [c["name"] for c in closures],
        "description": (
            f"Extracted from {primary.get('name', '?')}: "
            + ", ".join(c["name"] for c in closures)
        ),
    })
