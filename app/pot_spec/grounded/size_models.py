# FILE: app/pot_spec/grounded/size_models.py
"""
Size Analyzer — Shared data models and constants.

Defines the size policy constants (single source of truth for the pipeline)
and dataclasses used by the size analyzer and downstream stages.

v1.0 (2026-02-14): Extracted from size_analyzer.py for module cap compliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =============================================================================
# CONSTANTS — shared size policy (single source of truth)
# =============================================================================

MAX_FILE_LINES = 400
MAX_FILE_KB = 15
MAX_FUNCTION_LINES = 200
ABSOLUTE_FILE_KB_CEILING = 20
TARGET_FILE_KB = 10

# Heuristic: average chars per line in generated code
AVG_CHARS_PER_LINE = 55

# Project roots for resolving relative paths to absolute
DEFAULT_PROJECT_ROOTS = ["D:\\Orb", "D:\\orb-desktop"]


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SourceBlock:
    """A function, class, or top-level block in a source file."""
    name: str
    kind: str  # "function", "async_function", "class", "closure", "loop_body"
    start_line: int
    end_line: int
    line_count: int
    parent: Optional[str] = None  # enclosing function/class name
    nested_blocks: List["SourceBlock"] = field(default_factory=list)


@dataclass
class FileSizeEstimate:
    """Size estimate for a single output file."""
    rel_path: str
    estimated_lines: int
    estimated_kb: float
    source_file: Optional[str] = None  # which source file this maps to
    source_blocks: List[str] = field(default_factory=list)
    exceeds_cap: bool = False
    decomposition_suggested: bool = False
    suggested_splits: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rel_path": self.rel_path,
            "estimated_lines": self.estimated_lines,
            "estimated_kb": self.estimated_kb,
            "source_file": self.source_file,
            "source_blocks": self.source_blocks,
            "exceeds_cap": self.exceeds_cap,
            "decomposition_suggested": self.decomposition_suggested,
            "suggested_splits": self.suggested_splits,
        }


@dataclass
class SizeAnalysisResult:
    """Complete result of pre-segmentation size analysis."""
    original_file_scope: List[str]
    enriched_file_scope: List[str]  # may have additional files
    estimates: Dict[str, FileSizeEstimate]
    files_added: List[str]  # new files from decomposition
    files_exceeding_cap: List[str]
    source_files_analyzed: int
    is_refactor: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_count": len(self.original_file_scope),
            "enriched_count": len(self.enriched_file_scope),
            "files_added": self.files_added,
            "files_exceeding_cap": self.files_exceeding_cap,
            "source_files_analyzed": self.source_files_analyzed,
            "is_refactor": self.is_refactor,
            "estimates": {k: v.to_dict() for k, v in self.estimates.items()},
        }
