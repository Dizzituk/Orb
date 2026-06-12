# FILE: app/orchestrator/refactor_segmenter_models.py
# Purpose: Data models for the deterministic refactor segmenter.
# Called-by: app.orchestrator._deterministic_architecture_utils_2, app.orchestrator._refactor_segmenter_utils_2, app.orchestrator._refactor_segmenter_utils_3, app.orchestrator._refactor_segmenter_utils_4 (+3 more)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Data models for the deterministic refactor segmenter.

Shared between the segmenter and any consumers.
Kept separate from segmenter logic to stay under file size targets.

BUILD_ID: 2026-02-20-v1.0-refactor-segmenter-models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

REFACTOR_SEGMENTER_MODELS_BUILD_ID = "2026-02-20-v1.0-refactor-segmenter-models"
print(f"[REFACTOR_SEGMENTER_MODELS_LOADED] BUILD_ID={REFACTOR_SEGMENTER_MODELS_BUILD_ID}")


# =============================================================================
# NODE TYPES
# =============================================================================

class SymbolKind(Enum):
    """Classification of a symbol extracted from enrichment."""
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    CONSTANT = "constant"
    DATA_STRUCTURE = "data_structure"  # dicts, lists, compiled regexes, prompt strings


@dataclass
class Symbol:
    """A single symbol (function, class, constant) from the source monolith."""
    name: str
    kind: SymbolKind
    line_start: int = 0
    line_end: int = 0
    char_count: int = 0
    references: List[str] = field(default_factory=list)  # other symbols this one references
    is_private: bool = False  # starts with _
    is_dunder: bool = False   # starts with __

    @property
    def estimated_lines(self) -> int:
        if self.line_start and self.line_end:
            return self.line_end - self.line_start + 1
        if self.char_count:
            return max(1, self.char_count // 60)
        return 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "char_count": self.char_count,
            "estimated_lines": self.estimated_lines,
            "references": self.references,
            "is_private": self.is_private,
            "is_dunder": self.is_dunder,
        }


# =============================================================================
# FILE NODE
# =============================================================================

@dataclass
class FileNode:
    """
    A target file in the refactored package.

    Contains the symbols assigned to it and its dependency edges
    to other FileNodes.
    """
    file_path: str                              # relative path e.g. "app/overwatcher/sandbox_build_validator/_config.py"
    file_stem: str = ""                         # e.g. "_config"
    description: str = ""                       # from architecture file inventory
    symbols: List[Symbol] = field(default_factory=list)
    depends_on: Set[str] = field(default_factory=set)  # file_paths this node imports from
    depended_by: Set[str] = field(default_factory=set)  # file_paths that import from this node
    tier: int = -1                              # assigned during topological sort
    is_facade: bool = False                     # __init__.py or main re-export file
    is_data_only: bool = False                  # only constants/data, no logic

    @property
    def estimated_lines(self) -> int:
        base = sum(s.estimated_lines for s in self.symbols)
        # Add imports overhead (~2 lines per dependency + 5 for boilerplate)
        return base + len(self.depends_on) * 2 + 5

    @property
    def symbol_names(self) -> Set[str]:
        return {s.name for s in self.symbols}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_stem": self.file_stem,
            "description": self.description,
            "symbols": [s.to_dict() for s in self.symbols],
            "depends_on": sorted(self.depends_on),
            "depended_by": sorted(self.depended_by),
            "tier": self.tier,
            "is_facade": self.is_facade,
            "is_data_only": self.is_data_only,
            "estimated_lines": self.estimated_lines,
        }


# =============================================================================
# DEPENDENCY GRAPH
# =============================================================================

@dataclass
class DependencyGraph:
    """
    The complete file-level dependency graph for a refactor.

    Nodes are FileNodes, edges are "imports from" relationships.
    """
    nodes: Dict[str, FileNode] = field(default_factory=dict)  # keyed by file_path
    tiers: List[List[str]] = field(default_factory=list)       # tier[0] = leaf file_paths
    unassigned_symbols: List[Symbol] = field(default_factory=list)
    cycle_detected: bool = False
    warnings: List[str] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.nodes)

    @property
    def total_symbols(self) -> int:
        return sum(len(n.symbols) for n in self.nodes.values())

    @property
    def total_tiers(self) -> int:
        return len(self.tiers)

    def get_tier(self, tier_index: int) -> List[FileNode]:
        """Get all FileNodes in a given tier."""
        if tier_index < 0 or tier_index >= len(self.tiers):
            return []
        return [self.nodes[fp] for fp in self.tiers[tier_index] if fp in self.nodes]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_symbols": self.total_symbols,
            "total_tiers": self.total_tiers,
            "cycle_detected": self.cycle_detected,
            "warnings": self.warnings,
            "tiers": [
                [{"file": fp, "symbols": len(self.nodes[fp].symbols),
                  "estimated_lines": self.nodes[fp].estimated_lines}
                 for fp in tier_files]
                for tier_files in self.tiers
            ],
            "unassigned_symbols": [s.to_dict() for s in self.unassigned_symbols],
            "nodes": {fp: n.to_dict() for fp, n in self.nodes.items()},
        }


# =============================================================================
# SEGMENT PLAN
# =============================================================================

@dataclass
class SegmentPlan:
    """
    A single segment in the build plan.

    Groups one or more tiers into a buildable unit.
    """
    segment_index: int
    segment_id: str = ""
    title: str = ""
    file_paths: List[str] = field(default_factory=list)
    tiers_included: List[int] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # segment_ids this depends on
    estimated_lines: int = 0
    estimated_files: int = 0
    is_facade_segment: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "segment_id": self.segment_id,
            "title": self.title,
            "file_paths": self.file_paths,
            "tiers_included": self.tiers_included,
            "dependencies": self.dependencies,
            "estimated_lines": self.estimated_lines,
            "estimated_files": self.estimated_files,
            "is_facade_segment": self.is_facade_segment,
        }


@dataclass
class RefactorBuildPlan:
    """
    The complete deterministic build plan for a refactor job.

    This replaces LLM-driven segmentation for refactor jobs.
    """
    source_file: str                          # the monolith being refactored
    target_package: str                       # the package directory
    graph: DependencyGraph = field(default_factory=DependencyGraph)
    segments: List[SegmentPlan] = field(default_factory=list)
    facade_file: str = ""                     # __init__.py path
    public_symbols: List[str] = field(default_factory=list)  # symbols the facade must re-export
    warnings: List[str] = field(default_factory=list)

    @property
    def total_segments(self) -> int:
        return len(self.segments)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_file": self.source_file,
            "target_package": self.target_package,
            "total_segments": self.total_segments,
            "facade_file": self.facade_file,
            "public_symbols": self.public_symbols,
            "warnings": self.warnings,
            "graph": self.graph.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
        }
