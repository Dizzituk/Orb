# FILE: app/orchestrator/refactor_segmenter.py
# Purpose: Deterministic Refactor Segmenter.
# Called-by: app.orchestrator.refactor_pipeline
# Depends-on: app.orchestrator._refactor_segmenter_utils_2, app.orchestrator._refactor_segmenter_utils_3, app.orchestrator._refactor_segmenter_utils_4, app.orchestrator.refactor_segmenter_models
# Last-renovated: 2026-06-11
"""
Deterministic Refactor Segmenter.

Replaces LLM-driven segmentation for refactor jobs with a scan-based
approach that:
  1. Parses enrichment data to extract symbols and their references
  2. Maps symbols to target files using architecture file descriptions
  3. Builds a file-level dependency graph
  4. Topologically sorts files into build tiers (leaves first)
  5. Groups tiers into segments that produce a standard manifest

Zero LLM calls. The entire segmentation and ordering is deterministic.

BUILD_ID: 2026-02-20-v1.0-deterministic-refactor-segmenter
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from app.orchestrator.refactor_segmenter_models import (
    DependencyGraph,
    FileNode,
    RefactorBuildPlan,
    SegmentPlan,
    Symbol,
    SymbolKind,
)
from app.orchestrator._refactor_segmenter_utils_2 import REFACTOR_SEGMENTER_BUILD_ID, _DATA_CONSTANT_PATTERNS, _assign_remainder, _find_common_prefix, _match_by_architecture_mention, _match_by_reference_clustering, _parse_line_range, save_build_plan
from app.orchestrator._refactor_segmenter_utils_3 import MAX_SEGMENT_ESTIMATED_LINES, MAX_SEGMENT_FILES, _BUILTIN_NAMES, _build_file_keyword_index, _find_references, _make_segment, _match_by_semantic_affinity
from app.orchestrator._refactor_segmenter_utils_4 import _auto_generate_file_layout, build_dependency_edges, build_file_nodes, sort_into_tiers

logger = logging.getLogger(__name__)
print(f"[REFACTOR_SEGMENTER_LOADED] BUILD_ID={REFACTOR_SEGMENTER_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# Segment size targets

# Symbols that are never "references" — stdlib builtins, common names

# Data-heavy constant patterns — these contain data, not logic


# =============================================================================
# STEP 1: EXTRACT SYMBOLS FROM ENRICHMENT
# =============================================================================

def extract_symbols(enrichment: Dict[str, Any]) -> List[Symbol]:
    """
    Extract all symbols from enrichment data.

    Parses functions, classes, and constants into Symbol objects
    with their internal references detected.
    """
    if not enrichment:
        return []

    symbols: List[Symbol] = []
    source_extract = enrichment.get("source_extract", {})
    all_symbol_names: Set[str] = set()

    # Collect all names first (needed for reference detection)
    for func in enrichment.get("functions", []):
        all_symbol_names.add(func.get("name", ""))
    for cls in enrichment.get("classes", []):
        all_symbol_names.add(cls.get("name", ""))
    for const in enrichment.get("constants", []):
        all_symbol_names.add(const.get("name", ""))
    # Also add source_extract keys that might be constants
    for key in source_extract:
        all_symbol_names.add(key)
    all_symbol_names.discard("")

    # --- Functions ---
    for func in enrichment.get("functions", []):
        name = func.get("name", "")
        if not name:
            continue

        body = func.get("body", "")
        # Also get from source_extract if body is empty/stub
        if (not body or body.strip() in ("...", "pass")) and name in source_extract:
            body = source_extract[name]

        line_range = func.get("line_range", "")
        line_start, line_end = _parse_line_range(line_range)

        refs = _find_references(body, all_symbol_names, name)

        kind = SymbolKind.ASYNC_FUNCTION if func.get("is_async") else SymbolKind.FUNCTION

        symbols.append(Symbol(
            name=name,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            char_count=len(body),
            references=refs,
            is_private=name.startswith("_") and not name.startswith("__"),
            is_dunder=name.startswith("__") and name.endswith("__"),
        ))

    # --- Classes ---
    for cls in enrichment.get("classes", []):
        name = cls.get("name", "")
        if not name:
            continue

        body = ""
        if name in source_extract:
            body = source_extract[name]

        refs = _find_references(body, all_symbol_names, name) if body else []

        symbols.append(Symbol(
            name=name,
            kind=SymbolKind.CLASS,
            char_count=len(body),
            references=refs,
        ))

    # --- Constants ---
    for const in enrichment.get("constants", []):
        name = const.get("name", "")
        if not name:
            continue

        value_str = const.get("value", "")
        body = ""
        if name in source_extract:
            body = source_extract[name]

        # Classify: is this a simple constant or a data structure?
        kind = SymbolKind.CONSTANT
        if any(p.match(name) for p in _DATA_CONSTANT_PATTERNS):
            kind = SymbolKind.DATA_STRUCTURE
        elif value_str and len(value_str) > 200:
            kind = SymbolKind.DATA_STRUCTURE

        refs = _find_references(body or value_str, all_symbol_names, name)

        line_number = const.get("line_number", 0)
        # Estimate end line from value length
        est_lines = max(1, len(value_str) // 80) if value_str else 1

        symbols.append(Symbol(
            name=name,
            kind=kind,
            line_start=line_number,
            line_end=line_number + est_lines - 1,
            char_count=len(body or value_str),
            references=refs,
        ))

    logger.info(
        "[refactor_segmenter] Extracted %d symbols (%d functions, %d classes, %d constants)",
        len(symbols),
        sum(1 for s in symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)),
        sum(1 for s in symbols if s.kind == SymbolKind.CLASS),
        sum(1 for s in symbols if s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE)),
    )
    return symbols


# =============================================================================
# STEP 2: BUILD FILE NODES FROM ARCHITECTURE
# =============================================================================


# =============================================================================
# STEP 3: ASSIGN SYMBOLS TO FILES
# =============================================================================

def assign_symbols_to_files(
    symbols: List[Symbol],
    nodes: Dict[str, FileNode],
    architecture_text: str,
) -> List[Symbol]:
    """
    Assign each symbol to its best-matching target file.

    Uses a multi-pass strategy:
      Pass 1: Architecture description matching — check if the architecture
              explicitly mentions a symbol name in a file's design section
      Pass 2: Semantic affinity — match symbol kind + name patterns to
              file descriptions (e.g. "detection" functions → _detection.py)
      Pass 3: Reference clustering — symbols that reference each other
              should be in the same file when possible
      Pass 4: Remainder → assign to the most appropriate non-facade file

    Returns list of symbols that couldn't be assigned.
    """
    unassigned: List[Symbol] = []
    assigned_names: Dict[str, str] = {}  # symbol_name → file_path

    # Build description keyword index for each file
    file_keywords = _build_file_keyword_index(nodes, architecture_text)

    # --- Pass 1: Architecture mentions ---
    # Check if the architecture explicitly names a symbol in a file's section
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _match_by_architecture_mention(
            symbol.name, nodes, architecture_text,
        )
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file

    pass1_count = len(assigned_names)
    logger.info("[refactor_segmenter] Pass 1 (arch mention): assigned %d symbols", pass1_count)

    # --- Pass 2: Semantic affinity ---
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _match_by_semantic_affinity(symbol, nodes, file_keywords)
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file

    pass2_count = len(assigned_names) - pass1_count
    logger.info("[refactor_segmenter] Pass 2 (semantic): assigned %d symbols", pass2_count)

    # --- Pass 3: Reference clustering ---
    # Symbols that reference already-assigned symbols go to the same file
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _match_by_reference_clustering(symbol, assigned_names, nodes)
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file

    pass3_count = len(assigned_names) - pass1_count - pass2_count
    logger.info("[refactor_segmenter] Pass 3 (reference clustering): assigned %d symbols", pass3_count)

    # --- Pass 4: Remainder ---
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _assign_remainder(symbol, nodes)
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file
        else:
            unassigned.append(symbol)

    pass4_count = len(assigned_names) - pass1_count - pass2_count - pass3_count
    logger.info("[refactor_segmenter] Pass 4 (remainder): assigned %d, unassigned %d",
                pass4_count, len(unassigned))

    return unassigned


# =============================================================================
# STEP 4: BUILD DEPENDENCY EDGES
# =============================================================================


# =============================================================================
# STEP 5: TOPOLOGICAL SORT INTO TIERS
# =============================================================================


# =============================================================================
# STEP 6: GROUP TIERS INTO SEGMENTS
# =============================================================================

def group_tiers_into_segments(
    tiers: List[List[str]],
    nodes: Dict[str, FileNode],
    source_file: str,
) -> List[SegmentPlan]:
    """
    Group tiers into build segments.

    Rules:
      - Adjacent tiers can be merged if combined size < MAX_SEGMENT_ESTIMATED_LINES
        and combined files < MAX_SEGMENT_FILES
      - Facade files always get their own segment (last)
      - Each segment declares dependencies on earlier segments
    """
    segments: List[SegmentPlan] = []
    seg_index = 0

    # Separate facade tiers
    facade_tiers: List[int] = []
    logic_tiers: List[int] = []
    for i, tier_files in enumerate(tiers):
        if all(nodes[fp].is_facade for fp in tier_files):
            facade_tiers.append(i)
        else:
            logic_tiers.append(i)

    # Group logic tiers
    current_files: List[str] = []
    current_tiers: List[int] = []
    current_lines = 0

    for tier_idx in logic_tiers:
        tier_files = tiers[tier_idx]
        tier_lines = sum(nodes[fp].estimated_lines for fp in tier_files)

        # Check if adding this tier would exceed limits
        if current_files and (
            current_lines + tier_lines > MAX_SEGMENT_ESTIMATED_LINES
            or len(current_files) + len(tier_files) > MAX_SEGMENT_FILES
        ):
            # Flush current segment
            segments.append(_make_segment(
                seg_index, current_files, current_tiers,
                nodes, segments, source_file,
            ))
            seg_index += 1
            current_files = []
            current_tiers = []
            current_lines = 0

        current_files.extend(tier_files)
        current_tiers.append(tier_idx)
        current_lines += tier_lines

    # Flush remaining logic files
    if current_files:
        segments.append(_make_segment(
            seg_index, current_files, current_tiers,
            nodes, segments, source_file,
        ))
        seg_index += 1

    # Add facade segment(s)
    for tier_idx in facade_tiers:
        tier_files = tiers[tier_idx]
        segments.append(_make_segment(
            seg_index, tier_files, [tier_idx],
            nodes, segments, source_file,
            is_facade=True,
        ))
        seg_index += 1

    logger.info(
        "[refactor_segmenter] Grouped %d tiers into %d segments",
        len(tiers), len(segments),
    )
    return segments


# =============================================================================
# v6.1 FIX 14b: AUTO-GENERATE FILE LAYOUT
# =============================================================================


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_refactor_plan(
    enrichment: Dict[str, Any],
    architecture_text: str,
    source_file: str,
    target_package: str,
    facade_file: Optional[str] = None,
) -> RefactorBuildPlan:
    """
    Build a complete deterministic refactor plan.

    This is the main entry point. Takes enrichment data and architecture
    text, returns a RefactorBuildPlan that can be converted to a standard
    manifest.

    Args:
        enrichment: Enrichment data with functions, classes, constants, source_extract
        architecture_text: The architecture document with File Inventory
        source_file: Path to the source monolith being refactored
        target_package: Package directory for the refactored files
        facade_file: Path to __init__.py (auto-detected if not provided)
    """
    plan = RefactorBuildPlan(
        source_file=source_file,
        target_package=target_package,
    )

    # Step 1: Extract symbols
    symbols = extract_symbols(enrichment)
    if not symbols:
        plan.warnings.append("No symbols extracted from enrichment")
        return plan

    # Step 2: Build file nodes from architecture
    nodes = build_file_nodes(architecture_text, target_package)
    if not nodes:
        # v6.1 FIX 14b: Auto-generate file layout when no inventory provided.
        # This happens with Strategy 0 (direct source detection) where
        # the LLM didn't propose a package structure.
        nodes = _auto_generate_file_layout(symbols, target_package, source_file)
        if not nodes:
            plan.warnings.append("No files found in architecture and auto-layout failed")
            return plan
        plan.warnings.append(f"Auto-generated {len(nodes)} file nodes (Strategy 0)")

    # Detect facade
    if facade_file:
        plan.facade_file = facade_file
    else:
        for fp, node in nodes.items():
            if node.is_facade:
                plan.facade_file = fp
                break

    # Collect public symbols (non-private, non-dunder)
    plan.public_symbols = [
        s.name for s in symbols
        if not s.is_private and not s.is_dunder
    ]

    # Step 3: Assign symbols to files
    unassigned = assign_symbols_to_files(symbols, nodes, architecture_text)
    plan.graph.unassigned_symbols = unassigned

    # Mark data-only files
    for node in nodes.values():
        if node.symbols and all(
            s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE)
            for s in node.symbols
        ):
            node.is_data_only = True

    # Step 4: Build dependency edges
    build_dependency_edges(nodes)

    # Step 5: Topological sort
    tiers, cycle = sort_into_tiers(nodes)
    plan.graph.nodes = nodes
    plan.graph.tiers = tiers
    plan.graph.cycle_detected = cycle

    # Step 6: Group into segments
    plan.segments = group_tiers_into_segments(tiers, nodes, source_file)

    # Summary warnings
    if unassigned:
        plan.warnings.append(
            f"{len(unassigned)} symbol(s) could not be assigned to any file: "
            + ", ".join(s.name for s in unassigned)
        )
    if cycle:
        plan.warnings.append("Cycle detected in dependency graph — build order may be suboptimal")

    empty_files = [fp for fp, n in nodes.items() if not n.symbols and not n.is_facade]
    if empty_files:
        plan.warnings.append(
            f"{len(empty_files)} non-facade file(s) have no symbols assigned: "
            + ", ".join(os.path.basename(fp) for fp in empty_files)
        )

    logger.info(
        "[refactor_segmenter] Build plan complete: %d symbols → %d files → %d tiers → %d segments",
        len(symbols), len(nodes), len(tiers), len(plan.segments),
    )

    return plan


# =============================================================================
# PERSISTENCE
# =============================================================================
