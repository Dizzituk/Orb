# FILE: app/orchestrator/surgical_extractor.py
"""
Surgical Extractor — Deterministic symbol extraction from monolith files.

Extracts symbols from large Python files WITHOUT using an LLM.
Works by:
  1. AST-scanning the source to map every symbol's exact line range
  2. Scoring each symbol by "extractability" (coupling, size, purity)
  3. Programmatically copying selected symbols to a new module file
  4. Programmatically deleting those lines from the monolith
  5. Adding import statements so the monolith still references the symbols

Zero LLM calls. Pure mechanical cut-paste-import.

BUILD_ID: 2026-02-21-v1.0-surgical-extractor
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from app.orchestrator._surgical_extractor_utils import SURGICAL_EXTRACTOR_BUILD_ID, _build_new_module, _build_references, _collect_needed_imports, _find_import_insert_point, _node_to_location, analyse_file, select_extraction_cluster

logger = logging.getLogger(__name__)
print(f"[SURGICAL_EXTRACTOR_LOADED] BUILD_ID={SURGICAL_EXTRACTOR_BUILD_ID}")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SymbolLocation:
    """A symbol's exact position in the source file."""
    name: str
    kind: str  # "function", "async_function", "class", "constant", "assignment"
    line_start: int  # 1-indexed, inclusive
    line_end: int    # 1-indexed, inclusive
    char_count: int
    references: Set[str] = field(default_factory=set)      # symbols this one uses
    referenced_by: Set[str] = field(default_factory=set)    # symbols that use this one
    is_private: bool = False
    decorators_start: Optional[int] = None  # first decorator line if any


@dataclass
class ExtractionCandidate:
    """A scored symbol ready for extraction."""
    symbol: SymbolLocation
    coupling_score: float  # lower = easier to extract
    size_lines: int
    reason: str  # human-readable explanation


@dataclass
class ExtractionPlan:
    """What to extract in one pass."""
    source_file: str
    target_module: str  # new file path (relative)
    symbols: List[SymbolLocation]
    total_lines: int
    total_chars: int
    import_line: str  # e.g. "from .utils import func_a, func_b"


@dataclass
class ExtractionResult:
    """Result of one extraction pass."""
    success: bool
    source_file: str
    target_module: str
    symbols_extracted: List[str]
    lines_removed: int
    new_source_size: int
    error: str = ""


# =============================================================================
# PHASE 1: AST SCAN — MAP EVERY SYMBOL
# =============================================================================

def scan_symbols(source_code: str) -> List[SymbolLocation]:
    """
    Parse source and return every top-level symbol with exact line ranges.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.error("[surgical] Failed to parse source: %s", e)
        return []

    source_lines = source_code.split("\n")
    symbols: List[SymbolLocation] = []
    all_names: Set[str] = set()

    # First pass: collect all top-level symbol names and positions
    for node in ast.iter_child_nodes(tree):
        loc = _node_to_location(node, source_lines)
        if loc:
            symbols.append(loc)
            all_names.add(loc.name)

    # Second pass: build reference graph
    _build_references(symbols, all_names, source_lines)

    return symbols


# =============================================================================
# PHASE 2: COUPLING SCORER — RANK EXTRACTABILITY
# =============================================================================

def score_extractability(symbols: List[SymbolLocation]) -> List[ExtractionCandidate]:
    """
    Score each symbol by how easy it is to extract.
    Lower coupling_score = easier to extract.

    Scoring factors:
    - referenced_by count: more dependents = harder (they'd need import changes)
    - references count: more deps = harder (might pull in a chain)
    - size: smaller = safer per pass
    - private: private symbols are internal, less external coupling
    """
    candidates: List[ExtractionCandidate] = []

    for sym in symbols:
        # Skip module-level infrastructure
        if sym.name in ('logger', 'log', '__all__') or sym.name.startswith('__'):
            continue
        # Skip tiny assignments (single-line constants like BUILD_ID)
        if sym.kind == "assignment" and sym.line_end - sym.line_start < 1:
            continue

        # Coupling score: weighted combination
        ref_by_penalty = len(sym.referenced_by) * 3.0  # each dependent is costly
        ref_to_penalty = len(sym.references) * 1.0      # each dependency is minor
        size_factor = sym.char_count / 10000.0           # larger = slightly harder
        private_bonus = -1.0 if sym.is_private else 0.0  # private = easier

        score = ref_by_penalty + ref_to_penalty + size_factor + private_bonus

        # Build reason
        parts = []
        if sym.referenced_by:
            parts.append(f"used by {len(sym.referenced_by)}")
        if sym.references:
            parts.append(f"uses {len(sym.references)}")
        parts.append(f"{sym.char_count} chars")
        reason = ", ".join(parts) if parts else "standalone"

        candidates.append(ExtractionCandidate(
            symbol=sym,
            coupling_score=score,
            size_lines=sym.line_end - sym.line_start + 1,
            reason=reason,
        ))

    # Sort: lowest coupling first (easiest to extract)
    candidates.sort(key=lambda c: c.coupling_score)
    return candidates


# =============================================================================
# PHASE 3: CLUSTER SELECTION — PICK WHAT TO EXTRACT
# =============================================================================


# =============================================================================
# PHASE 4: SURGICAL EXTRACTION — CUT, PASTE, IMPORT
# =============================================================================

def build_extraction_plan(
    source_file: str,
    symbols: List[SymbolLocation],
    module_name: str = "",
) -> ExtractionPlan:
    """
    Build a plan for extracting symbols from the source file.
    """
    if not module_name:
        # Auto-name: _{source_stem}_utils
        source_stem = os.path.splitext(os.path.basename(source_file))[0]
        module_name = f"_{source_stem}_utils"

    # Build target path: same directory as source, new filename
    source_dir = os.path.dirname(source_file)
    target_path = os.path.join(source_dir, f"{module_name}.py")

    # Build import line using absolute package path
    # e.g. app/orchestrator/segment_loop.py extracts to
    #      app/orchestrator/_segment_loop_utils.py
    # Import: from app.orchestrator._segment_loop_utils import ...
    # Convert absolute path to Python package path relative to project root
    # D:\Orb\app\memory → app.memory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # D:\Orb
    rel_dir = os.path.relpath(source_dir, project_root)
    pkg_path = rel_dir.replace(os.sep, ".").replace("/", ".")
    symbol_names = sorted(s.name for s in symbols)
    import_line = f"from {pkg_path}.{module_name} import {', '.join(symbol_names)}"

    total_lines = sum(s.line_end - s.line_start + 1 for s in symbols)
    total_chars = sum(s.char_count for s in symbols)

    return ExtractionPlan(
        source_file=source_file,
        target_module=target_path,
        symbols=symbols,
        total_lines=total_lines,
        total_chars=total_chars,
        import_line=import_line,
    )


def execute_extraction(
    source_code: str,
    plan: ExtractionPlan,
) -> Tuple[str, str, ExtractionResult]:
    """
    Execute the extraction. Returns (new_source, new_module, result).

    Pure function — doesn't write to disk. Caller handles file I/O.
    """
    source_lines = source_code.split("\n")

    # Collect line ranges to remove (sorted by start, descending for safe removal)
    ranges_to_remove: List[Tuple[int, int]] = []
    for sym in plan.symbols:
        ranges_to_remove.append((sym.line_start, sym.line_end))
    ranges_to_remove.sort(key=lambda r: r[0], reverse=True)

    # Extract symbol bodies (before we modify anything)
    extracted_bodies: List[str] = []
    for sym in sorted(plan.symbols, key=lambda s: s.line_start):
        body = "\n".join(source_lines[sym.line_start - 1:sym.line_end])
        extracted_bodies.append(body)

    # Build the new module file
    new_module_parts = _build_new_module(source_code, plan, extracted_bodies)

    # Remove extracted lines from source (descending order to preserve indices)
    modified_lines = list(source_lines)
    lines_removed = 0
    for start, end in ranges_to_remove:
        # Also remove any blank lines immediately before the symbol (up to 2)
        actual_start = start
        for lookback in range(1, 3):
            check = start - 1 - lookback
            if check >= 0 and modified_lines[check].strip() == "":
                actual_start = check + 1  # 1-indexed
            else:
                break

        del modified_lines[actual_start - 1:end]
        lines_removed += (end - actual_start + 1)

    # Add import line near the top of the source (after existing imports)
    import_insert_idx = _find_import_insert_point(modified_lines)
    modified_lines.insert(import_insert_idx, plan.import_line)

    new_source = "\n".join(modified_lines)
    new_module = "\n".join(new_module_parts)

    result = ExtractionResult(
        success=True,
        source_file=plan.source_file,
        target_module=plan.target_module,
        symbols_extracted=[s.name for s in plan.symbols],
        lines_removed=lines_removed,
        new_source_size=len(new_source),
    )

    return new_source, new_module, result


# =============================================================================
# PUBLIC API — ONE-SHOT EXTRACTION
# =============================================================================


def extract_easiest(
    file_path: str,
    source_code: str,
    max_lines: int = 400,
    max_chars: int = 35_000,
    module_name: str = "",
) -> Tuple[Optional[str], Optional[str], Optional[ExtractionPlan], Optional[ExtractionResult]]:
    """
    Full pipeline: scan → score → select → extract.

    Returns (new_source, new_module_content, plan, result) or
    (None, None, None, None) if nothing extractable.
    """
    symbols = scan_symbols(source_code)
    if not symbols:
        logger.warning("[surgical] No symbols found in %s", file_path)
        return None, None, None, None

    candidates = score_extractability(symbols)
    if not candidates:
        logger.warning("[surgical] No extractable candidates in %s", file_path)
        return None, None, None, None

    cluster = select_extraction_cluster(candidates, max_lines, max_chars)
    if not cluster:
        logger.warning("[surgical] No viable cluster in %s", file_path)
        return None, None, None, None

    plan = build_extraction_plan(file_path, cluster, module_name)
    new_source, new_module, result = execute_extraction(source_code, plan)

    logger.info(
        "[surgical] Extracted %d symbols (%d lines, %d chars) from %s → %s",
        len(cluster), plan.total_lines, plan.total_chars,
        file_path, plan.target_module,
    )

    return new_source, new_module, plan, result
