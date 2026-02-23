# FILE: app/orchestrator/_codebase_scanner_health.py
"""
Smart health checks for the codebase scanner.

Replaces the old size-only OVERSIZED_FUNCTION / OVERSIZED_FILE checks with
structural analysis that identifies *genuine* refactoring opportunities.

Philosophy:
- A 1000-line file with 20 small functions is FINE — it's already modular.
- A 400-line file with one giant function is a PROBLEM.
- A 40KB data-only dict literal is NOT a concern — there's nothing to refactor.
- A single async orchestrator where splitting would break the yield chain is
  NOT a concern — flagging it wastes everyone's time.

What IS a concern:
- Multi-responsibility files: unrelated function clusters in the same file
- Extractable blocks: self-contained helper groups with no cross-references
- Monolithic functions: functions >150 lines that aren't async generators
- God classes: classes with 10+ methods spanning multiple concerns
- Tangled dependencies: high fan-in AND fan-out (hard to change safely)

BUILD_ID: 2026-02-23-v1.0-smart-health-checks
"""
from __future__ import annotations

import logging
from typing import Dict, List, Set

from app.orchestrator.codebase_scanner_models import (
    FileScanResult,
    HealthCategory,
    HealthIssue,
    HealthSeverity,
    SymbolInfo,
    SymbolKind,
)

logger = logging.getLogger(__name__)

# =============================================================================
# THRESHOLDS
# =============================================================================

# A function is "monolithic" if it exceeds this AND is not an async generator /
# single-orchestrator pattern.
MONOLITHIC_FUNCTION_LINES = 150

# A class is a "god class" if it has this many non-dunder methods.
GOD_CLASS_METHOD_COUNT = 10

# A file has "multi-responsibility" if it contains this many independent
# clusters of functions (groups with no internal cross-references).
MULTI_RESPONSIBILITY_CLUSTER_COUNT = 3

# Minimum file symbols before we bother analysing structure.
# Files with <3 top-level symbols can't meaningfully be "multi-responsibility".
MIN_SYMBOLS_FOR_STRUCTURE_CHECK = 3

# Minimum function lines to consider as an "extractable block".
MIN_EXTRACTABLE_BLOCK_LINES = 60

# Files under this line count are never flagged for structure issues.
# Small files are fine regardless of shape.
MIN_FILE_LINES_FOR_HEALTH = 200


# =============================================================================
# DATA-ONLY FILE DETECTION
# =============================================================================

def _is_data_only_file(result: FileScanResult) -> bool:
    """
    Return True if the file is mostly data declarations with negligible logic.

    Examples: intents.py (single dict literal), constants files, schema
    enumerations.  These files can't be meaningfully refactored — there's
    no logic to extract.
    """
    funcs = [
        s for s in result.symbols.values()
        if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)
    ]
    data = [
        s for s in result.symbols.values()
        if s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE)
    ]

    # If total function body lines are <20% of the file, it's data-dominated.
    func_lines = sum(s.estimated_lines for s in funcs)
    if result.line_count > 0 and func_lines / result.line_count < 0.20:
        return True

    # If there are more data symbols than functions, and data symbols are
    # individually large, it's a data file.
    if len(data) > len(funcs) and any(s.estimated_lines > 50 for s in data):
        return True

    return False


# =============================================================================
# SINGLE-ORCHESTRATOR DETECTION
# =============================================================================

def _is_single_orchestrator(result: FileScanResult) -> bool:
    """
    Return True if the file is dominated by a single large async function
    (the orchestrator pattern).

    These files *can't* be safely split without breaking async flow, shared
    locals, or yield chains.  Flagging them is noise.
    """
    funcs = [
        s for s in result.symbols.values()
        if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)
    ]
    if not funcs:
        return False

    largest = max(funcs, key=lambda s: s.estimated_lines)

    # If the largest function is async and occupies >60% of the file's
    # total function lines, this is a single-orchestrator file.
    total_func_lines = sum(s.estimated_lines for s in funcs)
    if total_func_lines == 0:
        return False

    ratio = largest.estimated_lines / total_func_lines
    return largest.is_async and ratio > 0.60 and largest.estimated_lines > 150


# =============================================================================
# MONOLITHIC FUNCTION CHECK
# =============================================================================

def check_monolithic_functions(result: FileScanResult) -> None:
    """
    Flag functions that are genuinely monolithic — large AND not excused by
    being an async orchestrator that can't be split.

    A function is monolithic if:
    - It exceeds MONOLITHIC_FUNCTION_LINES
    - It is NOT an async generator (yield-based orchestrators can't be split)
    - The file has other functions too (if it's the only function, it's an
      orchestrator pattern and we flag the file differently or not at all)
    """
    funcs = [
        (name, info) for name, info in result.symbols.items()
        if info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)
    ]
    if len(funcs) < 2:
        # Single-function file — either it's an orchestrator (excused) or
        # it's a helper with one function (probably fine).
        return

    for name, info in funcs:
        if info.estimated_lines <= MONOLITHIC_FUNCTION_LINES:
            continue

        # Check if it's an async generator (has yield in body).
        # We use a simple heuristic: if "yield" appears in the source.
        source_lower = info.source_code.lower()
        if info.is_async and ("yield " in source_lower or "yield\n" in source_lower):
            continue  # async generator — can't split safely

        result.health_issues.append(HealthIssue(
            category=HealthCategory.MONOLITHIC_FUNCTION,
            severity=HealthSeverity.WARNING,
            file_path=result.file_path,
            symbol_name=name,
            line_number=info.line_start,
            description=(
                f"'{name}' is {info.estimated_lines} lines with "
                f"{len(info.calls)} outgoing calls — consider extracting "
                f"helper functions for testability"
            ),
            suggestion=(
                "Look for sequential phases, repeated patterns, or blocks "
                "guarded by conditionals that could be standalone helpers"
            ),
        ))


# =============================================================================
# GOD CLASS CHECK
# =============================================================================

def check_god_classes(result: FileScanResult) -> None:
    """
    Flag classes with too many methods spanning multiple concerns.

    A class is a god class if:
    - It has > GOD_CLASS_METHOD_COUNT non-dunder methods
    - Its methods reference diverse external symbols (not just self)
    """
    for name, info in result.symbols.items():
        if info.kind != SymbolKind.CLASS:
            continue

        # Count non-dunder, non-property methods
        real_methods = [
            m for m in info.methods
            if not (m.startswith("__") and m.endswith("__"))
            and m != "to_dict"
            and m != "from_dict"
        ]

        if len(real_methods) <= GOD_CLASS_METHOD_COUNT:
            continue

        result.health_issues.append(HealthIssue(
            category=HealthCategory.GOD_CLASS,
            severity=HealthSeverity.WARNING,
            file_path=result.file_path,
            symbol_name=name,
            line_number=info.line_start,
            description=(
                f"Class '{name}' has {len(real_methods)} methods "
                f"({info.estimated_lines} lines) — may be doing too much"
            ),
            suggestion=(
                "Consider extracting method groups into mixins, "
                "helper classes, or standalone functions"
            ),
        ))


# =============================================================================
# EXTRACTABLE BLOCK DETECTION
# =============================================================================

def check_extractable_blocks(result: FileScanResult) -> None:
    """
    Find groups of functions that are self-contained and could live in
    their own module.

    A group is extractable if:
    - 2+ functions form a cluster (they call each other but nothing else
      in the file calls them)
    - The cluster's total size is > MIN_EXTRACTABLE_BLOCK_LINES
    - The rest of the file wouldn't break if the cluster moved
    """
    funcs = {
        name: info for name, info in result.symbols.items()
        if info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)
    }

    if len(funcs) < 4:
        return  # Too few functions to have meaningful clusters

    # Build internal call adjacency (only between functions in this file)
    func_names = set(funcs.keys())
    adjacency: Dict[str, Set[str]] = {n: set() for n in func_names}
    for name, info in funcs.items():
        for callee in info.calls:
            if callee in func_names:
                adjacency[name].add(callee)
                adjacency[callee].add(name)  # bidirectional for clustering

    # Find connected components via BFS
    visited: Set[str] = set()
    clusters: List[List[str]] = []

    for start in func_names:
        if start in visited:
            continue
        cluster: List[str] = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    queue.append(neighbour)
        clusters.append(cluster)

    if len(clusters) < MULTI_RESPONSIBILITY_CLUSTER_COUNT:
        return  # Not enough independent clusters

    # Find clusters that are self-contained and large enough to extract
    for cluster in clusters:
        cluster_lines = sum(funcs[n].estimated_lines for n in cluster)
        if cluster_lines < MIN_EXTRACTABLE_BLOCK_LINES:
            continue
        if len(cluster) < 2:
            continue

        # Check that nothing outside the cluster calls into it
        cluster_set = set(cluster)
        called_from_outside = False
        for name, info in funcs.items():
            if name in cluster_set:
                continue
            if any(c in cluster_set for c in info.calls):
                called_from_outside = True
                break

        if called_from_outside:
            continue  # Cluster is referenced externally — not cleanly extractable

        func_names_str = ", ".join(sorted(cluster)[:4])
        if len(cluster) > 4:
            func_names_str += f" (+{len(cluster) - 4} more)"

        result.health_issues.append(HealthIssue(
            category=HealthCategory.EXTRACTABLE_BLOCK,
            severity=HealthSeverity.INFO,
            file_path=result.file_path,
            description=(
                f"{len(cluster)} self-contained functions ({cluster_lines} lines) "
                f"could be extracted: {func_names_str}"
            ),
            suggestion="These functions only reference each other — safe to move to a submodule",
        ))


# =============================================================================
# MULTI-RESPONSIBILITY CHECK
# =============================================================================

def check_multi_responsibility(result: FileScanResult) -> None:
    """
    Flag files that mix unrelated function groups.

    Uses the call graph to find independent clusters of functions. If a file
    has 3+ clusters with no cross-references, it's doing too many things.

    Skipped for data-only files and single-orchestrator files.
    """
    if result.line_count < MIN_FILE_LINES_FOR_HEALTH:
        return
    if _is_data_only_file(result):
        return
    if _is_single_orchestrator(result):
        return

    funcs = {
        name: info for name, info in result.symbols.items()
        if info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)
    }

    if len(funcs) < MIN_SYMBOLS_FOR_STRUCTURE_CHECK:
        return

    # Build adjacency from call graph
    func_names = set(funcs.keys())
    adjacency: Dict[str, Set[str]] = {n: set() for n in func_names}
    for name, info in funcs.items():
        for callee in info.calls:
            if callee in func_names:
                adjacency[name].add(callee)
                adjacency[callee].add(name)

    # Find connected components
    visited: Set[str] = set()
    clusters: List[List[str]] = []
    for start in func_names:
        if start in visited:
            continue
        cluster: List[str] = []
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            cluster.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    queue.append(neighbour)
        if len(cluster) >= 2:  # Only count real clusters, not isolated functions
            clusters.append(cluster)

    if len(clusters) < MULTI_RESPONSIBILITY_CLUSTER_COUNT:
        return

    cluster_descs = []
    for cl in clusters[:4]:
        names = ", ".join(sorted(cl)[:3])
        if len(cl) > 3:
            names += f" +{len(cl) - 3}"
        cluster_descs.append(f"[{names}]")

    result.health_issues.append(HealthIssue(
        category=HealthCategory.MULTI_RESPONSIBILITY,
        severity=HealthSeverity.WARNING,
        file_path=result.file_path,
        description=(
            f"File has {len(clusters)} independent function groups with no "
            f"cross-references: {' | '.join(cluster_descs)}"
        ),
        suggestion=(
            "Functions in separate clusters serve different purposes — "
            "consider splitting into focused modules"
        ),
    ))


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def run_smart_health_checks(result: FileScanResult) -> None:
    """
    Run all smart structural health checks on a scanned file.

    This replaces the old _check_oversized_functions / _check_oversized_file
    pair.  It skips files that are genuinely fine (data-only, small,
    single-orchestrator) and only flags actionable structural problems.
    """
    if result.line_count < MIN_FILE_LINES_FOR_HEALTH:
        return  # Small files are fine

    if _is_data_only_file(result):
        return  # Nothing to refactor in a data file

    check_monolithic_functions(result)
    check_god_classes(result)
    check_extractable_blocks(result)
    check_multi_responsibility(result)
