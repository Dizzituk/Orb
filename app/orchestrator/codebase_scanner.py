# FILE: app/orchestrator/codebase_scanner.py
# Purpose: Enhanced Codebase Scanner — AST-based symbol extraction with call graph.
# Called-by: app.llm.local_tools.zobie.streams._codebase_report_structural, app.orchestrator.refactor_pipeline
# Depends-on: app.orchestrator, app.orchestrator._codebase_scanner_health, app.orchestrator._codebase_scanner_utils_2, app.orchestrator._codebase_scanner_utils_3 (+3 more)
# Last-renovated: 2026-06-11
"""
Enhanced Codebase Scanner — AST-based symbol extraction with call graph.

Scans a Python source file and produces a complete CodebaseGraph with:
  - Every function, class, constant, and import
  - Call graph edges: which symbols reference which other symbols
  - Reverse edges: which symbols are referenced by which
  - Health issues: dead code, unused imports, oversized functions

Zero LLM calls. Pure AST parsing and deterministic analysis.

BUILD_ID: 2026-02-20-v1.0-enhanced-codebase-scanner
"""

from __future__ import annotations

import ast
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from app.orchestrator.codebase_scanner_models import (
    CodebaseGraph,
    FileScanResult,
    HealthCategory,
    HealthIssue,
    HealthSeverity,
    ImportInfo,
    SymbolInfo,
    SymbolKind,
)
from app.orchestrator._codebase_scanner_utils_2 import CODEBASE_SCANNER_BUILD_ID, _BUILTINS, _build_signature, _get_name, _is_constant_name, _is_data_structure, _is_internal, _normalise_hash
from app.orchestrator._codebase_scanner_utils_3 import _STDLIB_MODULES, _build_call_graph, _build_cross_file_edges, _check_dead_code, _check_shadowed_builtins, _check_unused_imports, _detect_circular_imports, _detect_duplicate_functions
from app.orchestrator._codebase_scanner_health import run_smart_health_checks
from app.sandbox_fs import sandbox_read_text as _sbx_read_text

# Lazy import for JS scanner (only when JS files encountered)
_js_scanner = None


def _get_js_scanner():
    """Lazy-load JS scanner to avoid import overhead for Python-only scans."""
    global _js_scanner
    if _js_scanner is None:
        try:
            from app.orchestrator import js_scanner as _js
            _js_scanner = _js
        except ImportError:
            pass
    return _js_scanner

logger = logging.getLogger(__name__)
print(f"[CODEBASE_SCANNER_LOADED] BUILD_ID={CODEBASE_SCANNER_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# Python builtins that shouldn't be flagged as references

# Standard library top-level modules (common subset)

# Legacy thresholds (kept for reference — no longer used for health checks).
# Smart structural checks in _codebase_scanner_health.py replaced these.
# MAX_FUNCTION_LINES = 200
# MAX_FILE_LINES = 800


# =============================================================================
# SINGLE FILE SCANNER
# =============================================================================

def scan_file(file_path: str, source_code: str) -> FileScanResult:
    """
    Scan a single Python file and extract all symbols with their relationships.

    This is the core scanning function. It:
    1. Parses the AST
    2. Extracts every top-level symbol (function, class, constant, import)
    3. For each symbol, scans its body for references to other symbols
    4. Builds call/reference edges
    5. Detects health issues
    """
    result = FileScanResult(
        file_path=file_path,
        line_count=source_code.count("\n") + 1,
        char_count=len(source_code),
    )

    # Parse AST
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        result.parse_error = f"SyntaxError: {e}"
        logger.warning("[codebase_scanner] Parse error in %s: %s", file_path, e)
        return result

    # Phase 1: Extract all symbols
    _extract_symbols(tree, source_code, result)

    # Phase 2: Extract imports
    _extract_imports(tree, source_code, result)

    # Phase 3: Build call graph (reference analysis)
    _build_call_graph(result)

    # Phase 4: Detect health issues
    _detect_health_issues(result)

    logger.info(
        "[codebase_scanner] Scanned %s: %d symbols, %d imports, %d issues",
        file_path, len(result.symbols), len(result.imports),
        len(result.health_issues),
    )
    return result


# =============================================================================
# PHASE 1: SYMBOL EXTRACTION
# =============================================================================

def _extract_symbols(
    tree: ast.Module,
    source_code: str,
    result: FileScanResult,
) -> None:
    """Extract all top-level symbols from the AST."""

    for node in ast.iter_child_nodes(tree):

        # --- Functions ---
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_source = ast.get_source_segment(source_code, node) or ""
            sig = _build_signature(node, source_code)

            info = SymbolInfo(
                name=node.name,
                kind=SymbolKind.ASYNC_FUNCTION if isinstance(node, ast.AsyncFunctionDef) else SymbolKind.FUNCTION,
                source_code=func_source,
                signature=sig,
                docstring=ast.get_docstring(node) or "",
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                is_private=node.name.startswith("_") and not node.name.startswith("__"),
                is_dunder=node.name.startswith("__") and node.name.endswith("__"),
                decorators=[
                    ast.get_source_segment(source_code, d) or ""
                    for d in node.decorator_list
                ],
            )
            result.symbols[node.name] = info

        # --- Classes ---
        elif isinstance(node, ast.ClassDef):
            class_source = ast.get_source_segment(source_code, node) or ""
            methods = [
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            bases = [_get_name(b) for b in node.bases]

            info = SymbolInfo(
                name=node.name,
                kind=SymbolKind.CLASS,
                source_code=class_source,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                bases=bases,
                methods=methods,
            )
            result.symbols[node.name] = info

        # --- Constants (ALL_CAPS assignments) ---
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_constant_name(target.id):
                    source_segment = ast.get_source_segment(source_code, node) or ""

                    # Determine if it's a simple constant or data structure
                    kind = SymbolKind.CONSTANT
                    if len(source_segment) > 200 or _is_data_structure(node.value):
                        kind = SymbolKind.DATA_STRUCTURE

                    info = SymbolInfo(
                        name=target.id,
                        kind=kind,
                        source_code=source_segment,
                        line_start=node.lineno,
                        line_end=node.end_lineno or node.lineno,
                    )
                    result.symbols[target.id] = info

        # --- Constants (annotated assignments like X: Type = value) ---
        elif isinstance(node, ast.AnnAssign):
            # v6.1 FIX 19b: Handle type-annotated constants.
            # e.g. INTENT_DEFINITIONS: Dict[CanonicalIntent, IntentDefinition] = {...}
            # Python AST represents these as ast.AnnAssign (not ast.Assign).
            target = node.target
            if isinstance(target, ast.Name) and _is_constant_name(target.id) and node.value is not None:
                source_segment = ast.get_source_segment(source_code, node) or ""

                kind = SymbolKind.CONSTANT
                if len(source_segment) > 200 or _is_data_structure(node.value):
                    kind = SymbolKind.DATA_STRUCTURE

                info = SymbolInfo(
                    name=target.id,
                    kind=kind,
                    source_code=source_segment,
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                )
                result.symbols[target.id] = info

        # --- Module-level expressions ---
        elif isinstance(node, ast.Expr):
            source_segment = ast.get_source_segment(source_code, node) or ""
            if source_segment:
                result.module_level_code.append(source_segment)


# =============================================================================
# PHASE 2: IMPORT EXTRACTION
# =============================================================================

def _extract_imports(
    tree: ast.Module,
    source_code: str,
    result: FileScanResult,
) -> None:
    """Extract and classify all import statements."""

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw = ast.get_source_segment(source_code, node) or f"import {alias.name}"
                info = ImportInfo(
                    raw_statement=raw,
                    module=alias.name,
                    names=[alias.asname or alias.name.split(".")[-1]],
                    is_stdlib=_is_stdlib(alias.name),
                    is_internal=_is_internal(alias.name),
                    line_number=node.lineno,
                )
                result.imports.append(info)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.asname or alias.name for alias in (node.names or [])]
            raw = ast.get_source_segment(source_code, node) or f"from {module} import ..."

            info = ImportInfo(
                raw_statement=raw,
                module=module,
                names=names,
                is_relative=bool(node.level and node.level > 0),
                is_stdlib=_is_stdlib(module),
                is_internal=_is_internal(module),
                line_number=node.lineno,
            )
            result.imports.append(info)


def _is_stdlib(module: str) -> bool:
    """Check if a module is part of Python's standard library."""
    top = module.split(".")[0] if module else ""
    return top in _STDLIB_MODULES


# =============================================================================
# PHASE 3: CALL GRAPH
# =============================================================================


# =============================================================================
# PHASE 4: HEALTH CHECKS
# =============================================================================

def _detect_health_issues(result: FileScanResult) -> None:
    """Run all health checks on the scanned file."""
    _check_dead_code(result)
    _check_unused_imports(result)
    _check_shadowed_builtins(result)
    # v2.0: Smart structural checks replace dumb size-only checks.
    # Old _check_oversized_functions / _check_oversized_file removed —
    # they flagged data files, orchestrators, and already-modular files.
    run_smart_health_checks(result)


# =============================================================================
# LEGACY OVERSIZED CHECKS (removed v2.0)
# =============================================================================
# _check_oversized_functions and _check_oversized_file have been replaced by
# run_smart_health_checks() in _codebase_scanner_health.py.
#
# The old checks flagged files purely by line count / function size, which
# produced false positives for:
# - Data-only files (intents.py — pure dict, nothing to refactor)
# - Single async orchestrators (can't split without breaking yield chains)
# - Already-modular files (20 small functions = fine at any total size)
# - Files where helpers were already extracted to _utils modules
#
# The new checks analyse call-graph structure, cluster independence, and
# function composition to flag only genuinely refactorable code.


# =============================================================================
# MULTI-FILE SCANNER
# =============================================================================

def scan_codebase(
    file_paths: List[str],
    project_roots: Optional[List[str]] = None,
) -> CodebaseGraph:
    """
    Scan multiple files and build a cross-file codebase graph.

    After scanning each file individually, this runs cross-file analysis:
    - Connects call graph edges across file boundaries
    - Detects circular import chains
    - Detects orphaned files
    - Detects duplicate function bodies across files
    """
    if project_roots is None:
        project_roots = ["D:\\Orb", "D:\\orb-desktop"]

    graph = CodebaseGraph()

    # Scan each file
    for rel_path in file_paths:
        source_code = _read_source(rel_path, project_roots)
        if source_code is None:
            graph.scan_errors.append(f"Could not read: {rel_path}")
            continue

        # Route to appropriate scanner based on file type
        js_mod = _get_js_scanner()
        if js_mod and js_mod.is_js_file(rel_path):
            scan = js_mod.scan_js_file(rel_path, source_code)
        elif rel_path.endswith(".py"):
            scan = scan_file(rel_path, source_code)
        else:
            logger.debug("[codebase_scanner] Skipping unsupported file type: %s", rel_path)
            continue

        graph.files[rel_path] = scan

    # Cross-file analysis
    _build_cross_file_edges(graph)
    _detect_duplicate_functions(graph)
    _detect_circular_imports(graph)

    logger.info(
        "[codebase_scanner] Codebase scan complete: %d files, %d symbols, %d issues",
        graph.total_files, graph.total_symbols, graph.total_health_issues,
    )
    return graph


def _read_source(
    rel_path: str,
    project_roots: List[str],
) -> Optional[str]:
    """v3.5: Read a source file from the SANDBOX under any project root."""
    for root in project_roots:
        abs_path = os.path.join(root, rel_path.replace("/", os.sep))
        try:
            content = _sbx_read_text(abs_path)
            if content is not None:
                return content
        except Exception as e:
            logger.warning("[codebase_scanner] Error reading %s from sandbox: %s", abs_path, e)
    return None


# =============================================================================
# CROSS-FILE ANALYSIS
# =============================================================================
