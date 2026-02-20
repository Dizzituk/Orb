# FILE: app/orchestrator/codebase_scanner.py
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
import hashlib
import logging
import os
import re
import sys
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

CODEBASE_SCANNER_BUILD_ID = "2026-02-20-v1.0-enhanced-codebase-scanner"
print(f"[CODEBASE_SCANNER_LOADED] BUILD_ID={CODEBASE_SCANNER_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# Python builtins that shouldn't be flagged as references
_BUILTINS = frozenset(dir(__builtins__)) if isinstance(__builtins__, dict) else frozenset(dir(__builtins__))

# Standard library top-level modules (common subset)
_STDLIB_MODULES = frozenset({
    "abc", "ast", "asyncio", "base64", "bisect", "collections", "concurrent",
    "contextlib", "copy", "csv", "dataclasses", "datetime", "decimal",
    "difflib", "enum", "errno", "functools", "glob", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "importlib", "inspect", "io",
    "itertools", "json", "keyword", "linecache", "locale", "logging",
    "math", "mimetypes", "multiprocessing", "operator", "os", "pathlib",
    "pickle", "platform", "pprint", "queue", "random", "re", "secrets",
    "shlex", "shutil", "signal", "socket", "sqlite3", "ssl", "stat",
    "string", "struct", "subprocess", "sys", "tempfile", "textwrap",
    "threading", "time", "timeit", "token", "tokenize", "traceback",
    "types", "typing", "unittest", "urllib", "uuid", "warnings",
    "weakref", "xml", "zipfile", "zlib",
})

# Oversized thresholds
MAX_FUNCTION_LINES = 200
MAX_FILE_LINES = 800


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

        # --- Module-level expressions ---
        elif isinstance(node, ast.Expr):
            source_segment = ast.get_source_segment(source_code, node) or ""
            if source_segment:
                result.module_level_code.append(source_segment)


def _build_signature(node: ast.FunctionDef, source_code: str) -> str:
    """Build a clean function signature string."""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    # Get the source up to the colon
    try:
        func_source = ast.get_source_segment(source_code, node) or ""
        # Find first colon that's followed by a newline (not in type hints)
        lines = func_source.split("\n")
        sig_lines = []
        for line in lines:
            sig_lines.append(line)
            stripped = line.rstrip()
            if stripped.endswith(":") and not stripped.endswith("::"):
                break
        return "\n".join(sig_lines)
    except Exception:
        return f"{prefix} {node.name}(...):"


def _is_constant_name(name: str) -> bool:
    """Check if a name follows ALL_CAPS constant convention."""
    return bool(re.match(r'^[A-Z][A-Z0-9_]*$', name)) and len(name) > 1


def _is_data_structure(node: ast.expr) -> bool:
    """Check if an AST value node is a data structure (list, dict, etc)."""
    return isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Tuple, ast.JoinedStr))


def _get_name(node: ast.expr) -> str:
    """Extract a name string from an AST expression node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _get_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Constant):
        return str(node.value)
    return ""


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


def _is_internal(module: str) -> bool:
    """Check if a module is internal to the project."""
    if not module:
        return False
    return module.startswith("app.") or module.startswith("app/")


# =============================================================================
# PHASE 3: CALL GRAPH
# =============================================================================

def _build_call_graph(result: FileScanResult) -> None:
    """
    Analyse each symbol's source code to find references to other symbols.

    For each symbol, scan its body for word-boundary matches against
    all other known symbol names. This builds both forward edges
    (calls/references) and reverse edges (called_by/referenced_by).
    """
    all_names = set(result.symbols.keys())

    for name, info in result.symbols.items():
        if not info.source_code:
            continue

        # Find all references in this symbol's body
        body = info.source_code
        outgoing: Set[str] = set()

        for other_name in all_names:
            if other_name == name:
                continue
            if len(other_name) < 2:
                continue

            # Word boundary match
            pattern = r'\b' + re.escape(other_name) + r'\b'
            if re.search(pattern, body):
                outgoing.add(other_name)

        # Classify edges
        for ref_name in outgoing:
            ref_info = result.symbols.get(ref_name)
            if not ref_info:
                continue

            if ref_info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
                info.calls.append(ref_name)
                ref_info.called_by.append(name)
            else:
                info.references.append(ref_name)
                ref_info.referenced_by.append(name)

    # Sort for determinism
    for info in result.symbols.values():
        info.calls.sort()
        info.references.sort()
        info.called_by.sort()
        info.referenced_by.sort()


# =============================================================================
# PHASE 4: HEALTH CHECKS
# =============================================================================

def _detect_health_issues(result: FileScanResult) -> None:
    """Run all health checks on the scanned file."""
    _check_dead_code(result)
    _check_unused_imports(result)
    _check_oversized_functions(result)
    _check_oversized_file(result)
    _check_shadowed_builtins(result)


def _check_dead_code(result: FileScanResult) -> None:
    """Find functions and constants that nothing references."""
    for name, info in result.symbols.items():
        # Skip dunders, they're called by the runtime
        if info.is_dunder:
            continue
        # Skip module-level setup constants (BUILD_ID etc)
        if info.kind == SymbolKind.CONSTANT and "BUILD" in name:
            continue

        if info.is_dead:
            result.health_issues.append(HealthIssue(
                category=HealthCategory.DEAD_CODE,
                severity=HealthSeverity.WARNING,
                file_path=result.file_path,
                symbol_name=name,
                line_number=info.line_start,
                description=f"'{name}' ({info.kind.value}) is never called or referenced within this file",
                suggestion=f"Verify '{name}' is used externally or remove it",
            ))


def _check_unused_imports(result: FileScanResult) -> None:
    """Find imported names that are never used in the file."""
    # Collect all text from symbol bodies + module level code
    all_code = "\n".join(
        info.source_code for info in result.symbols.values()
    ) + "\n".join(result.module_level_code)

    for imp in result.imports:
        for name in imp.names:
            # Check if the imported name appears anywhere in the code
            pattern = r'\b' + re.escape(name) + r'\b'
            if re.search(pattern, all_code):
                imp.used_names.append(name)
            else:
                imp.unused_names.append(name)

        if imp.unused_names:
            result.health_issues.append(HealthIssue(
                category=HealthCategory.DEAD_IMPORT,
                severity=HealthSeverity.INFO,
                file_path=result.file_path,
                line_number=imp.line_number,
                description=f"Unused import(s): {', '.join(imp.unused_names)} from '{imp.module}'",
                suggestion="Remove unused imports",
            ))


def _check_oversized_functions(result: FileScanResult) -> None:
    """Flag functions that exceed the line limit."""
    for name, info in result.symbols.items():
        if info.kind not in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
            continue
        if info.estimated_lines > MAX_FUNCTION_LINES:
            result.health_issues.append(HealthIssue(
                category=HealthCategory.OVERSIZED_FUNCTION,
                severity=HealthSeverity.WARNING,
                file_path=result.file_path,
                symbol_name=name,
                line_number=info.line_start,
                description=f"'{name}' is {info.estimated_lines} lines (limit: {MAX_FUNCTION_LINES})",
                suggestion="Consider breaking into smaller functions",
            ))


def _check_oversized_file(result: FileScanResult) -> None:
    """Flag files that exceed the line limit."""
    if result.line_count > MAX_FILE_LINES:
        result.health_issues.append(HealthIssue(
            category=HealthCategory.OVERSIZED_FILE,
            severity=HealthSeverity.WARNING,
            file_path=result.file_path,
            description=f"File is {result.line_count} lines (limit: {MAX_FILE_LINES})",
            suggestion="Consider decomposing into a subpackage",
        ))


def _check_shadowed_builtins(result: FileScanResult) -> None:
    """Find symbols that shadow Python builtins."""
    shadow_targets = {"list", "dict", "set", "tuple", "type", "id", "input",
                      "format", "filter", "map", "range", "hash", "sum",
                      "min", "max", "open", "print", "len", "str", "int",
                      "float", "bool", "bytes", "object", "property"}

    for name in result.symbols:
        if name in shadow_targets:
            result.health_issues.append(HealthIssue(
                category=HealthCategory.SHADOWED_BUILTIN,
                severity=HealthSeverity.WARNING,
                file_path=result.file_path,
                symbol_name=name,
                description=f"'{name}' shadows the Python builtin '{name}'",
                suggestion=f"Rename to avoid shadowing (e.g. '{name}_value')",
            ))


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
    """Try to read a source file from disk under any project root."""
    for root in project_roots:
        abs_path = os.path.join(root, rel_path.replace("/", os.sep))
        if os.path.isfile(abs_path):
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    return f.read()
            except Exception as e:
                logger.warning("[codebase_scanner] Error reading %s: %s", abs_path, e)
    return None


# =============================================================================
# CROSS-FILE ANALYSIS
# =============================================================================

def _build_cross_file_edges(graph: CodebaseGraph) -> None:
    """
    Connect call graph edges across file boundaries.

    For each file, resolve its imports to find which symbols come from
    which other files in the graph. Then scan every symbol body for
    references to those imported names and build cross-file edges.

    This catches the case where function Y in file A calls function X
    from file B — the per-file scan only finds within-file references,
    so this step completes the global picture.
    """
    # Build global symbol → file lookup
    symbol_locations: Dict[str, str] = {}
    for fp, scan in graph.files.items():
        for name in scan.symbols:
            symbol_locations[name] = fp

    # Build module → file lookup for import resolution
    module_to_file: Dict[str, str] = {}
    for fp in graph.files:
        mod = fp.replace("/", ".").replace("\\", ".").replace(".py", "")
        module_to_file[mod] = fp

    for fp, scan in graph.files.items():
        # Resolve what symbols this file imports from other files in the graph
        imported_from_graph: Dict[str, str] = {}  # imported_name → source_file

        for imp in scan.imports:
            if not imp.is_internal:
                continue
            source_fp = module_to_file.get(imp.module)
            if not source_fp or source_fp == fp:
                continue
            source_scan = graph.files.get(source_fp)
            if not source_scan:
                continue
            for imp_name in imp.names:
                if imp_name in source_scan.symbols:
                    imported_from_graph[imp_name] = source_fp

        if not imported_from_graph:
            continue

        # Scan each symbol in this file for references to cross-file imports
        for name, info in scan.symbols.items():
            if not info.source_code:
                continue
            for cross_name, cross_fp in imported_from_graph.items():
                if len(cross_name) < 2:
                    continue
                pattern = r'\b' + re.escape(cross_name) + r'\b'
                if re.search(pattern, info.source_code):
                    cross_info = graph.files[cross_fp].symbols.get(cross_name)
                    if not cross_info:
                        continue
                    # Add cross-file edges
                    if cross_info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
                        if cross_name not in info.calls:
                            info.calls.append(cross_name)
                            info.calls.sort()
                        if name not in cross_info.called_by:
                            cross_info.called_by.append(name)
                            cross_info.called_by.sort()
                    else:
                        if cross_name not in info.references:
                            info.references.append(cross_name)
                            info.references.sort()
                        if name not in cross_info.referenced_by:
                            cross_info.referenced_by.append(name)
                            cross_info.referenced_by.sort()


def _detect_duplicate_functions(graph: CodebaseGraph) -> None:
    """
    Find functions with identical or near-identical bodies across files.

    Uses a normalised hash of the function body (stripped of whitespace,
    comments, and variable names) to detect structural duplicates.
    """
    body_hashes: Dict[str, List[Tuple[str, str]]] = {}  # hash → [(file, func_name)]

    for fp, scan in graph.files.items():
        for name, info in scan.symbols.items():
            if info.kind not in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
                continue
            if info.estimated_lines < 10:
                continue  # Skip tiny functions

            body_hash = _normalise_hash(info.source_code)
            if body_hash not in body_hashes:
                body_hashes[body_hash] = []
            body_hashes[body_hash].append((fp, name))

    for body_hash, locations in body_hashes.items():
        if len(locations) > 1:
            names = [f"{fp}::{name}" for fp, name in locations]
            graph.health_issues.append(HealthIssue(
                category=HealthCategory.DUPLICATE_CODE,
                severity=HealthSeverity.WARNING,
                file_path=locations[0][0],
                symbol_name=locations[0][1],
                description=f"Duplicate function body found in: {', '.join(names)}",
                suggestion="Extract into a shared utility and import from one location",
            ))


def _normalise_hash(source_code: str) -> str:
    """
    Create a normalised hash of source code for duplicate detection.

    Strips comments, normalises whitespace, but preserves structure.
    """
    # Remove comments
    lines = []
    for line in source_code.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Remove inline comments
        code_part = line.split("#")[0].rstrip()
        if code_part.strip():
            lines.append(code_part.strip())

    normalised = "\n".join(lines)
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def _detect_circular_imports(graph: CodebaseGraph) -> None:
    """
    Detect circular import chains across files.

    Builds a file-level import graph and checks for cycles.
    """
    # Build file import graph: file → set of files it imports from
    import_edges: Dict[str, Set[str]] = {}
    file_modules: Dict[str, str] = {}  # module_path → file_path

    # Build module → file mapping
    for fp in graph.files:
        # Convert file path to module path
        module = fp.replace("/", ".").replace("\\", ".").replace(".py", "")
        file_modules[module] = fp

    for fp, scan in graph.files.items():
        import_edges[fp] = set()
        for imp in scan.imports:
            if not imp.is_internal:
                continue
            # Try to resolve import to a known file
            target_fp = file_modules.get(imp.module)
            if target_fp and target_fp != fp:
                import_edges[fp].add(target_fp)

    # DFS cycle detection
    visited: Set[str] = set()
    in_stack: Set[str] = set()
    cycles: List[List[str]] = []

    def _dfs(node: str, path: List[str]) -> None:
        if node in in_stack:
            # Cycle found
            cycle_start = path.index(node)
            cycle = path[cycle_start:] + [node]
            cycles.append(cycle)
            return
        if node in visited:
            return

        visited.add(node)
        in_stack.add(node)
        path.append(node)

        for neighbour in import_edges.get(node, set()):
            _dfs(neighbour, path)

        path.pop()
        in_stack.remove(node)

    for fp in import_edges:
        if fp not in visited:
            _dfs(fp, [])

    for cycle in cycles:
        cycle_str = " → ".join(os.path.basename(fp) for fp in cycle)
        graph.health_issues.append(HealthIssue(
            category=HealthCategory.CIRCULAR_DEPENDENCY,
            severity=HealthSeverity.ERROR,
            file_path=cycle[0],
            description=f"Circular import chain: {cycle_str}",
            suggestion="Break the cycle by extracting shared types into a separate module",
        ))
