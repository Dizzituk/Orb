# Purpose: codebase scanner utils 3
# Called-by: app.orchestrator.codebase_scanner
# Depends-on: app.orchestrator._codebase_scanner_utils_2, app.orchestrator.codebase_scanner_models
# Last-renovated: 2026-06-11
from __future__ import annotations
import os
import re
from app.orchestrator._codebase_scanner_utils_2 import _normalise_hash
from app.orchestrator.codebase_scanner_models import CodebaseGraph, FileScanResult, HealthCategory, HealthIssue, HealthSeverity, SymbolKind
from typing import Dict, List, Set, Tuple


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
