# FILE: app/orchestrator/js_scanner.py
# Purpose: JavaScript/TypeScript scanner — regex-based symbol extraction.
# Called-by: app.orchestrator.codebase_scanner
# Depends-on: app.orchestrator.codebase_scanner_models
# Last-renovated: 2026-06-11
"""
JavaScript/TypeScript scanner — regex-based symbol extraction.

Python's ast module only handles Python. For JS/TS files we use regex
patterns to extract top-level exports, functions, classes, constants,
and import statements. Less precise than AST but sufficient for
dependency graph construction and health checks.

BUILD_ID: 2026-02-20-v1.0-js-scanner
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set

from app.orchestrator.codebase_scanner_models import (
    FileScanResult,
    HealthCategory,
    HealthIssue,
    HealthSeverity,
    ImportInfo,
    SymbolInfo,
    SymbolKind,
)

logger = logging.getLogger(__name__)

JS_SCANNER_BUILD_ID = "2026-02-20-v1.0-js-scanner"
print(f"[JS_SCANNER_LOADED] BUILD_ID={JS_SCANNER_BUILD_ID}")


# =============================================================================
# FILE TYPE DETECTION
# =============================================================================

_JS_EXTENSIONS = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})


def is_js_file(file_path: str) -> bool:
    """Check if a file path is a JavaScript/TypeScript file."""
    for ext in _JS_EXTENSIONS:
        if file_path.endswith(ext):
            return True
    return False


# =============================================================================
# PATTERNS
# =============================================================================

# Named export function: export function name(...) or export async function name(...)
_EXPORT_FUNC = re.compile(
    r'^export\s+(?:default\s+)?(?:async\s+)?function\s+(\w+)\s*\(',
    re.MULTILINE,
)

# Named export const/let/var with arrow or value
_EXPORT_CONST = re.compile(
    r'^export\s+(?:default\s+)?(?:const|let|var)\s+(\w+)\s*[=:]',
    re.MULTILINE,
)

# Named export class
_EXPORT_CLASS = re.compile(
    r'^export\s+(?:default\s+)?class\s+(\w+)',
    re.MULTILINE,
)

# Non-exported function declarations
_FUNC_DECL = re.compile(
    r'^(?:async\s+)?function\s+(\w+)\s*\(',
    re.MULTILINE,
)

# Non-exported const/let/var arrow functions
_CONST_ARROW = re.compile(
    r'^(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(?',
    re.MULTILINE,
)

# Non-exported class
_CLASS_DECL = re.compile(
    r'^class\s+(\w+)',
    re.MULTILINE,
)

# ALL_CAPS constants
_CONST_UPPER = re.compile(
    r'^(?:export\s+)?(?:const|let|var)\s+([A-Z][A-Z0-9_]+)\s*=',
    re.MULTILINE,
)

# Import statements
_IMPORT_STMT = re.compile(
    r'^import\s+(.+?)\s+from\s+[\'"]([^\'"]+)[\'"]',
    re.MULTILINE,
)

# Require statements
_REQUIRE_STMT = re.compile(
    r'^(?:const|let|var)\s+(.+?)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)',
    re.MULTILINE,
)

# Export list: export { name1, name2 } from '...'
_EXPORT_LIST = re.compile(
    r'^export\s*\{([^}]+)\}(?:\s*from\s*[\'"]([^\'"]+)[\'"])?',
    re.MULTILINE,
)

# TypeScript interface/type
_TS_INTERFACE = re.compile(
    r'^(?:export\s+)?(?:interface|type)\s+(\w+)',
    re.MULTILINE,
)


# =============================================================================
# SCANNER
# =============================================================================

def scan_js_file(file_path: str, source_code: str) -> FileScanResult:
    """
    Scan a JavaScript/TypeScript file using regex patterns.

    Extracts functions, classes, constants, imports, and exports.
    Less precise than AST but handles the common patterns.
    """
    result = FileScanResult(
        file_path=file_path,
        line_count=source_code.count("\n") + 1,
        char_count=len(source_code),
    )

    # Remove comments to avoid false matches
    cleaned = _strip_comments(source_code)

    # Extract symbols
    _extract_js_functions(cleaned, source_code, result)
    _extract_js_classes(cleaned, source_code, result)
    _extract_js_constants(cleaned, source_code, result)
    _extract_ts_types(cleaned, source_code, result)

    # Extract imports
    _extract_js_imports(cleaned, result)

    # Build internal call graph (same approach as Python scanner)
    _build_js_call_graph(result)

    # Health checks
    _check_js_health(result)

    logger.info(
        "[js_scanner] Scanned %s: %d symbols, %d imports",
        file_path, len(result.symbols), len(result.imports),
    )
    return result


# =============================================================================
# EXTRACTION
# =============================================================================

def _strip_comments(source: str) -> str:
    """Strip JS comments to avoid false regex matches."""
    # Remove single-line comments
    result = re.sub(r'//[^\n]*', '', source)
    # Remove multi-line comments
    result = re.sub(r'/\*.*?\*/', '', result, flags=re.DOTALL)
    return result


def _extract_js_functions(cleaned: str, original: str, result: FileScanResult) -> None:
    """Extract function declarations."""
    seen: Set[str] = set()

    for match in _EXPORT_FUNC.finditer(cleaned):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        is_async = "async" in match.group(0)
        body = _extract_body_around(original, name, match.start())
        result.symbols[name] = SymbolInfo(
            name=name,
            kind=SymbolKind.ASYNC_FUNCTION if is_async else SymbolKind.FUNCTION,
            source_code=body,
            is_async=is_async,
            line_start=cleaned[:match.start()].count("\n") + 1,
        )

    for match in _FUNC_DECL.finditer(cleaned):
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        is_async = "async" in match.group(0)
        body = _extract_body_around(original, name, match.start())
        result.symbols[name] = SymbolInfo(
            name=name,
            kind=SymbolKind.ASYNC_FUNCTION if is_async else SymbolKind.FUNCTION,
            source_code=body,
            is_async=is_async,
            is_private=name.startswith("_"),
            line_start=cleaned[:match.start()].count("\n") + 1,
        )


def _extract_js_classes(cleaned: str, original: str, result: FileScanResult) -> None:
    """Extract class declarations."""
    seen: Set[str] = set()

    for pattern in [_EXPORT_CLASS, _CLASS_DECL]:
        for match in pattern.finditer(cleaned):
            name = match.group(1)
            if name in seen:
                continue
            seen.add(name)
            body = _extract_body_around(original, name, match.start())
            result.symbols[name] = SymbolInfo(
                name=name,
                kind=SymbolKind.CLASS,
                source_code=body,
                line_start=cleaned[:match.start()].count("\n") + 1,
            )


def _extract_js_constants(cleaned: str, original: str, result: FileScanResult) -> None:
    """Extract ALL_CAPS constants."""
    for match in _CONST_UPPER.finditer(cleaned):
        name = match.group(1)
        if name in result.symbols:
            continue
        line_num = cleaned[:match.start()].count("\n") + 1
        # Get the full line as source
        lines = original.split("\n")
        source = lines[line_num - 1] if line_num <= len(lines) else ""
        result.symbols[name] = SymbolInfo(
            name=name,
            kind=SymbolKind.CONSTANT,
            source_code=source,
            line_start=line_num,
        )


def _extract_ts_types(cleaned: str, original: str, result: FileScanResult) -> None:
    """Extract TypeScript interface and type declarations."""
    for match in _TS_INTERFACE.finditer(cleaned):
        name = match.group(1)
        if name in result.symbols:
            continue
        body = _extract_body_around(original, name, match.start())
        result.symbols[name] = SymbolInfo(
            name=name,
            kind=SymbolKind.CLASS,  # Treat interfaces as class-like
            source_code=body,
            line_start=cleaned[:match.start()].count("\n") + 1,
        )


def _extract_js_imports(cleaned: str, result: FileScanResult) -> None:
    """Extract import and require statements."""
    for match in _IMPORT_STMT.finditer(cleaned):
        names_part = match.group(1).strip()
        module = match.group(2)
        names = _parse_import_names(names_part)
        result.imports.append(ImportInfo(
            raw_statement=match.group(0),
            module=module,
            names=names,
            is_internal=module.startswith(".") or module.startswith("@/"),
            line_number=cleaned[:match.start()].count("\n") + 1,
        ))

    for match in _REQUIRE_STMT.finditer(cleaned):
        names_part = match.group(1).strip()
        module = match.group(2)
        names = _parse_import_names(names_part)
        result.imports.append(ImportInfo(
            raw_statement=match.group(0),
            module=module,
            names=names,
            is_internal=module.startswith(".") or module.startswith("@/"),
            line_number=cleaned[:match.start()].count("\n") + 1,
        ))


def _parse_import_names(names_part: str) -> List[str]:
    """Parse the names from an import statement."""
    if not names_part:
        return []
    cleaned = names_part.strip()
    if '{' in cleaned:
        brace_match = re.search(r'\{([^}]+)\}', cleaned)
        if brace_match:
            inner = brace_match.group(1)
        else:
            inner = cleaned.strip('{} ')
        before_brace = cleaned.split('{')[0].strip().rstrip(',').strip()
        parts = [p.strip() for p in inner.split(',')]
        if before_brace and before_brace != '*':
            parts.insert(0, before_brace)
    else:
        parts = [p.strip() for p in cleaned.split(',')]

    result = []
    for part in parts:
        part = part.strip()
        if not part or part == '*':
            continue
        tokens = part.split(' as ')
        name = tokens[-1].strip()
        if name:
            result.append(name)
    return result


# =============================================================================
# BODY EXTRACTION
# =============================================================================

def _extract_body_around(source: str, name: str, approx_pos: int) -> str:
    """
    Extract the body of a function/class/constant from source code.

    Uses brace counting from the approximate position.
    """
    # Find the name near the position
    search_start = max(0, approx_pos - 50)
    search_region = source[search_start:]

    name_idx = search_region.find(name)
    if name_idx < 0:
        return ""

    abs_start = search_start + name_idx

    # Walk backwards to find the start of the declaration
    line_start = source.rfind("\n", 0, abs_start)
    if line_start < 0:
        line_start = 0
    else:
        line_start += 1

    # Find the opening brace
    brace_idx = source.find("{", abs_start)
    if brace_idx < 0:
        # No brace — single-line statement, take to end of line
        line_end = source.find("\n", abs_start)
        if line_end < 0:
            line_end = len(source)
        return source[line_start:line_end].strip()

    # Count braces to find the matching close
    depth = 0
    pos = brace_idx
    in_string = None
    while pos < len(source):
        ch = source[pos]

        # Handle strings
        if ch in ('"', "'", '`') and (pos == 0 or source[pos - 1] != '\\'):
            if in_string is None:
                in_string = ch
            elif in_string == ch:
                in_string = None
            pos += 1
            continue

        if in_string is None:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return source[line_start:pos + 1].strip()
        pos += 1

    # Fallback: couldn't find matching brace, take 50 lines
    lines = source[line_start:].split("\n")[:50]
    return "\n".join(lines).strip()


# =============================================================================
# CALL GRAPH + HEALTH
# =============================================================================

def _build_js_call_graph(result: FileScanResult) -> None:
    """Build within-file call graph using same approach as Python scanner."""
    all_names = set(result.symbols.keys())

    for name, info in result.symbols.items():
        if not info.source_code:
            continue
        for other_name in all_names:
            if other_name == name or len(other_name) < 2:
                continue
            pattern = r'\b' + re.escape(other_name) + r'\b'
            if re.search(pattern, info.source_code):
                ref_info = result.symbols.get(other_name)
                if not ref_info:
                    continue
                if ref_info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
                    info.calls.append(other_name)
                    ref_info.called_by.append(name)
                else:
                    info.references.append(other_name)
                    ref_info.referenced_by.append(name)

    for info in result.symbols.values():
        info.calls.sort()
        info.references.sort()
        info.called_by.sort()
        info.referenced_by.sort()


def _check_js_health(result: FileScanResult) -> None:
    """Run basic health checks on a JS/TS file."""
    # Dead code
    for name, info in result.symbols.items():
        if info.is_dunder:
            continue
        if info.is_dead:
            result.health_issues.append(HealthIssue(
                category=HealthCategory.DEAD_CODE,
                severity=HealthSeverity.WARNING,
                file_path=result.file_path,
                symbol_name=name,
                description=f"'{name}' is never called or referenced within this file",
            ))

    # Oversized file
    if result.line_count > 800:
        result.health_issues.append(HealthIssue(
            category=HealthCategory.OVERSIZED_FILE,
            severity=HealthSeverity.WARNING,
            file_path=result.file_path,
            description=f"File is {result.line_count} lines",
        ))
