# FILE: app/orchestrator/ast_helpers.py
"""
AST parsing utilities for cross-segment integration verification.

Provides Python AST and TypeScript regex-based extraction of:
- Class, function, and variable definitions
- Export declarations
- Import statements (what a file imports from where)
- Import path resolution (module path → absolute file path)

Used by integration_check.py to verify cross-segment references
without requiring a full type checker.

Design:
    - Host-direct filesystem access (same pattern as file_verifier.py)
    - Python: ast module for reliable parsing
    - TypeScript: regex-based (no TS parser dependency)
    - All functions are pure — no side effects, no LLM calls

Phase 3 of Pipeline Segmentation.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

AST_HELPERS_BUILD_ID = "2026-02-08-v1.0-initial"
print(f"[AST_HELPERS_LOADED] BUILD_ID={AST_HELPERS_BUILD_ID}")


# =============================================================================
# PYTHON AST EXTRACTION
# =============================================================================


def extract_python_definitions(file_path: str) -> Dict[str, Any]:
    """
    Parse a Python file with ast and return defined names.

    Returns:
        {
            "classes": ["ClassName", ...],
            "functions": ["function_name", ...],
            "variables": ["CONSTANT_NAME", ...],
            "exports": [...],  # from __all__ if defined
            "imports_from": [{"module": "app.x", "names": ["Y", "Z"]}, ...]
        }

    On parse failure returns a dict with empty lists and an "error" key.
    Uses host-direct filesystem access.
    """
    result: Dict[str, Any] = {
        "classes": [],
        "functions": [],
        "variables": [],
        "exports": [],
        "imports_from": [],
    }

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except OSError as e:
        logger.warning("[INTEGRATION_CHECK] Cannot read Python file %s: %s", file_path, e)
        result["error"] = str(e)
        return result

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as e:
        logger.warning("[INTEGRATION_CHECK] Syntax error parsing %s: %s", file_path, e)
        result["error"] = str(e)
        return result

    for node in ast.iter_child_nodes(tree):
        # --- Class definitions ---
        if isinstance(node, ast.ClassDef):
            result["classes"].append(node.name)

        # --- Function definitions (top-level only) ---
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result["functions"].append(node.name)

        # --- Variable assignments (top-level only) ---
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    # Capture __all__ as the exports list
                    if name == "__all__" and isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                result["exports"].append(elt.value)
                    else:
                        result["variables"].append(name)

        # --- AnnAssign (type-annotated assignments like x: int = 5) ---
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                result["variables"].append(node.target.id)

        # --- Import statements ---
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names = []
                for alias in node.names:
                    names.append(alias.asname if alias.asname else alias.name)
                result["imports_from"].append({
                    "module": node.module,
                    "names": names,
                })

    logger.debug(
        "[INTEGRATION_CHECK] Extracted from %s: %d classes, %d functions, %d variables, %d imports",
        file_path,
        len(result["classes"]),
        len(result["functions"]),
        len(result["variables"]),
        len(result["imports_from"]),
    )
    return result


# =============================================================================
# TYPESCRIPT / TSX REGEX EXTRACTION
# =============================================================================

# Named export patterns
_TS_EXPORT_FUNCTION_RE = re.compile(
    r"^export\s+(?:async\s+)?function\s+(\w+)", re.MULTILINE
)
_TS_EXPORT_CLASS_RE = re.compile(
    r"^export\s+class\s+(\w+)", re.MULTILINE
)
_TS_EXPORT_CONST_RE = re.compile(
    r"^export\s+(?:const|let|var)\s+(\w+)", re.MULTILINE
)
_TS_EXPORT_TYPE_RE = re.compile(
    r"^export\s+type\s+(\w+)", re.MULTILINE
)
_TS_EXPORT_INTERFACE_RE = re.compile(
    r"^export\s+interface\s+(\w+)", re.MULTILINE
)
_TS_EXPORT_ENUM_RE = re.compile(
    r"^export\s+enum\s+(\w+)", re.MULTILINE
)

# Default export: export default X (where X is a name, not an inline expression)
_TS_EXPORT_DEFAULT_NAME_RE = re.compile(
    r"^export\s+default\s+(?:function\s+|class\s+)?(\w+)", re.MULTILINE
)

# Re-export: export { X, Y } from './z'
_TS_REEXPORT_RE = re.compile(
    r"^export\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", re.MULTILINE
)

# Import statements: import { X, Y } from './z'  or  import X from './z'
_TS_IMPORT_NAMED_RE = re.compile(
    r"^import\s*\{([^}]+)\}\s*from\s*['\"]([^'\"]+)['\"]", re.MULTILINE
)
_TS_IMPORT_DEFAULT_RE = re.compile(
    r"^import\s+(\w+)\s+from\s*['\"]([^'\"]+)['\"]", re.MULTILINE
)


def extract_typescript_exports(file_path: str) -> Dict[str, Any]:
    """
    Parse a TypeScript/TSX file with regex and return exports and imports.

    Returns:
        {
            "exports": ["ComponentName", "hookName", "TypeName", ...],
            "default_export": "ComponentName" or None,
            "imports_from": [{"module": "../x", "names": ["Y", "Z"]}, ...]
        }

    Regex-based — no TS parser needed. Handles:
    - export function X
    - export class X
    - export const X
    - export default X
    - export type X
    - export interface X
    - export enum X
    - export { X, Y } from './z'
    """
    result: Dict[str, Any] = {
        "exports": [],
        "default_export": None,
        "imports_from": [],
    }

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        logger.warning("[INTEGRATION_CHECK] Cannot read TypeScript file %s: %s", file_path, e)
        result["error"] = str(e)
        return result

    exports_set: set = set()

    # Named exports
    for pattern in [
        _TS_EXPORT_FUNCTION_RE,
        _TS_EXPORT_CLASS_RE,
        _TS_EXPORT_CONST_RE,
        _TS_EXPORT_TYPE_RE,
        _TS_EXPORT_INTERFACE_RE,
        _TS_EXPORT_ENUM_RE,
    ]:
        for match in pattern.finditer(content):
            exports_set.add(match.group(1))

    # Default export
    for match in _TS_EXPORT_DEFAULT_NAME_RE.finditer(content):
        name = match.group(1)
        result["default_export"] = name
        exports_set.add(name)

    # Re-exports: export { X, Y } from './z'
    for match in _TS_REEXPORT_RE.finditer(content):
        names_str = match.group(1)
        for name_part in names_str.split(","):
            name_part = name_part.strip()
            # Handle "X as Y" — export the alias
            if " as " in name_part:
                _, alias = name_part.split(" as ", 1)
                exports_set.add(alias.strip())
            elif name_part:
                exports_set.add(name_part)

    result["exports"] = sorted(exports_set)

    # Import statements
    for match in _TS_IMPORT_NAMED_RE.finditer(content):
        names_str = match.group(1)
        module = match.group(2)
        names = []
        for name_part in names_str.split(","):
            name_part = name_part.strip()
            if " as " in name_part:
                original, _ = name_part.split(" as ", 1)
                names.append(original.strip())
            elif name_part:
                names.append(name_part)
        if names:
            result["imports_from"].append({"module": module, "names": names})

    for match in _TS_IMPORT_DEFAULT_RE.finditer(content):
        name = match.group(1)
        module = match.group(2)
        # Skip 'import type' which is a TS-only syntax
        if name != "type":
            result["imports_from"].append({"module": module, "names": [name]})

    logger.debug(
        "[INTEGRATION_CHECK] Extracted from %s: %d exports, %d imports",
        file_path, len(result["exports"]), len(result["imports_from"]),
    )
    return result


# =============================================================================
# IMPORT PATH RESOLUTION
# =============================================================================


def resolve_python_import(
    import_module: str,
    project_roots: List[str],
) -> Optional[str]:
    """
    Convert a Python import path to an absolute file path.

    E.g. 'app.services.transcription' → 'D:\\Orb\\app\\services\\transcription.py'

    Checks both module-as-file (.py) and module-as-package (__init__.py).
    Tries each project_root in order.

    Returns None if the file doesn't exist on disk.
    """
    # Convert dots to path separators
    parts = import_module.split(".")
    relative_path = os.path.join(*parts)

    for root in project_roots:
        # Check as .py file
        py_path = os.path.join(root, relative_path + ".py")
        if os.path.isfile(py_path):
            return os.path.normpath(py_path)

        # Check as package (__init__.py)
        init_path = os.path.join(root, relative_path, "__init__.py")
        if os.path.isfile(init_path):
            return os.path.normpath(init_path)

    return None


def resolve_typescript_import(
    import_path: str,
    importing_file: str,
    project_roots: List[str],
) -> Optional[str]:
    """
    Convert a TypeScript relative import to an absolute file path.

    E.g. '../services/api' imported from 'D:\\orb-desktop\\src\\components\\App.tsx'
         → 'D:\\orb-desktop\\src\\services\\api.ts'

    Handles:
    - Relative paths (./x, ../x)
    - Extension resolution: .ts, .tsx, .js, .jsx, /index.ts, /index.tsx
    - Alias paths starting with '@/' (resolved against project roots)

    Returns None if the file doesn't exist on disk.
    """
    # Common extensions to try, in priority order
    extensions = [".ts", ".tsx", ".js", ".jsx"]
    index_variants = [
        os.path.join("index.ts"),
        os.path.join("index.tsx"),
        os.path.join("index.js"),
        os.path.join("index.jsx"),
    ]

    if import_path.startswith("."):
        # Relative import — resolve from importing file's directory
        base_dir = os.path.dirname(importing_file)
        candidate_base = os.path.normpath(os.path.join(base_dir, import_path))
    elif import_path.startswith("@/"):
        # Alias — try each project root + src/
        stripped = import_path[2:]  # Remove @/
        for root in project_roots:
            for src_dir in ["src", ""]:
                candidate_base = os.path.normpath(
                    os.path.join(root, src_dir, stripped) if src_dir
                    else os.path.join(root, stripped)
                )
                resolved = _try_resolve_ts_path(candidate_base, extensions, index_variants)
                if resolved:
                    return resolved
        return None
    else:
        # Bare module import (e.g., 'react', 'lodash') — skip, these are npm packages
        return None

    return _try_resolve_ts_path(candidate_base, extensions, index_variants)


def _try_resolve_ts_path(
    candidate_base: str,
    extensions: List[str],
    index_variants: List[str],
) -> Optional[str]:
    """Try resolving a TypeScript path with various extensions."""
    # Already has an extension?
    if os.path.isfile(candidate_base):
        return os.path.normpath(candidate_base)

    # Try adding extensions
    for ext in extensions:
        full = candidate_base + ext
        if os.path.isfile(full):
            return os.path.normpath(full)

    # Try as directory with index file
    if os.path.isdir(candidate_base):
        for idx in index_variants:
            full = os.path.join(candidate_base, idx)
            if os.path.isfile(full):
                return os.path.normpath(full)

    return None


# =============================================================================
# UTILITY: Get all defined names from a file (language-agnostic dispatch)
# =============================================================================


def get_all_defined_names(file_path: str) -> set:
    """
    Return the set of all top-level names defined in a file.

    Dispatches to Python AST or TypeScript regex based on file extension.
    Returns an empty set on error.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        defs = extract_python_definitions(file_path)
        names = set()
        names.update(defs.get("classes", []))
        names.update(defs.get("functions", []))
        names.update(defs.get("variables", []))
        return names

    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        defs = extract_typescript_exports(file_path)
        return set(defs.get("exports", []))

    return set()


def get_all_imports(file_path: str) -> List[Dict[str, Any]]:
    """
    Return all import statements from a file.

    Each import is: {"module": "...", "names": ["X", "Y"]}
    Dispatches based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        defs = extract_python_definitions(file_path)
        return defs.get("imports_from", [])

    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        defs = extract_typescript_exports(file_path)
        return defs.get("imports_from", [])

    return []


__all__ = [
    "extract_python_definitions",
    "extract_typescript_exports",
    "resolve_python_import",
    "resolve_typescript_import",
    "get_all_defined_names",
    "get_all_imports",
]
