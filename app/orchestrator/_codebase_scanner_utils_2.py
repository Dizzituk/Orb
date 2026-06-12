# Purpose: codebase scanner utils 2
# Called-by: app.orchestrator._codebase_scanner_utils_3, app.orchestrator.codebase_scanner
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from __future__ import annotations
import ast
import hashlib
import re


CODEBASE_SCANNER_BUILD_ID = "2026-02-20-v1.0-enhanced-codebase-scanner"

_BUILTINS = frozenset(dir(__builtins__)) if isinstance(__builtins__, dict) else frozenset(dir(__builtins__))

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

def _is_internal(module: str) -> bool:
    """Check if a module is internal to the project."""
    if not module:
        return False
    return module.startswith("app.") or module.startswith("app/")

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
