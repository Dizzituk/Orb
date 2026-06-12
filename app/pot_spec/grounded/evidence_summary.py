# FILE: app/pot_spec/grounded/evidence_summary.py
# Purpose: v2.2: Signature-only evidence mode for cost-optimised prompts.
# Called-by: app.pot_spec.grounded._evidence_gathering_utils_8
# Depends-on: app.pot_spec.grounded._sbx_fs
# Last-renovated: 2026-06-11
"""
v2.2: Signature-only evidence mode for cost-optimised prompts.

When summary_mode=True, evidence packages include ONLY:
- File path, size, existence status
- Function/class signatures (from AST)
- Import statements
- NOT full file content

This reduces token count by ~70% for stages that don't need
full source code (SpecGate, Critique, Cohesion checks).

Called from: format_evidence_for_prompt() when summary_mode=True
"""
from __future__ import annotations

import ast
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_isdir, _sbx_read

logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.


# =========================================================================
# AST-based signature extraction
# =========================================================================

def extract_python_signatures(source: str) -> List[str]:
    """Extract class names, function signatures, and imports from Python source.

    Returns a list of human-readable signature lines, e.g.:
        'class MyClass(BaseClass):'
        'def my_func(self, arg1: str, arg2: int = 0) -> bool:'
        'from app.foo import bar'
    """
    signatures: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to regex for unparseable files
        return _extract_signatures_regex(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = ", ".join(_unparse_safe(b) for b in node.bases)
            sig = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
            signatures.append(sig)
            # Include public method signatures within the class
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not item.name.startswith("_") or item.name == "__init__":
                        prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                        args = _format_args(item.args)
                        ret = f" -> {_unparse_safe(item.returns)}" if item.returns else ""
                        signatures.append(f"  {prefix} {item.name}({args}){ret}:")

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                args = _format_args(node.args)
                ret = f" -> {_unparse_safe(node.returns)}" if node.returns else ""
                signatures.append(f"{prefix} {node.name}({args}){ret}:")

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = ", ".join(
                a.asname if a.asname else a.name for a in node.names[:6]
            )
            if len(node.names) > 6:
                names += f", ... (+{len(node.names) - 6})"
            signatures.append(f"from {module} import {names}")

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                signatures.append(f"import {name}")

    return signatures


def _format_args(args: ast.arguments) -> str:
    """Format function arguments into a readable string."""
    parts: List[str] = []
    # Regular args (skip 'self'/'cls')
    all_args = args.args
    defaults = args.defaults
    n_defaults = len(defaults)
    n_args = len(all_args)

    for i, arg in enumerate(all_args):
        if arg.arg in ("self", "cls"):
            parts.append(arg.arg)
            continue
        annotation = f": {_unparse_safe(arg.annotation)}" if arg.annotation else ""
        default_idx = i - (n_args - n_defaults)
        if default_idx >= 0 and default_idx < n_defaults:
            default = f" = {_unparse_safe(defaults[default_idx])}"
        else:
            default = ""
        parts.append(f"{arg.arg}{annotation}{default}")

    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for kwarg in args.kwonlyargs:
        annotation = f": {_unparse_safe(kwarg.annotation)}" if kwarg.annotation else ""
        parts.append(f"{kwarg.arg}{annotation}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")

    return ", ".join(parts)


def _unparse_safe(node) -> str:
    """Safely unparse an AST node to string."""
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def _extract_signatures_regex(source: str) -> List[str]:
    """Fallback regex extraction for unparseable files."""
    sigs: List[str] = []
    for line in source.splitlines()[:200]:
        stripped = line.strip()
        if re.match(r"^(async\s+)?def\s+\w+\(", stripped):
            sigs.append(stripped[:200])
        elif re.match(r"^class\s+\w+", stripped):
            sigs.append(stripped[:200])
        elif stripped.startswith("from ") and " import " in stripped:
            sigs.append(stripped[:200])
        elif stripped.startswith("import "):
            sigs.append(stripped[:200])
    return sigs


# =========================================================================
# Summary formatter
# =========================================================================

def format_evidence_summary(package) -> str:
    """Format evidence package in SUMMARY mode (signatures only, no full content).

    Produces ~70% fewer tokens than format_evidence_for_prompt() while
    retaining all structural information needed for SpecGate, Critique,
    and Cohesion checks.

    Args:
        package: EvidencePackage instance

    Returns:
        Formatted string with file paths, sizes, and signatures only.
    """
    lines = [
        "## Filesystem Evidence (Summary Mode — Signatures Only)",
        f"Timestamp: {package.ground_truth_timestamp}",
        f"Task Type: {package.task_type}",
        "",
        "### Target Files",
    ]

    if not package.target_files:
        lines.append("  (No files resolved)")
        return "\n".join(lines)

    for fe in package.target_files:
        # Status line (same as full mode)
        lines.append(fe.to_evidence_line())

        if not fe.exists or not fe.readable:
            continue

        # Extract signatures from content
        content = fe.full_content or fe.content_preview or ""
        if not content:
            continue

        path = fe.resolved_path or fe.original_reference
        ext = os.path.splitext(path)[1].lower()

        if ext == ".py":
            sigs = extract_python_signatures(content)
        else:
            # For non-Python, just show first 10 meaningful lines
            sigs = _extract_head_lines(content, max_lines=10)

        if sigs:
            lines.append(f"  Signatures ({len(sigs)}):")
            for sig in sigs[:30]:  # Cap at 30 signatures per file
                lines.append(f"    {sig}")
            if len(sigs) > 30:
                lines.append(f"    ... (+{len(sigs) - 30} more)")
        lines.append("")

    if package.validation_errors:
        lines.append("### Validation Errors")
        for err in package.validation_errors:
            lines.append(f"  - ⚠️ {err}")

    return "\n".join(lines)


def _extract_head_lines(content: str, max_lines: int = 10) -> List[str]:
    """Extract first N non-empty, non-comment lines from content."""
    result: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            result.append(stripped[:200])
            if len(result) >= max_lines:
                break
    return result


# =========================================================================
# Convenience: summarise a single file from disk
# =========================================================================

def summarise_file_signatures(path: str) -> Optional[str]:
    """Read a file and return its signatures as a formatted string.

    Useful for quick one-off summarisation without an EvidencePackage.
    Returns None if file cannot be read.
    """
    content = _sbx_read(path)
    if content is None:
        return None
    content = content[:50_000]  # 50KB cap

    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        sigs = extract_python_signatures(content)
    else:
        sigs = _extract_head_lines(content, max_lines=15)

    if not sigs:
        return None

    header = f"{path} [{len(content)} bytes] — {len(sigs)} signatures"
    body = "\n".join(f"  {s}" for s in sigs[:30])
    return f"{header}\n{body}"


__all__ = [
    "extract_python_signatures",
    "format_evidence_summary",
    "summarise_file_signatures",
]
