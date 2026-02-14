# FILE: app/pot_spec/grounded/python_parser.py
"""
Size Analyzer — Python AST Structure Parser.

Parses Python source files into SourceBlock trees using the ast module.
Identifies functions, classes, nested definitions, and extractable loop
bodies for size estimation and decomposition.

v1.0 (2026-02-14): Extracted from size_analyzer.py for module cap compliance.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any, Dict, List, Optional

from .size_models import MAX_FILE_LINES, SourceBlock

logger = logging.getLogger(__name__)


def parse_python_structure(source_code: str) -> List[SourceBlock]:
    """
    Parse a Python file into its top-level and nested structure using AST.

    Returns list of SourceBlock objects representing functions, classes,
    and their nested definitions with line ranges.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        logger.warning("[python_parser] AST parse failed: %s", exc)
        return []

    blocks: List[SourceBlock] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            end_line = _get_end_line(node, source_code)
            block = SourceBlock(
                name=node.name,
                kind=kind,
                start_line=node.lineno,
                end_line=end_line,
                line_count=end_line - node.lineno + 1,
            )
            # Find nested function/class definitions
            block.nested_blocks = _extract_nested_blocks(node, source_code)

            # If function exceeds cap and closures alone won't fix it,
            # also detect large loop bodies as extraction candidates.
            extractable_closure_lines = sum(
                nb.line_count for nb in block.nested_blocks
                if nb.line_count >= 50
            )
            remaining_after_closures = block.line_count - extractable_closure_lines
            if remaining_after_closures > MAX_FILE_LINES:
                loops = _detect_extractable_loops(node, source_code)
                if loops:
                    block.nested_blocks.extend(loops)
                    logger.info(
                        "[python_parser] Function '%s' (%d lines): found %d "
                        "extractable loop(s) totalling %d lines",
                        block.name, block.line_count,
                        len(loops), sum(lb.line_count for lb in loops),
                    )

            blocks.append(block)

        elif isinstance(node, ast.ClassDef):
            end_line = _get_end_line(node, source_code)
            block = SourceBlock(
                name=node.name,
                kind="class",
                start_line=node.lineno,
                end_line=end_line,
                line_count=end_line - node.lineno + 1,
            )
            block.nested_blocks = _extract_nested_blocks(node, source_code)
            blocks.append(block)

    return blocks


def _get_end_line(node: ast.AST, source_code: str) -> int:
    """Get the end line of an AST node, with fallback."""
    if hasattr(node, "end_lineno") and node.end_lineno is not None:
        return node.end_lineno
    return source_code.count("\n") + 1


def _extract_nested_blocks(
    parent_node: ast.AST,
    source_code: str,
) -> List[SourceBlock]:
    """Extract nested function/class definitions within a parent node."""
    nested: List[SourceBlock] = []
    parent_name = getattr(parent_node, "name", "?")

    for child in ast.walk(parent_node):
        if child is parent_node:
            continue
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_line = _get_end_line(child, source_code)
            nested.append(SourceBlock(
                name=child.name,
                kind="closure",
                start_line=child.lineno,
                end_line=end_line,
                line_count=end_line - child.lineno + 1,
                parent=parent_name,
            ))
        elif isinstance(child, ast.ClassDef):
            end_line = _get_end_line(child, source_code)
            nested.append(SourceBlock(
                name=child.name,
                kind="class",
                start_line=child.lineno,
                end_line=end_line,
                line_count=end_line - child.lineno + 1,
                parent=parent_name,
            ))

    return nested


def _detect_extractable_loops(
    parent_node: ast.AST,
    source_code: str,
    min_lines: int = 80,
) -> List[SourceBlock]:
    """
    Detect large loop bodies within a function that could be extracted.

    When a monolithic function has no nested defs large enough to extract,
    large for/while loop bodies are the next best extraction candidates.
    Walks the function subtree but skips nested function/class defs.
    """
    loops: List[SourceBlock] = []
    parent_name = getattr(parent_node, "name", "?")
    src_lines = source_code.split("\n")

    _skip_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def _walk_for_loops(node: ast.AST) -> List:
        """Walk AST, yielding loops but not descending into nested defs."""
        found = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _skip_types):
                continue
            if isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                found.append(child)
            else:
                found.extend(_walk_for_loops(child))
        return found

    all_loops = _walk_for_loops(parent_node)

    seen_lines: set = set()
    for child in all_loops:
        if child.lineno in seen_lines:
            continue
        seen_lines.add(child.lineno)

        end_line = _get_end_line(child, source_code)
        line_count = end_line - child.lineno + 1

        if line_count < min_lines:
            continue

        name = _derive_loop_name(child, src_lines)

        loops.append(SourceBlock(
            name=name,
            kind="loop_body",
            start_line=child.lineno,
            end_line=end_line,
            line_count=line_count,
            parent=parent_name,
        ))

    return loops


def _derive_loop_name(node: ast.AST, src_lines: List[str]) -> str:
    """
    Derive a descriptive name for a loop body from context.

    Checks preceding comments, loop variable names, then falls back to
    line-number based naming.
    """
    line_idx = node.lineno - 1

    # Scan up to 5 lines before for a comment
    for lookback in range(1, 6):
        check_idx = line_idx - lookback
        if check_idx < 0:
            break
        prev = src_lines[check_idx].strip()
        if not prev:
            continue
        if prev.startswith("#"):
            comment = prev.lstrip("# ").strip("=-").strip()
            if len(comment) > 3:
                clean = re.sub(r'[^a-zA-Z0-9_ ]', '', comment).strip()
                clean = re.sub(r'\s+', '_', clean).lower()[:40]
                if clean:
                    return clean
        break

    # Check loop variable/iterable for hints
    if isinstance(node, (ast.For, ast.AsyncFor)):
        target = None
        if isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node.target, ast.Tuple) and node.target.elts:
            for elt in reversed(node.target.elts):
                if isinstance(elt, ast.Name) and elt.id not in ('i', 'j', 'k', '_'):
                    target = elt.id
                    break
        if target and target not in ('i', 'j', 'k', 'x', 'item', '_'):
            return f"process_{target}_loop"

    return f"loop_at_line_{node.lineno}"
