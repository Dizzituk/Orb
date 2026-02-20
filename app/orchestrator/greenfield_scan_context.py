# FILE: app/orchestrator/greenfield_scan_context.py
"""
Greenfield Scan Context — feed codebase scan data into LLM architecture.

For greenfield jobs where the LLM designs the architecture, this module
scans the existing codebase around the target area and produces a context
block showing real interfaces, exports, and imports. The LLM architect
gets grounded data instead of guessing what exists.

Also provides deterministic import computation: given a file's function
bodies and the scan data, compute exactly which imports are needed.

BUILD_ID: 2026-02-20-v1.0-greenfield-scan-context
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.orchestrator.codebase_scanner_models import (
    CodebaseGraph,
    FileScanResult,
    SymbolInfo,
    SymbolKind,
)

logger = logging.getLogger(__name__)

GREENFIELD_SCAN_BUILD_ID = "2026-02-20-v1.0-greenfield-scan-context"
print(f"[GREENFIELD_SCAN_LOADED] BUILD_ID={GREENFIELD_SCAN_BUILD_ID}")


# =============================================================================
# SCAN CONTEXT FOR LLM ARCHITECT
# =============================================================================

def build_scan_context(
    graph: CodebaseGraph,
    target_files: List[str],
    max_chars: int = 8000,
) -> str:
    """
    Build a context block for the LLM architect from scan data.

    Scans the files that the new code will interact with (neighbours in
    the import graph) and produces a structured summary of their interfaces.

    Args:
        graph: The codebase graph from scanning.
        target_files: Files the new code will create or modify.
        max_chars: Maximum size of the context block.

    Returns:
        Markdown context block to inject into the architecture prompt.
    """
    # Find neighbour files — files that the targets import from or that
    # import from the targets' parent directories
    neighbours = _find_neighbours(graph, target_files)

    if not neighbours:
        return ""

    sections: List[str] = []
    sections.append("## Existing Codebase Context (from scan)\n")
    sections.append(
        "> The following interfaces exist in the codebase. "
        "Use these exact names and signatures when importing.\n"
    )

    total_chars = sum(len(s) for s in sections)

    for fp in sorted(neighbours):
        scan = graph.files.get(fp)
        if not scan:
            continue

        file_section = _format_file_interface(scan)
        if total_chars + len(file_section) > max_chars:
            sections.append(f"\n*(truncated — {len(neighbours) - len(sections) + 2} more files)*")
            break

        sections.append(file_section)
        total_chars += len(file_section)

    return "\n".join(sections)


def _find_neighbours(
    graph: CodebaseGraph,
    target_files: List[str],
) -> List[str]:
    """
    Find files that are neighbours of the target files.

    A neighbour is:
    - A file in the same directory as a target
    - A file that a target imports from
    - A file in a parent directory's __init__.py
    """
    target_dirs: Set[str] = set()
    for fp in target_files:
        target_dirs.add(os.path.dirname(fp))

    neighbours: Set[str] = set()
    for fp in graph.files:
        if fp in target_files:
            continue  # Skip the targets themselves

        fp_dir = os.path.dirname(fp)
        # Same directory
        if fp_dir in target_dirs:
            neighbours.add(fp)
            continue

        # Parent directory
        for td in target_dirs:
            if td.startswith(fp_dir):
                neighbours.add(fp)
                break

    return sorted(neighbours)


def _format_file_interface(scan: FileScanResult) -> str:
    """Format a file's public interface for the context block."""
    lines = [f"\n### `{scan.file_path}`\n"]

    # Exports
    public_symbols = [
        (name, info) for name, info in scan.symbols.items()
        if not info.is_private and not info.is_dunder
    ]

    if not public_symbols:
        lines.append("*(no public exports)*\n")
        return "\n".join(lines)

    for name, info in sorted(public_symbols, key=lambda x: x[0]):
        if info.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION):
            sig = info.signature or f"def {name}(...):"
            # Truncate long signatures
            if len(sig) > 120:
                sig = sig[:117] + "..."
            lines.append(f"- `{sig}`")
        elif info.kind == SymbolKind.CLASS:
            bases = f"({', '.join(info.bases)})" if info.bases else ""
            methods_str = ", ".join(info.methods[:5])
            if len(info.methods) > 5:
                methods_str += f", ... (+{len(info.methods) - 5})"
            lines.append(f"- `class {name}{bases}` — methods: {methods_str}")
        elif info.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE):
            lines.append(f"- `{name}` (constant)")

    lines.append("")
    return "\n".join(lines)


# =============================================================================
# DETERMINISTIC IMPORT COMPUTATION
# =============================================================================

def compute_imports_for_file(
    file_source: str,
    graph: CodebaseGraph,
    file_path: str,
    target_package: str = "",
) -> List[str]:
    """
    Given a file's source code and the codebase graph, deterministically
    compute which imports the file needs.

    For each name referenced in the source, look up where it's defined
    in the codebase graph and generate the correct import statement.

    Args:
        file_source: The source code of the file being written.
        graph: The complete codebase graph.
        file_path: Path of the file being written.
        target_package: If within a package, used for relative imports.

    Returns:
        List of import statement strings.
    """
    # Collect all symbol names from the graph
    all_symbols: Dict[str, Tuple[str, SymbolInfo]] = {}
    for fp, scan in graph.files.items():
        for name, info in scan.symbols.items():
            all_symbols[name] = (fp, info)

    # Find which names are referenced in the file source
    needed: Dict[str, str] = {}  # name → source_file
    for name, (source_fp, info) in all_symbols.items():
        if source_fp == file_path:
            continue  # Don't import from self
        if len(name) < 2:
            continue
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, file_source):
            needed[name] = source_fp

    if not needed:
        return []

    # Group by source file
    by_file: Dict[str, List[str]] = {}
    for name, source_fp in needed.items():
        by_file.setdefault(source_fp, []).append(name)

    # Generate import statements
    imports: List[str] = []
    file_dir = os.path.dirname(file_path)

    for source_fp, names in sorted(by_file.items()):
        source_dir = os.path.dirname(source_fp)
        module = os.path.splitext(os.path.basename(source_fp))[0]
        sorted_names = sorted(set(names))

        if source_dir == file_dir:
            # Same directory — relative import
            if len(sorted_names) <= 4:
                imports.append(f"from .{module} import {', '.join(sorted_names)}")
            else:
                imp = f"from .{module} import (\n"
                for n in sorted_names:
                    imp += f"    {n},\n"
                imp += ")"
                imports.append(imp)
        else:
            # Different directory — absolute import
            abs_module = source_fp.replace("/", ".").replace("\\", ".").replace(".py", "")
            if len(sorted_names) <= 4:
                imports.append(f"from {abs_module} import {', '.join(sorted_names)}")
            else:
                imp = f"from {abs_module} import (\n"
                for n in sorted_names:
                    imp += f"    {n},\n"
                imp += ")"
                imports.append(imp)

    return imports


# =============================================================================
# INJECTION INTO SEGMENT CONTEXT
# =============================================================================

def inject_scan_context_into_segment(
    segment_context: Dict[str, Any],
    graph: Optional[CodebaseGraph],
) -> None:
    """
    Inject scan context into a segment's context dict for the LLM.

    Called from segment_loop for greenfield jobs when scan data is available.
    Adds a 'scan_context' key with the formatted interface summary.
    """
    if graph is None:
        return

    file_scope = segment_context.get("file_scope", [])
    if not file_scope:
        return

    context_text = build_scan_context(graph, file_scope)
    if context_text:
        segment_context["scan_context"] = context_text
        logger.info(
            "[greenfield_scan] Injected %d chars of scan context for %s",
            len(context_text), segment_context.get("segment_id", "?"),
        )
