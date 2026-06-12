# FILE: app/orchestrator/deterministic_architecture.py
# Purpose: Deterministic Architecture Generator.
# Called-by: app.orchestrator._deterministic_architecture_utils_2, app.orchestrator.refactor_pipeline
# Depends-on: app.orchestrator._deterministic_architecture_utils_2, app.orchestrator.codebase_scanner_models, app.orchestrator.refactor_segmenter_models
# Last-renovated: 2026-06-11
"""
Deterministic Architecture Generator.

Produces architecture documents from codebase scan data — zero LLM calls.
For refactor jobs, the scan already knows every file, every function,
every import, and every dependency. This module templates that data into
the same markdown format the Overwatcher and Implementer expect.

The LLM's job shrinks from "design the architecture" to "copy this code
into this file with these imports".

BUILD_ID: 2026-02-20-v1.0-deterministic-architecture-generator
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from app.orchestrator.codebase_scanner_models import (
    CodebaseGraph,
    FileScanResult,
    SymbolInfo,
    SymbolKind,
)
from app.orchestrator.refactor_segmenter_models import (
    DependencyGraph,
    FileNode,
    RefactorBuildPlan,
    SegmentPlan,
)
from app.orchestrator._deterministic_architecture_utils_2 import DETERMINISTIC_ARCH_BUILD_ID, _build_context, _build_header, _infer_purpose, _name_in_source, _sort_symbols, generate_all_architectures, save_architecture

logger = logging.getLogger(__name__)
print(f"[DETERMINISTIC_ARCH_LOADED] BUILD_ID={DETERMINISTIC_ARCH_BUILD_ID}")


# =============================================================================
# MAIN ENTRY
# =============================================================================

def generate_segment_architecture(
    segment: SegmentPlan,
    plan: RefactorBuildPlan,
    source_scan: FileScanResult,
    completed_tiers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Generate a complete architecture document for a segment.

    Args:
        segment: The segment plan to generate architecture for.
        plan: The full refactor build plan (for context).
        source_scan: The scan of the source monolith.
        completed_tiers: Optional dict of {file_path: actual_source_code}
                         from tiers that have already been built.

    Returns:
        Markdown architecture document string.
    """
    # Pre-compute the set of file paths in this segment for import guards
    segment_file_set = frozenset(segment.file_paths)
    sections: List[str] = []

    sections.append(_build_header(segment, plan))
    sections.append(_build_context(plan, source_scan))
    sections.append(_build_file_inventory(segment, plan))
    sections.append(_build_dependency_map(segment, plan))

    # Per-file design sections
    for file_path in segment.file_paths:
        node = plan.graph.nodes.get(file_path)
        if not node:
            continue
        sections.append(_build_file_design(
            node, source_scan, plan, completed_tiers,
            segment_file_set=segment_file_set,
        ))

    sections.append(_build_implementation_rules(plan))

    doc = "\n\n".join(sections)

    logger.info(
        "[deterministic_arch] Generated architecture for %s: %d chars, %d files",
        segment.segment_id, len(doc), len(segment.file_paths),
    )
    return doc


# =============================================================================
# SECTION BUILDERS
# =============================================================================


def _build_file_inventory(segment: SegmentPlan, plan: RefactorBuildPlan) -> str:
    """Build the File Inventory with New Files / Modified Files subsections.

    The architecture_executor parser expects:
      ### New Files  (table with | File | Purpose | columns)
      ### Modified Files  (table with | File | Purpose | columns)
    """
    new_rows: List[str] = []
    mod_rows: List[str] = []

    for file_path in segment.file_paths:
        node = plan.graph.nodes.get(file_path)
        if not node:
            continue

        purpose = node.description or _infer_purpose(node)
        row = f"| `{file_path}` | {purpose} |"

        # All files go to New Files — the source monolith gets fully
        # rewritten as a thin facade, not edited in place.
        new_rows.append(row)

    lines = ["## File Inventory", ""]

    lines.append("### New Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("| --- | --- |")
    if new_rows:
        lines.extend(new_rows)
    else:
        lines.append("| *(none)* | |")
    lines.append("")

    lines.append("### Modified Files")
    lines.append("")
    lines.append("| File | Purpose |")
    lines.append("| --- | --- |")
    if mod_rows:
        lines.extend(mod_rows)
    else:
        lines.append("| *(none)* | |")

    return "\n".join(lines)


def _build_dependency_map(segment: SegmentPlan, plan: RefactorBuildPlan) -> str:
    """Build the inter-file dependency map."""
    lines = [
        "## Dependency Map",
        "",
        "Import relationships between files in this segment:",
        "",
    ]

    for file_path in segment.file_paths:
        node = plan.graph.nodes.get(file_path)
        if not node:
            continue

        basename = os.path.basename(file_path)
        deps_in_segment = [
            os.path.basename(d) for d in node.depends_on
            if d in segment.file_paths
        ]
        deps_external = [
            os.path.basename(d) for d in node.depends_on
            if d not in segment.file_paths
        ]

        if deps_in_segment or deps_external:
            lines.append(f"- `{basename}`:")
            if deps_in_segment:
                lines.append(f"  - imports from (this segment): {', '.join(deps_in_segment)}")
            if deps_external:
                lines.append(f"  - imports from (earlier tiers): {', '.join(deps_external)}")
        else:
            lines.append(f"- `{basename}`: no internal dependencies (leaf)")

    return "\n".join(lines)


def _build_file_design(
    node: FileNode,
    source_scan: FileScanResult,
    plan: RefactorBuildPlan,
    completed_tiers: Optional[Dict[str, str]],
    segment_file_set: Optional[frozenset] = None,
) -> str:
    """
    Build the per-file design section.

    This is the key section — it tells the Implementer exactly what to put
    in each file with the actual source code to transplant.
    """
    basename = os.path.basename(node.file_path)
    lines = [
        f"## File: `{node.file_path}`",
        "",
    ]

    # Facade special case
    if node.is_facade:
        return _build_facade_design(node, plan, lines)

    # v6.1 FIX 2: Source monolith becomes a thin facade, not a code carrier.
    # If this file IS the source monolith, generate facade content instead
    # of transplanting symbols into it.
    if node.file_path == plan.source_file:
        return _build_source_facade_design(node, plan, lines)

    # Import block
    imports = _compute_imports(node, plan, source_scan, segment_file_set)
    if imports:
        lines.append("### Imports")
        lines.append("")
        lines.append("```python")
        for imp in imports:
            lines.append(imp)
        lines.append("```")
        lines.append("")

    # Symbols to transplant
    if node.symbols:
        lines.append("### Contents")
        lines.append("")
        lines.append(f"**DIRECTIVE: TRANSPLANT VERBATIM.** Copy each symbol exactly "
                      f"as provided. Do not rewrite, simplify, or modify logic.")
        lines.append("")

        for symbol in _sort_symbols(node.symbols):
            source_info = source_scan.symbols.get(symbol.name)
            source_code = ""
            if source_info:
                source_code = source_info.source_code

            lines.append(f"#### `{symbol.name}` ({symbol.kind.value}, ~{symbol.estimated_lines}L)")
            lines.append("")

            if source_code:
                lines.append("```python")
                lines.append(source_code)
                lines.append("```")
            else:
                lines.append(f"*Source code not available — implement based on signature: `{symbol.signature}`*")

            lines.append("")

    # Exports
    exports = [s.name for s in node.symbols if not s.is_private]
    if exports:
        lines.append("### Exports")
        lines.append("")
        lines.append(f"```python")
        lines.append(f"__all__ = {exports}")
        lines.append(f"```")

    return "\n".join(lines)


def _build_facade_design(
    node: FileNode,
    plan: RefactorBuildPlan,
    lines: List[str],
) -> str:
    """Build the design section for a facade __init__.py."""
    lines.append("### Purpose")
    lines.append("")
    lines.append("Package facade — re-exports all public symbols so that:")
    lines.append("```python")
    lines.append(f"from {plan.target_package.rstrip('/').replace('/', '.')} import run_build_validation_loop")
    lines.append("```")
    lines.append("continues to work after the refactor.")
    lines.append("")

    lines.append("### Re-exports")
    lines.append("")
    lines.append("```python")

    # Group public symbols by their file
    file_symbols: Dict[str, List[str]] = {}
    for fp, fnode in plan.graph.nodes.items():
        if fnode.is_facade:
            continue
        public = [s.name for s in fnode.symbols if not s.is_private]
        if public:
            module = os.path.splitext(os.path.basename(fp))[0]
            file_symbols[module] = public

    for module, syms in sorted(file_symbols.items()):
        lines.append(f"from .{module} import (")
        for sym in sorted(syms):
            lines.append(f"    {sym},")
        lines.append(")")
        lines.append("")

    # __all__
    all_public = sorted(plan.public_symbols)
    lines.append(f"__all__ = [")
    for sym in all_public:
        lines.append(f'    "{sym}",')
    lines.append("]")

    lines.append("```")

    return "\n".join(lines)


def _build_source_facade_design(
    node: FileNode,
    plan: RefactorBuildPlan,
    lines: List[str],
) -> str:
    """Build design for the source monolith — thin backward-compat facade.

    v6.1 FIX 2: After refactoring, the original file path must still work
    for any external code that imports from it. We replace the monolith
    with a thin file that re-exports everything from the new subpackage.
    """
    pkg_module = plan.target_package.rstrip('/').replace('/', '.')

    lines.append("### Purpose")
    lines.append("")
    lines.append("Backward-compatibility facade. Replaces the monolith with re-exports")
    lines.append("from the new subpackage so existing imports continue to work.")
    lines.append("")
    lines.append("### Contents")
    lines.append("")
    lines.append("**DIRECTIVE: REPLACE ENTIRE FILE with the following facade.**")
    lines.append("")
    lines.append("```python")
    lines.append(f'"""Backward-compatibility facade — imports from {pkg_module}."""')
    lines.append(f"from {pkg_module} import *  # noqa: F401,F403")

    # Also re-export explicitly for type checkers
    if plan.public_symbols:
        lines.append("")
        lines.append("__all__ = [")
        for sym in sorted(plan.public_symbols):
            lines.append(f'    "{sym}",')
        lines.append("]")

    lines.append("```")
    return "\n".join(lines)


def _compute_imports(
    node: FileNode,
    plan: RefactorBuildPlan,
    source_scan: FileScanResult,
    segment_file_set: Optional[frozenset] = None,
) -> List[str]:
    """
    Compute the exact import statements a file needs.

    Sources:
    1. stdlib/third-party imports from the source monolith that are
       used by any symbol in this file
    2. Relative imports from sibling files in the same package
    """
    imports: List[str] = []
    used_stdlibs: Set[str] = set()

    # Collect all names referenced by symbols in this file
    all_refs: Set[str] = set()
    for sym in node.symbols:
        src_sym = source_scan.symbols.get(sym.name)
        if src_sym:
            all_refs.update(src_sym.calls)
            all_refs.update(src_sym.references)

    # 1. Stdlib/third-party imports from source that are needed
    for imp in source_scan.imports:
        # Check if any imported name is used by our symbols
        needed_names = []
        for name in imp.names:
            if name in all_refs or _name_in_source(name, node, source_scan):
                needed_names.append(name)

        if needed_names:
            if imp.module:
                if imp.is_relative:
                    # v6.1 FIX 19: Adjust relative imports for subpackage nesting.
                    # Source monolith sits at e.g. app/translation/intents.py
                    # so `from .schemas import X` is relative to app/translation/.
                    # But the output file is at app/translation/intents/core.py
                    # so `.schemas` would resolve to app/translation/intents/schemas
                    # (wrong). We need `..schemas` to reach app/translation/schemas.
                    # Extract original dot level from raw_statement (e.g. "from ..foo" -> 2)
                    _after_from = imp.raw_statement.strip()
                    if _after_from.startswith("from"):
                        _after_from = _after_from[4:].lstrip()
                    _orig_dots = len(_after_from) - len(_after_from.lstrip("."))
                    _orig_dots = max(_orig_dots, 1)  # at least 1 dot for relative
                    _new_dots = "." * (_orig_dots + 1)
                    _import_stmt = f"from {_new_dots}{imp.module} import {', '.join(needed_names)}"
                    imports.append(_import_stmt)
                    logger.debug(
                        "[det_arch] FIX 19: Adjusted relative import: %s -> %s",
                        imp.raw_statement.strip(), _import_stmt,
                    )
                elif len(needed_names) == len(imp.names):
                    imports.append(imp.raw_statement)
                else:
                    imports.append(f"from {imp.module} import {', '.join(needed_names)}")
            used_stdlibs.add(imp.module)

    # 2. Relative imports from sibling files
    for dep_path in sorted(node.depends_on):
        # v6.1 FIX 1: Only import from files that exist in this segment
        # (or in earlier segments that have already been built)
        if segment_file_set and dep_path not in segment_file_set:
            # Check if it's in the full plan (cross-segment dep)
            # — these will be available at runtime but shouldn't be
            # generated as relative imports within this architecture
            continue

        dep_node = plan.graph.nodes.get(dep_path)
        if not dep_node:
            continue

        dep_module = os.path.splitext(os.path.basename(dep_path))[0]

        # Find which symbols we need from this dependency
        dep_symbol_names = dep_node.symbol_names
        needed = []
        for sym in node.symbols:
            src_sym = source_scan.symbols.get(sym.name)
            if not src_sym:
                continue
            for ref in list(src_sym.calls) + list(src_sym.references):
                if ref in dep_symbol_names:
                    needed.append(ref)

        needed = sorted(set(needed))
        if needed:
            if len(needed) <= 4:
                imports.append(f"from .{dep_module} import {', '.join(needed)}")
            else:
                import_lines = f"from .{dep_module} import (\n"
                for n in needed:
                    import_lines += f"    {n},\n"
                import_lines += ")"
                imports.append(import_lines)

    return imports


# =============================================================================
# IMPLEMENTATION RULES
# =============================================================================

def _build_implementation_rules(plan: RefactorBuildPlan) -> str:
    """Build the implementation rules section."""
    return """## Implementation Rules

1. **TRANSPLANT VERBATIM** — Copy function/class bodies exactly as provided. Do not rewrite, simplify, refactor, or "improve" any logic.
2. **Imports only** — The only new code you write is the import block at the top of each file. All imports are specified above.
3. **`__all__` required** — Every file must declare `__all__` listing its public exports.
4. **No stubs** — Every function must have its complete body. If source code is provided, use it. Never write `pass` or `...` as a placeholder.
5. **Preserve signatures** — Function signatures (parameters, types, defaults, return types) must match the source exactly.
6. **Preserve decorators** — If a function has decorators in the source, include them.
7. **Preserve docstrings** — Copy docstrings verbatim from the source.
8. **No extra code** — Do not add logging, comments, or helper functions that are not in the source.
9. **File encoding** — UTF-8, no BOM, Unix line endings."""


# =============================================================================
# FULL PLAN → ARCHITECTURES
# =============================================================================


# =============================================================================
# PERSISTENCE
# =============================================================================
