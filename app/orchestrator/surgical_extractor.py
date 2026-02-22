# FILE: app/orchestrator/surgical_extractor.py
"""
Surgical Extractor — Deterministic symbol extraction from monolith files.

Extracts symbols from large Python files WITHOUT using an LLM.
Works by:
  1. AST-scanning the source to map every symbol's exact line range
  2. Scoring each symbol by "extractability" (coupling, size, purity)
  3. Programmatically copying selected symbols to a new module file
  4. Programmatically deleting those lines from the monolith
  5. Adding import statements so the monolith still references the symbols

Zero LLM calls. Pure mechanical cut-paste-import.

BUILD_ID: 2026-02-21-v1.0-surgical-extractor
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SURGICAL_EXTRACTOR_BUILD_ID = "2026-02-21-v1.0-surgical-extractor"
print(f"[SURGICAL_EXTRACTOR_LOADED] BUILD_ID={SURGICAL_EXTRACTOR_BUILD_ID}")


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class SymbolLocation:
    """A symbol's exact position in the source file."""
    name: str
    kind: str  # "function", "async_function", "class", "constant", "assignment"
    line_start: int  # 1-indexed, inclusive
    line_end: int    # 1-indexed, inclusive
    char_count: int
    references: Set[str] = field(default_factory=set)      # symbols this one uses
    referenced_by: Set[str] = field(default_factory=set)    # symbols that use this one
    is_private: bool = False
    decorators_start: Optional[int] = None  # first decorator line if any


@dataclass
class ExtractionCandidate:
    """A scored symbol ready for extraction."""
    symbol: SymbolLocation
    coupling_score: float  # lower = easier to extract
    size_lines: int
    reason: str  # human-readable explanation


@dataclass
class ExtractionPlan:
    """What to extract in one pass."""
    source_file: str
    target_module: str  # new file path (relative)
    symbols: List[SymbolLocation]
    total_lines: int
    total_chars: int
    import_line: str  # e.g. "from .utils import func_a, func_b"


@dataclass
class ExtractionResult:
    """Result of one extraction pass."""
    success: bool
    source_file: str
    target_module: str
    symbols_extracted: List[str]
    lines_removed: int
    new_source_size: int
    error: str = ""


# =============================================================================
# PHASE 1: AST SCAN — MAP EVERY SYMBOL
# =============================================================================

def scan_symbols(source_code: str) -> List[SymbolLocation]:
    """
    Parse source and return every top-level symbol with exact line ranges.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        logger.error("[surgical] Failed to parse source: %s", e)
        return []

    source_lines = source_code.split("\n")
    symbols: List[SymbolLocation] = []
    all_names: Set[str] = set()

    # First pass: collect all top-level symbol names and positions
    for node in ast.iter_child_nodes(tree):
        loc = _node_to_location(node, source_lines)
        if loc:
            symbols.append(loc)
            all_names.add(loc.name)

    # Second pass: build reference graph
    _build_references(symbols, all_names, source_lines)

    return symbols


def _node_to_location(
    node: ast.AST, source_lines: List[str]
) -> Optional[SymbolLocation]:
    """Convert an AST node to a SymbolLocation."""

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno

        # Include decorators
        dec_start = None
        if node.decorator_list:
            dec_start = node.decorator_list[0].lineno
            line_start = dec_start

        body_text = "\n".join(source_lines[line_start - 1:line_end])
        return SymbolLocation(
            name=node.name,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            char_count=len(body_text),
            is_private=node.name.startswith("_"),
            decorators_start=dec_start,
        )

    elif isinstance(node, ast.ClassDef):
        line_start = node.lineno
        line_end = node.end_lineno or node.lineno
        if node.decorator_list:
            line_start = node.decorator_list[0].lineno
        body_text = "\n".join(source_lines[line_start - 1:line_end])
        return SymbolLocation(
            name=node.name,
            kind="class",
            line_start=line_start,
            line_end=line_end,
            char_count=len(body_text),
            is_private=node.name.startswith("_"),
        )

    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                line_start = node.lineno
                line_end = node.end_lineno or node.lineno
                body_text = "\n".join(source_lines[line_start - 1:line_end])
                return SymbolLocation(
                    name=target.id,
                    kind="constant" if target.id.isupper() else "assignment",
                    line_start=line_start,
                    line_end=line_end,
                    char_count=len(body_text),
                    is_private=target.id.startswith("_"),
                )
        return None

    return None


def _build_references(
    symbols: List[SymbolLocation],
    all_names: Set[str],
    source_lines: List[str],
) -> None:
    """Scan each symbol's body for references to other symbols in the file."""
    # Build word boundary pattern for all symbol names
    name_pattern = re.compile(
        r'\b(' + '|'.join(re.escape(n) for n in sorted(all_names, key=len, reverse=True)) + r')\b'
    )

    for sym in symbols:
        body = "\n".join(source_lines[sym.line_start - 1:sym.line_end])
        found = set(name_pattern.findall(body))
        found.discard(sym.name)  # don't count self-reference
        sym.references = found

    # Build reverse references
    for sym in symbols:
        for ref_name in sym.references:
            for other in symbols:
                if other.name == ref_name:
                    other.referenced_by.add(sym.name)


# =============================================================================
# PHASE 2: COUPLING SCORER — RANK EXTRACTABILITY
# =============================================================================

def score_extractability(symbols: List[SymbolLocation]) -> List[ExtractionCandidate]:
    """
    Score each symbol by how easy it is to extract.
    Lower coupling_score = easier to extract.

    Scoring factors:
    - referenced_by count: more dependents = harder (they'd need import changes)
    - references count: more deps = harder (might pull in a chain)
    - size: smaller = safer per pass
    - private: private symbols are internal, less external coupling
    """
    candidates: List[ExtractionCandidate] = []

    for sym in symbols:
        # Skip module-level infrastructure
        if sym.name in ('logger', 'log', '__all__') or sym.name.startswith('__'):
            continue
        # Skip tiny assignments (single-line constants like BUILD_ID)
        if sym.kind == "assignment" and sym.line_end - sym.line_start < 1:
            continue

        # Coupling score: weighted combination
        ref_by_penalty = len(sym.referenced_by) * 3.0  # each dependent is costly
        ref_to_penalty = len(sym.references) * 1.0      # each dependency is minor
        size_factor = sym.char_count / 10000.0           # larger = slightly harder
        private_bonus = -1.0 if sym.is_private else 0.0  # private = easier

        score = ref_by_penalty + ref_to_penalty + size_factor + private_bonus

        # Build reason
        parts = []
        if sym.referenced_by:
            parts.append(f"used by {len(sym.referenced_by)}")
        if sym.references:
            parts.append(f"uses {len(sym.references)}")
        parts.append(f"{sym.char_count} chars")
        reason = ", ".join(parts) if parts else "standalone"

        candidates.append(ExtractionCandidate(
            symbol=sym,
            coupling_score=score,
            size_lines=sym.line_end - sym.line_start + 1,
            reason=reason,
        ))

    # Sort: lowest coupling first (easiest to extract)
    candidates.sort(key=lambda c: c.coupling_score)
    return candidates


# =============================================================================
# PHASE 3: CLUSTER SELECTION — PICK WHAT TO EXTRACT
# =============================================================================

def select_extraction_cluster(
    candidates: List[ExtractionCandidate],
    max_lines: int = 400,
    max_chars: int = 35_000,
    max_symbols: int = 8,
) -> List[SymbolLocation]:
    """
    Pick the best cluster of symbols to extract in one pass.
    Takes the easiest candidates until hitting a size cap.
    """
    selected: List[SymbolLocation] = []
    total_lines = 0
    total_chars = 0

    for candidate in candidates:
        if len(selected) >= max_symbols:
            break
        sym = candidate.symbol
        new_lines = total_lines + (sym.line_end - sym.line_start + 1)
        new_chars = total_chars + sym.char_count

        if selected and (new_lines > max_lines or new_chars > max_chars):
            break

        selected.append(sym)
        total_lines = new_lines
        total_chars = new_chars

    return selected


# =============================================================================
# PHASE 4: SURGICAL EXTRACTION — CUT, PASTE, IMPORT
# =============================================================================

def build_extraction_plan(
    source_file: str,
    symbols: List[SymbolLocation],
    module_name: str = "",
) -> ExtractionPlan:
    """
    Build a plan for extracting symbols from the source file.
    """
    if not module_name:
        # Auto-name: _{source_stem}_utils
        source_stem = os.path.splitext(os.path.basename(source_file))[0]
        module_name = f"_{source_stem}_utils"

    # Build target path: same directory as source, new filename
    source_dir = os.path.dirname(source_file)
    target_path = os.path.join(source_dir, f"{module_name}.py")

    # Build import line using absolute package path
    # e.g. app/orchestrator/segment_loop.py extracts to
    #      app/orchestrator/_segment_loop_utils.py
    # Import: from app.orchestrator._segment_loop_utils import ...
    # Convert absolute path to Python package path relative to project root
    # D:\Orb\app\memory → app.memory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # D:\Orb
    rel_dir = os.path.relpath(source_dir, project_root)
    pkg_path = rel_dir.replace(os.sep, ".").replace("/", ".")
    symbol_names = sorted(s.name for s in symbols)
    import_line = f"from {pkg_path}.{module_name} import {', '.join(symbol_names)}"

    total_lines = sum(s.line_end - s.line_start + 1 for s in symbols)
    total_chars = sum(s.char_count for s in symbols)

    return ExtractionPlan(
        source_file=source_file,
        target_module=target_path,
        symbols=symbols,
        total_lines=total_lines,
        total_chars=total_chars,
        import_line=import_line,
    )


def execute_extraction(
    source_code: str,
    plan: ExtractionPlan,
) -> Tuple[str, str, ExtractionResult]:
    """
    Execute the extraction. Returns (new_source, new_module, result).

    Pure function — doesn't write to disk. Caller handles file I/O.
    """
    source_lines = source_code.split("\n")

    # Collect line ranges to remove (sorted by start, descending for safe removal)
    ranges_to_remove: List[Tuple[int, int]] = []
    for sym in plan.symbols:
        ranges_to_remove.append((sym.line_start, sym.line_end))
    ranges_to_remove.sort(key=lambda r: r[0], reverse=True)

    # Extract symbol bodies (before we modify anything)
    extracted_bodies: List[str] = []
    for sym in sorted(plan.symbols, key=lambda s: s.line_start):
        body = "\n".join(source_lines[sym.line_start - 1:sym.line_end])
        extracted_bodies.append(body)

    # Build the new module file
    new_module_parts = _build_new_module(source_code, plan, extracted_bodies)

    # Remove extracted lines from source (descending order to preserve indices)
    modified_lines = list(source_lines)
    lines_removed = 0
    for start, end in ranges_to_remove:
        # Also remove any blank lines immediately before the symbol (up to 2)
        actual_start = start
        for lookback in range(1, 3):
            check = start - 1 - lookback
            if check >= 0 and modified_lines[check].strip() == "":
                actual_start = check + 1  # 1-indexed
            else:
                break

        del modified_lines[actual_start - 1:end]
        lines_removed += (end - actual_start + 1)

    # Add import line near the top of the source (after existing imports)
    import_insert_idx = _find_import_insert_point(modified_lines)
    modified_lines.insert(import_insert_idx, plan.import_line)

    new_source = "\n".join(modified_lines)
    new_module = "\n".join(new_module_parts)

    result = ExtractionResult(
        success=True,
        source_file=plan.source_file,
        target_module=plan.target_module,
        symbols_extracted=[s.name for s in plan.symbols],
        lines_removed=lines_removed,
        new_source_size=len(new_source),
    )

    return new_source, new_module, result


def _build_new_module(
    source_code: str,
    plan: ExtractionPlan,
    extracted_bodies: List[str],
) -> List[str]:
    """Build the content of the new extracted module."""
    parts: List[str] = []

    # Always add future annotations — prevents NameError for type hints
    # referencing symbols that stayed in the parent module
    parts.append("from __future__ import annotations")

    # Collect imports needed by the extracted symbols
    needed_imports = _collect_needed_imports(source_code, plan.symbols, plan)
    if needed_imports:
        parts.extend(needed_imports)
        parts.append("")

    # Add each extracted symbol body
    for body in extracted_bodies:
        parts.append("")
        parts.append(body)

    # Add trailing newline
    parts.append("")

    return parts


def _collect_needed_imports(
    source_code: str,
    symbols: List[SymbolLocation],
    plan: Optional[ExtractionPlan] = None,
) -> List[str]:
    """
    Find which imports from the source file are needed by the extracted symbols.
    Also copies local type aliases and small assignments that are referenced.
    
    CIRCULAR IMPORT GUARD: If an import points back to the source file's own
    module, we skip it — the new _utils file cannot import from its parent
    at module level without creating a circular import.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    source_lines = source_code.split("\n")
    extracted_names = {s.name for s in symbols}

    # Build the source module path for circular import detection
    # e.g. D:\Orb\app\llm\weaver_stream.py → "app.llm.weaver_stream"
    source_module_path = ""
    if plan:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        rel = os.path.relpath(plan.source_file, project_root)
        source_module_path = rel.replace(os.sep, ".").replace("/", ".")[:-3]  # strip .py

    # Collect all names used in extracted symbol bodies
    used_names: Set[str] = set()
    for sym in symbols:
        body = "\n".join(source_lines[sym.line_start - 1:sym.line_end])
        try:
            body_tree = ast.parse(body)
            for node in ast.walk(body_tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)
                # Catch annotation references (e.g. param: ProgressCallback)
                elif isinstance(node, ast.arg) and node.annotation:
                    for ann_node in ast.walk(node.annotation):
                        if isinstance(ann_node, ast.Name):
                            used_names.add(ann_node.id)
        except SyntaxError:
            used_names.update(re.findall(r'\b[A-Za-z_]\w*\b', body))

    # Remove names we're extracting (they'll be in the same file)
    used_names -= extracted_names

    # Find source imports that provide these names
    needed: List[str] = []
    names_provided_by_imports: Set[str] = set()
    # Track names that come from the parent (circular) — these need lazy import
    circular_names: Set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                check_name = alias.asname or alias.name
                if check_name in used_names:
                    needed.append(ast.get_source_segment(source_code, node) or "")
                    names_provided_by_imports.add(check_name)
                    break

        elif isinstance(node, ast.ImportFrom):
            if not node.names:
                continue
            imported_names = {
                (a.asname or a.name) for a in node.names
            }
            matched = imported_names & used_names
            if matched:
                module = ("." * (node.level or 0)) + (node.module or "")
                
                # CIRCULAR IMPORT GUARD: skip imports from the source file itself
                if source_module_path and module == source_module_path:
                    logger.info(
                        "[surgical] CIRCULAR GUARD: skipping import of %s from parent %s",
                        matched, module
                    )
                    circular_names.update(matched)
                    names_provided_by_imports.update(matched)
                    continue
                
                names_str = ", ".join(sorted(matched))
                needed.append(f"from {module} import {names_str}")
                names_provided_by_imports.update(matched)

    # Find local assignments (type aliases, constants) that are referenced
    # but not provided by imports and not being extracted.
    # Walk into Try/If/With blocks (where module-level constants often live)
    # but NOT into FunctionDef/ClassDef bodies (those are local variables).
    still_needed = used_names - names_provided_by_imports - extracted_names
    if still_needed:
        def _walk_module_level(node):
            """Yield assignments from module-level and try/if/with blocks only."""
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue  # Don't descend into function/class bodies
                if isinstance(child, ast.Assign):
                    yield child
                elif isinstance(child, (ast.Try, ast.If, ast.With, ast.ExceptHandler)):
                    yield from _walk_module_level(child)

        for node in _walk_module_level(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if not isinstance(target, ast.Name):
                        continue
                    name = target.id
                    if name not in still_needed:
                        continue
                    # Copy the assignment, dedenting if inside a try/if/with block
                    raw_lines = source_lines[node.lineno - 1:(node.end_lineno or node.lineno)]
                    # Skip large assignments (>5 lines) — data structures that
                    # would cause circular imports if copied
                    if len(raw_lines) > 5:
                        logger.debug(
                            f"[surgical] Skipping large assignment {name} "
                            f"({len(raw_lines)} lines) — too big to copy"
                        )
                        continue
                    import textwrap
                    assign_text = textwrap.dedent("\n".join(raw_lines))
                    # Also resolve imports needed by this assignment's RHS
                    try:
                        rhs_tree = ast.parse(assign_text)
                        for rhs_node in ast.walk(rhs_tree):
                            if isinstance(rhs_node, ast.Name):
                                used_names.add(rhs_node.id)
                    except SyntaxError:
                        pass
                    needed.append(assign_text)
                    still_needed.discard(name)

        # Second import pass: pick up imports needed by copied assignments
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.names:
                imported_names = {(a.asname or a.name) for a in node.names}
                new_matched = (imported_names & used_names) - names_provided_by_imports
                if new_matched:
                    module = ("." * (node.level or 0)) + (node.module or "")
                    names_str = ", ".join(sorted(new_matched))
                    needed.append(f"from {module} import {names_str}")
                    names_provided_by_imports.update(new_matched)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    check_name = alias.asname or alias.name
                    if check_name in used_names and check_name not in names_provided_by_imports:
                        needed.append(ast.get_source_segment(source_code, node) or "")
                        names_provided_by_imports.add(check_name)

    # Always include common stdlib
    has_logging = any("logger" in "\n".join(source_lines[s.line_start-1:s.line_end]) for s in symbols)
    if has_logging and not any("import logging" in n for n in needed):
        needed.insert(0, "import logging")

    # Sort: stdlib imports → from imports → local assignments → logger setup
    stdlib_imports = [n for n in needed if n.startswith("import ")]
    from_imports = [n for n in needed if n.startswith("from ")]
    local_assigns = [n for n in needed if not n.startswith(("import ", "from "))]

    ordered: List[str] = []
    ordered.extend(sorted(set(stdlib_imports)))
    ordered.extend(sorted(set(from_imports)))
    # Logger setup right after imports
    if has_logging:
        ordered.append("logger = logging.getLogger(__name__)")
    ordered.extend(local_assigns)

    return ordered


def _find_import_insert_point(lines: List[str]) -> int:
    """Find the line index (0-based) where a new import should be inserted.
    
    Correctly handles multi-line imports like:
        from ..config import (
            FOO,
            BAR,
        )
    by tracking parenthesised blocks and never inserting inside them.
    """
    last_import_idx = 0
    in_paren = 0  # track nested parentheses depth

    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Track parentheses across multi-line imports
        open_count = stripped.count("(")
        close_count = stripped.count(")")
        
        if in_paren > 0:
            # Inside a multi-line import — keep scanning until closed
            in_paren += open_count - close_count
            if in_paren <= 0:
                in_paren = 0
                last_import_idx = i + 1  # import block ended here
            continue
        
        if stripped.startswith(("import ", "from ")):
            in_paren = open_count - close_count
            if in_paren < 0:
                in_paren = 0
            if in_paren == 0:
                last_import_idx = i + 1  # single-line import
            # else: multi-line import started, will continue tracking
        elif stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            if last_import_idx > 0:
                break

    return last_import_idx


# =============================================================================
# PUBLIC API — ONE-SHOT EXTRACTION
# =============================================================================

def analyse_file(file_path: str, source_code: str) -> List[ExtractionCandidate]:
    """
    Scan a file and return ranked extraction candidates.
    Use this to preview what the extractor would do.
    """
    symbols = scan_symbols(source_code)
    return score_extractability(symbols)


def extract_easiest(
    file_path: str,
    source_code: str,
    max_lines: int = 400,
    max_chars: int = 35_000,
    module_name: str = "",
) -> Tuple[Optional[str], Optional[str], Optional[ExtractionPlan], Optional[ExtractionResult]]:
    """
    Full pipeline: scan → score → select → extract.

    Returns (new_source, new_module_content, plan, result) or
    (None, None, None, None) if nothing extractable.
    """
    symbols = scan_symbols(source_code)
    if not symbols:
        logger.warning("[surgical] No symbols found in %s", file_path)
        return None, None, None, None

    candidates = score_extractability(symbols)
    if not candidates:
        logger.warning("[surgical] No extractable candidates in %s", file_path)
        return None, None, None, None

    cluster = select_extraction_cluster(candidates, max_lines, max_chars)
    if not cluster:
        logger.warning("[surgical] No viable cluster in %s", file_path)
        return None, None, None, None

    plan = build_extraction_plan(file_path, cluster, module_name)
    new_source, new_module, result = execute_extraction(source_code, plan)

    logger.info(
        "[surgical] Extracted %d symbols (%d lines, %d chars) from %s → %s",
        len(cluster), plan.total_lines, plan.total_chars,
        file_path, plan.target_module,
    )

    return new_source, new_module, plan, result
