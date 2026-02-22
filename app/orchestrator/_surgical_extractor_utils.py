import ast
import logging
import re
from typing import List, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


SURGICAL_EXTRACTOR_BUILD_ID = "2026-02-21-v1.0-surgical-extractor"

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

def _build_new_module(
    source_code: str,
    plan: ExtractionPlan,
    extracted_bodies: List[str],
) -> List[str]:
    """Build the content of the new extracted module."""
    parts: List[str] = []

    # Collect imports needed by the extracted symbols
    needed_imports = _collect_needed_imports(source_code, plan.symbols)
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
) -> List[str]:
    """
    Find which imports from the source file are needed by the extracted symbols.
    Also copies local type aliases and small assignments that are referenced.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    source_lines = source_code.split("\n")
    extracted_names = {s.name for s in symbols}

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
                    # Skip large assignments (>5 lines) — these are data structures
                    # that would cause circular imports if copied. The symbols that
                    # need them simply can't be extracted.
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
    """Find the line index (0-based) where a new import should be inserted."""
    last_import_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            last_import_idx = i + 1
        elif stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            if last_import_idx > 0:
                break

    return last_import_idx

def analyse_file(file_path: str, source_code: str) -> List[ExtractionCandidate]:
    """
    Scan a file and return ranked extraction candidates.
    Use this to preview what the extractor would do.
    """
    symbols = scan_symbols(source_code)
    return score_extractability(symbols)
