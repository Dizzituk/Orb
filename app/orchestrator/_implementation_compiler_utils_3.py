from __future__ import annotations
import logging
import os
import re
from app.orchestrator._implementation_compiler_utils_2 import _extract_import_names
from app.orchestrator.compiler_models import FileFunction, FileImport
from typing import Dict, List, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _parse_file_inventory(architecture_text: str) -> List[Dict[str, str]]:
    """
    Extract file inventory entries from architecture document.

    Returns list of {path, operation, description} dicts.
    """
    files: List[Dict[str, str]] = []
    in_inventory = False
    past_header = False

    for line in architecture_text.split("\n"):
        stripped = line.strip()

        # Detect section start
        if re.match(r'#{1,4}\s*.*[Ff]ile\s*[Ii]nventory', stripped):
            in_inventory = True
            past_header = False
            continue

        # Detect section end
        if in_inventory and (stripped.startswith('#') or stripped == '---'):
            if past_header:
                in_inventory = False
                continue

        if not in_inventory:
            continue

        if not stripped.startswith('|'):
            continue

        if re.match(r'\|[-\s|]+\|', stripped):
            past_header = True
            continue

        if 'File' in stripped and 'Purpose' in stripped:
            continue

        lower = stripped.lower()
        if '*(none' in lower or '_(none' in lower:
            continue

        match = re.search(r'\|\s*`([^`]+)`\s*\|\s*([^|]+)', stripped)
        if match:
            fpath = match.group(1).strip()
            desc = match.group(2).strip()
            if fpath and fpath.lower() != 'file':
                operation = "CREATE"
                desc_lower = desc.lower()
                if 'modify' in desc_lower or 'update' in desc_lower or 'edit' in desc_lower:
                    operation = "MODIFY"
                # v6.1 FIX 10: Recognise QUARANTINE as a facade file
                elif 'quarantine' in desc_lower:
                    operation = "FACADE"
                # v6.1 FIX 23c: Recognise facade/re-export __init__.py files
                elif ('facade' in desc_lower or 're-export' in desc_lower
                      or 'reexport' in desc_lower):
                    operation = "FACADE"
                files.append({
                    "path": fpath,
                    "operation": operation,
                    "description": desc,
                })

    return files

def _assign_functions_to_files(
    file_inventory: List[Dict[str, str]],
    source_extract: Dict[str, str],
    func_by_name: Dict[str, Dict],
    const_by_name: Dict[str, Dict],
    architecture_text: str = "",
) -> Dict[str, List[FileFunction]]:
    """
    Assign enrichment functions/constants to their target files.

    Strategy:
    0. Parse architecture file sections for explicit assignments (v6.1 FIX 23b)
    1. Match function names against file stems
    2. Remaining functions go to the largest non-init file
    """
    result: Dict[str, List[FileFunction]] = {
        entry["path"]: [] for entry in file_inventory
        # v6.1 FIX 10b: Exclude FACADE files from function assignment.
        # Facade files must only contain re-exports, not transplanted code.
        if entry.get("operation") != "FACADE"
    }

    assigned: Set[str] = set()
    file_stems: Dict[str, str] = {}

    for entry in file_inventory:
        if entry.get("operation") == "FACADE":
            continue
        fpath = entry["path"]
        stem = os.path.basename(fpath).replace(".py", "").lower().lstrip("_")
        file_stems[stem] = fpath

    # Build all available functions
    all_funcs: Dict[str, FileFunction] = {}

    for name, body in source_extract.items():
        func_meta = func_by_name.get(name, {})
        is_async = func_meta.get("is_async", False) if func_meta else False
        signature = func_meta.get("signature", "") if func_meta else ""
        docstring = func_meta.get("docstring", "") if func_meta else ""
        line_count = body.count("\n") + 1 if body else 0

        all_funcs[name] = FileFunction(
            name=name, kind="function", signature=signature,
            body=body, line_count=line_count,
            is_async=is_async, docstring=docstring,
        )

    for name, const_data in const_by_name.items():
        if name in all_funcs:
            continue
        value = const_data.get("value", const_data.get("code", const_data.get("body", "")))
        all_funcs[name] = FileFunction(
            name=name, kind="constant", signature=f"{name} = ...",
            body=value, line_count=value.count("\n") + 1 if value else 1,
        )

    # Pass 0 (v6.1 FIX 23b): Parse architecture for explicit assignments.
    # Architecture has "## File: `path`" sections with "#### `symbol_name`" headers.
    # This is the authoritative mapping from the deterministic pipeline.
    if architecture_text:
        import re as _re
        _current_file = ""
        for _line in architecture_text.split("\n"):
            # Match: ## File: `app/foo/bar.py`
            _file_match = _re.match(r'^## File:\s*`([^`]+)`', _line)
            if _file_match:
                _current_file = _file_match.group(1)
                continue
            # Match: #### `symbol_name` (function, ~6L) or (data_structure, ~468L)
            _sym_match = _re.match(r'^####\s*`(\w+)`', _line)
            if _sym_match and _current_file:
                _sym_name = _sym_match.group(1)
                if _sym_name in all_funcs and _sym_name not in assigned:
                    if _current_file in result:
                        result[_current_file].append(all_funcs[_sym_name])
                        assigned.add(_sym_name)
        if assigned:
            logger.info(
                "[IMPL_COMPILER] FIX 23b: Architecture assigned %d/%d symbols",
                len(assigned), len(all_funcs),
            )

    # Pass 1: File stem affinity matching (fallback for unassigned)
    for name, func in all_funcs.items():
        if name in assigned:
            continue
        name_lower = name.lower().lstrip("_")
        name_words = {w for w in name_lower.split("_") if len(w) > 2}

        best_match = ""
        best_score = 0

        for stem, fpath in file_stems.items():
            if not stem or stem == "__init__":
                continue
            stem_words = {w for w in stem.split("_") if len(w) > 2}
            if not stem_words:
                continue

            overlap = name_words & stem_words
            score = len(overlap)

            if stem in name_lower:
                score += 2
            elif name_lower in stem:
                score += 1

            if score > best_score:
                best_score = score
                best_match = fpath

        if best_match and best_score >= 1:
            result[best_match].append(func)
            assigned.add(name)

    # Pass 2: Log unassigned symbols but do NOT dump them.
    # v6.1 FIX 26b: Unassigned symbols belong to OTHER segments.
    # Dumping them into the largest file caused 146KB briefs and
    # LLM duplication failures. If the architecture didn't assign
    # a symbol to this segment, it's handled by another segment.
    unassigned_funcs = [
        (name, func) for name, func in all_funcs.items()
        if name not in assigned
    ]

    if unassigned_funcs:
        unassigned_names = [name for name, _ in unassigned_funcs]
        logger.info(
            "[IMPL_COMPILER] FIX 26b: %d symbols not in this segment's "
            "architecture (belong to other segments): %s",
            len(unassigned_names),
            ", ".join(sorted(unassigned_names)[:10]),
        )

    return result

def _resolve_imports_for_file(
    functions: List[FileFunction],
    enrichment_imports: List[str],
    file_path: str,
    file_inventory: List[Dict[str, str]],
    file_function_map: Dict[str, List[FileFunction]],
    sibling_exports: Dict[str, Dict[str, str]],
) -> List[FileImport]:
    """Determine what imports this file needs."""
    imports: List[FileImport] = []
    seen_statements: Set[str] = set()

    # Collect all names used in this file's function bodies
    used_names: Set[str] = set()
    for func in functions:
        if func.body:
            for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', func.body):
                used_names.add(match.group(1))

    defined_names = {f.name for f in functions}

    # Source 1: Enrichment imports (stdlib and third-party)
    for imp_line in enrichment_imports:
        imported_names = _extract_import_names(imp_line)
        if imported_names & used_names:
            if imp_line not in seen_statements:
                imports.append(FileImport(
                    statement=imp_line, source_segment="",
                    symbols=sorted(imported_names & used_names),
                ))
                seen_statements.add(imp_line)

    # Source 2: Cross-file imports within this segment
    for other_entry in file_inventory:
        other_path = other_entry["path"]
        if other_path == file_path:
            continue
        other_funcs = file_function_map.get(other_path, [])
        other_names = {f.name for f in other_funcs}
        needed = (used_names & other_names) - defined_names
        if needed:
            other_module = os.path.basename(other_path).replace(".py", "")
            imp_statement = f"from .{other_module} import {', '.join(sorted(needed))}"
            if imp_statement not in seen_statements:
                imports.append(FileImport(
                    statement=imp_statement, source_segment="",
                    symbols=sorted(needed),
                ))
                seen_statements.add(imp_statement)

    # Source 3: Cross-segment imports
    for sym_name, sym_info in sibling_exports.items():
        if sym_name in used_names and sym_name not in defined_names:
            already_covered = any(sym_name in imp.symbols for imp in imports)
            if not already_covered:
                imports.append(FileImport(
                    statement=f"# Cross-segment: {sym_name} from {sym_info['segment_id']}",
                    source_segment=sym_info["segment_id"],
                    symbols=[sym_name],
                ))

    return imports
