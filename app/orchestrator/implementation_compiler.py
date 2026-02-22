# FILE: app/orchestrator/implementation_compiler.py
"""
Implementation Compiler — Unified per-file brief builder.

Sits between architecture generation and the Overwatcher/Implementer.
Takes all scattered evidence sources (enrichment, contracts, reconciliation,
source evidence, feedback) and compiles them into a single structured
per-file brief. The Implementer receives one focused document per file
instead of a patchwork of injections.

v1.0 (2026-02-20): Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from app.orchestrator.compiler_models import (
    CompilerProfile,
    CompilationResult,
    FileBrief,
    FileFunction,
    FileImport,
    build_instruction,
    detect_profile,
)

logger = logging.getLogger(__name__)

BUILD_ID = "2026-02-20-v1.0-implementation-compiler"
print(f"[IMPLEMENTATION_COMPILER_LOADED] BUILD_ID={BUILD_ID}")


# =============================================================================
# ARCHITECTURE PARSER
# =============================================================================


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


def _extract_file_design_notes(
    architecture_text: str,
    file_path: str,
) -> str:
    """
    Extract design notes for a specific file from the architecture document.

    Looks for subsections referencing this file path or stem.
    """
    file_stem = os.path.basename(file_path).replace(".py", "")
    lines = architecture_text.split("\n")
    capture = False
    captured: List[str] = []

    # v6.1 FIX 23c: Track the heading level that started capture.
    # Only stop when hitting a heading at the same or higher level.
    # Previously, ### subheadings (Purpose, Re-exports) inside a
    # ## File section would terminate capture immediately.
    capture_level = 0

    for line in lines:
        stripped = line.strip()

        _h_match = re.match(r'^(#{2,4})\s', stripped)
        if _h_match:
            _level = len(_h_match.group(1))
            if file_stem in stripped or file_path in stripped:
                capture = True
                capture_level = _level
                continue
            elif capture and _level <= capture_level:
                # Same or higher level heading = new section
                break
            # Deeper subheading within section — include it

        if capture:
            captured.append(line)

    raw = "\n".join(captured).strip()

    # v6.1 FIX 23d: Strip content that the compiler provides separately
    # to avoid duplicating function bodies in the brief.
    import re as _re
    # 1. Strip "### Contents" sections entirely
    raw = _re.sub(
        r'### Contents.*?(?=### [A-Z]|## [A-Z]|$)',
        '',
        raw,
        flags=_re.DOTALL,
    )
    # 2. Strip individual #### `symbol` blocks with their code fences.
    #    These appear under various subsection headers and duplicate
    #    the function bodies the compiler embeds via enrichment.
    raw = _re.sub(
        r'####\s*`\w+`\s*\([^)]*\).*?(?=####\s*`|### [A-Z]|## [A-Z]|$)',
        '',
        raw,
        flags=_re.DOTALL,
    ).strip()

    return raw


# =============================================================================
# MAIN COMPILER
# =============================================================================


def compile_implementation_briefs(
    architecture_text: str,
    enrichment: Optional[Dict[str, Any]],
    segment_id: str,
    source_file_evidence: Optional[Dict[str, str]] = None,
    interface_contract: str = "",
    sibling_interfaces: str = "",
    cohesion_feedback: str = "",
    implementation_feedback: str = "",
    import_validation_feedback: str = "",
    sibling_enrichments: Optional[Dict[str, Dict]] = None,
) -> CompilationResult:
    """
    Main entry point: compile all evidence into per-file briefs.

    Args:
        architecture_text: Generated architecture document
        enrichment: This segment's enrichment data from Stage 4B
        segment_id: Current segment ID
        source_file_evidence: {path: content} of source files
        interface_contract: Skeleton contract markdown
        sibling_interfaces: AST-extracted interfaces from completed siblings
        cohesion_feedback: Issues from cohesion checker (regen)
        implementation_feedback: Errors from previous implementation attempt
        import_validation_feedback: Import validation failures
        sibling_enrichments: {seg_id: enrichment_dict} for all sibling segments

    Returns:
        CompilationResult with per-file briefs
    """
    profile = detect_profile(architecture_text, enrichment, source_file_evidence)

    logger.info(
        "[IMPL_COMPILER] Compiling briefs for %s (profile=%s)",
        segment_id, profile.value,
    )

    # Step 1: Get file inventory from architecture
    file_inventory = _parse_file_inventory(architecture_text)
    if not file_inventory:
        logger.warning("[IMPL_COMPILER] No file inventory found in architecture")
        return CompilationResult(
            briefs=[],
            profile=profile,
            warnings=["No file inventory found in architecture document"],
        )

    # Step 2: Build enrichment lookups
    source_extract = {}
    enrichment_functions: Dict[str, Dict] = {}
    enrichment_constants: Dict[str, Dict] = {}
    enrichment_imports: List[str] = []

    if enrichment:
        source_extract = enrichment.get("source_extract", {})
        for f in enrichment.get("functions", []):
            if isinstance(f, dict):
                enrichment_functions[f.get("name", "")] = f
        for c in enrichment.get("constants", []):
            if isinstance(c, dict):
                enrichment_constants[c.get("name", "")] = c
        enrichment_imports = enrichment.get("imports", [])

    # Step 3: Build sibling export map
    sibling_exports: Dict[str, Dict[str, str]] = {}
    if sibling_enrichments:
        for sib_id, sib_data in sibling_enrichments.items():
            if sib_id == segment_id:
                continue
            for exp_sym in sib_data.get("exports", []):
                sibling_exports[exp_sym] = {
                    "segment_id": sib_id,
                    "source_file": sib_data.get("source_file", ""),
                }

    # Step 4: Assign functions to files
    file_function_map = _assign_functions_to_files(
        file_inventory=file_inventory,
        source_extract=source_extract,
        func_by_name=enrichment_functions,
        const_by_name=enrichment_constants,
        architecture_text=architecture_text,
    )

    # Step 5: Collect feedback
    all_feedback = _collect_feedback(
        cohesion_feedback, implementation_feedback, import_validation_feedback,
    )

    # Step 6: Build briefs
    briefs: List[FileBrief] = []
    total_functions = 0
    total_lines = 0
    warnings: List[str] = []

    for file_entry in file_inventory:
        fpath = file_entry["path"]
        operation = file_entry["operation"]

        # v6.1 FIX 10: FACADE files get a minimal re-export brief.
        # The architecture contains the exact facade code in its
        # design notes section — extract and use it directly.
        if operation == "FACADE":
            design_notes = _extract_file_design_notes(architecture_text, fpath)
            logger.info(
                "[IMPL_COMPILER] v6.1 Facade file: %s (design_notes=%d chars)",
                fpath, len(design_notes),
            )
            facade_instruction = (
                "**FACADE MODE — REPLACE ENTIRE FILE**\n\n"
                "This file is a backward-compatibility facade. It must contain "
                "ONLY re-exports from the new subpackage. Do NOT transplant "
                "any function bodies — just re-export.\n\n"
                "The exact facade code is specified in the Design Notes below. "
                "Write EXACTLY that code and nothing else."
            )
            brief = FileBrief(
                file_path=fpath,
                operation="CREATE",
                segment_id=segment_id,
                functions=[],
                imports=[],
                exports=[],
                consumed_by=[],
                consumes_from=[],
                instruction=facade_instruction,
                feedback=[],
                design_notes=design_notes,
                estimated_lines=10,
                profile="facade",
            )
            briefs.append(brief)
            total_lines += 10
            continue

        functions = file_function_map.get(fpath, [])
        total_functions += len(functions)

        file_imports = _resolve_imports_for_file(
            functions, enrichment_imports, fpath,
            file_inventory, file_function_map, sibling_exports,
        )

        file_exports = [f.name for f in functions]

        consumed_by = _build_consumed_by(
            fpath, file_exports, file_function_map, file_inventory,
        )
        consumes_from = _build_consumes_from(
            functions, fpath, file_function_map, file_inventory,
        )

        design_notes = _extract_file_design_notes(architecture_text, fpath)
        estimated_lines = sum(f.line_count for f in functions) + len(file_imports) + 10
        total_lines += estimated_lines
        instruction = build_instruction(profile, fpath, functions)
        file_feedback = _filter_feedback_for_file(all_feedback, fpath)

        brief = FileBrief(
            file_path=fpath,
            operation=operation,
            segment_id=segment_id,
            functions=functions,
            imports=file_imports,
            exports=file_exports,
            consumed_by=consumed_by,
            consumes_from=consumes_from,
            instruction=instruction,
            feedback=file_feedback,
            design_notes=design_notes,
            estimated_lines=estimated_lines,
            profile=profile.value,
        )
        briefs.append(brief)

    # Check for unassigned functions
    assigned_names: Set[str] = set()
    for funcs in file_function_map.values():
        for f in funcs:
            assigned_names.add(f.name)

    all_available = set(source_extract.keys()) | set(enrichment_functions.keys()) | set(enrichment_constants.keys())
    unassigned = all_available - assigned_names
    if unassigned:
        warnings.append(
            f"Unassigned symbols ({len(unassigned)}): {', '.join(sorted(unassigned)[:10])}"
        )

    result = CompilationResult(
        briefs=briefs, profile=profile,
        total_functions=total_functions,
        total_estimated_lines=total_lines,
        warnings=warnings,
    )

    logger.info(
        "[IMPL_COMPILER] Compiled %d brief(s), %d function(s), ~%d lines (profile=%s)",
        len(briefs), total_functions, total_lines, profile.value,
    )
    return result


# =============================================================================
# FUNCTION-TO-FILE ASSIGNMENT
# =============================================================================


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


# =============================================================================
# IMPORT RESOLUTION
# =============================================================================


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


def _extract_import_names(import_line: str) -> Set[str]:
    """Extract imported symbol names from an import statement."""
    names: Set[str] = set()
    line = import_line.strip()

    if line.startswith("from "):
        match = re.search(r'import\s+(.+)', line)
        if match:
            imports_part = match.group(1).strip("()")
            for part in imports_part.split(","):
                part = part.strip()
                if " as " in part:
                    names.add(part.split(" as ")[-1].strip())
                elif part and part != "*":
                    names.add(part.strip())
    elif line.startswith("import "):
        imports_part = line[7:]
        for part in imports_part.split(","):
            part = part.strip()
            if " as " in part:
                names.add(part.split(" as ")[-1].strip())
            elif part:
                names.add(part.split(".")[0].strip())

    return names


# =============================================================================
# CROSS-FILE DEPENDENCY MAPS
# =============================================================================


def _build_consumed_by(
    file_path: str,
    file_exports: List[str],
    file_function_map: Dict[str, List[FileFunction]],
    file_inventory: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Build map of which sibling files consume symbols from this file."""
    consumed_by: Dict[str, List[str]] = {}

    for entry in file_inventory:
        other_path = entry["path"]
        if other_path == file_path:
            continue
        other_funcs = file_function_map.get(other_path, [])
        for other_func in other_funcs:
            if not other_func.body:
                continue
            for export_name in file_exports:
                if export_name in other_func.body:
                    consumed_by.setdefault(other_path, [])
                    if export_name not in consumed_by[other_path]:
                        consumed_by[other_path].append(export_name)

    return consumed_by


def _build_consumes_from(
    functions: List[FileFunction],
    file_path: str,
    file_function_map: Dict[str, List[FileFunction]],
    file_inventory: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Build map of which sibling files this file consumes from."""
    consumes_from: Dict[str, List[str]] = {}
    our_names = {f.name for f in functions}

    used_names: Set[str] = set()
    for func in functions:
        if func.body:
            for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', func.body):
                used_names.add(match.group(1))

    for entry in file_inventory:
        other_path = entry["path"]
        if other_path == file_path:
            continue
        other_funcs = file_function_map.get(other_path, [])
        other_names = {f.name for f in other_funcs}
        needed = (used_names & other_names) - our_names
        if needed:
            consumes_from[other_path] = sorted(needed)

    return consumes_from


# =============================================================================
# FEEDBACK COLLECTION
# =============================================================================


def _collect_feedback(
    cohesion_feedback: str,
    implementation_feedback: str,
    import_validation_feedback: str,
) -> List[str]:
    """Collect all feedback into a flat list."""
    feedback: List[str] = []

    if cohesion_feedback:
        feedback.append(f"[COHESION] {cohesion_feedback.strip()}")

    if implementation_feedback:
        for line in implementation_feedback.strip().split("\n"):
            if line.strip():
                feedback.append(f"[IMPL] {line.strip()}")

    if import_validation_feedback:
        for line in import_validation_feedback.strip().split("\n"):
            if line.strip():
                feedback.append(f"[IMPORT] {line.strip()}")

    return feedback


def _filter_feedback_for_file(
    all_feedback: List[str],
    file_path: str,
) -> List[str]:
    """Filter feedback entries relevant to a specific file."""
    file_stem = os.path.basename(file_path).replace(".py", "")
    relevant: List[str] = []

    for fb in all_feedback:
        if file_stem in fb or file_path in fb:
            relevant.append(fb)
        elif not any(kw in fb for kw in [".py", "/"]):
            relevant.append(fb)

    return relevant


# =============================================================================
# PERSISTENCE
# =============================================================================


def save_compilation_result(
    result: CompilationResult,
    job_dir_path: str,
    segment_id: str,
) -> None:
    """Persist compilation result and individual briefs to disk."""
    seg_dir = os.path.join(job_dir_path, "segments", segment_id)
    compiler_dir = os.path.join(seg_dir, "compiler")
    os.makedirs(compiler_dir, exist_ok=True)

    summary_path = os.path.join(compiler_dir, "compilation_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("[IMPL_COMPILER] Summary saved: %s", summary_path)
    except Exception as e:
        logger.warning("[IMPL_COMPILER] Failed to save summary: %s", e)

    for brief in result.briefs:
        safe_name = os.path.basename(brief.file_path).replace(".py", "")
        brief_path = os.path.join(compiler_dir, f"brief_{safe_name}.md")
        try:
            with open(brief_path, "w", encoding="utf-8") as f:
                f.write(brief.to_markdown())
            logger.info("[IMPL_COMPILER] Brief saved: %s", brief_path)
        except Exception as e:
            logger.warning("[IMPL_COMPILER] Failed to save brief %s: %s", brief_path, e)
