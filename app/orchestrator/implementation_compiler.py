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
from app.orchestrator._implementation_compiler_utils_2 import BUILD_ID, _build_consumed_by, _build_consumes_from, _collect_feedback, _extract_file_design_notes, _extract_import_names, _filter_feedback_for_file, save_compilation_result
from app.orchestrator._implementation_compiler_utils_3 import _assign_functions_to_files, _parse_file_inventory, _resolve_imports_for_file

logger = logging.getLogger(__name__)
print(f"[IMPLEMENTATION_COMPILER_LOADED] BUILD_ID={BUILD_ID}")


# =============================================================================
# ARCHITECTURE PARSER
# =============================================================================


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


# =============================================================================
# IMPORT RESOLUTION
# =============================================================================


# =============================================================================
# CROSS-FILE DEPENDENCY MAPS
# =============================================================================


# =============================================================================
# FEEDBACK COLLECTION
# =============================================================================


# =============================================================================
# PERSISTENCE
# =============================================================================
