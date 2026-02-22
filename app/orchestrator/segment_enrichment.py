# FILE: app/orchestrator/segment_enrichment.py
"""
Stage 4B: Segment Enrichment

Runs ONCE per job after segmentation produces the manifest, BEFORE any
Critical Pipeline calls.  For each segment it produces an enrichment
bundle containing:

  Layer 1 — Deterministic Source Extraction (AST parsing, no LLM)
    Parse the monolith with Python ast and extract the exact code that
    belongs to each segment's scope.

  Layer 2 — Deterministic Cross-Segment Symbol Map (no LLM)
    Build export/import/binding maps across all segments and flag
    unresolved references.

  Layer 3 — LLM Intelligence Pass (one call per job)
    A single LLM call producing implementation ordering, per-segment
    design guidance, risk flags, and cross-segment integration notes.
    Also resolves any symbols that couldn't be deterministically assigned
    in Layer 1.

The enrichment bundle flows into build_segment_context() and from there
into the Critical Pipeline prompt so that architectures are designed from
hard evidence rather than prose descriptions.

Failure handling:  Enrichment is NON-FATAL.  If any layer fails the
pipeline continues exactly as it does today.

v1.0 (2026-02-16): Initial implementation
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from app.orchestrator._segment_enrichment_utils import ENRICHMENT_PROVIDER, _build_enrichment_user_prompt, _extract_names_from_import, _get_function_signature, _get_name, _is_constant_name, _pick_primary_source, load_enrichment
from app.orchestrator._segment_enrichment_utils import ENRICHMENT_MAX_TOKENS, ENRICHMENT_MODEL, ENRICHMENT_SYSTEM_PROMPT, ENRICHMENT_TIMEOUT, SegmentEnrichment, _apply_llm_assignments, _build_per_segment_extractions, _load_experience_patterns
from app.orchestrator._segment_enrichment_utils import _build_symbol_map, _generate_implementation_intelligence, _save_enrichment
from app.orchestrator._segment_enrichment_utils import BUILD_ID, enrich_segments

logger = logging.getLogger(__name__)
print(f"[SEGMENT_ENRICHMENT_LOADED] BUILD_ID={BUILD_ID}")

# =============================================================================
# ENV Configuration
# =============================================================================


# =============================================================================
# Data Models
# =============================================================================


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


# =============================================================================
# LAYER 1: AST EXTRACTION
# =============================================================================


def _extract_all_symbols(source_code: str) -> Dict[str, list]:
    """
    Parse the monolith with Python AST and extract every top-level symbol.

    Returns a dict with keys: constants, functions, classes, imports,
    module_level.  Each value is a list of dicts with symbol metadata.
    """
    tree = ast.parse(source_code)

    result = {
        "constants": [],
        "functions": [],
        "classes": [],
        "imports": [],
        "module_level": [],
    }

    for node in ast.iter_child_nodes(tree):

        # --- Constants: ALL_CAPS assignments ---
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _is_constant_name(target.id):
                    source_segment = ast.get_source_segment(source_code, node)
                    result["constants"].append({
                        "name": target.id,
                        "value": source_segment or "",
                        "line_number": node.lineno,
                    })

        # --- Functions ---
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_source = ast.get_source_segment(source_code, node)
            result["functions"].append({
                "name": node.name,
                "signature": _get_function_signature(node, source_code),
                "docstring": ast.get_docstring(node) or "",
                "body": func_source or "",
                "line_range": f"{node.lineno}-{node.end_lineno}",
                "is_async": isinstance(node, ast.AsyncFunctionDef),
                "decorators": [
                    ast.get_source_segment(source_code, d) or ""
                    for d in node.decorator_list
                ],
            })

        # --- Classes ---
        elif isinstance(node, ast.ClassDef):
            class_source = ast.get_source_segment(source_code, node)
            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(item.name)
            result["classes"].append({
                "name": node.name,
                "bases": [_get_name(b) for b in node.bases],
                "methods": methods,
                "body": class_source or "",
                "line_range": f"{node.lineno}-{node.end_lineno}",
            })

        # --- Import statements ---
        elif isinstance(node, ast.Import):
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment:
                result["imports"].append(source_segment)

        elif isinstance(node, ast.ImportFrom):
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment:
                result["imports"].append(source_segment)

        # --- Module-level expressions (print(), logger setup, if __name__) ---
        elif isinstance(node, ast.Expr):
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment:
                result["module_level"].append({
                    "code": source_segment,
                    "line_number": node.lineno,
                })

        elif isinstance(node, ast.If):
            # Capture if __name__ == "__main__" blocks
            source_segment = ast.get_source_segment(source_code, node)
            if source_segment and "__name__" in source_segment:
                result["module_level"].append({
                    "code": source_segment,
                    "line_number": node.lineno,
                })

    return result


# =============================================================================
# LAYER 1b: DETERMINISTIC SYMBOL ASSIGNMENT
# =============================================================================

def _assign_symbols_deterministic(
    segments: list,
    all_symbols: Dict[str, list],
    source_path: str,
) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]]]:
    """
    Assign extracted symbols to segments using deterministic heuristics.

    Strategy (Option C — deterministic first, LLM resolves ambiguity):

    1. CONSTANTS segment: If a segment's file_scope targets a file with
       "constant" in the name (or the segment title says "constants"),
       ALL constants go there.

    2. EXPLICIT NAME MATCH: Scan each segment's title, requirements text,
       and file_scope filename for explicit mentions of function/class names.
       If a symbol name appears in a segment's text, assign it there.

    3. FILENAME HEURISTIC: If a segment targets e.g. "sandbox_helpers.py"
       and a function is named "_sandbox_*" or "sandbox_*", assign it there.

    4. FACADE / __init__ segment: If a segment targets "__init__.py" or
       its title mentions "facade", it gets module-level code and the
       import-time BUILD_ID print statement.

    Anything not matched goes into the unassigned list for the LLM.

    Args:
        segments: List of SegmentSpec objects
        all_symbols: Output from _extract_all_symbols
        source_path: Path to the monolith (for logging)

    Returns:
        (assignments, unassigned) where:
        - assignments = {segment_id: [symbol_name, ...]}
        - unassigned = [{name, type, ...}, ...] for LLM resolution
    """
    assignments: Dict[str, List[str]] = {seg.segment_id: [] for seg in segments}
    assigned_names: Set[str] = set()

    # Build a lookup of what text is associated with each segment
    seg_text_map: Dict[str, str] = {}
    seg_filename_map: Dict[str, List[str]] = {}  # segment_id -> target filename stems
    constants_segment_id: Optional[str] = None
    facade_segment_id: Optional[str] = None

    for seg in segments:
        # Combine all text for matching
        text_parts = [seg.title.lower()]
        text_parts.extend(r.lower() for r in seg.requirements)
        text_parts.extend(ac.lower() for ac in seg.acceptance_criteria)
        seg_text_map[seg.segment_id] = " ".join(text_parts)

        # Extract the target filename stems (e.g. "constants", "sandbox_helpers")
        # v1.2: Collect ALL filename stems for multi-file segments
        seg_filename_map[seg.segment_id] = []
        for fpath in seg.file_scope:
            fname = os.path.basename(fpath).replace(".py", "").lower()
            seg_filename_map[seg.segment_id].append(fname)

            # Detect constants segment
            if "constant" in fname:
                constants_segment_id = seg.segment_id
            # Detect facade / __init__ segment
            if fname == "__init__" or "facade" in fname:
                facade_segment_id = seg.segment_id

        # Also detect from title
        title_lower = seg.title.lower()
        if "constant" in title_lower and not constants_segment_id:
            constants_segment_id = seg.segment_id
        if ("facade" in title_lower or "__init__" in title_lower) and not facade_segment_id:
            facade_segment_id = seg.segment_id

    # --- Pass 1: Constants → constants segment ---
    if constants_segment_id:
        for const in all_symbols["constants"]:
            name = const["name"]
            assignments[constants_segment_id].append(name)
            assigned_names.add(name)
        logger.info(
            "[SEGMENT_ENRICHMENT] Pass 1: %d constants → %s",
            len(all_symbols["constants"]), constants_segment_id,
        )

    # --- Pass 2: Explicit name matching in segment text ---
    for func in all_symbols["functions"]:
        if func["name"] in assigned_names:
            continue
        name_lower = func["name"].lower()
        for seg in segments:
            seg_text = seg_text_map[seg.segment_id]
            # Look for the function name explicitly mentioned in the segment's text
            if name_lower in seg_text or func["name"] in seg_text:
                assignments[seg.segment_id].append(func["name"])
                assigned_names.add(func["name"])
                break

    for cls in all_symbols["classes"]:
        if cls["name"] in assigned_names:
            continue
        name_lower = cls["name"].lower()
        for seg in segments:
            seg_text = seg_text_map[seg.segment_id]
            if name_lower in seg_text or cls["name"] in seg_text:
                assignments[seg.segment_id].append(cls["name"])
                assigned_names.add(cls["name"])
                break

    # --- Pass 3: Filename stem heuristic ---
    # If a function name contains the target filename stem, assign it there.
    # e.g. _resolve_path → path_resolution segment, _sandbox_write → sandbox_helpers
    # v1.2: Iterates ALL filename stems for multi-file segments, and adds
    # bidirectional matching (function words checked against stem too).
    for func in all_symbols["functions"]:
        if func["name"] in assigned_names:
            continue
        name_lower = func["name"].lower().lstrip("_")
        # v1.2: Split function name into words for bidirectional matching
        func_words = [w for w in func["name"].lower().lstrip("_").split("_") if len(w) > 2]
        _matched = False
        for seg_id, fname_stems in seg_filename_map.items():
            if _matched:
                break
            for fname_stem in fname_stems:
                if not fname_stem or fname_stem in ("__init__", "constants"):
                    continue
                # Check if the function name contains significant words from the
                # filename stem.  Split stem on underscores and check each word.
                stem_words = [w for w in fname_stem.split("_") if len(w) > 2]
                if not stem_words:
                    continue
                # Forward match: stem word appears in function name
                fwd_matches = sum(1 for w in stem_words if w in name_lower)
                # v1.2: Reverse match: function word appears in filename stem
                # (e.g. "update" in func matches "updates" in _state_updates)
                rev_matches = sum(
                    1 for fw in func_words
                    if any(fw in sw or sw in fw for sw in stem_words)
                )
                total_matches = max(fwd_matches, rev_matches)
                if total_matches >= 1 and len(stem_words) <= 3:
                    assignments[seg_id].append(func["name"])
                    assigned_names.add(func["name"])
                    _matched = True
                    break
                elif total_matches >= 2:
                    assignments[seg_id].append(func["name"])
                    assigned_names.add(func["name"])
                    _matched = True
                    break

    # --- Pass 4: Module-level code → facade/__init__ segment ---
    if facade_segment_id:
        for ml in all_symbols["module_level"]:
            # BUILD_ID print statements and if __name__ blocks go to facade
            code = ml.get("code", "")
            if "BUILD_ID" in code or "__name__" in code:
                assignments[facade_segment_id].append(f"__module_level_L{ml['line_number']}")
                assigned_names.add(f"__module_level_L{ml['line_number']}")

    # --- Collect unassigned symbols for LLM ---
    unassigned: List[Dict[str, Any]] = []

    for func in all_symbols["functions"]:
        if func["name"] not in assigned_names:
            unassigned.append({
                "name": func["name"],
                "type": "function",
                "signature": func["signature"],
                "docstring": func.get("docstring", "")[:200],
                "line_range": func.get("line_range", ""),
            })

    for cls in all_symbols["classes"]:
        if cls["name"] not in assigned_names:
            unassigned.append({
                "name": cls["name"],
                "type": "class",
                "bases": cls.get("bases", []),
                "methods": cls.get("methods", []),
                "line_range": cls.get("line_range", ""),
            })

    # Constants that weren't assigned (no constants segment found)
    for const in all_symbols["constants"]:
        if const["name"] not in assigned_names:
            unassigned.append({
                "name": const["name"],
                "type": "constant",
                "value_preview": const.get("value", "")[:100],
            })

    logger.info(
        "[SEGMENT_ENRICHMENT] Assignment summary: %s",
        {seg_id: len(names) for seg_id, names in assignments.items() if names},
    )

    return assignments, unassigned


# =============================================================================
# LAYER 1c: BUILD PER-SEGMENT EXTRACTIONS
# =============================================================================


# =============================================================================
# LAYER 2: CROSS-SEGMENT SYMBOL MAP
# =============================================================================


# =============================================================================
# LAYER 3: LLM INTELLIGENCE PASS
# =============================================================================


# =============================================================================
# EXPERIENCE MEMORY INTEGRATION
# =============================================================================


# =============================================================================
# PERSISTENCE
# =============================================================================
