"""
Skeleton Contracts — Deterministic Interface Binding for Segments.

v5.6 of Pipeline Evolution.

Generates interface contracts DETERMINISTICALLY from the manifest alone.
Zero LLM calls. Runs between segmentation and architecture generation.

For each segment, the skeleton defines:
  - File scope constraint (ONLY these files may be touched)
  - Export contracts (files that downstream segments depend on)
  - Import contracts (files from upstream segments this segment needs)
  - Cross-segment bindings (the dependency graph edges)

The contract markdown is injected into each segment's Critical Pipeline
prompt as a hard constraint, preventing:
  - Scope creep (touching files outside the segment's scope)
  - Phantom segments (referencing segments that don't exist)
  - Interface drift (inventing alternative imports)

v1.0 (2026-02-12): Initial implementation — deterministic skeleton.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.orchestrator._skeleton_contracts_utils_4 import SKELETON_CONTRACTS_BUILD_ID, SegmentSkeleton, generate_skeleton_contract, load_skeleton_contract, save_skeleton_contract
from app.orchestrator._skeleton_contracts_utils_5 import ExportBinding, SkeletonContractSet

logger = logging.getLogger(__name__)
print(f"[SKELETON_CONTRACTS_LOADED] BUILD_ID={SKELETON_CONTRACTS_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================


# =============================================================================
# GENERATOR — Pure logic, no LLM calls
# =============================================================================


# =============================================================================
# POST-ENRICHMENT AUGMENTATION
# =============================================================================


def augment_skeleton_with_enrichment(
    contract_set: SkeletonContractSet,
    enrichment_data: Dict[str, Any],
    job_dir: Optional[str] = None,
) -> int:
    """
    v2.0: Wire enrichment-extracted symbol names into skeleton export bindings.

    The skeleton is generated BEFORE enrichment (deterministic, zero LLM calls).
    Enrichment runs AFTER (AST extraction + optional LLM resolution).
    This function bridges the gap: it reads the enrichment's `exports` and
    `functions` lists and populates each ExportBinding's `names` and
    `signatures` fields.

    This means the architecture generator prompt will now say:
        "_evidence.py must export: build_evidence_bundle, verify_contracts_fulfilled"
    instead of just:
        "_evidence.py is consumed by seg-05, seg-06, seg-07"

    Args:
        contract_set: The skeleton contract set to augment (modified in place).
        enrichment_data: Dict of {segment_id: enrichment_dict} from enrich_segments().
        job_dir: Optional job directory — if provided, re-saves the augmented skeleton.

    Returns:
        Number of export bindings that were augmented with names.
    """
    augmented_count = 0

    # v2.3: Build a cross-segment symbol ownership map.
    # For each function/class in every segment's enrichment, record which
    # segment's file_scope it canonically belongs to. This lets us detect
    # when a segment's "export" is actually a re-export from a sibling.
    # Key: symbol_name, Value: (owning_segment_id, canonical_file_path)
    _symbol_ownership: Dict[str, tuple] = {}
    for _map_skel in contract_set.skeletons:
        _map_enr = enrichment_data.get(_map_skel.segment_id)
        if not _map_enr:
            continue
        _map_funcs = _map_enr.get("functions", [])
        _map_classes = _map_enr.get("classes", [])
        # For each function/class, check if its name matches a file in this
        # segment's file_scope (the canonical home after refactor).
        # We also check enrichment-level source_file if available.
        for _sym in (_map_funcs + _map_classes):
            _sym_name = _sym.get("name", "")
            if not _sym_name:
                continue
            # If this symbol is already owned by another segment, the
            # first-registered owner wins (earlier segments take priority
            # since they're upstream).
            if _sym_name in _symbol_ownership:
                continue
            _symbol_ownership[_sym_name] = (
                _map_skel.segment_id,
                _map_skel.file_scope[0] if len(_map_skel.file_scope) == 1 else "",
            )
    # For multi-file segments, try to refine the canonical file using
    # the file-stem heuristic (e.g. "build_evidence_bundle" → "_evidence.py")
    for _map_skel in contract_set.skeletons:
        if len(_map_skel.file_scope) <= 1:
            continue
        _map_enr = enrichment_data.get(_map_skel.segment_id)
        if not _map_enr:
            continue
        for _sym in (_map_enr.get("functions", []) + _map_enr.get("classes", [])):
            _sym_name = _sym.get("name", "")
            if not _sym_name:
                continue
            if _symbol_ownership.get(_sym_name, ("",))[0] != _map_skel.segment_id:
                continue  # Only refine if we own this symbol
            # Try file-stem match
            _name_lower = _sym_name.lower()
            for _fp in _map_skel.file_scope:
                _stem = os.path.splitext(os.path.basename(_fp))[0].lstrip("_").lower()
                if _stem in _name_lower or _name_lower in _stem:
                    _symbol_ownership[_sym_name] = (_map_skel.segment_id, _fp)
                    break

    if _symbol_ownership:
        logger.info(
            "[skeleton_contracts] v2.3 Symbol ownership map: %d symbol(s) across %d segment(s)",
            len(_symbol_ownership),
            len(set(v[0] for v in _symbol_ownership.values())),
        )

    for skeleton in contract_set.skeletons:
        seg_id = skeleton.segment_id
        seg_enrichment = enrichment_data.get(seg_id)
        if not seg_enrichment:
            continue

        # Get the enrichment's exports list (symbol names) and functions list
        enriched_exports: List[str] = seg_enrichment.get("exports", [])
        enriched_functions: List[Dict[str, Any]] = seg_enrichment.get("functions", [])
        enriched_classes: List[Dict[str, Any]] = seg_enrichment.get("classes", [])
        enriched_constants: List[Dict[str, Any]] = seg_enrichment.get("constants", [])

        if not enriched_exports:
            continue

        # Build a signature lookup: name -> signature string
        sig_lookup: Dict[str, str] = {}
        for func in enriched_functions:
            fname = func.get("name", "")
            fsig = func.get("signature", "")
            if fname and fsig:
                sig_lookup[fname] = fsig
        for cls in enriched_classes:
            cname = cls.get("name", "")
            if cname:
                sig_lookup[cname] = f"class {cname}"

        # Now determine which exports belong to which file in this segment.
        # Enrichment gives us a flat list of export names for the segment,
        # and the skeleton has per-file ExportBindings.  For single-file
        # segments (most common), all exports go to that one file.  For
        # multi-file segments, we use the function's source_file or line_range
        # to match.

        # v2.1: Terminal segments (no downstream consumers) have zero
        # ExportBindings from generate_skeleton_contract().  But they still
        # need contract enforcement — especially for segments like the main
        # orchestration loop that define critical functions.  Create
        # self-referencing ExportBindings for each file in scope so the
        # contract injection system can enforce function signatures.
        if len(skeleton.exports) == 0 and enriched_exports:
            for fp in skeleton.file_scope:
                skeleton.exports.append(ExportBinding(
                    file_path=fp,
                    consumed_by=["__self__"],
                ))
            logger.info(
                "[skeleton_contracts] v2.1 Created %d self-referencing export(s) "
                "for terminal segment %s",
                len(skeleton.exports), seg_id,
            )

        if len(skeleton.exports) == 1:
            # Simple case: all exports belong to the single exported file
            binding = skeleton.exports[0]
            binding.names = enriched_exports[:]
            binding.signatures = [
                sig_lookup[name] for name in enriched_exports
                if name in sig_lookup
            ]
            # v2.3: Detect re-exports — symbols canonically owned by another segment
            binding.re_exports = []
            for _name in enriched_exports:
                _owner = _symbol_ownership.get(_name)
                if _owner and _owner[0] != seg_id and _owner[1]:
                    binding.re_exports.append([_name, _owner[1]])
                    logger.info(
                        "[skeleton_contracts] v2.3 %s/%s: '%s' is re-export from %s (%s)",
                        seg_id, binding.file_path, _name, _owner[0], _owner[1],
                    )
            augmented_count += 1
            logger.info(
                "[skeleton_contracts] v2.0 Augmented %s: %s with %d export name(s)"
                " (%d re-export(s))",
                seg_id, binding.file_path, len(binding.names), len(binding.re_exports),
            )
        elif len(skeleton.exports) > 1:
            # Multi-file segment: try to assign exports to specific files.
            # Use function body/line_range to guess which file each symbol
            # belongs to.  Enrichment functions have a 'line_range' field
            # and the skeleton has file_scope.  If we can't determine the
            # file, distribute evenly (better than nothing).
            #
            # Build file -> function-name mapping from enrichment functions.
            # Each function has a source context but no explicit file assignment
            # (they all come from the monolith).  The segment's file_scope tells
            # us which files this segment owns.  For refactor jobs, each file
            # typically handles one responsibility area.
            #
            # Heuristic: match function names to file names.
            # e.g. "build_evidence_bundle" -> "_evidence.py" (contains "evidence")
            # First pass: try strong matches (name contains file stem or vice versa)
            _assigned_names: set = set()
            for binding in skeleton.exports:
                _file_stem = os.path.splitext(os.path.basename(binding.file_path))[0]
                _file_stem_clean = _file_stem.lstrip("_").lower()
                matched_names = []
                matched_sigs = []
                for name in enriched_exports:
                    _name_lower = name.lower()
                    if _file_stem_clean in _name_lower or _name_lower in _file_stem_clean:
                        matched_names.append(name)
                        _assigned_names.add(name)
                        if name in sig_lookup:
                            matched_sigs.append(sig_lookup[name])

                if matched_names:
                    binding.names = matched_names
                    binding.signatures = matched_sigs
                    # v2.3: Detect re-exports for multi-file segments
                    binding.re_exports = []
                    for _name in matched_names:
                        _owner = _symbol_ownership.get(_name)
                        if _owner and _owner[0] != seg_id and _owner[1]:
                            binding.re_exports.append([_name, _owner[1]])
                    augmented_count += 1
                    logger.info(
                        "[skeleton_contracts] v2.0 Augmented %s: %s with %d export name(s)"
                        " (%d re-export(s))",
                        seg_id, binding.file_path, len(binding.names),
                        len(binding.re_exports),
                    )

            # Second pass: log unassigned exports but DO NOT blindly assign them.
            # v2.2 FIX: The previous logic dumped unmatched function names onto
            # whatever file binding happened to be empty or first. This caused
            # functions defined in one file (e.g. cohesion.py) to appear as
            # required exports of a different file (e.g. job_runner.py), which
            # then failed signature checking because the function was only
            # re-imported, not defined there.
            #
            # If a function name doesn't match any file stem, we simply skip it.
            # The function is still enforced on the file where it IS matched by
            # the first-pass heuristic or by the single-file fast path.
            _unassigned = [n for n in enriched_exports if n not in _assigned_names]
            if _unassigned:
                logger.info(
                    "[skeleton_contracts] v2.2 %d unassigned export(s) for %s "
                    "(skipped, not blindly assigned): %s",
                    len(_unassigned), seg_id, _unassigned,
                )

            # If heuristic didn't match anything, put all exports on the first binding
            # as a fallback (still better than empty)
            _any_matched = any(exp.names for exp in skeleton.exports)
            if not _any_matched and skeleton.exports:
                skeleton.exports[0].names = enriched_exports[:]
                skeleton.exports[0].signatures = [
                    sig_lookup[name] for name in enriched_exports
                    if name in sig_lookup
                ]
                augmented_count += 1
                logger.info(
                    "[skeleton_contracts] v2.0 Fallback augment %s: all %d exports on %s",
                    seg_id, len(enriched_exports), skeleton.exports[0].file_path,
                )

    # Re-save if job_dir provided
    if job_dir and augmented_count > 0:
        try:
            save_skeleton_contract(contract_set, job_dir)
            logger.info(
                "[skeleton_contracts] v2.0 Re-saved augmented skeleton: %d binding(s) enriched",
                augmented_count,
            )
        except Exception as e:
            logger.warning("[skeleton_contracts] v2.0 Failed to re-save augmented skeleton: %s", e)

    return augmented_count


# =============================================================================
# PERSISTENCE
# =============================================================================


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ExportBinding",
    "SegmentSkeleton",
    "SkeletonContractSet",
    "generate_skeleton_contract",
    "augment_skeleton_with_enrichment",
    "save_skeleton_contract",
    "load_skeleton_contract",
    "SKELETON_CONTRACTS_BUILD_ID",
]
