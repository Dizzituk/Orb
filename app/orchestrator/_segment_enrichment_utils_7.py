from __future__ import annotations
import logging
from app.orchestrator._segment_enrichment_utils import SegmentEnrichment, _apply_llm_assignments, _build_per_segment_extractions, _load_experience_patterns
from app.orchestrator._segment_enrichment_utils import _build_symbol_map, _generate_implementation_intelligence, _save_enrichment
from app.orchestrator._segment_enrichment_utils import _pick_primary_source
from app.orchestrator.segment_enrichment import _assign_symbols_deterministic, _extract_all_symbols, logger
from typing import Any, Dict, List
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


BUILD_ID = "2026-02-18-v1.3-llm-assignment-conflict-guard"

async def enrich_segments(
    manifest: Any,
    source_evidence: Dict[str, str],
    job_dir_path: str,
    db: Any,
    project_id: int,
) -> Dict[str, Dict]:
    """
    Main entry point for Stage 4B.

    Runs all three layers sequentially and returns a dict of
    {segment_id: enrichment_dict} for every segment in the manifest.

    Args:
        manifest: SegmentManifest instance
        source_evidence: {relative_path: file_content} from _load_source_file_evidence
        job_dir_path: Path to the job directory on disk
        db: SQLAlchemy session for experience memory queries
        project_id: Project ID for experience memory queries

    Returns:
        {segment_id: enrichment_dict} — empty dict if enrichment fails entirely
    """
    if not source_evidence:
        logger.info("[SEGMENT_ENRICHMENT] No source evidence — skipping enrichment")
        return {}

    segments = manifest.segments
    if not segments or len(segments) < 2:
        logger.info("[SEGMENT_ENRICHMENT] < 2 segments — skipping enrichment")
        return {}

    # Identify the primary source file (the monolith being refactored).
    # In a typical refactor there's one large source file that all segments
    # extract from.  If multiple source files exist, process the largest.
    source_path, source_code = _pick_primary_source(source_evidence)
    if not source_code:
        logger.warning("[SEGMENT_ENRICHMENT] No parseable source code found")
        return {}

    # v1.1: Warn about segments with empty file_scope (segmentation quality issue).
    # These segments can't receive any symbol assignments because they own no files.
    _empty_scope_segs = [s.segment_id for s in segments if not s.file_scope]
    if _empty_scope_segs:
        logger.warning(
            "[SEGMENT_ENRICHMENT] v1.1 %d segment(s) have EMPTY file_scope "
            "(segmentation quality issue — cannot assign symbols): %s",
            len(_empty_scope_segs), _empty_scope_segs,
        )
        print(
            f"[SEGMENT_ENRICHMENT] ⚠️ {len(_empty_scope_segs)} segment(s) have "
            f"empty file_scope: {_empty_scope_segs}"
        )

    logger.info(
        "[SEGMENT_ENRICHMENT] Primary source: %s (%d chars)",
        source_path, len(source_code),
    )

    # =====================================================================
    # Layer 1: Deterministic AST extraction
    # =====================================================================
    try:
        all_symbols = _extract_all_symbols(source_code)
    except SyntaxError as e:
        logger.warning("[SEGMENT_ENRICHMENT] AST parse failed: %s", e)
        return {}

    logger.info(
        "[SEGMENT_ENRICHMENT] Layer 1: Extracted %d constants, %d functions, %d classes",
        len(all_symbols["constants"]),
        len(all_symbols["functions"]),
        len(all_symbols["classes"]),
    )

    # =====================================================================
    # Layer 1b: Deterministic assignment — match symbols to segments
    # =====================================================================
    # Step 1: Assign everything we can confidently match from the spec text
    assignments, unassigned = _assign_symbols_deterministic(
        segments, all_symbols, source_path,
    )

    logger.info(
        "[SEGMENT_ENRICHMENT] Layer 1b: %d symbols assigned deterministically, "
        "%d unassigned",
        sum(len(v) for v in assignments.values()),
        len(unassigned),
    )

    # =====================================================================
    # Layer 2: Cross-segment symbol map (from deterministic assignments)
    # =====================================================================
    per_segment_extractions = _build_per_segment_extractions(
        segments, assignments, all_symbols, source_code,
    )
    symbol_map = _build_symbol_map(segments, per_segment_extractions)

    logger.info(
        "[SEGMENT_ENRICHMENT] Layer 2: %d cross-segment bindings, %d unresolved",
        sum(
            len(syms)
            for seg_consumers in symbol_map["consumed_by"].values()
            for syms in seg_consumers.values()
        ),
        len(symbol_map["unresolved"]),
    )

    # =====================================================================
    # Layer 3: LLM intelligence pass (one call per job)
    # Resolves unassigned symbols AND generates ordering/guidance/risk
    # =====================================================================
    experience_patterns = _load_experience_patterns(db, manifest)

    llm_intelligence = await _generate_implementation_intelligence(
        manifest=manifest,
        symbol_map=symbol_map,
        extractions=per_segment_extractions,
        unassigned_symbols=unassigned,
        experience_patterns=experience_patterns,
        source_path=source_path,
    )

    # Apply LLM assignments for previously unassigned symbols
    if llm_intelligence and llm_intelligence.get("symbol_assignments"):
        _apply_llm_assignments(
            llm_intelligence["symbol_assignments"],
            assignments,
            per_segment_extractions,
            all_symbols,
            source_code,
            segments,
        )
        # Rebuild symbol map with newly assigned symbols
        symbol_map = _build_symbol_map(segments, per_segment_extractions)
        logger.info(
            "[SEGMENT_ENRICHMENT] Layer 3: LLM resolved %d additional symbol(s)",
            len(llm_intelligence.get("symbol_assignments", {})),
        )

    # =====================================================================
    # v1.4: POST-ASSIGNMENT DUPLICATE FUNCTION DETECTION
    # After both deterministic and LLM assignments, check for functions
    # that appear in more than one segment. This prevents the exact bug
    # from sg-8d29f79f where run_segmented_job was placed in both
    # seg-02 (_loop.py) AND seg-06 (_utils.py).
    # =====================================================================
    _seen_functions: Dict[str, List[str]] = {}  # func_name -> [seg_id, ...]
    for _seg_id, _seg_symbols in assignments.items():
        for _sym_name in _seg_symbols:
            # Only check functions (not constants/classes — those may legitimately
            # appear as re-exports or shared types)
            _is_func = any(f["name"] == _sym_name for f in all_symbols.get("functions", []))
            if _is_func:
                _seen_functions.setdefault(_sym_name, []).append(_seg_id)

    _duplicates = {k: v for k, v in _seen_functions.items() if len(v) > 1}
    if _duplicates:
        for _dup_name, _dup_segs in _duplicates.items():
            # Find the function's line count to assess severity
            _func_info = next(
                (f for f in all_symbols.get("functions", []) if f["name"] == _dup_name),
                None,
            )
            _line_count = 0
            if _func_info and "body" in _func_info:
                _line_count = _func_info["body"].count("\n") + 1

            _severity = "BLOCKING" if _line_count > 100 else "WARNING"
            logger.warning(
                "[SEGMENT_ENRICHMENT] v1.4 DUPLICATE FUNCTION %s: '%s' (%d lines) "
                "assigned to segments: %s. Only the first segment should own it; "
                "others should import it.",
                _severity, _dup_name, _line_count, _dup_segs,
            )
            print(
                f"[SEGMENT_ENRICHMENT] v1.4 {_severity}: '{_dup_name}' ({_line_count} lines) "
                f"duplicated across: {', '.join(_dup_segs)}"
            )

            # Auto-fix: keep the function in the FIRST segment only,
            # remove from all subsequent segments.
            _owner_seg = _dup_segs[0]
            for _remove_seg in _dup_segs[1:]:
                if _dup_name in assignments.get(_remove_seg, []):
                    assignments[_remove_seg].remove(_dup_name)
                    logger.info(
                        "[SEGMENT_ENRICHMENT] v1.4 Removed '%s' from %s (owner: %s)",
                        _dup_name, _remove_seg, _owner_seg,
                    )
                # Also remove from per_segment_extractions
                _rse = per_segment_extractions.get(_remove_seg, {})
                if "functions" in _rse:
                    _rse["functions"] = [
                        f for f in _rse["functions"] if f["name"] != _dup_name
                    ]

        # Rebuild symbol map after dedup
        symbol_map = _build_symbol_map(segments, per_segment_extractions)
        logger.info(
            "[SEGMENT_ENRICHMENT] v1.4 Dedup complete: %d duplicate(s) resolved",
            len(_duplicates),
        )

    # =====================================================================
    # Assemble enrichment bundles per segment
    # =====================================================================
    enrichment_data: Dict[str, Dict] = {}

    for seg in segments:
        seg_id = seg.segment_id
        seg_extract = per_segment_extractions.get(seg_id, {})
        seg_intelligence = {}
        if llm_intelligence and llm_intelligence.get("segments"):
            seg_intelligence = llm_intelligence["segments"].get(seg_id, {})

        enrichment = SegmentEnrichment(
            # Layer 1
            source_extract={
                s["name"]: s.get("body", s.get("value", ""))
                for category in ("constants", "functions", "classes")
                for s in seg_extract.get(category, [])
            },
            constants=seg_extract.get("constants", []),
            functions=seg_extract.get("functions", []),
            classes=seg_extract.get("classes", []),
            imports=seg_extract.get("imports", []),
            # Layer 2
            exports=list(symbol_map["exports"].get(seg_id, set())),
            consumes=symbol_map["consumes"].get(seg_id, {}),
            consumed_by=symbol_map["consumed_by"].get(seg_id, {}),
            unresolved=[
                u for u in symbol_map["unresolved"]
                if u.startswith(f"{seg_id} ")
            ],
            # Layer 3
            implementation_order=seg_intelligence.get("implementation_order", 0),
            design_guidance=seg_intelligence.get("design_guidance", ""),
            risk_level=seg_intelligence.get("risk_level", "low"),
            risk_notes=seg_intelligence.get("risk_notes", ""),
            # Metadata
            source_file=source_path,
            extraction_stats={
                "constants": len(seg_extract.get("constants", [])),
                "functions": len(seg_extract.get("functions", [])),
                "classes": len(seg_extract.get("classes", [])),
                "imports": len(seg_extract.get("imports", [])),
            },
        )

        enrichment_data[seg_id] = enrichment.to_dict()

    # Persist to disk
    _save_enrichment(enrichment_data, job_dir_path)

    logger.info(
        "[SEGMENT_ENRICHMENT] Complete: %d segment(s) enriched",
        len(enrichment_data),
    )
    return enrichment_data
