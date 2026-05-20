from __future__ import annotations
import logging
from .file_verifier import verify_segment_files
from .segment_schemas import SegmentManifest, SegmentSpec
from app.pot_spec.grounded._segmentation_utils_2 import _generate_segment_id
from app.pot_spec.grounded._segmentation_utils_3 import _distribute_requirements, _infer_layer_dependencies, _load_architecture_file_list, _merge_small_segments, group_files_by_layer, needs_segmentation
from typing import Any, Dict, List, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


class ArchLayer:
    """Architectural layer names used for boundary detection."""
    BACKEND_SERVICES = "backend-services"
    BACKEND_API = "backend-api"
    BACKEND_PIPELINE = "backend-pipeline"
    BACKEND_CONFIG = "backend-config"
    FRONTEND_COMPONENTS = "frontend-components"
    FRONTEND_HOOKS = "frontend-hooks"
    FRONTEND_SERVICES = "frontend-services"
    FRONTEND_INTEGRATION = "frontend-integration"
    INFRASTRUCTURE = "infrastructure"
    TESTS = "tests"

def generate_segments(
    file_scope: List[str],
    requirements: List[str],
    acceptance_criteria: List[str],
    parent_spec_id: Optional[str] = None,
    parent_spec_hash: Optional[str] = None,
    concept_groups: Optional[List[Dict[str, Any]]] = None,
) -> Optional[SegmentManifest]:
    """
    Generate a segmented manifest from a job's file scope.
    
    This is the main segmentation entry point. It:
    1. Checks if segmentation is needed
    2. (v5.5) If concept_groups provided, uses concept-aware grouping (Phase 3B)
    3. Otherwise loads architecture map for path resolution (v1.1)
    4. Groups files by architectural layer (legacy fallback)
    5. Infers dependency ordering
    6. Merges small segments
    7. Generates SegmentSpec objects
    8. Runs file verification per segment (v1.1)
    9. Distributes requirements to segments (v1.1)
    10. Builds and validates the manifest
    
    Returns None if segmentation is not needed (job runs single-pass).
    Returns None if validation fails (graceful fallback to single-pass).
    
    Args:
        file_scope: All files the job creates or modifies
        requirements: Job requirements
        acceptance_criteria: Job acceptance criteria
        parent_spec_id: ID of the parent SPoT spec
        parent_spec_hash: Hash of the parent SPoT spec
        concept_groups: (v5.5 Phase 3B) Pre-computed concept segments from
                        smart_segmentation.py. List of dicts with keys:
                        title, files, concepts, depends_on.
    """
    from .segmentation import _build_manifest_from_concepts, _topological_sort, validate_manifest
    # Check if segmentation is needed (skipped if concept_groups already provided)
    if concept_groups is None:
        should_segment, reason = needs_segmentation(file_scope, requirements)
        if not should_segment:
            logger.info("[segmentation] Skipping segmentation: %s", reason)
            return None
        logger.info("[segmentation] Segmentation triggered (legacy): %s", reason)
    else:
        logger.info("[segmentation] Using concept-aware grouping (Phase 3B): %d groups",
                     len(concept_groups))

    # v1.1: Load architecture map for path resolution and boundary detection
    arch_paths = _load_architecture_file_list()

    # =====================================================================
    # v5.5 PHASE 3B: Concept-aware grouping (preferred path)
    # =====================================================================
    if concept_groups and len(concept_groups) >= 2:
        logger.info("[segmentation] v5.5 Building segments from concept groups")
        return _build_manifest_from_concepts(
            concept_groups=concept_groups,
            file_scope=file_scope,
            requirements=requirements,
            acceptance_criteria=acceptance_criteria,
            parent_spec_id=parent_spec_id,
            parent_spec_hash=parent_spec_hash,
            arch_paths=arch_paths,
        )

    # =====================================================================
    # Legacy path: layer-based grouping
    # =====================================================================
    # Group files by architectural layer
    layer_groups = group_files_by_layer(file_scope)
    layers_present = list(layer_groups.keys())

    logger.info(
        "[segmentation] File distribution: %s",
        {layer: len(files) for layer, files in layer_groups.items()},
    )

    # Infer dependency ordering
    layer_deps = _infer_layer_dependencies(layers_present)

    # Merge small segments
    layer_groups, layer_deps = _merge_small_segments(layer_groups, layer_deps)

    # Topological sort to determine segment order
    sorted_layers = _topological_sort(layer_groups.keys(), layer_deps)
    if sorted_layers is None:
        logger.error("[segmentation] Topological sort failed — cycle detected in layer dependencies")
        return None  # Fallback to single pass

    # Build a set of all files in scope for evidence_files calculation
    all_scope_files: Set[str] = set()
    for files in layer_groups.values():
        all_scope_files.update(files)

    # Generate segment specs
    segments: List[SegmentSpec] = []
    layer_to_segment_id: Dict[str, str] = {}

    for index, layer in enumerate(sorted_layers, start=1):
        seg_id = _generate_segment_id(index, layer)
        layer_to_segment_id[layer] = seg_id

        files = layer_groups[layer]
        dep_layers = layer_deps.get(layer, [])
        dep_segment_ids = [layer_to_segment_id[dl] for dl in dep_layers if dl in layer_to_segment_id]

        # v1.1: Compute evidence_files — files from dependency layers that
        # this segment needs to READ for context (boundary files)
        evidence_files: List[str] = []
        for dep_layer in dep_layers:
            if dep_layer in layer_groups:
                for dep_file in layer_groups[dep_layer]:
                    if dep_file not in files:  # Don't include own files
                        evidence_files.append(dep_file)

        # v1.1: Run file verification for this segment's files
        grounding = None
        try:
            grounding = verify_segment_files(
                file_scope=files,
                evidence_files=evidence_files,
                known_arch_paths=set(arch_paths) if arch_paths else None,
            )
            logger.info(
                "[segmentation] v1.1 File verification for %s: "
                "%d verified, %d stale, %d create_targets",
                seg_id,
                len(grounding.verified_files),
                len(grounding.stale_entries),
                len(grounding.create_targets),
            )
        except Exception as verify_err:
            logger.warning(
                "[segmentation] v1.1 File verification failed for %s (non-fatal): %s",
                seg_id, verify_err,
            )

        segment = SegmentSpec(
            segment_id=seg_id,
            title=f"{layer.replace('-', ' ').title()} — {len(files)} files",
            parent_spec_id=parent_spec_id,
            requirements=[],  # Populated by _distribute_requirements below
            file_scope=files,
            evidence_files=evidence_files,
            dependencies=dep_segment_ids,
            estimated_files=len(files),
            grounding_data=grounding,
        )

        # v1.2 (2026-04-12): Phase 1 Job 7b - attach contracts from group dict
        try:
            if isinstance(group, dict):
                _exp = _build_contract_from_dict(group.get("exposes"))
                _con = _build_contract_from_dict(group.get("consumes"))
                segment.exposes = _exp
                segment.consumes = _con
        except Exception as _contract_err:
            logger.debug("[segmentation] Job 7b contract attach failed: %s", _contract_err)

        # v1.1 (2026-04-12): Phase 1 Job 5c — per-segment target tagging.
        try:
            from app.pipeline_v2.target_registry import resolve_target_for_files
            _tid, _hits = resolve_target_for_files(files)
            segment.target_id = _tid
            if _tid is None and len(_hits) > 1:
                logger.warning(
                    "[segmentation] MIXED-TARGET segment %s spans %s — "
                    "target_id left None; downstream write_file will refuse "
                    "ambiguous writes. Segment splitter needed (Phase 1.5).",
                    seg_id, sorted(_hits),
                )
            elif _tid is None:
                logger.debug(
                    "[segmentation] segment %s has unresolvable paths "
                    "(no registered root matched): files=%s",
                    seg_id, files[:3],
                )
        except Exception as _tag_err:
            logger.debug("[segmentation] target tagging failed: %s", _tag_err)
        segments.append(segment)

    # v1.1: Distribute requirements to segments (heuristic, not trivial broadcast)
    requirement_map = _distribute_requirements(requirements, segments)

    # v1.2 (2026-04-18): Backfill target_id for segments left None by the
    # path-based resolver. See _manifest_builder._backfill_single_target_segments
    # for full rationale — same bug, same fix, applied here for the legacy
    # layer-based path.
    try:
        from ._manifest_builder import _backfill_single_target_segments
        _backfill_single_target_segments(segments)
    except Exception as _bf_err:
        logger.debug("[segmentation] v1.2 backfill skipped: %s", _bf_err)

    # Assign requirements to each segment based on the map
    seg_id_to_seg = {s.segment_id: s for s in segments}
    for req, seg_ids in requirement_map.items():
        for sid in seg_ids:
            if sid in seg_id_to_seg:
                seg_id_to_seg[sid].requirements.append(req)

    # Build manifest
    manifest = SegmentManifest(
        parent_spec_id=parent_spec_id,
        parent_spec_hash=parent_spec_hash,
        segments=segments,
        requirement_map=requirement_map,
    )

    # Validate
    valid, errors = validate_manifest(manifest)
    if not valid:
        logger.warning(
            "[segmentation] Manifest validation failed — falling back to single pass. Errors: %s",
            errors,
        )
        return None  # Graceful fallback

    logger.info("[segmentation] Manifest generated successfully: %s", manifest.summary())
    return manifest


def _build_contract_from_dict(contract_data) -> "object | None":
    """Build an InterfaceContract from the dict shape LLM produces in smart_segmentation.
    Returns None if the dict is empty/missing/malformed rather than an empty contract, so
    downstream (Job 7 verifier) can distinguish 'no declaration' from 'empty declaration'.
    v1.0 (2026-04-12): Phase 1 Job 7b.
    """
    if not isinstance(contract_data, dict):
        return None
    try:
        from app.pot_spec.grounded._segment_schemas_utils_1 import InterfaceContract
    except Exception:
        return None
    contract = InterfaceContract(
        class_names=list(contract_data.get("class_names", []) or []),
        method_signatures=list(contract_data.get("method_signatures", []) or []),
        endpoint_paths=list(contract_data.get("endpoint_paths", []) or []),
        export_names=list(contract_data.get("export_names", []) or []),
    )
    return None if contract.is_empty() else contract
