# FILE: app/pot_spec/grounded/_manifest_builder.py
# Purpose: v5.5 PHASE 3B: Build SegmentManifest from concept-aware groupings.
# Called-by: app.pot_spec.grounded._segmentation_utils_4, app.pot_spec.grounded._spec_runner_utils_12, app.pot_spec.grounded.segmentation
# Depends-on: app.pipeline_v2.target_registry, app.pot_spec.grounded._segment_schemas_utils_1, app.pot_spec.grounded._segmentation_utils_3, app.pot_spec.grounded.contract_verifier (+3 more)
# Last-renovated: 2026-06-11
"""
v5.5 PHASE 3B: Build SegmentManifest from concept-aware groupings.

Converts the output of smart_segmentation.generate_concept_segments() into
the same SegmentManifest format that the legacy layer-based path produces.

Includes:
- v1.2 FIX: Transitive dependency closure for same-package files
- v1.4 FIX: Refactor source files belong in integration segment only
- v1.5 FIX: Deterministic cycle breaking (facade detection, Kahn's)
- v1.6: Monolith replacement detection
- v3.0: Safety net deduplication
- v3.1: Reference-only file demotion + evidence-based dependency inference
"""
from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set

from .segment_schemas import SegmentManifest, SegmentSpec
from .file_verifier import verify_segment_files
from app.pot_spec.grounded._segmentation_utils_3 import _distribute_requirements

logger = logging.getLogger(__name__)


def _build_manifest_from_concepts(
    concept_groups: List[Dict[str, Any]],
    file_scope: List[str],
    requirements: List[str],
    acceptance_criteria: List[str],
    parent_spec_id: Optional[str],
    parent_spec_hash: Optional[str],
    arch_paths: List[str],
    *,
    _topological_sort_fn=None,
    _validate_manifest_fn=None,
) -> Optional[SegmentManifest]:
    """
    Build a SegmentManifest from concept-aware groupings (Phase 3B).

    The _topological_sort_fn and _validate_manifest_fn kwargs are injected
    by the parent module to avoid circular imports.
    """
    segments = _build_segment_specs(
        concept_groups, parent_spec_id, arch_paths,
    )

    # v1.2 (2026-04-18): Phase 1 Job 5d — backfill target_id for segments
    # whose file paths don't match any registered target signals. This
    # fixes the "seg-01 has no target_id — refusing to route" failure
    # mode observed for cross-cutting Android config files (build.gradle,
    # AndroidManifest.xml, MainActivity.kt) whose relative paths don't
    # include package-level discriminators like 'drivercopilot/'. When
    # the overall job has a single primary target, segments with ambiguous
    # paths should inherit it rather than being left stranded.
    _backfill_single_target_segments(segments)

    _resolve_dependencies(segments, concept_groups)
    _break_cycles(segments, _topological_sort_fn)
    _relocate_existing_files(segments)
    _demote_reference_only_files(segments, file_scope)
    _apply_transitive_deps(segments)
    _infer_evidence_dependencies(segments)
    _assign_requirements(segments, requirements)

    manifest = SegmentManifest(
        parent_spec_id=parent_spec_id,
        parent_spec_hash=parent_spec_hash,
        segments=segments,
        requirement_map=_distribute_requirements(requirements, segments),
    )

    # v1.1 (2026-04-12): Phase 1 Job 6 — classify cross-target dependencies.
    try:
        from app.pot_spec.grounded.segmentation import classify_dependencies
        classify_dependencies(manifest)
    except Exception as _cd_err:
        logger.debug("[segmentation] cross-target classification failed: %s", _cd_err)

    # v1.2 (2026-04-12): Phase 1 Job 7 — verify interface contracts across
    # segments. Diagnostic only at this stage; errors and warnings are logged
    # but do not fail the manifest build. Phase 3 verifier may decide to
    # escalate errors. Report is stashed on manifest for downstream consumers.
    try:
        from app.pot_spec.grounded.contract_verifier import verify_manifest_contracts
        contract_report = verify_manifest_contracts(manifest)
        manifest._contract_report = contract_report  # stash for Phase 3
        if contract_report.errors():
            for issue in contract_report.errors():
                logger.warning(
                    "[segmentation] Job 7 CONTRACT ERROR: %s %s[%s] (%s -> %s) — %s",
                    issue.kind.value, issue.category, issue.identifier,
                    issue.producer_segment or "?", issue.consumer_segment or "?",
                    issue.detail,
                )
    except Exception as _cv_err:
        logger.debug("[segmentation] contract verification failed: %s", _cv_err)

    if _validate_manifest_fn:
        valid, errors = _validate_manifest_fn(manifest)
        if not valid:
            logger.warning(
                "[segmentation] v5.5 Concept manifest validation failed: %s — "
                "falling back to legacy layer-based segmentation",
                errors,
            )
            return None

    logger.info("[segmentation] v5.5 Concept manifest generated: %s", manifest.summary())
    return manifest


# =============================================================================
# STEP 1: Build SegmentSpec objects from concept groups
# =============================================================================

def _build_segment_specs(
    concept_groups: List[Dict[str, Any]],
    parent_spec_id: Optional[str],
    arch_paths: List[str],
) -> List[SegmentSpec]:
    """Create SegmentSpec objects from concept group dicts."""
    segments: List[SegmentSpec] = []

    for idx, group in enumerate(concept_groups):
        title = group.get("title", f"Segment {idx + 1}")
        files = group.get("files", [])

        slug = title.lower().replace(" ", "-").replace("_", "-")
        slug = re.sub(r'[^a-z0-9\-]', '', slug)[:30]
        seg_id = f"seg-{idx + 1:02d}-{slug}" if slug else f"seg-{idx + 1:02d}"

        # Evidence files from dependency groups
        dep_indices = group.get("depends_on", [])
        evidence_files: List[str] = []
        for dep_idx in dep_indices:
            if 0 <= dep_idx < len(concept_groups):
                for dep_file in concept_groups[dep_idx].get("files", []):
                    if dep_file not in files:
                        evidence_files.append(dep_file)

        grounding = None
        try:
            grounding = verify_segment_files(
                file_scope=files,
                evidence_files=evidence_files,
                known_arch_paths=set(arch_paths) if arch_paths else None,
            )
        except Exception as verify_err:
            logger.warning("[segmentation] v5.5 File verification failed for %s: %s",
                           seg_id, verify_err)

        # v3.0: deduplicate file_scope
        files = list(dict.fromkeys(files))

        # v1.2 (2026-04-12): Phase 1 Job 7b — pass exposes/consumes through
        _exposes = _build_contract_from_dict(group.get("exposes"))
        _consumes = _build_contract_from_dict(group.get("consumes"))
        segment = SegmentSpec(
            segment_id=seg_id,
            title=f"{title} — {len(files)} file(s)",
            parent_spec_id=parent_spec_id,
            requirements=[],
            acceptance_criteria=[],
            file_scope=files,
            evidence_files=evidence_files,
            dependencies=[],
            estimated_files=len(files),
            grounding_data=grounding,
            exposes=_exposes,
            consumes=_consumes,
        )

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

    return segments


# =============================================================================
# STEP 1b: Backfill target_id for single-target jobs (v1.2 2026-04-18)
# =============================================================================

def _backfill_single_target_segments(segments: List[SegmentSpec]) -> None:
    """For single-target jobs, assign the job's primary target to any
    segment whose target_id resolution came back None.

    Why this exists: resolve_target_for_files() matches paths against
    each profile's registered signals (project_root prefix, folder name,
    path_signals, file extension). Generic Android config paths like
    'app/build.gradle.kts' or 'AndroidManifest.xml' or 'MainActivity.kt'
    contain none of the project-specific discriminators, so the resolver
    returns None even though the segment clearly belongs to the job's
    sole target. Downstream llm_tools.route_write then rejects writes
    to the segment with 'refusing to route', bricking the build.

    This pass reads the ambient job target hint (set by the spec_runner
    when the job starts — see target_registry.set_job_target_hint). If
    exactly one target is in scope, None target_ids are filled in.
    Multi-target jobs are left alone — ambiguity there is genuine and
    needs the segment splitter to handle it.

    Safe to call even when no job hint is set: it's a no-op.
    """
    try:
        from app.pipeline_v2.target_registry import get_job_target_hint
        hint = get_job_target_hint()
    except Exception as e:
        logger.debug("[segmentation] v1.2 backfill: could not read job hint: %s", e)
        return

    if not hint or len(hint) != 1:
        return

    primary = next(iter(hint))
    backfilled = 0
    for seg in segments:
        if seg.target_id is None:
            seg.target_id = primary
            backfilled += 1
            logger.info(
                "[segmentation] v1.2 Backfilled target_id=%s on %s "
                "(resolver returned None, single-target job hint available)",
                primary, seg.segment_id,
            )

    if backfilled:
        logger.info(
            "[segmentation] v1.2 Backfilled %d segment(s) with job primary target=%s",
            backfilled, primary,
        )


# =============================================================================
# STEP 2: Resolve dependency indices → segment IDs
# =============================================================================

def _resolve_dependencies(
    segments: List[SegmentSpec],
    concept_groups: List[Dict[str, Any]],
) -> None:
    """Map dependency indices from concept groups to segment IDs."""
    index_to_seg_id = {idx: seg.segment_id for idx, seg in enumerate(segments)}

    for idx, group in enumerate(concept_groups):
        dep_indices = group.get("depends_on", [])
        dep_ids = [index_to_seg_id[d] for d in dep_indices
                    if d in index_to_seg_id and d != idx]
        segments[idx].dependencies = dep_ids


# =============================================================================
# STEP 3: Deterministic cycle breaking (v1.5/v1.6)
# =============================================================================

def _break_cycles(segments: List[SegmentSpec], topological_sort_fn) -> None:
    """Detect and break cycles in the dependency graph."""
    if len(segments) < 2:
        return

    facade_ids = _detect_facade_segments(segments)

    if facade_ids:
        _remove_facade_back_edges(segments, facade_ids)
        _ensure_facade_terminal(segments, facade_ids)

    _general_cycle_breaker(segments, topological_sort_fn)


def _detect_facade_segments(segments: List[SegmentSpec]) -> set:
    """Identify facade/init segments that should be terminal."""
    facade_ids: set = set()

    all_files_normalised = []
    for seg in segments:
        for f in seg.file_scope:
            all_files_normalised.append(f.replace("\\", "/"))

    subpackage_dirs = set()
    for fnorm in all_files_normalised:
        parts = fnorm.rsplit("/", 1)
        if len(parts) == 2:
            subpackage_dirs.add(parts[0])

    for seg in segments:
        has_init = any(
            f.replace("\\", "/").endswith("__init__.py")
            for f in seg.file_scope
        )

        # v1.6: Detect original monolith being replaced by subpackage
        has_monolith_being_replaced = False
        for f in seg.file_scope:
            fn = f.replace("\\", "/")
            if fn.endswith(".py") and not fn.endswith("__init__.py"):
                stem = fn[:-3]
                if stem in subpackage_dirs:
                    has_monolith_being_replaced = True
                    logger.info(
                        "[segmentation] v1.6 Monolith replacement detected: %s → %s/ subpackage",
                        fn, stem,
                    )
                    break

        title_lower = (seg.title or "").lower()
        title_has_facade_keyword = any(
            kw in title_lower
            for kw in ["facade", "façade", "init", "package init", "re-export",
                        "package initialization", "package entry",
                        "stream", "integration"]
        )
        is_facade = (
            (has_init and title_has_facade_keyword)
            or has_monolith_being_replaced
        )
        if is_facade:
            facade_ids.add(seg.segment_id)
            logger.info(
                "[segmentation] v1.6 Facade segment detected: %s (init=%s, monolith_replace=%s)",
                seg.segment_id, has_init, has_monolith_being_replaced,
            )

    return facade_ids


def _remove_facade_back_edges(segments: List[SegmentSpec], facade_ids: set) -> None:
    """Remove edges where non-facade segments depend on facade segments."""
    for seg in segments:
        if seg.segment_id in facade_ids:
            continue
        before = len(seg.dependencies)
        seg.dependencies = [d for d in seg.dependencies if d not in facade_ids]
        if len(seg.dependencies) < before:
            logger.info(
                "[segmentation] v1.5 Removed facade dependency from %s",
                seg.segment_id,
            )


def _ensure_facade_terminal(segments: List[SegmentSpec], facade_ids: set) -> None:
    """Ensure facade segments depend on ALL non-facade segments."""
    non_facade_ids = [s.segment_id for s in segments if s.segment_id not in facade_ids]
    for seg in segments:
        if seg.segment_id in facade_ids:
            existing = set(seg.dependencies)
            for nf_id in non_facade_ids:
                if nf_id not in existing:
                    seg.dependencies.append(nf_id)
            seg.dependencies = sorted(set(seg.dependencies))
            logger.info(
                "[segmentation] v1.5 Facade %s now depends on %d segments (terminal)",
                seg.segment_id, len(seg.dependencies),
            )


def _general_cycle_breaker(segments: List[SegmentSpec], topological_sort_fn) -> None:
    """General cycle breaker using Kahn's algorithm."""
    seg_by_id = {s.segment_id: s for s in segments}
    max_rounds = len(segments)

    for cycle_round in range(max_rounds):
        edges = {s.segment_id: list(s.dependencies) for s in segments}
        if topological_sort_fn:
            sorted_result = topological_sort_fn(set(edges.keys()), edges)
        else:
            sorted_result = True  # Skip if no sort function
        if sorted_result is not None:
            break

        # Find nodes in cycle
        in_deg = {sid: len(edges.get(sid, [])) for sid in edges}
        succs = {sid: [] for sid in edges}
        for sid, deps in edges.items():
            for d in deps:
                if d in succs:
                    succs[d].append(sid)
        queue = [sid for sid, deg in in_deg.items() if deg == 0]
        processed = set()
        while queue:
            n = queue.pop(0)
            processed.add(n)
            for s in succs.get(n, []):
                in_deg[s] -= 1
                if in_deg[s] == 0:
                    queue.append(s)
        in_cycle = set(edges.keys()) - processed

        if not in_cycle:
            break

        # Break cycle: cut edge from node with fewest deps
        best_edge = None
        best_score = float('inf')
        for sid in in_cycle:
            for dep in seg_by_id[sid].dependencies:
                if dep in in_cycle:
                    score = len(seg_by_id[dep].dependencies)
                    if score < best_score:
                        best_score = score
                        best_edge = (sid, dep)

        if best_edge:
            from_seg, to_seg = best_edge
            seg_by_id[from_seg].dependencies = [
                d for d in seg_by_id[from_seg].dependencies if d != to_seg
            ]
            logger.warning(
                "[segmentation] v1.5 CYCLE BREAK: removed edge %s → %s (round %d)",
                from_seg, to_seg, cycle_round + 1,
            )
        else:
            logger.error("[segmentation] v1.5 Cannot find edge to break cycle in: %s", in_cycle)
            break


# =============================================================================
# STEP 4: Relocate existing files to integration segment (v1.4)
# =============================================================================

def _relocate_existing_files(segments: List[SegmentSpec]) -> None:
    """Move existing source files from helper segments to integration segment only."""
    if len(segments) < 2:
        return

    integration_seg = max(segments, key=lambda s: len(s.dependencies))
    integration_id = integration_seg.segment_id

    all_verified: set = set()
    for seg in segments:
        if seg.grounding_data and isinstance(seg.grounding_data, dict):
            for vf in seg.grounding_data.get("verified_files", []):
                vf_path = vf.get("path", "") if isinstance(vf, dict) else str(vf)
                if vf_path:
                    all_verified.add(vf_path.replace("\\", "/").lower())

    if not all_verified:
        return

    moved_count = 0
    for seg in segments:
        if seg.segment_id == integration_id:
            continue

        new_scope = []
        for f in seg.file_scope:
            f_norm = f.replace("\\", "/").lower()
            if f_norm in all_verified:
                if f not in seg.evidence_files:
                    seg.evidence_files.append(f)
                moved_count += 1
                logger.info(
                    "[segmentation] v1.4 Moved existing file %s from %s file_scope → evidence_files",
                    f, seg.segment_id,
                )
            else:
                new_scope.append(f)

        if len(new_scope) < len(seg.file_scope):
            seg.file_scope = new_scope
            seg.estimated_files = len(new_scope)

    if moved_count > 0:
        logger.info(
            "[segmentation] v1.4 Relocated %d existing file(s) to evidence — integration: %s",
            moved_count, integration_id,
        )


# =============================================================================
# STEP 4B: Demote reference-only files from foreign domains (v3.1)
# =============================================================================

def _demote_reference_only_files(
    segments: List[SegmentSpec],
    original_file_scope: List[str],
) -> None:
    """
    v3.1: Detect and demote files from unrelated domains to evidence_files.

    When the spec references a file from an existing domain purely as a pattern
    (e.g. app/builds/schemas.py mentioned as reference when building
    app/education/schemas.py), the file scope extractor picks it up as a
    modification target. This function detects such cross-domain reference
    files and demotes them to evidence_files.

    Heuristic: if a segment contains files from two different app/ subpackages,
    and one subpackage has only existing (verified) files while the other has
    CREATE targets, the existing files from the foreign subpackage are reference-only.
    """
    demoted_count = 0

    for seg in segments:
        if len(seg.file_scope) < 2:
            continue

        grounding = seg.grounding_data
        if not grounding or not isinstance(grounding, dict):
            continue

        # Build set of verified (existing) file paths
        verified_paths = set()
        for vf in grounding.get("verified_files", []):
            vf_path = vf.get("path", "") if isinstance(vf, dict) else str(vf)
            if vf_path:
                verified_paths.add(vf_path.replace("\\", "/").lower())

        # Build set of CREATE target file paths
        create_paths = set()
        for ct in grounding.get("create_targets", []):
            ct_path = ct.get("path", "") if isinstance(ct, dict) else str(ct)
            if ct_path:
                create_paths.add(ct_path.replace("\\", "/").lower())

        if not verified_paths or not create_paths:
            continue

        # Detect subpackage directories for each category
        def _get_subpkg(path_norm: str) -> str:
            """Extract app/X subpackage from a path, or empty string."""
            parts = path_norm.split("/")
            if len(parts) >= 2 and parts[0] == "app":
                return f"app/{parts[1]}"
            return ""

        create_pkgs = {_get_subpkg(p) for p in create_paths} - {""}
        verified_pkgs = {_get_subpkg(p) for p in verified_paths} - {""}

        # Foreign packages = verified packages that have NO create targets
        foreign_pkgs = verified_pkgs - create_pkgs
        if not foreign_pkgs:
            continue

        # Demote files from foreign packages: file_scope → evidence_files
        new_scope = []
        for f in seg.file_scope:
            f_norm = f.replace("\\", "/").lower()
            f_pkg = _get_subpkg(f_norm)
            if f_pkg in foreign_pkgs and f_norm in verified_paths:
                if f not in seg.evidence_files:
                    seg.evidence_files.append(f)
                demoted_count += 1
                logger.info(
                    "[segmentation] v3.1 Demoted reference-only file %s from %s "
                    "(foreign pkg '%s', not in create targets)",
                    f, seg.segment_id, f_pkg,
                )
            else:
                new_scope.append(f)

        if len(new_scope) < len(seg.file_scope):
            seg.file_scope = new_scope
            seg.estimated_files = len(new_scope)

    if demoted_count > 0:
        logger.info(
            "[segmentation] v3.1 Demoted %d reference-only file(s) from foreign domains",
            demoted_count,
        )


# =============================================================================
# STEP 5: Transitive dependency closure (v1.2)
# =============================================================================

def _apply_transitive_deps(segments: List[SegmentSpec]) -> None:
    """Infer missing cross-segment dependencies for same-package files."""
    all_files = [f for seg in segments for f in seg.file_scope]
    normalised = [f.replace("\\", "/") for f in all_files]

    if len(normalised) < 2:
        return

    parts_list = [f.rsplit("/", 1) for f in normalised if "/" in f]
    if not parts_list:
        return

    dirs = [p[0] for p in parts_list]
    dir_counts = Counter(dirs)
    most_common_dir, most_common_count = dir_counts.most_common(1)[0]
    common_dir = most_common_dir if most_common_count >= len(dirs) * 0.6 else None

    if not common_dir:
        return

    logger.info("[segmentation] v1.2 Common package detected: %s — applying transitive deps", common_dir)

    changed = True
    rounds = 0
    while changed and rounds < 10:
        changed = False
        rounds += 1
        for seg in segments:
            new_deps = set(seg.dependencies)
            for dep_id in list(seg.dependencies):
                dep_seg = next((s for s in segments if s.segment_id == dep_id), None)
                if dep_seg:
                    for transitive_dep in dep_seg.dependencies:
                        if transitive_dep not in new_deps and transitive_dep != seg.segment_id:
                            new_deps.add(transitive_dep)
                            changed = True
            if len(new_deps) > len(seg.dependencies):
                added = new_deps - set(seg.dependencies)
                logger.info("[segmentation] v1.2 Added transitive deps to %s: %s",
                            seg.segment_id, list(added))
                seg.dependencies = sorted(new_deps)
                for added_dep_id in added:
                    added_seg = next((s for s in segments if s.segment_id == added_dep_id), None)
                    if added_seg:
                        for dep_file in added_seg.file_scope:
                            if dep_file not in seg.evidence_files and dep_file not in seg.file_scope:
                                seg.evidence_files.append(dep_file)

    logger.info("[segmentation] v1.2 Transitive dependency closure complete (%d rounds)", rounds)


# =============================================================================
# STEP 5B: Infer missing dependencies from evidence_files (v3.1)
# =============================================================================

def _infer_evidence_dependencies(segments: List[SegmentSpec]) -> None:
    """
    v3.1: If segment X lists files from segment Y in evidence_files,
    X should depend on Y (Y must be built first so X can read its outputs).

    This catches cases where the LLM segmenter forgets a dependency edge
    but correctly placed files in evidence_files. For example, seg-08
    (main.py integration) lists app/education/__init__.py in evidence but
    has no dependency on the facade segment that creates it.

    Skips facade segments to avoid creating cycles (facades already depend
    on everything).
    """
    # Build file → owner segment mapping (only for CREATE targets / file_scope)
    file_to_owner: Dict[str, str] = {}
    for seg in segments:
        for f in seg.file_scope:
            f_norm = f.replace("\\", "/").lower()
            file_to_owner[f_norm] = seg.segment_id

    # Detect facades (they already depend on everything)
    facade_ids = set()
    for seg in segments:
        if any(f.replace("\\", "/").endswith("__init__.py") for f in seg.file_scope):
            title_lower = (seg.title or "").lower()
            if any(kw in title_lower for kw in ["facade", "init", "package"]):
                facade_ids.add(seg.segment_id)

    added_count = 0
    for seg in segments:
        if seg.segment_id in facade_ids:
            continue

        existing_deps = set(seg.dependencies)
        for ev_file in seg.evidence_files:
            ev_norm = ev_file.replace("\\", "/").lower()
            owner_id = file_to_owner.get(ev_norm)
            if owner_id and owner_id != seg.segment_id and owner_id not in existing_deps:
                # Don't add dependency on facades (would create cycles)
                if owner_id in facade_ids:
                    continue
                seg.dependencies.append(owner_id)
                existing_deps.add(owner_id)
                added_count += 1
                logger.info(
                    "[segmentation] v3.1 Inferred dependency: %s → %s "
                    "(evidence file %s owned by %s)",
                    seg.segment_id, owner_id, ev_file, owner_id,
                )

    if added_count > 0:
        logger.info(
            "[segmentation] v3.1 Inferred %d missing dependencies from evidence_files",
            added_count,
        )


# =============================================================================
# STEP 6: Assign requirements to segments
# =============================================================================

def _assign_requirements(
    segments: List[SegmentSpec],
    requirements: List[str],
) -> None:
    """Distribute requirements across segments."""
    requirement_map = _distribute_requirements(requirements, segments)
    seg_id_to_seg = {s.segment_id: s for s in segments}
    for req, seg_ids in requirement_map.items():
        for sid in seg_ids:
            if sid in seg_id_to_seg:
                seg_id_to_seg[sid].requirements.append(req)


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
