# FILE: app/pot_spec/grounded/_spec_runner_segmentation.py
"""
Spec runner Step 4b: Segmentation check and manifest generation.

Handles needle classification, deterministic refactor detection,
concept-aware grouping, consumer exclusion, and manifest output.
Extracted from spec_runner.py.
"""
from __future__ import annotations

import hashlib
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.pot_spec.grounded._spec_runner_utils_11 import (
    _extract_requirements_from_spec,
    _get_job_dir_for_segmentation,
)
from app.pot_spec.grounded._spec_runner_utils_12 import _write_segmentation_output
from app.pot_spec.grounded._spec_runner_utils_13 import (
    _extract_acceptance_from_spec,
    _extract_file_scope_from_spec,
)
from app.pot_spec.grounded._spec_runner_deterministic_refactor import (
    detect_refactor_pairs,
    build_deterministic_manifest,
)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.
_SBX_FS_OK = False

logger = logging.getLogger(__name__)


async def run_segmentation_check(
    spot_markdown: Optional[str],
    combined_text: str,
    multi_file_op: Any,
    job_id: str,
    goal: str,
    round_n: int,
    is_create_job: bool = False,
) -> Tuple[Any, Optional[Any], bool]:
    """
    Run Step 4b: Segmentation check.

    Returns:
        (segmentation_manifest_or_None, needle_estimate_or_None, early_return)

    If early_return is True, caller should build a SpecGateResult from the
    manifest and return immediately (the job is segmented).
    """
    segmentation_manifest = None
    _needle_estimate = None

    try:
        from .segmentation import needs_segmentation, generate_segments

        # Extract file scope from spec
        _file_scope = _extract_file_scope_from_spec(
            spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
        )

        # v6.1 FIX 25: Also merge file paths from raw user intent
        _user_scope = _extract_file_scope_from_spec(
            combined_text, grounding_data=None, multi_file_op=None,
        )
        if _user_scope:
            _existing = {fp.replace("\\", "/").lower() for fp in _file_scope}
            _added = []
            for _ufp in _user_scope:
                if _ufp.replace("\\", "/").lower() not in _existing:
                    _file_scope.append(_ufp)
                    _added.append(_ufp)
            if _added:
                logger.info(
                    "[spec_runner] FIX 25: Merged %d file(s) from user intent: %s",
                    len(_added), _added[:5],
                )

        # v5.17: Resolve file scope paths against real filesystem.
        # LLM-extracted paths may use wrong locations (e.g. src/types.ts
        # instead of src/types/index.ts, or src/Sidebar.tsx instead of
        # src/components/sidebar/Sidebar.tsx). Resolve before segmentation.
        _file_scope = _resolve_file_scope_paths(_file_scope)

        logger.info("[spec_runner] v5.17 File scope: %d file(s)", len(_file_scope))
        print(f"[spec_runner] v5.17 FILE_SCOPE: {len(_file_scope)} file(s): {_file_scope[:5]}")

        if not _file_scope:
            return None, None, False

        # v5.6: Pre-segmentation size analysis
        _size_metadata = _run_size_analysis(_file_scope, spot_markdown)

        # Needle classification
        _should_segment, _seg_reason, _needle_estimate = await _classify_needles(
            spot_markdown, _file_scope,
        )

        # v6.1: Deterministic refactor check
        # v6.2: SKIP for create jobs — create specs mention 'extract'/'refactor'
        # in reference patterns but are NOT refactor jobs
        _deterministic_manifest = None
        if not is_create_job:
            _deterministic_manifest = _try_deterministic_refactor(
                _file_scope, spot_markdown, job_id,
            )
        else:
            logger.info("[spec_runner] v6.2 Skipping deterministic refactor check (create job)")

        if _deterministic_manifest:
            segmentation_manifest = _deterministic_manifest
            _should_segment = True
            _seg_reason = "Deterministic refactor"

        if not _should_segment:
            logger.info("[spec_runner] v4.8 No segmentation needed: %s", _seg_reason)
            return None, _needle_estimate, False

        logger.info("[spec_runner] v5.5 Segmentation triggered: %s", _seg_reason)
        print(f"[spec_runner] v5.5 SEGMENTATION: {_seg_reason}")

        # Extract spec metadata
        _requirements = _extract_requirements_from_spec(spot_markdown)
        _acceptance = _extract_acceptance_from_spec(spot_markdown)

        # v5.18: Consumer exclusion for file→package refactors
        _file_scope, _deferred_consumers = _exclude_external_consumers(_file_scope)

        # v5.5 PHASE 3B: Concept-aware grouping
        _concept_groups = None
        _target_segs = _needle_estimate.recommended_segment_count if _needle_estimate else 0
        if _target_segs >= 2 and not _deterministic_manifest:
            _concept_groups = await _try_concept_grouping(
                spot_markdown, _file_scope, _target_segs, _requirements, _size_metadata,
            )

        # Generate manifest (skip LLM segmenter for deterministic)
        if not _deterministic_manifest:
            segmentation_manifest = generate_segments(
                file_scope=_file_scope,
                requirements=_requirements,
                acceptance_criteria=_acceptance,
                parent_spec_id=f"sg-{uuid.uuid4().hex[:12]}",
                parent_spec_hash=hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else None,
                concept_groups=_concept_groups,
            )

        if segmentation_manifest:
            if _deferred_consumers:
                segmentation_manifest.deferred_consumer_files = _deferred_consumers

            _write_segmentation_output(job_id, segmentation_manifest)
            logger.info("[spec_runner] v4.8 Segmentation complete: %s", segmentation_manifest.summary())
            print(f"[spec_runner] v4.8 SEGMENTED: {segmentation_manifest.summary()}")
            return segmentation_manifest, _needle_estimate, True
        else:
            logger.info("[spec_runner] v4.8 Segmentation returned None — single pass")
            return None, _needle_estimate, False

    except ImportError:
        logger.debug("[spec_runner] v4.8 Segmentation module not available")
    except Exception as seg_err:
        logger.warning("[spec_runner] v4.8 Segmentation failed (non-fatal): %s", seg_err)
        print(f"[spec_runner] v4.8 SEGMENTATION FAILED (non-fatal): {seg_err}")

    return None, _needle_estimate, False


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _resolve_file_scope_paths(file_scope: List[str]) -> List[str]:
    """v5.17: Resolve LLM-extracted file paths against the real filesystem.

    LLM analysis often produces paths that are close but wrong:
    - src/types.ts instead of src/types/index.ts (barrel export)
    - src/Sidebar.tsx instead of src/components/sidebar/Sidebar.tsx

    Resolution strategies (in order):
    1. Direct check: path exists at a project root -> keep as-is
    2. Barrel fallback: foo.ts -> foo/index.ts or foo/index.tsx
    3. Architecture index search: find the file by name in known paths

    Returns a new list with resolved paths (same length, deduped).
    """
    from app.pot_spec.grounded._spec_runner_utils_13 import _discover_project_roots

    disc = _discover_project_roots()
    roots = disc.get("roots", [])
    if not roots:
        roots = [r"D:\Orb", r"D:\orb-desktop"]

    # Load architecture index for filename search fallback
    arch_paths: List[str] = []
    try:
        from app.pot_spec.grounded._segmentation_utils_3 import _load_architecture_file_list
        arch_paths = _load_architecture_file_list()
    except Exception:
        pass

    resolved: List[str] = []
    seen: set = set()

    for raw_path in file_scope:
        norm = raw_path.replace("/", os.sep).replace("\\", os.sep)
        result = _try_resolve_single(norm, roots, arch_paths)
        key = result.replace("\\", "/").lower()
        if key not in seen:
            seen.add(key)
            resolved.append(result)

    if resolved != file_scope:
        _changes = [
            (old, new) for old, new in zip(file_scope, resolved)
            if old.replace("\\", "/").lower() != new.replace("\\", "/").lower()
        ]
        if _changes:
            logger.info(
                "[spec_runner] v5.17 Resolved %d path(s): %s",
                len(_changes),
                [f"{o} -> {n}" for o, n in _changes[:5]],
            )

    return resolved


def _try_resolve_single(
    rel_path: str,
    roots: List[str],
    arch_paths: List[str],
) -> str:
    """Try to resolve a single relative path to its real location."""
    # Strategy 1: Direct existence check against project roots
    for root in roots:
        candidate = os.path.join(root, rel_path)
        if os.path.exists(candidate):
            return rel_path  # Path is correct as-is

    # Strategy 2: Barrel export fallback (TypeScript)
    # src/types.ts -> src/types/index.ts
    if rel_path.endswith((".ts", ".tsx")):
        stem = rel_path.rsplit(".", 1)[0]
        ext = rel_path.rsplit(".", 1)[1]
        for barrel_name in ("index.ts", "index.tsx"):
            barrel_path = stem + os.sep + barrel_name
            for root in roots:
                if os.path.exists(os.path.join(root, barrel_path)):
                    logger.info(
                        "[spec_runner] v5.17 Barrel resolve: %s -> %s",
                        rel_path, barrel_path,
                    )
                    return barrel_path

    # Strategy 3: Architecture index filename search
    # src/Sidebar.tsx -> src/components/sidebar/Sidebar.tsx
    if arch_paths:
        filename = os.path.basename(rel_path)
        filename_lower = filename.lower()
        matches = [
            ap for ap in arch_paths
            if ap.lower().endswith(os.sep + filename_lower)
            or ap.lower().endswith("/" + filename_lower)
        ]
        if len(matches) == 1:
            # Unique match — safe to use
            resolved = matches[0]
            # Normalise to relative path (strip project root prefix)
            for root in roots:
                root_prefix = root.replace("/", os.sep)
                if not root_prefix.endswith(os.sep):
                    root_prefix += os.sep
                if resolved.startswith(root_prefix):
                    resolved = resolved[len(root_prefix):]
                    break
            if resolved != rel_path:
                logger.info(
                    "[spec_runner] v5.17 Index resolve: %s -> %s",
                    rel_path, resolved,
                )
            return resolved
        elif len(matches) > 1:
            # Multiple matches — try to pick the one most similar to the LLM path
            # Prefer matches that share directory segments with the original
            rel_parts = set(rel_path.replace("\\", "/").lower().split("/")[:-1])
            best = None
            best_score = -1
            for m in matches:
                m_parts = set(m.replace("\\", "/").lower().split("/")[:-1])
                score = len(rel_parts & m_parts)
                if score > best_score:
                    best_score = score
                    best = m
            if best and best_score > 0:
                resolved = best
                for root in roots:
                    root_prefix = root.replace("/", os.sep)
                    if not root_prefix.endswith(os.sep):
                        root_prefix += os.sep
                    if resolved.startswith(root_prefix):
                        resolved = resolved[len(root_prefix):]
                        break
                if resolved != rel_path:
                    logger.info(
                        "[spec_runner] v5.17 Index resolve (best of %d): %s -> %s",
                        len(matches), rel_path, resolved,
                    )
                return resolved

    # No resolution found — return as-is (will be treated as CREATE target)
    return rel_path


def _run_size_analysis(
    file_scope: List[str],
    spot_markdown: Optional[str],
) -> Dict[str, Any]:
    """Run pre-segmentation size analysis, expanding scope if needed."""
    try:
        from .size_analyzer import analyze_file_sizes
        result = analyze_file_sizes(file_scope=file_scope, spec_markdown=spot_markdown)
        if result.files_added:
            logger.info(
                "[spec_runner] v5.6 Size analysis expanded scope: +%d files",
                len(result.files_added),
            )
            file_scope.clear()
            file_scope.extend(result.enriched_file_scope)
        return {
            est.rel_path: est.to_dict()
            for est in result.estimates.values()
        } if result.estimates else {}
    except (ImportError, Exception) as err:
        logger.debug("[spec_runner] Size analyzer unavailable: %s", err)
        return {}


async def _classify_needles(
    spot_markdown: Optional[str],
    file_scope: List[str],
) -> Tuple[bool, str, Any]:
    """Classify cognitive load via needle count. Falls back to legacy."""
    try:
        from .needle_classifier import classify_needles
        estimate = await classify_needles(
            spec_markdown=spot_markdown, file_scope=file_scope,
        )
        logger.info(
            "[spec_runner] v5.5 Needle estimate: %d (%s)",
            estimate.needle_estimate, estimate.difficulty_tier,
        )
        return estimate.needs_segmentation, (
            f"Needle count {estimate.needle_estimate} ({estimate.difficulty_tier})"
        ), estimate
    except (ImportError, Exception) as err:
        logger.debug("[spec_runner] Needle classifier unavailable: %s", err)
        try:
            from .segmentation import needs_segmentation
            should_seg, reason = needs_segmentation(file_scope)
            return should_seg, reason, None
        except ImportError:
            return False, "segmentation not available", None


def _try_deterministic_refactor(
    file_scope: List[str],
    spot_markdown: Optional[str],
    job_id: str,
) -> Any:
    """Try deterministic refactor detection. Returns manifest or None."""
    try:
        is_refactor, reason, pairs = detect_refactor_pairs(file_scope, spot_markdown)
        if is_refactor and pairs:
            return build_deterministic_manifest(pairs, job_id)
    except Exception as err:
        logger.warning("[spec_runner] v6.1 Deterministic refactor error: %s", err)
    return None


def _exclude_external_consumers(
    file_scope: List[str],
) -> Tuple[List[str], List[str]]:
    """
    v5.18: Exclude external consumer files from segmentation scope.
    Returns (filtered_scope, deferred_consumers).
    """
    _norm_scope = [f.replace("\\", "/") for f in file_scope]
    _refactor_packages: Dict[str, str] = {}

    for _fp in _norm_scope:
        if _fp.endswith(".py") and "/" in _fp:
            _stem = _fp.rsplit("/", 1)[-1].replace(".py", "")
            _parent = _fp.rsplit("/", 1)[0]
            _pkg_prefix = f"{_parent}/{_stem}/"
            _has_pkg_files = any(
                _other.startswith(_pkg_prefix) for _other in _norm_scope if _other != _fp
            )
            if _has_pkg_files:
                _refactor_packages[_pkg_prefix] = _fp

    if not _refactor_packages:
        return file_scope, []

    _products = set()
    for _pkg_prefix, _mono_path in _refactor_packages.items():
        _products.add(_mono_path)
        for _fp2 in _norm_scope:
            if _fp2.startswith(_pkg_prefix):
                _products.add(_fp2)

    _deferred = []
    for _fp3 in _norm_scope:
        if _fp3 not in _products:
            _abs = os.path.join("D:\\Orb", _fp3.replace("/", os.sep))
            if (_sbx_isfile(_abs) if _SBX_FS_OK else os.path.isfile(_abs)):
                _deferred.append(_fp3)

    if _deferred:
        filtered = [
            f for f in file_scope
            if f.replace("\\", "/") not in set(_deferred)
        ]
        logger.info(
            "[spec_runner] v5.18 External consumer exclusion: deferred %d file(s)",
            len(_deferred),
        )
        return filtered, _deferred

    return file_scope, []


async def _try_concept_grouping(
    spot_markdown: Optional[str],
    file_scope: List[str],
    target_segments: int,
    requirements: List[str],
    size_metadata: Dict[str, Any],
) -> Any:
    """Try concept-aware segmentation grouping."""
    try:
        from .smart_segmentation import generate_concept_segments
        groups = await generate_concept_segments(
            spec_markdown=spot_markdown,
            file_scope=file_scope,
            target_segments=target_segments,
            requirements=requirements,
            size_metadata=size_metadata,
        )
        if groups:
            logger.info("[spec_runner] v5.5 Concept grouping: %d groups", len(groups))
        return groups
    except (ImportError, Exception) as err:
        logger.debug("[spec_runner] Smart segmentation unavailable: %s", err)
        return None
