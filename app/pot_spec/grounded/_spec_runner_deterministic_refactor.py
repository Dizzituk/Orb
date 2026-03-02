# FILE: app/pot_spec/grounded/_spec_runner_deterministic_refactor.py
"""
v6.1: Deterministic refactor detection and manifest generation.

Extracted from spec_runner.py Step 4b. Identifies file→package refactor
patterns and generates segment manifests without LLM calls.
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.
_SBX_FS_OK = False


def detect_refactor_pairs(
    file_scope: List[str],
    spot_markdown: str,
) -> Tuple[bool, str, "list[tuple[str,str]]"]:
    """
    Detect file→package refactor patterns and return source/target pairs.

    Returns:
        (is_refactor, reason, refactor_pairs)
    """
    try:
        from app.orchestrator.refactor_pipeline import is_refactor_job
    except ImportError:
        logger.debug("[det_refactor] refactor_pipeline not available")
        return False, "", []

    _is_refactor, _refactor_reason = is_refactor_job(spot_markdown, file_scope)
    logger.info(
        "[det_refactor] v6.1 Refactor detection: is_refactor=%s, reason=%s, file_scope=%d files",
        _is_refactor, _refactor_reason, len(file_scope),
    )

    if not _is_refactor:
        return False, _refactor_reason, []

    logger.info("[det_refactor] v6.1 DETERMINISTIC REFACTOR detected: %s", _refactor_reason)
    print(f"[spec_runner] v6.1 DETERMINISTIC REFACTOR: {_refactor_reason}")

    _scope_norm = [fp.replace("\\", "/") for fp in file_scope]
    _refactor_pairs: list[tuple[str, str]] = []
    _matched_sources: set[str] = set()

    # Strategy 0: Direct source detection — any existing .py > 10KB on disk
    for _fp_norm in _scope_norm:
        if not _fp_norm.endswith(".py") or "/" not in _fp_norm:
            continue
        if _fp_norm.endswith("__init__.py"):
            continue
        _abs = os.path.join("D:\\Orb", _fp_norm.replace("/", os.sep))
        if not (_sbx_isfile(_abs) if _SBX_FS_OK else os.path.isfile(_abs)):
            continue
        try:
            _fsize = os.path.getsize(_abs)
            if _fsize < 10_000:
                continue
        except OSError:
            continue
        _stem = _fp_norm.rsplit("/", 1)[-1].replace(".py", "")
        _parent = _fp_norm.rsplit("/", 1)[0]
        _auto_pkg = f"{_parent}/{_stem}/"
        _refactor_pairs.append((_fp_norm, _auto_pkg))
        _matched_sources.add(_fp_norm)
        logger.info(
            "[det_refactor] v6.1 Strategy 0 (direct): %s -> %s (%d bytes)",
            _fp_norm, _auto_pkg, _fsize,
        )

    if _refactor_pairs:
        logger.info(
            "[det_refactor] v6.1 Strategy 0 found %d pair(s) — skipping Strategy 1/2",
            len(_refactor_pairs),
        )
        return True, _refactor_reason, _refactor_pairs

    # Strategy 1: Exact stem match (file.py → file/ package)
    for _fp_norm in _scope_norm:
        if _fp_norm.endswith(".py") and "/" in _fp_norm:
            _stem = _fp_norm.rsplit("/", 1)[-1].replace(".py", "")
            _parent = _fp_norm.rsplit("/", 1)[0]
            _pkg_prefix = f"{_parent}/{_stem}/"
            _has_pkg = any(
                f2.startswith(_pkg_prefix)
                for f2 in _scope_norm if f2 != _fp_norm
            )
            _check_path = os.path.join("D:\\Orb", _fp_norm.replace("/", os.sep))
            _on_disk = (_sbx_isfile(_check_path) if _SBX_FS_OK else os.path.isfile(_check_path))
            if _has_pkg and _on_disk:
                _refactor_pairs.append((_fp_norm, _pkg_prefix))
                _matched_sources.add(_fp_norm)
                logger.info(
                    "[det_refactor] v6.1 Strategy 1 match: %s -> %s",
                    _fp_norm, _pkg_prefix,
                )

    if _refactor_pairs:
        return True, _refactor_reason, _refactor_pairs

    # Strategy 2: Source .py on disk + __init__.py in child dir
    _unmatched = [
        fp for fp in _scope_norm
        if fp.endswith(".py") and "/" in fp and fp not in _matched_sources
        and (_sbx_isfile(os.path.join("D:\\Orb", fp.replace("/", os.sep))) if _SBX_FS_OK else os.path.isfile(os.path.join("D:\\Orb", fp.replace("/", os.sep))))
    ]
    for _fp_norm in _unmatched:
        _parent = _fp_norm.rsplit("/", 1)[0]
        for _f2 in _scope_norm:
            if _f2 == _fp_norm:
                continue
            if _f2.startswith(_parent + "/") and "__init__.py" in _f2:
                _pkg_dir = _f2.rsplit("__init__.py", 1)[0]
                _refactor_pairs.append((_fp_norm, _pkg_dir))
                _matched_sources.add(_fp_norm)
                logger.info(
                    "[det_refactor] v6.1 Strategy 2 match: %s -> %s",
                    _fp_norm, _pkg_dir,
                )
                break

    if not _refactor_pairs:
        logger.info(
            "[det_refactor] v6.1 Refactor detected but couldn't identify source/target pairs"
        )

    return True if _refactor_pairs else False, _refactor_reason, _refactor_pairs


def build_deterministic_manifest(
    refactor_pairs: "list[tuple[str,str]]",
    job_id: str,
):
    """
    Run deterministic refactor for each source/target pair and merge into manifest.

    Returns:
        SegmentManifest or None
    """
    try:
        from app.orchestrator.refactor_pipeline import run_deterministic_refactor
        from app.pot_spec.grounded.segment_schemas import (
            SegmentManifest,
            SegmentSpec as _DetSegSpec,
        )
    except ImportError:
        logger.debug("[det_refactor] Required modules not available")
        return None

    from app.pot_spec.grounded._spec_runner_utils_11 import _get_job_dir_for_segmentation

    _det_job_dir = _get_job_dir_for_segmentation(job_id)
    _all_segments = []
    _all_sources = []

    for _src, _tgt in refactor_pairs:
        logger.info(
            "[det_refactor] v6.1 Running deterministic refactor: %s -> %s",
            _src, _tgt,
        )
        # v6.1 FIX 20: Always use auto-layout (empty inventory)
        _det_plan, _det_archs, _det_manifest_data = run_deterministic_refactor(
            source_file_path=_src,
            architecture_file_inventory="",
            target_package=_tgt,
            job_dir=_det_job_dir,
            spec_id=f"sg-{uuid.uuid4().hex[:12]}",
        )

        for _seg_data in _det_manifest_data["segments"]:
            _seg = _DetSegSpec.from_dict(_seg_data)
            _seg.deterministic_source = _src
            _all_segments.append(_seg)
        _all_sources.append(_src)

    manifest = SegmentManifest(
        segments=_all_segments,
        total_segments=len(_all_segments),
        total_files=sum(len(s.file_scope) for s in _all_segments),
        deterministic_sources=_all_sources,
    )
    logger.info(
        "[det_refactor] v6.1 Deterministic manifest: %d segments, %d files, %d source(s)",
        len(_all_segments),
        sum(len(s.file_scope) for s in _all_segments),
        len(_all_sources),
    )
    print(
        f"[spec_runner] v6.1 DETERMINISTIC: "
        f"{len(_all_segments)} segments, "
        f"{sum(len(s.file_scope) for s in _all_segments)} files, "
        f"{len(_all_sources)} source(s)"
    )
    return manifest
