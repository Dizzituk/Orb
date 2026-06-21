# FILE: app/pipeline_v2/scaffold_engine.py
# Purpose: ASTRA v2.2 Scaffold Engine — deterministic file generator (orchestration spine).
# Called-by: app.pipeline_v2.orchestrator
# Depends-on: app.pipeline_v2.scaffold_paths, app.pipeline_v2.scaffold_templates, app.pipeline_v2.models (+2 lazy)
# Last-renovated: 2026-06-21
"""
ASTRA v2.2 Scaffold Engine — deterministic file generator.

Reads the SpecGate manifest and produces skeleton files. No LLM calls.
80-90% of each file is laid down here.

BATCH 4 split: path/existence resolution moved to scaffold_paths.py and the
per-file skeleton templates moved to scaffold_templates.py; both are re-exported
below so this module's public surface (run_scaffold_engine) is unchanged.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from app.pipeline_v2.models import ScaffoldFile, ScaffoldResult

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

from app.pipeline_v2.scaffold_paths import (
    _exists_on_host,
    _resolve_for_log,
    _WALK_SKIP_DIRS,
    _build_project_basename_index,
    _maybe_redirect_to_existing_path,
)
from app.pipeline_v2.scaffold_templates import (
    _generate_skeleton,
    _skeleton_android_xml,
    _skeleton_python,
    _skeleton_typescript,
    _skeleton_css,
    _to_kebab,
)

logger = logging.getLogger(__name__)


async def run_scaffold_engine(
    manifest: Dict[str, Any],
    spec: Dict[str, Any],
    job_dir: str,
    on_progress: Optional[Callable[[str], None]] = None,
    profile: Optional["BuildTargetProfile"] = None,
) -> ScaffoldResult:
    """Run the Scaffold Engine to produce deterministic file skeletons.

    Args:
        manifest: Segment manifest from SpecGate.
        spec: The verified spec content.
        job_dir: Job directory for saving artifacts.
        on_progress: Progress callback for UI updates.
        profile: Build target profile (determines language, paths).

    Returns:
        ScaffoldResult with all skeleton files.
    """
    t_start = time.time()
    emit = on_progress or (lambda msg: None)
    result = ScaffoldResult()

    lang = profile.language if profile else "python"
    emit(f"🏗️ Scaffold Engine: Generating {lang} file skeletons...")

    segments = manifest.get("segments", [])
    skeleton_contract = _load_skeleton_contract(job_dir)
    # Collect all files across segments
    all_files: List[Dict[str, Any]] = []
    for seg in segments:
        seg_id = seg.get("segment_id", "")
        file_scope = seg.get("file_scope", [])
        requirements = seg.get("requirements", [])
        grounding = seg.get("grounding_data") or {}
        # v2.1 (2026-04-12): Phase 1 Job 15 — capture per-segment target_id
        # so scaffold writes route to the correct repo for multi-target jobs.
        seg_target_id = seg.get("target_id")

        for fp in file_scope:
            is_new = _is_create_file(fp, grounding)
            all_files.append({
                "path": fp,
                "segment_id": seg_id,
                "target_id": seg_target_id,
                "is_new": is_new,
                "requirements": requirements,
                "grounding": grounding,
            })

    emit(f"   Files to scaffold: {len(all_files)}")

    # Generate skeleton for each file
    from app.pipeline_v2.sandbox_tools import write_file as sandbox_write

    # v2.3 (2026-04-18): track how many CREATE writes were demoted to MODIFY
    # because the file already existed on disk. Surfaced in the final summary.
    demoted_existing = 0

    # v2.4 (2026-04-18): per-run cache of project basename indices, keyed by
    # target_id. Built lazily on first use per target so we only walk each
    # project tree once per scaffold run. For jobs with a single target this
    # is a single walk; multi-target jobs get one walk per touched target.
    basename_index_cache: Dict[str, Dict[str, List[str]]] = {}
    redirected_count = 0

    for file_info in all_files:
        fp = file_info["path"]
        is_new = file_info["is_new"]

        # v2.1 (2026-04-12): Phase 1 Job 15 — resolve per-file profile from
        # the segment's target_id. Falls back to the passed-in profile if
        # the segment has no target (single-target or legacy jobs).
        file_profile = profile
        _tid = file_info.get("target_id")
        if _tid:
            try:
                from app.pipeline_v2.target_registry import get_profile
                _resolved = get_profile(_tid)
                if _resolved is not None:
                    file_profile = _resolved
            except Exception as _pe:
                logger.debug("[scaffold_engine] profile lookup failed for target_id=%s: %s", _tid, _pe)

        # v2.4 (2026-04-18): BASENAME-AWARE PATH REDIRECTION.
        # Before anything else, check if the spec path's basename already
        # exists somewhere in the target project tree. If there's a single
        # unambiguous match, redirect `fp` to that existing path. This
        # prevents the scaffold from creating bogus duplicates like
        # drivercopilot/DriverCopilotDatabase.kt at the package root when
        # the real file lives at drivercopilot/data/DriverCopilotDatabase.kt.
        # Zero matches or multiple matches → no redirect, use fp as-is.
        if is_new and file_profile is not None:
            redirected_fp = _maybe_redirect_to_existing_path(
                fp, file_profile, basename_index_cache,
            )
            if redirected_fp is not None and redirected_fp != fp:
                logger.info(
                    "[scaffold_engine] v2.4 PATH-REDIRECT: spec said '%s' → "
                    "existing path '%s' (basename match in target tree)",
                    fp, redirected_fp,
                )
                emit(f"   📍 [REDIRECT] {fp} → {redirected_fp} (matched existing file)")
                fp = redirected_fp
                redirected_count += 1

        # v2.3 (2026-04-18): HARD NON-DESTRUCTIVE GUARD.
        # If grounding said this file is new but it already exists on disk,
        # demote to MODIFY rather than overwriting it. This protects host
        # files when SpecGate's grounding data is incomplete (for example,
        # when the sandbox is offline during inspection and verified_files
        # didn't populate). The agentic builder owns MODIFY edits via its
        # read_file / write_file tool loop — that's the right place for
        # changes to existing code, not blind scaffold overwrites.
        if is_new and _exists_on_host(fp, file_profile):
            is_new = False
            demoted_existing += 1
            abs_path = _resolve_for_log(fp, file_profile)
            logger.info(
                "[scaffold_engine] v2.3 DEMOTED to MODIFY — file exists on host: %s",
                abs_path,
            )
            emit(f"   🛡️ [SKIP-EXISTS] {fp} — already on disk, queued for agentic builder")

        if is_new:
            skeleton = _generate_skeleton(
                fp,
                file_info["requirements"],
                skeleton_contract,
                spec,
                file_profile,
            )
            scaffold_file = ScaffoldFile(
                path=fp,
                content=skeleton,
                is_new=True,
                char_count=len(skeleton),
            )
            result.files.append(scaffold_file)

            # Write to sandbox using per-file profile (multi-target aware)
            ok = await sandbox_write(fp, skeleton, profile=file_profile)
            status = "✅" if ok else "❌"
            _tgt = file_profile.project_id if file_profile else "?"
            emit(f"   {status} [CREATE] {fp} -> {_tgt} ({len(skeleton):,} chars)")
        else:
            # MODIFY files: don't write a skeleton, just record them
            scaffold_file = ScaffoldFile(
                path=fp,
                content="",
                is_new=False,
                char_count=0,
            )
            result.files.append(scaffold_file)
            # Only emit the explicit MODIFY marker if this wasn't already
            # reported as a SKIP-EXISTS above (avoid duplicate log lines).
            if file_info["is_new"]:
                pass  # already emitted SKIP-EXISTS above
            else:
                emit(f"   📝 [MODIFY] {fp}")

    result.total_files = len(all_files)
    result.duration_seconds = time.time() - t_start

    # v2.1: Copy Gradle wrapper for Android greenfield projects
    if profile and profile.language == "kotlin":
        try:
            from app.pipeline_v2.scaffolds.android_config_scaffolds import copy_gradle_wrapper
            if copy_gradle_wrapper(profile.project_root.replace('/', os.sep)):
                emit("   Copied Gradle wrapper (gradlew, jar, properties)")
        except Exception as _gw_err:
            emit(f"   Gradle wrapper copy failed: {_gw_err}")
    create_count = sum(1 for f in result.files if f.is_new)
    modify_count = sum(1 for f in result.files if not f.is_new)
    summary_bits = []
    if demoted_existing:
        summary_bits.append(f"{demoted_existing} existing file(s) preserved")
    if redirected_count:
        summary_bits.append(f"{redirected_count} path(s) redirected to existing locations")
    summary_tail = (", " + ", ".join(summary_bits)) if summary_bits else ""
    emit(f"\n🏗️ Scaffold complete: {create_count} skeletons written, "
         f"{modify_count} MODIFY files queued{summary_tail} "
         f"({result.duration_seconds:.1f}s)")

    return result


def _is_create_file(file_path: str, grounding: Dict) -> bool:
    """Determine if a file is CREATE (new) or MODIFY (existing)."""
    grounding = grounding or {}
    create_targets = grounding.get("create_targets", [])
    for ct in create_targets:
        if isinstance(ct, dict):
            ct_path = ct.get("path", "").replace("\\", "/")
        else:
            ct_path = str(ct).replace("\\", "/")
        if ct_path.lower() == file_path.replace("\\", "/").lower():
            return True

    verified = grounding.get("verified_files", [])
    for vf in verified:
        if isinstance(vf, dict):
            vf_path = vf.get("path", "").replace("\\", "/")
        else:
            vf_path = str(vf).replace("\\", "/")
        if vf_path.lower() == file_path.replace("\\", "/").lower():
            return False

    new_files = grounding.get("new_files", [])
    for nf in new_files:
        nf_path = (nf.get("path", "") if isinstance(nf, dict) else str(nf)).replace("\\", "/")
        if nf_path.lower() == file_path.replace("\\", "/").lower():
            return True

    return True


def _load_skeleton_contract(job_dir: str) -> Dict[str, Any]:
    """Load skeleton contract from job directory."""
    import json
    contract_path = os.path.join(job_dir, "segments", "skeleton_contract.json")
    if os.path.exists(contract_path):
        try:
            with open(contract_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("[scaffold] Could not load skeleton contract: %s", e)
    return {}
