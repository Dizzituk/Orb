# FILE: app/pipeline_v2/file_target_resolution.py
# Purpose: Resolve which build target owns a set of file paths (+ ambient job hint).
# Called-by: app.pipeline_v2.target_registry (re-export shim; importers resolve through it)
# Depends-on: app.pipeline_v2.target_profiles
# Last-renovated: 2026-06-21
"""
File-target resolution — split out of target_registry.py (SPLIT BATCH 9, 2026-06-21).

resolve_target_for_files / _resolve_single_file: the 4-tier path->target_id
cascade. Co-located with the ambient job-target hint (_job_target_hint +
set_job_target_hint / get_job_target_hint) BECAUSE set_job_target_hint REBINDS the
module global that resolve_target_for_files reads — they must share one module for
the ambient-hint mechanism to stay consistent (deliberate deviation from the V3
plan, which placed the hint pair in the shim). Reads the shared _REGISTRY from
target_profiles. Moved VERBATIM; target_registry.py re-exports these names so all
importers resolve unchanged.
"""
from __future__ import annotations

import logging

from app.pipeline_v2.target_profiles import _REGISTRY

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Job-level target hint (ambient context for resolve_target_for_files)
# ═══════════════════════════════════════════════════════════════════
# When the pipeline_bridge layer persists target_ids on the BuildProject
# (Phase 0 Job 3), the spec_runner should read those back and call
# set_job_target_hint() before segmentation begins. resolve_target_for_files
# falls back to this hint when callers do not pass job_target_ids
# explicitly — removes the need to thread context through every layer.
# v1.0 (2026-04-12): Phase 1 Job 14.

_job_target_hint: "set | None" = None


def set_job_target_hint(target_ids) -> None:
    """Set the ambient job-level target hint.

    target_ids: iterable of project_id strings, or None to clear.
    """
    global _job_target_hint
    if target_ids is None:
        _job_target_hint = None
        return
    try:
        _job_target_hint = set(target_ids)
        logger.info("[target_registry] Job target hint set: %s", sorted(_job_target_hint))
    except TypeError:
        _job_target_hint = None
        logger.warning("[target_registry] set_job_target_hint: invalid target_ids=%r", target_ids)


def get_job_target_hint() -> "set | None":
    """Return the current ambient job target hint (or None)."""
    return _job_target_hint


def _resolve_single_file(norm: str, candidate_ids=None) -> "str | None":
    """Resolve a single (normalised, lowercased, forward-slash) path to one
    target_id, or None if no profile matches.

    Tier 1: absolute path starts with a profile's project_root.
    Tier 2: project folder name appears as a path segment (e.g. /astra-bridge/).
    Tier 3: path_signals (distinctive relative directories) — scored, clear winner required.
    Tier 4: file_extension alone (only when exactly one profile's ext matches).

    If candidate_ids is provided, only profiles whose project_id is in that
    set are considered (used to scope resolution to the current job's targets).
    """
    pool = [p for p in _REGISTRY.values()
            if candidate_ids is None or p.project_id in candidate_ids]
    # Tier 1: absolute root match
    for profile in pool:
        root = profile.project_root.replace("\\", "/").rstrip("/").lower()
        if norm.startswith(root + "/") or norm == root:
            return profile.project_id
    # Tier 2: project-folder-name in path
    for profile in pool:
        root = profile.project_root.replace("\\", "/").rstrip("/").lower()
        seg = root.split("/")[-1]
        if seg and ("/" + seg + "/") in ("/" + norm + "/"):
            return profile.project_id
    # Tier 3: path_signals scoring (relative-path LLM output)
    sig_scores = {}
    for profile in pool:
        signals = getattr(profile, "path_signals", None) or []
        score = sum(3 for sig in signals if sig.lower() in norm)
        if score > 0:
            sig_scores[profile.project_id] = score
    if sig_scores:
        best_score = max(sig_scores.values())
        winners = [pid for pid, s in sig_scores.items() if s == best_score]
        if len(winners) == 1:
            return winners[0]
    # Tier 4: file_extension as last resort
    ext_winners = [p.project_id for p in pool
                   if p.file_extension and norm.endswith(p.file_extension.lower())]
    if len(ext_winners) == 1:
        return ext_winners[0]
    return None


def resolve_target_for_files(file_paths, job_target_ids=None) -> "tuple[str | None, set[str]]":
    """Resolve which registered target owns a list of file paths.

    Returns (target_id, all_hits) where:
      - target_id is the single target_id if all files belong to one target
        AND all files resolved cleanly.
      - target_id is None if files span multiple targets OR any file failed
        to resolve (genuinely ambiguous segment).
      - all_hits is the full set of target_ids touched (for diagnostics).

    Used by smart_segmentation (Phase 1 Job 5) to tag each segment with
    its owning target. Mixed segments (target_id=None) should be split
    along target lines before being written to the manifest.

    job_target_ids: optional set of project_ids to restrict resolution to.
        When the pipeline_bridge layer has already detected the set of
        targets for this job (Phase 0), passing it here avoids false
        matches against unrelated profiles.

    v1.0 (2026-04-11): Phase 1 Job 5 — initial absolute-path resolver.
    v1.1 (2026-04-12): Phase 1 Job 14 — added path_signals + extension
        fallback tiers for LLM-generated relative paths, plus job-scoped
        candidate restriction. Any unresolved file forces target_id=None.
    """
    # Fall back to ambient job hint when caller did not pass explicit scope.
    if job_target_ids is None and _job_target_hint is not None:
        job_target_ids = _job_target_hint
    hits = set()
    unresolved = 0
    for raw in file_paths:
        if not raw:
            continue
        norm = str(raw).replace("\\", "/").lower()
        tid = _resolve_single_file(norm, candidate_ids=job_target_ids)
        if tid is not None:
            hits.add(tid)
        else:
            unresolved += 1
    # If any file failed to resolve, flag the whole segment as mixed/ambiguous
    # rather than silently claiming a partial target.
    if unresolved > 0:
        return (None, hits)
    if len(hits) == 1:
        return (next(iter(hits)), hits)
    return (None, hits)
