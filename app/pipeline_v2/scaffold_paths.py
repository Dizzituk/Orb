# FILE: app/pipeline_v2/scaffold_paths.py
# Purpose: Scaffold Engine path/existence resolution (non-destructive guard + basename redirect).
# Called-by: app.pipeline_v2.scaffold_engine (shim)
# Depends-on: app.pipeline_v2.sandbox_tools (lazy), app.pipeline_v2.build_targets (type-only)
# Last-renovated: 2026-06-21
"""
Scaffold Engine path/existence resolution.

Split out of scaffold_engine.py (BATCH 4) verbatim. Pure filesystem reasoning:
host-existence checks (v2.3 non-destructive guard) and basename-aware path
redirection (v2.4). Never reads or generates templates.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from app.pipeline_v2.build_targets import BuildTargetProfile

logger = logging.getLogger(__name__)


def _exists_on_host(
    file_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> bool:
    """Return True if `file_path` (as scaffold would write it) already
    exists on the host filesystem.

    Resolves relative paths against the profile's project_root via
    sandbox_tools._resolve_path, then calls os.path.exists. Works for
    all target types because host-mode writes and sandbox-mode writes
    both land on host-visible disk (the sandbox mounts the host repo).

    Fails safe: any resolution or IO error returns False, so the
    scaffold engine will attempt the write normally (preserving
    existing behaviour rather than mysteriously skipping files).
    """
    try:
        from app.pipeline_v2.sandbox_tools import _resolve_path
        abs_path = _resolve_path(file_path, profile)
        return os.path.exists(abs_path.replace("/", os.sep))
    except Exception as e:
        logger.debug(
            "[scaffold_engine] _exists_on_host check failed for %s: %s — defaulting to False",
            file_path, e,
        )
        return False


def _resolve_for_log(
    file_path: str,
    profile: Optional["BuildTargetProfile"] = None,
) -> str:
    """Resolve a relative path to its absolute host path for log messages.

    Separate from _exists_on_host so log formatting stays resilient even
    if resolution throws.
    """
    try:
        from app.pipeline_v2.sandbox_tools import _resolve_path
        return _resolve_path(file_path, profile)
    except Exception:
        return file_path


_WALK_SKIP_DIRS = frozenset({
    ".git", ".gradle", ".idea", ".vscode", "build", ".build",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env", ".env", "dist", "out", "target",
    "bin", "obj", ".next", ".nuxt", ".cache", "coverage",
})


def _build_project_basename_index(project_root: str) -> Dict[str, List[str]]:
    """Walk the project tree once and build a {basename: [relative_path, ...]}
    index.

    Paths are returned relative to project_root, forward-slash normalised.
    Well-known noise directories (build output, VCS, vendored deps) are
    skipped. Multiple files with the same basename are all kept in the
    list — the caller decides how to handle ambiguity.

    Returns an empty dict if the root doesn't exist or can't be read.
    """
    index: Dict[str, List[str]] = {}
    root_os = project_root.replace("/", os.sep)
    if not os.path.isdir(root_os):
        logger.debug(
            "[scaffold_engine] v2.4 basename index: root not a directory: %s",
            root_os,
        )
        return index

    try:
        for dirpath, dirnames, filenames in os.walk(root_os):
            # In-place prune to skip noise dirs. os.walk respects dirnames edits.
            dirnames[:] = [d for d in dirnames if d not in _WALK_SKIP_DIRS]
            for fname in filenames:
                abs_file = os.path.join(dirpath, fname)
                rel = os.path.relpath(abs_file, root_os).replace(os.sep, "/")
                index.setdefault(fname.lower(), []).append(rel)
    except Exception as e:
        logger.warning(
            "[scaffold_engine] v2.4 basename index walk failed for %s: %s",
            root_os, e,
        )
        return {}

    logger.debug(
        "[scaffold_engine] v2.4 basename index built for %s: %d unique basenames, %d total files",
        root_os, len(index), sum(len(v) for v in index.values()),
    )
    return index


def _maybe_redirect_to_existing_path(
    spec_path: str,
    profile: "BuildTargetProfile",
    basename_index_cache: Dict[str, Dict[str, List[str]]],
) -> Optional[str]:
    """If `spec_path` is an unrooted/ambiguous path and its basename matches
    exactly one existing file in the target project tree, return the
    existing relative path. Otherwise return None.

    Rules:
      - Absolute paths (drive letter prefix) are left alone — the caller
        clearly wanted that exact location.
      - Paths that already contain a directory separator are only
        redirected when the resolver's default would place them somewhere
        clearly wrong (checked by: does the spec path itself exist on
        disk relative to project root? if yes, no redirect needed; if no,
        try basename lookup).
      - Bare basenames (no slashes) always get basename lookup.
      - Multiple basename matches → None (ambiguous, caller keeps spec path).
      - Zero matches → None (truly new file, caller keeps spec path).
    """
    try:
        norm = (spec_path or "").replace("\\", "/").strip()
        if not norm:
            return None
        # Absolute path — trust the caller, don't redirect.
        if len(norm) > 1 and norm[1] == ":":
            return None

        basename = norm.rsplit("/", 1)[-1]
        if not basename or "." not in basename:
            # No usable basename (e.g. directory-ish path)
            return None

        project_root = (profile.project_root or "").replace("\\", "/").rstrip("/")
        if not project_root:
            return None

        # If the spec path already resolves to something that exists on disk,
        # no need to redirect — the path is already correct.
        try:
            candidate_abs = profile.resolve_path(norm).replace("/", os.sep)
            if os.path.exists(candidate_abs):
                return None
        except Exception:
            pass  # fall through to basename lookup

        # Build / fetch the basename index for this target, cached per run.
        cache_key = profile.project_id or project_root
        index = basename_index_cache.get(cache_key)
        if index is None:
            index = _build_project_basename_index(project_root)
            basename_index_cache[cache_key] = index

        matches = index.get(basename.lower(), [])
        if len(matches) == 0:
            return None  # truly new file
        if len(matches) > 1:
            logger.info(
                "[scaffold_engine] v2.4 basename '%s' ambiguous in %s — "
                "%d matches (%s) — no redirect applied",
                basename, profile.project_id, len(matches),
                ", ".join(matches[:4]) + ("…" if len(matches) > 4 else ""),
            )
            return None

        # Exactly one match — redirect.
        return matches[0]
    except Exception as e:
        logger.debug(
            "[scaffold_engine] v2.4 redirect helper failed for '%s': %s — no redirect",
            spec_path, e,
        )
        return None
