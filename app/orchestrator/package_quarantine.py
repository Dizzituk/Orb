# FILE: app/orchestrator/package_quarantine.py
"""
Pre-execution quarantine for file->package refactors.

When a segmented job converts a .py file into a package directory
(e.g. architecture_executor.py -> architecture_executor/), the original
file must be moved out of the way BEFORE any segments execute.

The per-segment shadow check in architecture_executor.py v2.9 cannot
handle this because it only sees one segment's files at a time — the
__init__.py for the new package is typically in the LAST segment while
the files that need the directory exist in earlier segments.

This module scans the FULL manifest at job level, detects file->package
patterns, quarantines the originals via sandbox, and provides rollback
if the job fails.

v1.0 (2026-02-14): Initial implementation — job-level quarantine
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Set, Tuple
from app.orchestrator._package_quarantine_utils import FRONTEND_PREFIX, FRONTEND_ROOT, PACKAGE_QUARANTINE_BUILD_ID, QUARANTINE_DIR_NAME, QUARANTINE_SUFFIX, cleanup_quarantine, promote_quarantined_in_architecture, rollback_quarantine

logger = logging.getLogger(__name__)
print(f"[PACKAGE_QUARANTINE_LOADED] BUILD_ID={PACKAGE_QUARANTINE_BUILD_ID}")

# ── Constants ────────────────────────────────────────────────────────


# ── Data models ──────────────────────────────────────────────────────

@dataclass
class QuarantineEntry:
    """One file that was quarantined."""
    original_path: str          # Absolute path of original .py file
    quarantine_path: str        # Absolute path after rename (.pre_refactor)
    package_dir: str            # Absolute path of the package directory
    rel_module: str             # Relative module path (e.g. app/overwatcher/architecture_executor)
    status: str = "pending"     # pending | quarantined | restored | failed


@dataclass
class QuarantineResult:
    """Result of the quarantine operation."""
    entries: List[QuarantineEntry] = field(default_factory=list)
    directories_created: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def has_quarantined(self) -> bool:
        return any(e.status == "quarantined" for e in self.entries)

    @property
    def all_ok(self) -> bool:
        return len(self.errors) == 0

    @property
    def quarantined_rel_paths(self) -> Set[str]:
        """Return relative .py paths that were quarantined (normalised with forward slashes)."""
        return {
            (e.rel_module + ".py").replace("\\", "/")
            for e in self.entries
            if e.status == "quarantined"
        }


# ── Path resolution (duplicated from arch_executor to avoid import) ──

def _resolve_path(rel_path: str, sandbox_base: str) -> str:
    """Resolve a relative path to absolute, handling frontend/backend roots."""
    normalized = rel_path.replace("\\", "/")
    if normalized.startswith(FRONTEND_PREFIX):
        frontend_rel = normalized[len(FRONTEND_PREFIX):]
        return f"{FRONTEND_ROOT}\\{frontend_rel.replace('/', chr(92))}"
    return f"{sandbox_base}\\{normalized.replace('/', chr(92))}"


# ── Detection ────────────────────────────────────────────────────────

def detect_file_to_package_refactors(
    manifest_dict: dict,
) -> List[Tuple[str, str]]:
    """Scan all segments in the manifest for file->package refactor patterns.

    Looks for cases where:
      1. Multiple segments create files inside directory X/
      2. At least one segment creates X/__init__.py
      3. X.py currently exists (checked later via sandbox)

    v6.1 FIX 8: For deterministic refactor jobs (manifest has
    'deterministic_source'), the source monolith path is explicit.
    We use it directly instead of deriving from the package name,
    which handles the case where the LLM renamed the package.

    Args:
        manifest_dict: The full manifest dict with 'segments' key.

    Returns:
        List of (dir_segment, init_segment_id) tuples.
        dir_segment is the relative directory path (e.g. "app/overwatcher/architecture_executor").
        init_segment_id is the segment that creates the __init__.py.
    """
    segments = manifest_dict.get("segments", [])

    # Collect all file paths and which segment declares them
    all_files: List[str] = []
    init_owners: Dict[str, str] = {}  # dir_segment -> segment_id that has __init__.py

    for seg in segments:
        seg_id = seg.get("segment_id", "")
        for fpath in seg.get("file_scope", []):
            normalized = fpath.replace("\\", "/")
            all_files.append(normalized)

            # Check if this is an __init__.py
            if normalized.endswith("/__init__.py"):
                dir_segment = normalized.rsplit("/__init__.py", 1)[0]
                init_owners[dir_segment] = seg_id

    if not init_owners:
        return []

    # v6.1 FIX 8 + FIX 13: deterministic_source(s) override.
    # The manifest tells us exactly which file(s) to quarantine.
    # Support both list (deterministic_sources) and single (deterministic_source).
    det_sources = manifest_dict.get("deterministic_sources", [])
    if not det_sources:
        _single = manifest_dict.get("deterministic_source")
        if _single:
            det_sources = [_single]

    if det_sources:
        # v6.1 FIX 17c: Match by stem name, not parent directory.
        # Old logic used parent dir as dict key, which broke when two
        # sources shared the same parent (e.g. conduct_policy.py and
        # sandbox_build_validator.py both in app/overwatcher/).
        _pairs = []
        for det_source in det_sources:
            det_source_norm = det_source.replace("\\", "/")
            # Derive expected package: parent/stem  e.g. app/overwatcher/conduct_policy
            _stem = det_source_norm.rsplit("/", 1)[-1].replace(".py", "") if "/" in det_source_norm else det_source_norm.replace(".py", "")
            _parent = det_source_norm.rsplit("/", 1)[0] if "/" in det_source_norm else ""
            _expected_pkg = f"{_parent}/{_stem}" if _parent else _stem

            _matched = False
            if _expected_pkg in init_owners:
                # Direct stem match — most precise
                init_seg_id = init_owners[_expected_pkg]
                logger.info(
                    "[quarantine] v6.1 FIX 17c stem match: %s → package %s",
                    det_source, _expected_pkg,
                )
                _pairs.append((_expected_pkg, init_seg_id))
                _matched = True
            else:
                # Fallback: scan init_owners for any package whose stem matches
                for dir_segment, init_seg_id in init_owners.items():
                    pkg_stem = dir_segment.rsplit("/", 1)[-1] if "/" in dir_segment else dir_segment
                    if pkg_stem == _stem:
                        logger.info(
                            "[quarantine] v6.1 FIX 17c stem fallback: %s → package %s",
                            det_source, dir_segment,
                        )
                        _pairs.append((dir_segment, init_seg_id))
                        _matched = True
                        break
            if not _matched:
                logger.warning(
                    "[quarantine] v6.1 FIX 17c: no package found for source %s",
                    det_source,
                )
        if _pairs:
            return _pairs

    # For each directory that gets an __init__.py, check if other segments
    # also write files into it (confirming it's a real package, not just
    # a stray __init__.py)
    refactors: List[Tuple[str, str]] = []
    for dir_segment, init_seg_id in init_owners.items():
        prefix = dir_segment + "/"
        files_in_dir = [f for f in all_files if f.startswith(prefix)]
        # Need at least 2 files (the __init__.py + at least 1 module)
        if len(files_in_dir) >= 2:
            refactors.append((dir_segment, init_seg_id))

    return refactors


# ── Quarantine execution ─────────────────────────────────────────────

def run_quarantine(
    manifest_dict: dict,
    sandbox_base: str,
    client,  # SandboxClient instance
    on_progress=None,
) -> QuarantineResult:
    """Detect and quarantine files that will be replaced by packages.

    This must be called BEFORE any segments execute.

    Steps for each detected refactor:
      1. Check if {dir_segment}.py exists on disk via sandbox
      2. Move it into a .quarantined/ folder in the parent directory
      3. Create the package directory
      4. Record the quarantine for potential rollback

    Args:
        manifest_dict: Full manifest dict.
        sandbox_base: Resolved sandbox base path (e.g. "D:\\Orb").
        client: SandboxClient for filesystem operations.
        on_progress: Optional callback for status messages.

    Returns:
        QuarantineResult with entries and any errors.
    """
    _emit = on_progress or (lambda msg: None)
    result = QuarantineResult()

    refactors = detect_file_to_package_refactors(manifest_dict)
    if not refactors:
        logger.debug("[quarantine] No file->package refactors detected")
        return result

    logger.info(
        "[quarantine] Detected %d file->package refactor(s): %s",
        len(refactors),
        [r[0] for r in refactors],
    )
    _emit(f"[quarantine] Detected {len(refactors)} file->package refactor(s)")

    # v6.1 FIX 8 + FIX 13 + FIX 17c: Build lookup from package stem -> source file.
    # For multi-file refactors, each package has its own source monolith.
    # FIX 17c: Keyed by stem (not parent dir) to handle multiple sources
    # in the same parent directory.
    det_sources = manifest_dict.get("deterministic_sources", [])
    if not det_sources:
        _single = manifest_dict.get("deterministic_source")
        if _single:
            det_sources = [_single]

    # Build map: stem_name -> source_file  (e.g. "conduct_policy" -> "app/overwatcher/conduct_policy.py")
    _source_by_stem: Dict[str, str] = {}
    for ds in det_sources:
        ds_norm = ds.replace("\\", "/")
        _ds_stem = ds_norm.rsplit("/", 1)[-1].replace(".py", "") if "/" in ds_norm else ds_norm.replace(".py", "")
        _source_by_stem[_ds_stem] = ds_norm

    for dir_segment, init_seg_id in refactors:
        # v6.1 FIX 17c: Match by package stem name
        pkg_stem = dir_segment.rsplit("/", 1)[-1] if "/" in dir_segment else dir_segment
        _matched_source = _source_by_stem.get(pkg_stem)
        if _matched_source:
            module_py = _matched_source
            logger.info(
                "[quarantine] v6.1 FIX 17c stem match for quarantine: %s (package: %s)",
                module_py, dir_segment,
            )
        elif det_sources and len(det_sources) == 1:
            # Single source fallback
            module_py = det_sources[0].replace("\\", "/")
            logger.info(
                "[quarantine] v6.1 Using single deterministic_source fallback: %s (package: %s)",
                module_py, dir_segment,
            )
        else:
            module_py = dir_segment + ".py"
        abs_original = _resolve_path(module_py, sandbox_base)
        abs_package_dir = _resolve_path(dir_segment, sandbox_base)

        # v2.0: Quarantine into a .quarantined/ folder in the parent directory
        # e.g. D:\Orb\app\overwatcher\.quarantined\architecture_executor.py
        original_name = PureWindowsPath(abs_original).name
        parent_dir = str(PureWindowsPath(abs_original).parent)
        quarantine_dir = f"{parent_dir}\\.quarantined"
        abs_quarantine = f"{quarantine_dir}\\{original_name}"

        entry = QuarantineEntry(
            original_path=abs_original,
            quarantine_path=abs_quarantine,
            package_dir=abs_package_dir,
            rel_module=dir_segment,
        )
        result.entries.append(entry)

        # Step 1: Check if the original .py file exists
        try:
            check_cmd = (
                f'if (Test-Path -Path "{abs_original}" -PathType Leaf) '
                f'{{ "EXISTS" }} else {{ "NONE" }}'
            )
            check_result = client.shell_run(check_cmd, timeout_seconds=10)
            if not (check_result.stdout and "EXISTS" in check_result.stdout):
                logger.info(
                    "[quarantine] %s does not exist - no quarantine needed",
                    module_py,
                )
                entry.status = "skipped"
                _emit(f"  [INFO] {module_py} not found - skip quarantine")
                continue
        except Exception as e:
            error_msg = f"Failed to check {module_py}: {e}"
            logger.error("[quarantine] %s", error_msg)
            result.errors.append(error_msg)
            entry.status = "failed"
            continue

        # Step 2: Move original into .quarantined/ folder
        # v2.0: Use a folder instead of a rename suffix.
        # Move-Item -Force overwrites the target if it already exists.
        # We verify the source is gone after the move to catch silent failures.
        try:
            move_cmd = (
                f'$ErrorActionPreference = "Stop"; '
                f'$quarDir = "{quarantine_dir}"; '
                f'$src = "{abs_original}"; '
                f'$dst = "{abs_quarantine}"; '
                f'if (-not (Test-Path $quarDir)) {{ '
                f'  New-Item -Path $quarDir -ItemType Directory -Force | Out-Null '
                f'}}; '
                f'Move-Item -Path $src -Destination $dst -Force; '
                f'if (Test-Path $src) {{ "STILL_EXISTS" }} else {{ "MOVED" }}'
            )
            move_result = client.shell_run(move_cmd, timeout_seconds=15)
            stdout = (move_result.stdout or "").strip()
            stderr = (move_result.stderr or "").strip()

            if "MOVED" in stdout:
                entry.status = "quarantined"
                logger.info(
                    "[quarantine] QUARANTINED: %s -> %s",
                    abs_original, abs_quarantine,
                )
                _emit(
                    f"  [quarantine] Quarantined: {module_py} -> .quarantined/{original_name}"
                )
            elif "STILL_EXISTS" in stdout:
                error_msg = (
                    f"Move appeared to run but source still exists for {module_py}. "
                    f"stderr: {stderr}"
                )
                logger.error("[quarantine] %s", error_msg)
                result.errors.append(error_msg)
                entry.status = "failed"
                continue
            else:
                error_msg = (
                    f"Move failed for {module_py}: "
                    f"stdout={stdout}, stderr={stderr}"
                )
                logger.error("[quarantine] %s", error_msg)
                result.errors.append(error_msg)
                entry.status = "failed"
                continue
        except Exception as e:
            error_msg = f"Move exception for {module_py}: {e}"
            logger.error("[quarantine] %s", error_msg)
            result.errors.append(error_msg)
            entry.status = "failed"
            continue

        # Step 3: Create the package directory
        try:
            mkdir_cmd = (
                f'if (-not (Test-Path -Path "{abs_package_dir}" -PathType Container)) {{ '
                f'New-Item -Path "{abs_package_dir}" -ItemType Directory -Force | Out-Null; '
                f'"CREATED" }} else {{ "EXISTS" }}'
            )
            mkdir_result = client.shell_run(mkdir_cmd, timeout_seconds=10)
            if mkdir_result.stdout and ("CREATED" in mkdir_result.stdout or "EXISTS" in mkdir_result.stdout):
                result.directories_created.append(abs_package_dir)
                logger.info("[quarantine] Directory ready: %s", abs_package_dir)
                _emit(f"  [quarantine] Package directory created: {dir_segment}/")
            else:
                error_msg = (
                    f"mkdir failed for {dir_segment}/: "
                    f"{mkdir_result.stdout or mkdir_result.stderr or 'no output'}"
                )
                logger.warning("[quarantine] %s", error_msg)
                result.errors.append(error_msg)
        except Exception as e:
            error_msg = f"mkdir exception for {dir_segment}/: {e}"
            logger.error("[quarantine] %s", error_msg)
            result.errors.append(error_msg)

    # Summary
    quarantined = sum(1 for e in result.entries if e.status == "quarantined")
    skipped = sum(1 for e in result.entries if e.status == "skipped")
    failed = sum(1 for e in result.entries if e.status == "failed")

    if quarantined > 0:
        _emit(
            f"[OK] Quarantine complete: {quarantined} file(s) quarantined, "
            f"{len(result.directories_created)} dir(s) created"
            f"{f', {skipped} skipped' if skipped else ''}"
            f"{f', {failed} FAILED' if failed else ''}"
        )
    elif skipped > 0:
        _emit(f"[INFO] Quarantine: {skipped} file(s) already handled, nothing to do")

    return result


# ── Rollback ─────────────────────────────────────────────────────────


# ── Architecture text promotion ─────────────────────────────────────


# ── Cleanup (success path) ───────────────────────────────────────────
