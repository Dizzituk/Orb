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

logger = logging.getLogger(__name__)

PACKAGE_QUARANTINE_BUILD_ID = "2026-02-14-v2.0-quarantine-folder"
print(f"[PACKAGE_QUARANTINE_LOADED] BUILD_ID={PACKAGE_QUARANTINE_BUILD_ID}")

# ── Constants ────────────────────────────────────────────────────────

QUARANTINE_SUFFIX = ".quarantined"  # Legacy: kept for rollback compat
QUARANTINE_DIR_NAME = ".quarantined"  # Subfolder next to the original file
FRONTEND_PREFIX = "orb-desktop/"
FRONTEND_ROOT = r"D:\orb-desktop"


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

    for dir_segment, init_seg_id in refactors:
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

def rollback_quarantine(
    quarantine_result: QuarantineResult,
    client,  # SandboxClient instance
    on_progress=None,
) -> bool:
    """Restore quarantined files if the job fails.

    Steps for each quarantined entry:
      1. Remove the package directory (if empty or if we created it)
      2. Move file from .quarantined/ folder back to original location

    Args:
        quarantine_result: The result from run_quarantine().
        client: SandboxClient for filesystem operations.
        on_progress: Optional callback for status messages.

    Returns:
        True if all rollbacks succeeded, False if any failed.
    """
    _emit = on_progress or (lambda msg: None)
    all_ok = True

    quarantined_entries = [
        e for e in quarantine_result.entries if e.status == "quarantined"
    ]
    if not quarantined_entries:
        return True

    _emit(f"[ROLLBACK] Rolling back {len(quarantined_entries)} quarantined file(s)...")

    for entry in quarantined_entries:
        # Step 1: Remove the package directory if it's empty
        try:
            rmdir_cmd = (
                f'if (Test-Path -Path "{entry.package_dir}" -PathType Container) {{ '
                f'$items = Get-ChildItem -Path "{entry.package_dir}" -Force; '
                f'if ($items.Count -eq 0) {{ '
                f'Remove-Item -Path "{entry.package_dir}" -Force; "REMOVED" '
                f'}} else {{ "NOT_EMPTY" }} '
                f'}} else {{ "GONE" }}'
            )
            rmdir_result = client.shell_run(rmdir_cmd, timeout_seconds=10)
            dir_status = (rmdir_result.stdout or "").strip()
            logger.info(
                "[quarantine] Rollback dir %s: %s",
                entry.package_dir, dir_status,
            )
        except Exception as e:
            logger.warning(
                "[quarantine] Rollback rmdir failed for %s: %s",
                entry.package_dir, e,
            )

        # Step 2: Move from .quarantined/ folder back to original location
        try:
            restore_cmd = (
                f'$ErrorActionPreference = "Stop"; '
                f'$src = "{entry.quarantine_path}"; '
                f'$dst = "{entry.original_path}"; '
                f'if (Test-Path $src) {{ '
                f'  Move-Item -Path $src -Destination $dst -Force; '
                f'  if (Test-Path $dst) {{ "RESTORED" }} else {{ "MOVE_FAILED" }} '
                f'}} else {{ "MISSING" }}'
            )
            restore_result = client.shell_run(restore_cmd, timeout_seconds=15)
            stdout = (restore_result.stdout or "").strip()
            if "RESTORED" in stdout:
                entry.status = "restored"
                logger.info(
                    "[quarantine] RESTORED: %s",
                    entry.original_path,
                )
                _emit(f"  [OK] Restored: {entry.rel_module}.py")
            else:
                logger.error(
                    "[quarantine] Restore failed for %s: stdout=%s stderr=%s",
                    entry.original_path, stdout,
                    (restore_result.stderr or "").strip(),
                )
                _emit(f"  [ERROR] Restore failed: {entry.rel_module}.py")
                all_ok = False
        except Exception as e:
            logger.error(
                "[quarantine] Restore exception for %s: %s",
                entry.original_path, e,
            )
            all_ok = False

    return all_ok


# ── Architecture text promotion ─────────────────────────────────────

def promote_quarantined_in_architecture(
    arch_text: str,
    quarantined_paths: Set[str],
) -> str:
    """Move quarantined files from Modified Files to New Files in architecture markdown.

    When a file has been quarantined (renamed to .pre_refactor), the Implementer
    can't MODIFY it because it no longer exists at the original path. This function rewrites the
    File Inventory section to list the file under New Files instead.

    Args:
        arch_text: The architecture markdown content.
        quarantined_paths: Set of relative paths that were quarantined
                           (e.g. {"app/overwatcher/architecture_executor.py"}).

    Returns:
        Modified architecture text, or original if no changes needed.
    """
    if not quarantined_paths:
        return arch_text

    import re

    modified = arch_text
    promoted = []

    for qpath in quarantined_paths:
        # Normalise for matching: the arch text may use forward or backslashes
        qpath_fwd = qpath.replace("\\", "/")
        qpath_bk = qpath.replace("/", "\\")

        # Find the row in Modified Files table that contains this path
        # Pattern: | `path` | description |
        pattern = re.compile(
            r'(\|\s*`?' + re.escape(qpath_fwd) + r'`?\s*\|[^\n]*\n)',
            re.IGNORECASE,
        )
        if not pattern.search(modified):
            # Try backslash variant
            pattern = re.compile(
                r'(\|\s*`?' + re.escape(qpath_bk) + r'`?\s*\|[^\n]*\n)',
                re.IGNORECASE,
            )

        match = pattern.search(modified)
        if not match:
            continue

        row_text = match.group(1)

        # Check this row is under Modified Files (not already under New Files)
        # Find the position and look backwards for the section header
        pos = match.start()
        preceding = modified[:pos]
        last_new = preceding.rfind("New Files")
        last_mod = preceding.rfind("Modified Files")
        if last_mod < 0 or last_new > last_mod:
            # Already under New Files or can't determine — skip
            continue

        # Remove row from Modified Files
        modified = modified[:match.start()] + modified[match.end():]

        # Add row to New Files table
        # Strategy: find the header separator line (|------|--------|) under New Files,
        # then insert our row right after it (before any existing rows or blank lines)
        # Find "New Files" table: header row + separator (with dashes)
        # Separator must contain actual dashes to distinguish from empty data rows
        # Separator line must contain at least 3 dashes (distinguishes from empty data rows)
        nf_header = re.search(
            r'(New Files[^\n]*\n\|[^\n]+\n\|[^\n]*---[^\n]*\n)',
            modified,
            re.DOTALL,
        )
        if nf_header:
            insert_pos = nf_header.end()
            # Skip any empty placeholder rows like "|  |  |" (no dashes, just spaces)
            rest = modified[insert_pos:]
            # Match rows that are effectively empty (only whitespace and pipes)
            empty_rows = re.match(r'((?:\|[\s]*\|[\s]*\|\s*\n)*)', rest)
            if empty_rows and empty_rows.group(1):
                # Replace empty placeholder(s) with our real row
                modified = modified[:insert_pos] + row_text + modified[insert_pos + empty_rows.end():]
            else:
                modified = modified[:insert_pos] + row_text + modified[insert_pos:]
        else:
            # Can't find New Files table — log and skip
            logger.warning("[quarantine] Could not find New Files table for promotion of %s", qpath)

        promoted.append(qpath_fwd)

    if promoted:
        logger.info(
            "[quarantine] Promoted %d quarantined file(s) from MODIFY->CREATE in architecture: %s",
            len(promoted), promoted,
        )

    return modified


# ── Cleanup (success path) ───────────────────────────────────────────

def cleanup_quarantine(
    quarantine_result: QuarantineResult,
    client,  # SandboxClient instance
    on_progress=None,
) -> None:
    """Remove .quarantined/ backup folders after successful job completion.

    Only call this when ALL segments have completed successfully.

    Args:
        quarantine_result: The result from run_quarantine().
        client: SandboxClient for filesystem operations.
        on_progress: Optional callback for status messages.
    """
    _emit = on_progress or (lambda msg: None)

    quarantined_entries = [
        e for e in quarantine_result.entries if e.status == "quarantined"
    ]
    if not quarantined_entries:
        return

    # Collect unique quarantine directories to remove
    quarantine_dirs = set()
    for entry in quarantined_entries:
        qdir = str(PureWindowsPath(entry.quarantine_path).parent)
        quarantine_dirs.add(qdir)

    for qdir in quarantine_dirs:
        try:
            delete_cmd = (
                f'if (Test-Path -Path "{qdir}" -PathType Container) {{ '
                f'Remove-Item -Path "{qdir}" -Recurse -Force; '
                f'"DELETED" }} else {{ "GONE" }}'
            )
            delete_result = client.shell_run(delete_cmd, timeout_seconds=10)
            status = (delete_result.stdout or "").strip()
            if status in ("DELETED", "GONE"):
                logger.info("[quarantine] Cleanup: removed %s", qdir)
            else:
                logger.warning("[quarantine] Cleanup uncertain for %s: %s", qdir, status)
        except Exception as e:
            logger.warning("[quarantine] Cleanup failed for %s: %s", qdir, e)

    _emit(
        f"[CLEANUP] Quarantine cleanup: {len(quarantine_dirs)} .quarantined/ folder(s) removed"
    )
