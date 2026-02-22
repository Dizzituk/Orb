from __future__ import annotations
import logging
from app.orchestrator.package_quarantine import QuarantineResult, logger
from pathlib import PureWindowsPath
from typing import Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


PACKAGE_QUARANTINE_BUILD_ID = "2026-02-21-v3.0-host-quarantine-stem-match"

QUARANTINE_SUFFIX = ".quarantined"  # Legacy: kept for rollback compat

QUARANTINE_DIR_NAME = ".quarantined"  # Subfolder next to the original file

FRONTEND_PREFIX = "orb-desktop/"

FRONTEND_ROOT = r"D:\orb-desktop"

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
