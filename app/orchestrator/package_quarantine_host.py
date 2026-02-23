# FILE: app/orchestrator/package_quarantine_host.py
"""
Host-side quarantine for file->package refactors.

v6.1 FIX 17b: Direct filesystem operations on the HOST (D:/Orb),
replacing the sandbox-client approach that operated on the wrong machine.

The sandbox client (shell_run over HTTP to 192.168.250.2) was checking
file existence and moving files inside the Windows Sandbox, not on the
host. This meant monolith files on the host were never actually moved,
causing Python import conflicts when both the monolith .py file and the
new package directory coexisted.

This module uses os.path, shutil, and os.makedirs — simple, direct,
no network dependency, no sandbox state confusion.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import PureWindowsPath
from typing import Dict

from app.orchestrator.package_quarantine import (
    QUARANTINE_DIR_NAME,
    QuarantineEntry,
    QuarantineResult,
    _resolve_path,
    detect_file_to_package_refactors,
)

logger = logging.getLogger(__name__)

PACKAGE_QUARANTINE_HOST_BUILD_ID = "2026-02-21-v1.0-host-quarantine"
print(f"[PACKAGE_QUARANTINE_HOST_LOADED] BUILD_ID={PACKAGE_QUARANTINE_HOST_BUILD_ID}")


def run_host_quarantine(
    manifest_dict: dict,
    host_base: str = "D:\\Orb",
    on_progress=None,
) -> QuarantineResult:
    """Quarantine monolith files on the HOST filesystem using direct operations.

    v6.1 FIX 17b: Unlike run_quarantine() which uses the sandbox client
    (shell_run over HTTP to Windows Sandbox), this operates directly on
    the host at D:\\Orb using os/shutil. This is correct because the
    monoliths live on the host and must be moved there.

    Steps for each detected refactor:
      1. Check if {source}.py exists on host disk
      2. Create .quarantined/ folder in the parent directory
      3. Move the monolith into .quarantined/
      4. Create the package directory (if it doesn't exist)

    Args:
        manifest_dict: Full manifest dict.
        host_base: Host codebase root (default D:\\Orb).
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
        "[quarantine] FIX 17b host quarantine: %d refactor(s): %s",
        len(refactors), [r[0] for r in refactors],
    )
    _emit(f"\U0001f4e6 Host quarantine: {len(refactors)} file->package refactor(s)")

    # Build stem -> source lookup (same as run_quarantine FIX 17c)
    det_sources = manifest_dict.get("deterministic_sources", [])
    if not det_sources:
        _single = manifest_dict.get("deterministic_source")
        if _single:
            det_sources = [_single]

    _source_by_stem: Dict[str, str] = {}
    for ds in det_sources:
        ds_norm = ds.replace("\\", "/")
        _ds_stem = (
            ds_norm.rsplit("/", 1)[-1].replace(".py", "")
            if "/" in ds_norm else ds_norm.replace(".py", "")
        )
        _source_by_stem[_ds_stem] = ds_norm

    for dir_segment, _init_seg_id in refactors:
        # Resolve source file path
        pkg_stem = dir_segment.rsplit("/", 1)[-1] if "/" in dir_segment else dir_segment
        _matched_source = _source_by_stem.get(pkg_stem)
        if _matched_source:
            module_py = _matched_source
        elif det_sources and len(det_sources) == 1:
            module_py = det_sources[0].replace("\\", "/")
        else:
            module_py = dir_segment + ".py"

        # Resolve absolute paths on HOST
        abs_original = _resolve_path(module_py, host_base)
        abs_package_dir = _resolve_path(dir_segment, host_base)
        original_name = PureWindowsPath(abs_original).name
        parent_dir = str(PureWindowsPath(abs_original).parent)
        quarantine_dir = os.path.join(parent_dir, QUARANTINE_DIR_NAME)
        abs_quarantine = os.path.join(quarantine_dir, original_name)

        entry = QuarantineEntry(
            original_path=abs_original,
            quarantine_path=abs_quarantine,
            package_dir=abs_package_dir,
            rel_module=dir_segment,
        )
        result.entries.append(entry)

        # Step 1: Check existence
        if not os.path.isfile(abs_original):
            logger.info(
                "[quarantine] FIX 17b: %s does not exist on host — skip",
                module_py,
            )
            entry.status = "skipped"
            _emit(f"  [INFO] {module_py} not found on host — skip")
            continue

        # Step 2: Create .quarantined/ dir and move
        try:
            os.makedirs(quarantine_dir, exist_ok=True)
            shutil.move(abs_original, abs_quarantine)

            if not os.path.isfile(abs_original) and os.path.isfile(abs_quarantine):
                entry.status = "quarantined"
                logger.info(
                    "[quarantine] FIX 17b QUARANTINED: %s -> %s",
                    abs_original, abs_quarantine,
                )
                _emit(f"  \U0001f4e6 Quarantined: {module_py} \u2192 .quarantined/{original_name}")
            else:
                error_msg = f"Move appeared to run but source still exists: {module_py}"
                logger.error("[quarantine] FIX 17b: %s", error_msg)
                result.errors.append(error_msg)
                entry.status = "failed"
                continue
        except Exception as e:
            error_msg = f"Move failed for {module_py}: {e}"
            logger.error("[quarantine] FIX 17b: %s", error_msg)
            result.errors.append(error_msg)
            entry.status = "failed"
            continue

        # Step 3: Create package directory
        try:
            os.makedirs(abs_package_dir, exist_ok=True)
            result.directories_created.append(abs_package_dir)
            logger.info("[quarantine] FIX 17b directory ready: %s", abs_package_dir)
            _emit(f"  \U0001f4e6 Package directory: {dir_segment}/")
        except Exception as e:
            error_msg = f"mkdir failed for {dir_segment}/: {e}"
            logger.warning("[quarantine] FIX 17b: %s", error_msg)
            result.errors.append(error_msg)

    # Summary
    quarantined = sum(1 for e in result.entries if e.status == "quarantined")
    skipped = sum(1 for e in result.entries if e.status == "skipped")
    failed = sum(1 for e in result.entries if e.status == "failed")

    if quarantined > 0:
        _emit(
            f"\U0001f4e6 Host quarantine complete: {quarantined} file(s) moved, "
            f"{len(result.directories_created)} dir(s) created"
            f"{f', {skipped} skipped' if skipped else ''}"
            f"{f', {failed} FAILED' if failed else ''}"
        )
    elif skipped > 0:
        _emit(f"[INFO] Host quarantine: {skipped} file(s) already handled")

    return result


def rollback_host_quarantine(
    quarantine_result: QuarantineResult,
    on_progress=None,
) -> bool:
    """Restore host-quarantined files if the job fails.

    v6.1 FIX 17b: Direct filesystem rollback (no sandbox client).

    Returns:
        True if all rollbacks succeeded.
    """
    _emit = on_progress or (lambda msg: None)
    all_ok = True

    quarantined_entries = [
        e for e in quarantine_result.entries if e.status == "quarantined"
    ]
    if not quarantined_entries:
        return True

    _emit(f"[ROLLBACK] Rolling back {len(quarantined_entries)} host-quarantined file(s)...")

    for entry in quarantined_entries:
        # Remove empty package directory
        try:
            if os.path.isdir(entry.package_dir):
                contents = os.listdir(entry.package_dir)
                if not contents:
                    os.rmdir(entry.package_dir)
                    logger.info("[quarantine] FIX 17b rollback: removed empty %s", entry.package_dir)
        except Exception as e:
            logger.warning("[quarantine] FIX 17b rollback rmdir: %s", e)

        # Move file back
        try:
            if os.path.isfile(entry.quarantine_path):
                shutil.move(entry.quarantine_path, entry.original_path)
                if os.path.isfile(entry.original_path):
                    entry.status = "restored"
                    logger.info("[quarantine] FIX 17b RESTORED: %s", entry.original_path)
                    _emit(f"  [OK] Restored: {entry.rel_module}.py")
                else:
                    logger.error("[quarantine] FIX 17b restore move didn't work: %s", entry.original_path)
                    all_ok = False
            else:
                logger.warning("[quarantine] FIX 17b quarantine file missing: %s", entry.quarantine_path)
                all_ok = False
        except Exception as e:
            logger.error("[quarantine] FIX 17b restore error: %s", e)
            all_ok = False

    return all_ok
