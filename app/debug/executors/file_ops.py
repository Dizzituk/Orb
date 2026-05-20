# FILE: app/debug/executors/file_ops.py
"""
File operation executors: move_file, create_folder, move_files_batch.

All operations are restricted to user folders (Documents, Pictures, Desktop,
Downloads, Music, Videos, OneDrive, etc.) plus D:/Orb/output. Cannot touch
ASTRA's protected codebase paths or Windows system folders.

No delete operations - destructive ops require a separate, explicitly-confirmed
flow (per Taz's hard rules). Move/rename covers ~99% of organisation needs.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

from app.debug.executors._paths import is_host_write_blocked

logger = logging.getLogger(__name__)


def _allowed_roots() -> List[Path]:
    """Resolved list of paths a user is allowed to write to."""
    roots: List[Path] = []
    try:
        from app.drive.file_utils import get_category_paths
        roots.extend(Path(p) for p in get_category_paths().values())
    except Exception:
        pass
    # Always include outputs and debug uploads as safe targets
    roots.append(Path("D:/Orb/output"))
    roots.append(Path("D:/Orb/data/debug_uploads"))
    return roots


def _path_inside_allowed(target: Path, roots: List[Path]) -> bool:
    """Return True if target is inside any allowed root."""
    try:
        target_resolved = target.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            target_resolved.relative_to(root.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def _validate_path_for_write(path_str: str) -> tuple[bool, str]:
    """Return (ok, error_message). If ok is True, error_message is empty."""
    if not path_str:
        return False, "Error: path is required."

    p = Path(path_str)
    roots = _allowed_roots()

    # Allow if inside any safe root, OR if parent exists and is inside a safe root
    # (lets you create new subfolders under e.g. Documents)
    if not _path_inside_allowed(p, roots):
        if not _path_inside_allowed(p.parent, roots):
            return False, (
                f"Access denied: {path_str} is outside allowed user folders. "
                f"Use get_user_folders to see valid base paths."
            )

    # Belt and braces - block protected ASTRA dirs
    try:
        resolved = str(p.resolve())
    except Exception:
        resolved = str(p)
    if is_host_write_blocked(resolved):
        return False, (
            f"Access denied: {path_str} is inside a protected ASTRA directory. "
            f"Host-side code is read-only; changes must go through the sandbox."
        )

    return True, ""


# =============================================================================
# CREATE FOLDER
# =============================================================================

async def execute_create_folder(params: Dict[str, Any]) -> str:
    """Create a directory (with parents) inside an allowed user folder."""
    path = params.get("path", "").strip()
    ok, err = _validate_path_for_write(path)
    if not ok:
        return err

    target = Path(path)
    if target.exists():
        if target.is_dir():
            return f"Folder already exists: {path}"
        return f"Error: {path} exists but is not a directory."

    try:
        target.mkdir(parents=True, exist_ok=True)
        logger.info("[executors.file_ops] Created folder: %s", path)
        return f"Created folder: {path}"
    except PermissionError:
        return f"Permission denied creating folder: {path}"
    except Exception as e:
        return f"Folder creation failed: {e}"


# =============================================================================
# MOVE FILE
# =============================================================================

async def execute_move_file(params: Dict[str, Any]) -> str:
    """Move (or rename) a single file. Both source and destination must be in user folders."""
    source = params.get("source", "").strip()
    destination = params.get("destination", "").strip()
    overwrite = bool(params.get("overwrite", False))

    if not source or not destination:
        return "Error: both source and destination are required."

    src_ok, src_err = _validate_path_for_write(source)
    if not src_ok:
        return f"Source path rejected: {src_err}"
    dst_ok, dst_err = _validate_path_for_write(destination)
    if not dst_ok:
        return f"Destination path rejected: {dst_err}"

    src = Path(source)
    dst = Path(destination)

    if not src.exists():
        return f"Source file not found: {source}"
    if not src.is_file():
        return f"Source is not a file (use create_folder for directories): {source}"

    if dst.exists() and not overwrite:
        return (
            f"Destination already exists: {destination}. "
            f"Pass overwrite=true to replace it, or choose a different name."
        )

    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        logger.info("[executors.file_ops] Moved: %s -> %s", source, destination)

        # Best-effort manifest update so search_my_files reflects the move
        try:
            from app.drive.manifest_scanner import index_single_file
            index_single_file(str(dst))
        except Exception:
            pass

        return f"Moved: {source} -> {destination}"
    except PermissionError:
        return f"Permission denied moving {source}"
    except Exception as e:
        return f"Move failed: {e}"


# =============================================================================
# MOVE FILES BATCH
# =============================================================================

async def execute_move_files_batch(params: Dict[str, Any]) -> str:
    """Move many files in a single tool call.

    Skip-and-continue semantics: a failure on one file does not abort the batch.
    Returns a summary with succeeded count and a list of failures.
    """
    moves = params.get("moves", [])
    overwrite = bool(params.get("overwrite", False))

    if not isinstance(moves, list) or not moves:
        return "Error: moves must be a non-empty list of {source, destination} objects."

    succeeded: List[str] = []
    failed: List[Dict[str, str]] = []

    # Pre-validate everything before touching disk so we fail fast on a bad batch
    for i, m in enumerate(moves):
        if not isinstance(m, dict):
            return f"Error: moves[{i}] is not an object."
        if "source" not in m or "destination" not in m:
            return f"Error: moves[{i}] missing 'source' or 'destination'."

    try:
        from app.drive.manifest_scanner import index_single_file
        manifest_update_available = True
    except Exception:
        index_single_file = None  # type: ignore
        manifest_update_available = False

    for m in moves:
        source = (m.get("source") or "").strip()
        destination = (m.get("destination") or "").strip()

        src_ok, src_err = _validate_path_for_write(source)
        if not src_ok:
            failed.append({"source": source, "destination": destination, "error": f"src: {src_err}"})
            continue
        dst_ok, dst_err = _validate_path_for_write(destination)
        if not dst_ok:
            failed.append({"source": source, "destination": destination, "error": f"dst: {dst_err}"})
            continue

        src = Path(source)
        dst = Path(destination)

        if not src.exists():
            failed.append({"source": source, "destination": destination, "error": "source not found"})
            continue
        if not src.is_file():
            failed.append({"source": source, "destination": destination, "error": "source is not a file"})
            continue
        if dst.exists() and not overwrite:
            failed.append({"source": source, "destination": destination, "error": "destination exists (overwrite=false)"})
            continue

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            succeeded.append(destination)
            if manifest_update_available and index_single_file is not None:
                try:
                    index_single_file(str(dst))
                except Exception:
                    pass
        except Exception as e:
            failed.append({"source": source, "destination": destination, "error": str(e)})

    logger.info(
        "[executors.file_ops] Batch move complete: %d succeeded, %d failed",
        len(succeeded), len(failed),
    )

    lines = [f"Batch move complete: {len(succeeded)} succeeded, {len(failed)} failed."]
    if failed:
        lines.append("")
        lines.append("Failures:")
        # Cap reported failures at 50 so the response stays readable
        for f in failed[:50]:
            lines.append(f"  {f['source']} -> {f['destination']}: {f['error']}")
        if len(failed) > 50:
            lines.append(f"  ... and {len(failed) - 50} more.")
    return "\n".join(lines)
