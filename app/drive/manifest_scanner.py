# FILE: app/drive/manifest_scanner.py
"""
Filesystem manifest scanner — Tier 1 of the boot scan.

Walks all category paths, records file metadata, and compares
against the previous boot's manifest to detect changes.

Designed to be FAST (<5s for ~17,000 files). No content reading,
no API calls, no embeddings. Pure filesystem stat operations.

Called from: app/drive/boot_scan.py → run_boot_scan()
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

from sqlalchemy.orm import Session as DbSession

from app.drive.file_utils import (
    get_category_paths,
    get_file_extension,
    classify_file,
)
from app.drive.manifest_models import DriveFileManifest

logger = logging.getLogger(__name__)

# Skip directories that are internal / build artifacts
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    ".gradle", ".idea", ".vs", "$RECYCLE.BIN", "System Volume Information",
}

# Skip hidden files and system files
SKIP_PREFIXES = (".", "~$", "Thumbs.db", "desktop.ini")


@dataclass
class ScanReport:
    """Result of a manifest scan."""
    timestamp: str = ""
    scan_generation: int = 0
    duration_ms: int = 0
    total_files: int = 0
    new_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    unchanged_files: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def _walk_category(
    cat_id: str,
    cat_path: Path,
) -> List[dict]:
    """
    Walk a category directory and return file metadata dicts.

    Returns list of dicts with keys:
        path, filename, extension, category, file_class, size_bytes, mtime
    """
    files = []
    if not cat_path.exists():
        return files

    try:
        for root, dirs, filenames in os.walk(str(cat_path)):
            # Prune skip directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

            for fname in filenames:
                if any(fname.startswith(p) for p in SKIP_PREFIXES):
                    continue

                filepath = os.path.join(root, fname)
                try:
                    stat = os.stat(filepath)
                    ext = get_file_extension(fname)
                    files.append({
                        "path": filepath,
                        "filename": fname,
                        "extension": ext,
                        "category": cat_id,
                        "file_class": classify_file(ext),
                        "size_bytes": stat.st_size,
                        "mtime": datetime.fromtimestamp(
                            stat.st_mtime, tz=timezone.utc
                        ),
                    })
                except (OSError, PermissionError):
                    continue
    except (OSError, PermissionError) as e:
        logger.warning("[manifest_scanner] Error walking %s: %s", cat_path, e)

    return files


def scan_manifest(db: DbSession) -> ScanReport:
    """
    Scan all category paths and update the drive_file_manifest table.

    Detects new, modified, deleted, and unchanged files by comparing
    current filesystem state against DB records.

    Returns a ScanReport with change counts.
    """
    start = time.monotonic()
    report = ScanReport(timestamp=datetime.utcnow().isoformat())

    # Determine scan generation (increment from latest)
    latest_gen = db.query(
        DriveFileManifest.scan_generation
    ).order_by(DriveFileManifest.scan_generation.desc()).first()
    generation = (latest_gen[0] + 1) if latest_gen else 1
    report.scan_generation = generation

    # ── Phase 1: Walk filesystem ─────────────────────────────────
    disk_files: Dict[str, dict] = {}
    cat_paths = get_category_paths()

    for cat_id, cat_path in cat_paths.items():
        files = _walk_category(cat_id, cat_path)
        for f in files:
            disk_files[f["path"]] = f
        report.categories[cat_id] = len(files)

    report.total_files = len(disk_files)

    # ── Phase 2: Load existing manifest from DB ──────────────────
    db_records: Dict[str, DriveFileManifest] = {}
    for rec in db.query(DriveFileManifest).all():
        db_records[rec.path] = rec

    disk_paths = set(disk_files.keys())
    db_paths = set(db_records.keys())

    # ── Phase 3: Detect changes ──────────────────────────────────

    # New files (on disk, not in DB)
    new_paths = disk_paths - db_paths
    for path in new_paths:
        f = disk_files[path]
        rec = DriveFileManifest(
            path=f["path"],
            filename=f["filename"],
            extension=f["extension"],
            category=f["category"],
            file_class=f["file_class"],
            size_bytes=f["size_bytes"],
            mtime=f["mtime"],
            content_indexed=False,
            first_seen_at=datetime.utcnow(),
            last_seen_at=datetime.utcnow(),
            scan_generation=generation,
        )
        db.add(rec)
    report.new_files = len(new_paths)

    # Deleted files (in DB, not on disk)
    deleted_paths = db_paths - disk_paths
    if deleted_paths:
        db.query(DriveFileManifest).filter(
            DriveFileManifest.path.in_(deleted_paths)
        ).delete(synchronize_session="fetch")
    report.deleted_files = len(deleted_paths)

    # Existing files — check for modifications
    common_paths = disk_paths & db_paths
    for path in common_paths:
        f = disk_files[path]
        rec = db_records[path]

        # Check if modified (size or mtime changed)
        size_changed = rec.size_bytes != f["size_bytes"]
        mtime_changed = (
            rec.mtime is None
            or abs((rec.mtime.replace(tzinfo=timezone.utc) - f["mtime"]).total_seconds()) > 2
        )

        if size_changed or mtime_changed:
            rec.size_bytes = f["size_bytes"]
            rec.mtime = f["mtime"]
            rec.content_indexed = False  # Mark for re-indexing
            rec.last_seen_at = datetime.utcnow()
            rec.scan_generation = generation
            report.modified_files += 1
        else:
            rec.last_seen_at = datetime.utcnow()
            rec.scan_generation = generation
            report.unchanged_files += 1

    db.commit()

    elapsed = int((time.monotonic() - start) * 1000)
    report.duration_ms = elapsed

    logger.info(
        "[manifest_scanner] Scan complete: %d files, %d new, %d modified, "
        "%d deleted, %d unchanged (%dms)",
        report.total_files, report.new_files, report.modified_files,
        report.deleted_files, report.unchanged_files, elapsed,
    )

    return report
