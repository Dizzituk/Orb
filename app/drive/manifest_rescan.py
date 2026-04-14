# FILE: app/drive/manifest_rescan.py
"""
On-demand manifest rescan helpers.

Two entry points:
  - rescan_all(): full re-walk of all category paths (same as boot scan)
  - rescan_path(path): index/refresh a single file path now

These are callable from tools, endpoints, or background jobs — any caller
that suspects the manifest is stale without restarting the backend.

Also provides a lightweight `index_single_file` fallback if the one in
manifest_scanner.py is not present.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.db import SessionLocal
from app.drive.file_utils import (
    classify_file,
    get_category_paths,
    get_file_extension,
)
from app.drive.manifest_models import DriveFileManifest

logger = logging.getLogger(__name__)


def _category_for(path: str) -> str:
    """Match a path to one of the configured category roots."""
    p = Path(path).resolve()
    for cat_id, cat_path in get_category_paths().items():
        try:
            cat_resolved = cat_path.resolve()
        except Exception:
            continue
        try:
            p.relative_to(cat_resolved)
            return cat_id
        except ValueError:
            continue
    return "other"


def rescan_path(path: str) -> Dict[str, Any]:
    """
    Refresh a single file's row in the manifest.

    If the file exists on disk, the row is inserted or updated.
    If it doesn't exist, any existing row is deleted.
    """
    if not path:
        return {"status": "error", "error": "empty path"}

    db = SessionLocal()
    try:
        existing = db.query(DriveFileManifest).filter(
            DriveFileManifest.path == path
        ).first()

        if not os.path.isfile(path):
            # File gone — remove row if present
            if existing:
                db.delete(existing)
                db.commit()
                return {"status": "removed", "path": path}
            return {"status": "not_found", "path": path}

        try:
            stat = os.stat(path)
        except OSError as e:
            return {"status": "error", "path": path, "error": str(e)}

        fname = os.path.basename(path)
        ext = get_file_extension(fname)
        category = _category_for(path)
        file_class = classify_file(ext)
        mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        now = datetime.utcnow()

        if existing:
            existing.filename = fname
            existing.extension = ext
            existing.category = category
            existing.file_class = file_class
            existing.size_bytes = stat.st_size
            existing.mtime = mtime
            existing.last_seen_at = now
            existing.content_indexed = False
            action = "updated"
        else:
            rec = DriveFileManifest(
                path=path,
                filename=fname,
                extension=ext,
                category=category,
                file_class=file_class,
                size_bytes=stat.st_size,
                mtime=mtime,
                content_indexed=False,
                first_seen_at=now,
                last_seen_at=now,
                scan_generation=0,
            )
            db.add(rec)
            action = "created"

        db.commit()
        return {
            "status": action,
            "path": path,
            "category": category,
            "size": stat.st_size,
        }
    except Exception as e:
        db.rollback()
        logger.exception("[manifest_rescan] rescan_path failed: %s", path)
        return {"status": "error", "path": path, "error": str(e)}
    finally:
        db.close()


def rescan_all() -> Dict[str, Any]:
    """
    Run a full manifest scan right now (same logic as boot scan).

    Useful as a fallback when search_my_files returns empty results
    and the user knows the file should exist.
    """
    try:
        from app.drive.manifest_scanner import scan_manifest
    except Exception as e:
        return {"status": "error", "error": f"scanner import failed: {e}"}

    db = SessionLocal()
    try:
        report = scan_manifest(db)
        return {
            "status": "ok",
            "total_files": report.total_files,
            "new_files": report.new_files,
            "modified_files": report.modified_files,
            "deleted_files": report.deleted_files,
            "unchanged_files": report.unchanged_files,
            "duration_ms": report.duration_ms,
        }
    except Exception as e:
        logger.exception("[manifest_rescan] rescan_all failed")
        return {"status": "error", "error": str(e)}
    finally:
        db.close()
