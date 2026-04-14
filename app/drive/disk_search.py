# FILE: app/drive/disk_search.py
"""
Live filesystem search — bypasses the manifest cache.

Walks the actual disk for filename matches across the user's category
paths. Slower than search_my_files (which queries cached SQLite) but
authoritative: if a file exists, this finds it.

Designed as the fallback when search_my_files returns no results for
a file the user insists exists. Auto-heals the manifest by reindexing
any matches it finds.

Performance: a single category walk for a typical user-folders setup
runs in 100-500ms depending on tree depth. Hard cap of MAX_RESULTS
prevents pathological queries from hanging.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.drive.file_utils import (
    classify_file,
    get_category_paths,
    get_file_extension,
)

logger = logging.getLogger(__name__)

# Hard caps to prevent runaway searches
MAX_RESULTS = 50
MAX_FILES_SCANNED = 100_000

# Same skip list the boot scanner and watcher use
SKIP_DIR_NAMES = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    ".gradle", ".idea", ".vs", "$RECYCLE.BIN",
    "System Volume Information",
}
SKIP_PREFIXES = (".", "~$", "Thumbs.db", "desktop.ini")


def _matches(filename: str, query_lower: str) -> bool:
    """Substring match, case-insensitive. Same semantics as search_my_files."""
    return query_lower in filename.lower()


def search_disk(
    query: str,
    category: Optional[str] = None,
    extension: Optional[str] = None,
    max_results: int = MAX_RESULTS,
) -> Dict[str, Any]:
    """
    Walk the filesystem for files matching the query.

    Args:
        query: filename substring (case-insensitive). Required.
        category: optional category id to limit the walk (e.g. "documents").
                  If None, walks all user categories.
        extension: optional file extension filter (without dot).
        max_results: cap on returned matches (default 50, hard max 50).

    Returns:
        {
            "status": "ok",
            "results": [{path, filename, category, size, ext}, ...],
            "scanned": <int>,        # files inspected
            "elapsed_ms": <int>,
            "truncated": <bool>,     # True if MAX_RESULTS hit
        }
    """
    import time
    start = time.monotonic()

    if not query or not query.strip():
        return {"status": "error", "error": "query is required"}

    query_lower = query.strip().lower()
    ext_lower = (extension or "").strip().lower().lstrip(".")
    cap = min(max_results, MAX_RESULTS)

    paths = get_category_paths()

    # Optionally limit to one category
    if category:
        category = category.strip().lower()
        if category not in paths:
            return {
                "status": "error",
                "error": f"unknown category {category!r}; valid: {sorted(paths.keys())}",
            }
        paths = {category: paths[category]}

    results: List[Dict[str, Any]] = []
    scanned = 0
    truncated = False

    for cat_id, cat_path in paths.items():
        if not cat_path.exists():
            continue
        if len(results) >= cap or scanned >= MAX_FILES_SCANNED:
            break

        try:
            for root, dirs, files in os.walk(str(cat_path)):
                # Prune skip dirs in-place so os.walk doesn't descend
                dirs[:] = [d for d in dirs if d not in SKIP_DIR_NAMES]

                for fname in files:
                    scanned += 1
                    if scanned >= MAX_FILES_SCANNED:
                        truncated = True
                        break
                    if any(fname.startswith(p) for p in SKIP_PREFIXES):
                        continue
                    if not _matches(fname, query_lower):
                        continue
                    file_ext = get_file_extension(fname)
                    if ext_lower and file_ext != ext_lower:
                        continue

                    fpath = os.path.join(root, fname)
                    try:
                        stat = os.stat(fpath)
                    except OSError:
                        continue

                    results.append({
                        "path": fpath,
                        "filename": fname,
                        "category": cat_id,
                        "size_bytes": stat.st_size,
                        "extension": file_ext,
                        "file_class": classify_file(file_ext),
                    })

                    if len(results) >= cap:
                        truncated = True
                        break

                if len(results) >= cap or scanned >= MAX_FILES_SCANNED:
                    break
        except (OSError, PermissionError) as e:
            logger.warning("[disk_search] error walking %s: %s", cat_path, e)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Best-effort: heal the manifest for anything we found that wasn't
    # already there. Failures here are non-fatal — the search result is
    # still returned.
    healed = 0
    if results:
        try:
            from app.drive.manifest_rescan import rescan_path
            for r in results:
                outcome = rescan_path(r["path"])
                if outcome.get("status") in ("created", "updated"):
                    healed += 1
        except Exception as e:
            logger.warning("[disk_search] manifest heal failed: %s", e)

    logger.info(
        "[disk_search] query=%r found=%d scanned=%d healed=%d in %dms",
        query, len(results), scanned, healed, elapsed_ms,
    )

    return {
        "status": "ok",
        "results": results,
        "scanned": scanned,
        "elapsed_ms": elapsed_ms,
        "truncated": truncated,
        "manifest_healed": healed,
    }
