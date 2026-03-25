# FILE: app/sandbox_walk.py
"""
Recursive directory walker for the sandbox environment.

Replaces os.walk() for all codebase scanning operations. Uses the
sandbox_fs API to list files, ensuring ASTRA's code awareness comes
from the sandbox clone, not the host.

CRITICAL RULE: When ASTRA scans or works on its own code, it MUST
use the sandbox (192.168.250.2:8765). The host D:\\Orb filesystem
is NEVER the source of truth for code operations.

The sandbox /fs/tree API returns a flat list of ALL files recursively.
We group them by directory to produce the same (dirpath, dirnames,
filenames) tuples that os.walk() yields.

Called from:
  - app/memory/domains/dependency_scanner.py (import graph)
  - app/rag/rescan.py (code chunk indexing)
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Generator, List, Set, Tuple

from app.sandbox_fs import sandbox_read_text

logger = logging.getLogger(__name__)

# Directories to skip during codebase walks
DEFAULT_SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".eggs",
    "data", "jobs", ".architecture",
}


def _fetch_all_files(root: str, max_files: int = 10000) -> list[dict]:
    """
    Fetch all files under root from the sandbox /fs/tree endpoint.

    Returns the raw list of file dicts from the API.
    """
    from app.sandbox_fs import _post

    status, data, error = _post("/fs/tree", {
        "roots": [root],
        "max_files": max_files,
        "include_size": True,
    })

    if status != 200 or not data:
        logger.error(
            "[sandbox_walk] Failed to fetch files from sandbox: "
            "status=%s, error=%s", status, error,
        )
        return []

    files = data.get("files", [])
    logger.info("[sandbox_walk] Fetched %d files from sandbox (%s)", len(files), root)
    return files


def sandbox_walk(
    root: str,
    skip_dirs: Set[str] | None = None,
) -> Generator[Tuple[str, List[str], List[str]], None, None]:
    """
    Recursive directory walk via sandbox API.

    Yields (dirpath, dirnames, filenames) tuples — same interface
    as os.walk() but reads from the sandbox controller.

    Makes a SINGLE API call to /fs/tree to get all files, then
    groups them by directory. Much faster than recursive listing.

    Args:
        root: Starting directory path (e.g. 'D:\\\\Orb\\\\app').
        skip_dirs: Set of directory names to skip.

    Yields:
        Tuples of (dirpath, [dir_names], [file_names]).
    """
    if skip_dirs is None:
        skip_dirs = DEFAULT_SKIP_DIRS

    all_files = _fetch_all_files(root)
    if not all_files:
        return

    # Normalise root path
    root_norm = root.replace("/", "\\").rstrip("\\")

    # Group files by their parent directory
    dir_files: dict[str, list[str]] = defaultdict(list)
    all_dirs: set[str] = set()

    for entry in all_files:
        path = entry.get("path", "").replace("/", "\\")
        name = entry.get("name", "")

        if not path or not name:
            continue

        # Get the parent directory
        if "\\" in path:
            parent = path.rsplit("\\", 1)[0]
        else:
            continue

        # Check if any path component is in skip_dirs
        rel_to_root = path[len(root_norm):].lstrip("\\") if path.startswith(root_norm) else path
        parts = rel_to_root.split("\\")
        if any(p in skip_dirs for p in parts[:-1]):  # Check dir parts, not filename
            continue

        dir_files[parent].append(name)

        # Track all intermediate directories
        current = root_norm
        for part in parts[:-1]:
            current = current + "\\" + part
            all_dirs.add(current)

    # Yield in sorted order (root first, then subdirectories)
    visited = set()

    def _yield_dir(dirpath: str):
        if dirpath in visited:
            return
        visited.add(dirpath)

        filenames = sorted(dir_files.get(dirpath, []))

        # Find immediate subdirectories
        subdirs = []
        for d in sorted(all_dirs):
            if d.startswith(dirpath + "\\"):
                rest = d[len(dirpath) + 1:]
                if "\\" not in rest:  # Immediate child only
                    dirname = rest
                    if dirname not in skip_dirs:
                        subdirs.append(dirname)

        yield dirpath, subdirs, filenames

        # Recurse into subdirectories
        for sd in subdirs:
            yield from _yield_dir(dirpath + "\\" + sd)

    yield from _yield_dir(root_norm)


def sandbox_read_python(filepath: str) -> str | None:
    """
    Read a Python file from the sandbox.

    Returns the file content as a string, or None on failure.
    """
    try:
        content = sandbox_read_text(filepath)
        return content
    except Exception as e:
        logger.debug("[sandbox_walk] Failed to read %s: %s", filepath, e)
        return None
