# FILE: app/drive/file_utils.py
"""
Utility functions for local file operations.
Handles path resolution, file metadata, thumbnails, and safety checks.
"""

import os
import mimetypes
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


# ─── Allowed root directories ───────────────────────────────────

def get_user_home() -> Path:
    """Get the current user's home directory."""
    return Path.home()


def get_category_paths() -> dict[str, Path]:
    """
    Map category IDs to their Windows filesystem paths.
    Uses Windows known-folder environment resolution where possible,
    falling back to OneDrive paths, then standard home paths.
    """
    home = get_user_home()

    # Use the Windows shell known-folder API via Environment
    # This correctly resolves OneDrive-redirected folders
    import ctypes.wintypes
    paths = {}

    # Try .NET known folder resolution first (handles OneDrive redirection)
    try:
        import os as _os
        docs = _os.environ.get("OneDrive", "")
        if docs and Path(docs, "Documents").exists():
            paths["documents"] = Path(docs, "Documents")
        else:
            paths["documents"] = home / "Documents"

        if docs and Path(docs, "Pictures").exists():
            paths["pictures"] = Path(docs, "Pictures")
        else:
            paths["pictures"] = home / "Pictures"

        if docs and Path(docs, "Desktop").exists():
            paths["desktop"] = Path(docs, "Desktop")
        else:
            paths["desktop"] = home / "Desktop"
    except Exception:
        paths["documents"] = home / "Documents"
        paths["pictures"] = home / "Pictures"
        paths["desktop"] = home / "Desktop"

    # Music and Videos are typically NOT in OneDrive
    paths["music"] = home / "Music"
    paths["videos"] = home / "Videos"

    # Screenshots is a subfolder of Pictures
    paths["screenshots"] = paths["pictures"] / "Screenshots"

    # ASTRA generated output files
    paths["astra_output"] = Path(r"D:\Orb\output")

    return paths


# ─── Path safety ────────────────────────────────────────────────

def is_safe_path(requested: Path, allowed_roots: list[Path]) -> bool:
    """
    Ensure the resolved path is within one of the allowed root dirs.
    Prevents directory traversal attacks.
    """
    resolved = requested.resolve()
    return any(
        resolved == root.resolve() or str(resolved).startswith(str(root.resolve()) + os.sep)
        for root in allowed_roots
    )


def get_allowed_roots() -> list[Path]:
    """Get all allowed root paths for file access."""
    return list(get_category_paths().values())


# ─── File metadata ──────────────────────────────────────────────

def get_file_extension(filename: str) -> str:
    """Extract lowercase extension without dot."""
    _, ext = os.path.splitext(filename)
    return ext.lstrip(".").lower()


IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp", "svg", "bmp", "ico", "tiff"}
AUDIO_EXTENSIONS = {"mp3", "wav", "flac", "ogg", "aac", "m4a", "wma"}
VIDEO_EXTENSIONS = {"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv"}
DOCUMENT_EXTENSIONS = {
    "docx", "doc", "pdf", "xlsx", "xls", "pptx", "ppt",
    "txt", "md", "csv", "json", "xml", "html", "htm",
    "py", "js", "ts", "tsx", "jsx", "css", "sql",
}


def classify_file(ext: str) -> str:
    """Classify a file extension into a broad category."""
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "other"


def get_file_info(filepath: Path) -> Optional[dict]:
    """
    Build a metadata dict for a single file.
    Returns None if the file doesn't exist or can't be read.
    """
    try:
        stat = filepath.stat()
        ext = get_file_extension(filepath.name)
        mime_type, _ = mimetypes.guess_type(str(filepath))

        return {
            "name": filepath.name,
            "path": str(filepath),
            "ext": ext,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "mime_type": mime_type or "application/octet-stream",
            "file_class": classify_file(ext),
            "is_hidden": filepath.name.startswith("."),
            "is_directory": False,
        }
    except (OSError, PermissionError):
        return None


def get_directory_info(dirpath: Path) -> Optional[dict]:
    """Build metadata dict for a directory."""
    try:
        stat = dirpath.stat()
        # Count immediate children
        children = list(dirpath.iterdir())
        file_count = sum(1 for c in children if c.is_file())
        dir_count = sum(1 for c in children if c.is_dir())

        return {
            "name": dirpath.name,
            "path": str(dirpath),
            "ext": "",
            "size": 0,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat(),
            "mime_type": "inode/directory",
            "file_class": "folder",
            "is_hidden": dirpath.name.startswith("."),
            "is_directory": True,
            "file_count": file_count,
            "dir_count": dir_count,
        }
    except (OSError, PermissionError):
        return None
