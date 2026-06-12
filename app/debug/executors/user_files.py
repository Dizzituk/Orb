# FILE: app/debug/executors/user_files.py
# Purpose: User file executors: search/read/write user-personal folders, manifest
# Called-by: app.debug.executors
# Depends-on: app.cloud.rclone_service, app.db, app.debug.executors._paths, app.debug.size_warning (+8 more)
# Last-renovated: 2026-06-11
"""
User file executors: search/read/write user-personal folders, manifest
maintenance, vision (image reading), web search, cloud storage.

Grouped together because they all operate on USER data (Pictures,
Documents, Desktop, OneDrive, Google Drive) rather than on ASTRA's
own codebase.

If this file approaches 30 KB, the natural split is:
  - user_files.py: search/read/write user files + manifest tools
  - external_services.py: vision + web + cloud
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

from app.debug.executors._paths import is_host_write_blocked
from app.debug.size_warning import add_size_warning as _size_warn

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================

def _format_search_header(shown: int, total: int) -> str:
    """Build the result-set header, flagging truncation when total > shown."""
    if total > shown:
        return f"Found {total} file(s) (showing first {shown} - refine query for the rest):"
    return f"Found {shown} file(s):"


# =============================================================================
# USER FILE READ/SEARCH
# =============================================================================

async def execute_search_my_files(params: Dict[str, Any]) -> str:
    """Search the drive_file_manifest for user files by name/category/extension."""
    query = params.get("query", "").strip()
    category = params.get("category", "").strip().lower()
    extension = params.get("extension", "").strip().lower().lstrip(".")

    if not query and not category and not extension:
        return "Please provide a search query, category, or extension."

    if query:
        query = query.replace(" ", "%")

    try:
        from app.db import SessionLocal
        from app.drive.manifest_models import DriveFileManifest

        db = SessionLocal()
        try:
            q = db.query(DriveFileManifest)

            if query:
                from sqlalchemy import or_
                q = q.filter(or_(
                    DriveFileManifest.filename.ilike(f"%{query}%"),
                    DriveFileManifest.path.ilike(f"%{query}%"),
                ))
            if category:
                q = q.filter(DriveFileManifest.category == category)
            if extension:
                q = q.filter(DriveFileManifest.extension == extension)

            total_count = q.count()
            results = q.order_by(DriveFileManifest.filename).limit(250).all()

            if not results:
                return f"No files found matching: query={query!r}, category={category!r}, extension={extension!r}"

            lines = [_format_search_header(len(results), total_count)]
            for r in results:
                size_kb = r.size_bytes / 1024
                size_str = f"{size_kb:.0f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
                indexed = "indexed" if r.content_indexed else "not indexed"
                lines.append(
                    f"  [{r.category}] {r.filename} ({size_str}, {r.file_class}, {indexed})"
                )
                lines.append(f"    Path: {r.path}")
            return "\n".join(lines)
        finally:
            db.close()
    except Exception as e:
        return f"Search failed: {e}"


async def execute_read_user_file(params: Dict[str, Any]) -> str:
    """Read a user file by extracting its text content."""
    path = params.get("path", "").strip()
    if not path:
        return "Please provide a file path."

    if not os.path.isfile(path):
        return f"File not found: {path}"

    try:
        from app.drive.file_utils import get_category_paths
        allowed_roots = [str(p) for p in get_category_paths().values()]
        allowed_roots.append(os.path.join("D:", os.sep, "Orb", "output"))
        allowed_roots.append(os.path.join("D:", os.sep, "Orb", "data", "debug_uploads"))

        path_norm = os.path.normpath(path)
        if not any(path_norm.startswith(os.path.normpath(r)) for r in allowed_roots):
            return f"Access denied: {path} is outside allowed user file areas."
    except Exception:
        pass

    try:
        from app.llm.file_analyzer import extract_text
        text, err = extract_text(file_path=path, filename=os.path.basename(path))
        if text:
            if len(text) > 50000:
                return text[:50000] + f"\n\n... [TRUNCATED - {len(text)} chars total]"
            return _size_warn(text, path)
        elif err:
            return f"Could not extract text from {os.path.basename(path)}: {err}"
        else:
            return f"No readable content in {os.path.basename(path)}"
    except Exception as e:
        return f"Read failed: {e}"


# =============================================================================
# USER FILE WRITE
# =============================================================================

# Extensions that hold binary data and CANNOT be created via write_user_file
# (which writes UTF-8 text). Each maps to the right tool/path the caller
# should use instead. The error message is read by the chat LLM, so it has
# to be specific enough to redirect the model rather than just say "no".
#
# v0.2 (2026-05-01): Added after gpt-5.4-mini was caught writing literal
# "PNG placeholder" text to a .png file and reporting success. The model
# had no real HTML→PNG converter so it improvised with the only writing
# tool it had. This guard makes the failure loud and tells the model
# what to do instead.
_BINARY_EXTENSIONS_GUIDANCE = {
    # Images — always go through image generation, not file writing
    ".png":  "PNG images are binary. Ask for image generation directly (e.g. 'make me a chart') and the image pipeline will render and save it. Do NOT write text to a .png path.",
    ".jpg":  "JPG images are binary. Ask for image generation directly (e.g. 'make me an image of X') and the image pipeline will render and save it.",
    ".jpeg": "JPEG images are binary. Ask for image generation directly (e.g. 'make me an image of X') and the image pipeline will render and save it.",
    ".webp": "WebP images are binary. Ask for image generation directly.",
    ".gif":  "GIF images are binary. Ask for image generation directly.",
    ".bmp":  "BMP images are binary. Ask for image generation directly.",
    ".tiff": "TIFF images are binary. Ask for image generation directly.",
    ".tif":  "TIFF images are binary. Ask for image generation directly.",
    ".ico":  "ICO images are binary. Ask for image generation directly.",
    ".heic": "HEIC images are binary. Ask for image generation directly.",
    # Documents — each has a dedicated builder tool
    ".pdf":  "PDF files are binary. Use the create_pdf tool with structured content blocks or a brief.",
    ".docx": "DOCX files are binary. Use the create_docx tool with structured content blocks or a brief.",
    ".xlsx": "XLSX files are binary. Use the create_xlsx tool with sheet definitions or a brief.",
    ".pptx": "PPTX files are binary. There is no PPTX builder yet — ask the user how to proceed.",
    ".doc":  "Legacy DOC files are binary. Use create_docx (writes .docx) instead.",
    ".xls":  "Legacy XLS files are binary. Use create_xlsx (writes .xlsx) instead.",
    ".ppt":  "Legacy PPT files are binary. There is no PowerPoint builder available.",
    # Archives / media — outright unsupported via this tool
    ".zip":  "ZIP archives are binary and cannot be written via write_user_file.",
    ".tar":  "TAR archives are binary and cannot be written via write_user_file.",
    ".gz":   "GZ archives are binary and cannot be written via write_user_file.",
    ".7z":   "7-Zip archives are binary and cannot be written via write_user_file.",
    ".rar":  "RAR archives are binary and cannot be written via write_user_file.",
    ".mp4":  "MP4 video files are binary. There is no video creation tool available via write_user_file.",
    ".mov":  "MOV video files are binary. There is no video creation tool available via write_user_file.",
    ".mkv":  "MKV video files are binary. There is no video creation tool available via write_user_file.",
    ".webm": "WebM video files are binary. There is no video creation tool available via write_user_file.",
    ".avi":  "AVI video files are binary. There is no video creation tool available via write_user_file.",
    ".mp3":  "MP3 audio files are binary. There is no audio synthesis tool available via write_user_file.",
    ".wav":  "WAV audio files are binary. There is no audio synthesis tool available via write_user_file.",
    ".flac": "FLAC audio files are binary. There is no audio synthesis tool available via write_user_file.",
    ".ogg":  "OGG audio files are binary. There is no audio synthesis tool available via write_user_file.",
    # Executables — should never be written
    ".exe":  "Executable files cannot be created via write_user_file.",
    ".dll":  "DLL files cannot be created via write_user_file.",
    ".so":   "Shared object files cannot be created via write_user_file.",
    ".dylib": "Dylib files cannot be created via write_user_file.",
}


def _check_binary_extension(path: str) -> str | None:
    """If `path` has a binary extension, return a guidance string for the
    caller. Otherwise return None. SVG is intentionally NOT in the list
    because it is XML text and is legitimately writable as UTF-8.
    """
    lower = path.lower()
    for ext, guidance in _BINARY_EXTENSIONS_GUIDANCE.items():
        if lower.endswith(ext):
            return guidance
    return None


async def execute_write_user_file(params: Dict[str, Any]) -> str:
    """Write a TEXT file to the user's personal folders.

    This tool only writes UTF-8 text. Binary extensions (.png, .pdf, .docx,
    archives, media files) are rejected with guidance pointing the caller
    at the right tool. Previously the function would silently write text
    content to any extension, including .png — producing 0 KB "placeholder"
    files that looked like successful saves to the LLM. See the v0.2
    docstring on _BINARY_EXTENSIONS_GUIDANCE for the failure trace.
    """
    path = params.get("path", "").strip()
    content = params.get("content", "")

    if not path:
        return "Error: path is required."
    if not content:
        return "Error: content is required (file would be empty)."

    # v0.2: Refuse binary extensions BEFORE doing any path validation,
    # because the guidance is more useful than a permission error.
    binary_guidance = _check_binary_extension(path)
    if binary_guidance:
        logger.warning(
            "[executors.user_files] write_user_file rejected binary extension: %s",
            path,
        )
        return f"Error: write_user_file cannot create binary files. {binary_guidance}"

    try:
        from app.drive.file_utils import get_category_paths, is_safe_path

        allowed_roots = list(get_category_paths().values())
        target = Path(path)

        if not is_safe_path(target, allowed_roots):
            return (
                f"Access denied: {path} is outside allowed user folders. "
                f"Use get_user_folders to see valid base paths."
            )

        if is_host_write_blocked(str(target.resolve())):
            logger.warning("[executors.user_files] user-file write blocked by host protection: %s", path)
            return (
                f"Access denied: {path} is inside a protected ASTRA directory. "
                f"Host-side code is read-only; changes must go through the sandbox."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        size_kb = len(content.encode("utf-8")) / 1024

        logger.info("[executors.user_files] User file write: %s (%.1f KB)", path, size_kb)

        try:
            from app.drive.manifest_scanner import index_single_file
            index_single_file(str(target))
        except Exception:
            pass

        return f"Successfully wrote {len(content)} chars ({size_kb:.1f} KB) to {path}"
    except PermissionError:
        return f"Permission denied writing to {path}"
    except Exception as e:
        return f"Write failed: {e}"


async def execute_get_user_folders(params: Dict[str, Any]) -> str:
    """Return resolved paths for all user personal folders."""
    try:
        from app.drive.file_utils import get_category_paths

        paths = get_category_paths()
        lines = ["User folders (use these as base paths for write_user_file):"]
        for category, folder_path in sorted(paths.items()):
            exists = folder_path.exists()
            lines.append(
                f"  {category}: {folder_path}  ({'OK' if exists else 'NOT FOUND'})"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Error resolving user folders: {e}"


# =============================================================================
# MANIFEST TOOLS (rescan, reindex, live disk search)
# =============================================================================

async def execute_rescan_manifest(params: Dict[str, Any]) -> str:
    """Force a full rescan of user folders and refresh the file manifest."""
    try:
        from app.drive.manifest_rescan import rescan_all
        result = rescan_all()
    except Exception as e:
        return f"Rescan failed: {e}"

    if result.get("status") != "ok":
        return f"Rescan error: {result.get('error', 'unknown')}"

    return (
        f"Manifest rescanned: {result['total_files']} files total "
        f"({result['new_files']} new, {result['modified_files']} modified, "
        f"{result['deleted_files']} removed) in {result['duration_ms']}ms."
    )


async def execute_reindex_file(params: Dict[str, Any]) -> str:
    """Refresh a single file's entry in the manifest."""
    path = params.get("path", "").strip()
    if not path:
        return "Error: path is required."

    try:
        from app.drive.manifest_rescan import rescan_path
        result = rescan_path(path)
    except Exception as e:
        return f"Reindex failed: {e}"

    status = result.get("status", "unknown")
    if status == "error":
        return f"Reindex error for {path}: {result.get('error', 'unknown')}"
    if status == "not_found":
        return f"No action: {path} does not exist and was not in the manifest."
    if status == "removed":
        return f"Removed stale manifest entry for {path} (file no longer exists)."
    return (
        f"Manifest {status}: {path} "
        f"(category={result.get('category', '?')}, "
        f"size={result.get('size', 0)} bytes)."
    )


async def execute_search_disk_live(params: Dict[str, Any]) -> str:
    """Search the actual filesystem (bypassing the manifest cache)."""
    query = params.get("query", "").strip()
    category = params.get("category", "").strip() or None
    extension = params.get("extension", "").strip() or None

    if not query:
        return "Error: query is required."

    try:
        from app.drive.disk_search import search_disk
        result = search_disk(query, category=category, extension=extension)
    except Exception as e:
        return f"Live disk search failed: {e}"

    if result.get("status") != "ok":
        return f"Search error: {result.get('error', 'unknown')}"

    matches = result.get("results", [])
    elapsed = result.get("elapsed_ms", 0)
    scanned = result.get("scanned", 0)
    healed = result.get("manifest_healed", 0)

    if not matches:
        return (
            f"No files matching {query!r} on disk "
            f"(scanned {scanned} files in {elapsed}ms). "
            f"The file genuinely does not exist in your user folders."
        )

    lines = [
        f"Found {len(matches)} file(s) on disk for {query!r} "
        f"(scanned {scanned} in {elapsed}ms"
        + (f", healed manifest for {healed}" if healed else "")
        + "):"
    ]
    for r in matches:
        size_kb = r["size_bytes"] / 1024
        size_str = f"{size_kb:.0f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
        lines.append(f"  [{r['category']}] {r['filename']} ({size_str}, {r['file_class']})")
        lines.append(f"    Path: {r['path']}")
    if result.get("truncated"):
        lines.append(f"  ... results capped; refine the query for more.")
    return "\n".join(lines)


# =============================================================================
# VISION (Gemini image reading)
# =============================================================================

async def execute_read_image(params: Dict[str, Any]) -> str:
    """Read the visual content of an image file via Gemini Vision.

    Wraps app.llm.gemini_vision.ask_about_image so the chat model can OCR
    screenshots, describe photos, and extract text from images on disk.
    Supports any format Pillow understands (PNG, JPEG, WebP, GIF, BMP, TIFF).
    """
    path = params.get("path", "").strip()
    question = params.get("question", "").strip()

    if not path:
        return "Error: path is required."

    if not os.path.isfile(path):
        return f"Image not found: {path}"

    if not question:
        question = (
            "Describe what is shown in this image. Include any visible text "
            "(verbatim where possible), UI elements, app names, status bar "
            "content, timestamps, and any notable details a viewer would care about."
        )

    try:
        from app.llm.gemini_vision import ask_about_image
        result = ask_about_image(image_source=path, user_question=question)
    except Exception as e:
        return f"Vision call failed: {e}"

    if result.get("error"):
        provider = result.get("provider", "unknown")
        return f"Vision error ({provider}): {result['error']}"

    answer = (result.get("answer") or "").strip()
    if not answer:
        return "Vision returned no description."

    provider = result.get("provider", "unknown")
    model = result.get("model", "unknown")
    return f"[{provider}/{model}]\n{answer}"


# =============================================================================
# WEB SEARCH
# =============================================================================

async def execute_web_search(params: Dict[str, Any]) -> str:
    """Search the web via ASTRA's existing Brave/DDG infrastructure."""
    query = str(params.get("query", "")).strip()
    if not query:
        return "Error: query is required."

    max_results = int(params.get("max_results", 5))
    max_results = max(1, min(10, max_results))

    try:
        from app.tools.registry import web_search_handler
        result = await web_search_handler(
            {"query": query, "max_results": max_results},
            context=None,
        )
        results = result.get("results", [])
        provider = result.get("provider", "unknown")
        if not results:
            return f"No results found for: {query} (provider: {provider})"

        lines = [f"Web search results for: {query} (via {provider})", ""]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', 'Untitled')}")
            lines.append(f"   URL: {r.get('url', '')}")
            lines.append(f"   {r.get('snippet', '')}")
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        logger.error("[executors.user_files] web_search failed: %s", e)
        return f"Web search error: {e}"


# =============================================================================
# CLOUD STORAGE (Google Drive via rclone)
# =============================================================================

async def execute_cloud_upload(params: Dict[str, Any]) -> str:
    """Upload a local file to Google Drive via rclone."""
    local_path = params.get("local_path", "")
    cloud_path = params.get("cloud_path", "")

    if not local_path or not cloud_path:
        return "Error: both local_path and cloud_path are required."

    if not Path(local_path).exists():
        return f"Error: local file not found: {local_path}"

    try:
        from app.cloud.rclone_service import upload_file
        result = await upload_file(local_path, cloud_path)
        if result.get("success"):
            size_kb = result.get("size", 0) / 1024
            return f"Uploaded to Google Drive: {cloud_path} ({size_kb:.0f}KB)"
        else:
            return f"Upload failed: {result.get('error', 'unknown error')}"
    except Exception as e:
        logger.error("[executors.user_files] cloud_upload failed: %s", e)
        return f"Upload error: {e}"


async def execute_cloud_list(params: Dict[str, Any]) -> str:
    """List files on Google Drive at a given path."""
    cloud_path = params.get("path", "")

    try:
        from app.cloud.rclone_service import list_files
        items = await list_files(cloud_path)
        if not items:
            return f"No files found at: {cloud_path or '(root)'}"

        lines = [f"Google Drive: {cloud_path or '(root)'} ({len(items)} items)"]
        for item in items:
            kind = "[DIR]" if item.get("is_dir") else "[FILE]"
            size = ""
            if not item.get("is_dir") and item.get("size"):
                size_kb = item["size"] / 1024
                size = f" ({size_kb:.0f}KB)"
            lines.append(f"  {kind} {item['name']}{size}")
        return "\n".join(lines)
    except Exception as e:
        logger.error("[executors.user_files] cloud_list failed: %s", e)
        return f"Cloud list error: {e}"
