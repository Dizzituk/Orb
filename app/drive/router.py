# FILE: app/drive/router.py
# Purpose: ASTRA Drive — FastAPI router for local file system operations.
# Called-by: main
# Depends-on: app.drive.file_routing, app.drive.file_utils, app.drive.thumbnail, app.llm.file_analyzer
# Last-renovated: 2026-06-11
"""
ASTRA Drive — FastAPI router for local file system operations.

v0.14.0: read-content now extracts text from DOCX/XLSX/PPTX/PDF
         via the universal extractor, instead of returning raw base64.

Endpoints:
  GET  /drive/categories        — list available categories with file counts
  GET  /drive/files              — list files in a category (with optional search)
  GET  /drive/file/info          — get metadata for a single file
  GET  /drive/read-content       — read file content (text or base64 for binary)
  GET  /drive/file-route         — get model routing for a file type
  GET  /drive/thumbnail          — get thumbnail for an image file
  POST /drive/rename             — rename a file
  POST /drive/move               — move a file to a new location
  POST /drive/delete             — delete a file (moves to recycle bin)
  POST /drive/create-folder      — create a new folder
  GET  /drive/storage-stats      — storage usage breakdown per category
  POST /drive/open               — open a file in the default OS application
"""

import base64
import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .file_utils import (
    get_category_paths,
    get_allowed_roots,
    is_safe_path,
    get_file_info,
    get_directory_info,
    get_file_extension,
    classify_file,
)
from .file_routing import get_file_route, get_mime_type
from .thumbnail import generate_thumbnail


router = APIRouter(prefix="/drive", tags=["Drive"])

# Max file size for inline content (10MB)
MAX_CONTENT_SIZE = 10 * 1024 * 1024
# Max text content length (500KB of text)
MAX_TEXT_SIZE = 500 * 1024

# Text-based extensions that can be read directly as UTF-8
TEXT_EXTENSIONS = {
    "txt", "md", "rst", "csv", "json", "xml", "html", "htm", "css", "sql",
    "py", "js", "ts", "tsx", "jsx", "log", "yml", "yaml", "toml",
    "ini", "cfg", "env", "sh", "bat", "ps1", "conf",
    "r", "rb", "go", "rs", "java", "kt", "swift",
    "c", "cpp", "h", "hpp", "cs", "lua", "pl", "php",
    "tex", "bib", "srt", "vtt", "ics",
    "gitignore", "dockerignore", "editorconfig",
    "makefile", "cmake", "tf", "hcl", "proto", "graphql",
}

# Structured document extensions that need a parser to extract text
EXTRACTABLE_EXTENSIONS = {
    "docx", "doc", "xlsx", "xls", "pptx", "pdf",
}


# ─── Request / Response models ──────────────────────────────────

class RenameRequest(BaseModel):
    path: str
    new_name: str

class MoveRequest(BaseModel):
    source_path: str
    dest_dir: str

class DeleteRequest(BaseModel):
    path: str

class CreateFolderRequest(BaseModel):
    parent_path: str
    folder_name: str

class OpenRequest(BaseModel):
    path: str


# ─── Endpoints ──────────────────────────────────────────────────

@router.get("/categories")
def list_categories():
    """List all available file categories with file counts and total sizes."""
    categories = []
    cat_paths = get_category_paths()

    for cat_id, cat_path in cat_paths.items():
        if not cat_path.exists():
            categories.append({
                "id": cat_id,
                "path": str(cat_path),
                "exists": False,
                "file_count": 0,
                "total_size": 0,
            })
            continue

        file_count = 0
        total_size = 0
        try:
            for item in cat_path.rglob("*"):
                if item.is_file() and not item.name.startswith("."):
                    file_count += 1
                    try:
                        total_size += item.stat().st_size
                    except OSError:
                        pass
        except PermissionError:
            pass

        categories.append({
            "id": cat_id,
            "path": str(cat_path),
            "exists": True,
            "file_count": file_count,
            "total_size": total_size,
        })

    total_files = sum(c["file_count"] for c in categories)
    total_bytes = sum(c["total_size"] for c in categories)
    categories.insert(0, {
        "id": "all",
        "path": "",
        "exists": True,
        "file_count": total_files,
        "total_size": total_bytes,
    })

    return {"categories": categories}


@router.get("/files")
def list_files(
    category: str = Query("all"),
    search: Optional[str] = Query(None),
    sort_by: str = Query("modified"),
    sort_dir: str = Query("desc"),
    path: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List files AND directories in a category or specific directory."""
    cat_paths = get_category_paths()
    allowed = get_allowed_roots()
    scan_dirs: list[Path] = []

    if path:
        target = Path(path)
        if not is_safe_path(target, allowed):
            raise HTTPException(403, "Access denied: path outside allowed directories")
        if not target.exists() or not target.is_dir():
            raise HTTPException(404, "Directory not found")
        scan_dirs = [target]
    elif category == "all":
        scan_dirs = [p for p in cat_paths.values() if p.exists()]
    else:
        if category not in cat_paths:
            raise HTTPException(400, f"Unknown category: {category}")
        cat_dir = cat_paths[category]
        if not cat_dir.exists():
            return {"files": [], "total": 0, "current_path": None}
        scan_dirs = [cat_dir]

    dirs = []
    files = []
    for scan_dir in scan_dirs:
        try:
            for entry in scan_dir.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    dinfo = get_directory_info(entry)
                    if dinfo:
                        for cid, cpath in cat_paths.items():
                            try:
                                if entry.resolve().is_relative_to(cpath.resolve()):
                                    dinfo["category"] = cid
                                    break
                            except (ValueError, OSError):
                                pass
                        else:
                            dinfo["category"] = "other"
                        dirs.append(dinfo)
                elif entry.is_file():
                    info = get_file_info(entry)
                    if info:
                        for cid, cpath in cat_paths.items():
                            try:
                                if entry.resolve().is_relative_to(cpath.resolve()):
                                    info["category"] = cid
                                    break
                            except (ValueError, OSError):
                                pass
                        else:
                            info["category"] = "other"
                        files.append(info)
        except PermissionError:
            continue

    if search:
        search_lower = search.lower()
        dirs = [d for d in dirs if search_lower in d["name"].lower()]
        files = [f for f in files if search_lower in f["name"].lower()]

    reverse = sort_dir == "desc"
    if sort_by == "name":
        dirs.sort(key=lambda f: f["name"].lower(), reverse=reverse)
        files.sort(key=lambda f: f["name"].lower(), reverse=reverse)
    elif sort_by == "size":
        dirs.sort(key=lambda f: f["name"].lower())
        files.sort(key=lambda f: f.get("size", 0), reverse=reverse)
    else:
        dirs.sort(key=lambda f: f.get("modified", ""), reverse=reverse)
        files.sort(key=lambda f: f.get("modified", ""), reverse=reverse)

    items = dirs + files
    total = len(items)
    paged = items[offset : offset + limit]
    current_path = str(scan_dirs[0]) if len(scan_dirs) == 1 else None

    return {"files": paged, "total": total, "offset": offset, "limit": limit, "current_path": current_path}


@router.get("/file/info")
def file_info(path: str = Query(...)):
    """Get detailed metadata for a single file."""
    target = Path(path)
    if not is_safe_path(target, get_allowed_roots()):
        raise HTTPException(403, "Access denied")
    if not target.exists():
        raise HTTPException(404, "File not found")
    info = get_directory_info(target) if target.is_dir() else get_file_info(target)
    if not info:
        raise HTTPException(500, "Could not read file metadata")
    return info


@router.get("/read-content")
def read_content(path: str = Query(..., description="Full path to the file")):
    """
    Read file content for ASTRA analysis.

    v0.14.0: Now uses the universal text extractor for structured documents
    (DOCX, XLSX, PPTX, PDF) so models receive extracted text, not raw binary.

    Returns:
      - For text files: { mode: "text", content: "...", mime_type: "..." }
      - For extractable docs: { mode: "text", content: "...", extracted_from: "xlsx" }
      - For binary files: { mode: "base64", content: "base64...", mime_type: "..." }
      - Includes routing info: { route: { provider, model, content_mode } }

    Size limits: 10MB max for binary, 500KB for text.
    """
    target = Path(path)
    if not is_safe_path(target, get_allowed_roots()):
        raise HTTPException(403, "Access denied")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")

    file_size = target.stat().st_size
    if file_size > MAX_CONTENT_SIZE:
        raise HTTPException(413, f"File too large: {file_size} bytes (max {MAX_CONTENT_SIZE})")

    ext = get_file_extension(target.name)
    mime = get_mime_type(ext)
    route = get_file_route(ext)

    # ── Plain text files: read as UTF-8 ─────────────────────────
    if ext in TEXT_EXTENSIONS:
        if file_size > MAX_TEXT_SIZE:
            raise HTTPException(413, f"Text file too large: {file_size} bytes (max {MAX_TEXT_SIZE})")
        try:
            content = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise HTTPException(500, f"Failed to read file: {e}")

        return {
            "mode": "text",
            "content": content,
            "mime_type": mime,
            "file_name": target.name,
            "file_size": file_size,
            "route": {
                "provider": route.provider,
                "model": route.model,
                "content_mode": route.content_mode,
            },
        }

    # ── Structured documents: extract text via universal extractor ─
    if ext in EXTRACTABLE_EXTENSIONS:
        try:
            from app.llm.file_analyzer import extract_text
            text, error = extract_text(file_path=str(target))

            if text:
                return {
                    "mode": "text",
                    "content": text,
                    "mime_type": mime,
                    "file_name": target.name,
                    "file_size": file_size,
                    "extracted_from": ext,
                    "route": {
                        "provider": route.provider,
                        "model": route.model,
                        "content_mode": "text",  # Override: we extracted text
                    },
                }
            else:
                # Extraction failed — log it and fall through to base64
                print(f"[drive/read-content] Text extraction failed for {target.name}: {error}")
        except Exception as e:
            print(f"[drive/read-content] Extractor error for {target.name}: {e}")

    # ── Binary files: read as base64 ────────────────────────────
    try:
        raw = target.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
    except Exception as e:
        raise HTTPException(500, f"Failed to read file: {e}")

    return {
        "mode": "base64",
        "content": b64,
        "mime_type": mime,
        "file_name": target.name,
        "file_size": file_size,
        "route": {
            "provider": route.provider,
            "model": route.model,
            "content_mode": route.content_mode,
        },
    }


@router.get("/file-route")
def file_route(ext: str = Query(..., description="File extension (without dot)")):
    """Get model routing info for a file extension."""
    route = get_file_route(ext)
    return {
        "provider": route.provider,
        "model": route.model,
        "content_mode": route.content_mode,
        "mime_prefix": route.mime_prefix,
    }


@router.get("/thumbnail")
def get_thumbnail(path: str = Query(...), size: int = Query(200, ge=50, le=400)):
    """Get a base64-encoded thumbnail for an image file."""
    target = Path(path)
    if not is_safe_path(target, get_allowed_roots()):
        raise HTTPException(403, "Access denied")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")
    ext = get_file_extension(target.name)
    if classify_file(ext) != "image":
        raise HTTPException(400, "Not an image file")
    thumb = generate_thumbnail(target, size)
    if not thumb:
        raise HTTPException(500, "Could not generate thumbnail")
    return {"thumbnail": thumb, "path": str(target)}


@router.post("/open")
def open_file(req: OpenRequest):
    """Open a file using the default Windows application."""
    target = Path(req.path)
    if not is_safe_path(target, get_allowed_roots()):
        raise HTTPException(403, "Access denied")
    if not target.exists():
        raise HTTPException(404, "File not found")
    try:
        os.startfile(str(target))
        return {"success": True, "path": str(target)}
    except OSError as e:
        raise HTTPException(500, f"Failed to open file: {e}")


@router.post("/rename")
def rename_file(req: RenameRequest):
    target = Path(req.path)
    allowed = get_allowed_roots()
    if not is_safe_path(target, allowed):
        raise HTTPException(403, "Access denied")
    if not target.exists():
        raise HTTPException(404, "File not found")
    if "/" in req.new_name or "\\" in req.new_name:
        raise HTTPException(400, "Invalid filename")
    new_path = target.parent / req.new_name
    if new_path.exists():
        raise HTTPException(409, f"'{req.new_name}' already exists")
    try:
        target.rename(new_path)
        return {"success": True, "old_path": str(target), "new_path": str(new_path)}
    except OSError as e:
        raise HTTPException(500, f"Rename failed: {e}")


@router.post("/move")
def move_file(req: MoveRequest):
    source = Path(req.source_path)
    dest_dir = Path(req.dest_dir)
    allowed = get_allowed_roots()
    if not is_safe_path(source, allowed):
        raise HTTPException(403, "Source access denied")
    if not is_safe_path(dest_dir, allowed):
        raise HTTPException(403, "Destination access denied")
    if not source.exists():
        raise HTTPException(404, "Source not found")
    if not dest_dir.exists() or not dest_dir.is_dir():
        raise HTTPException(404, "Destination not found")
    new_path = dest_dir / source.name
    if new_path.exists():
        raise HTTPException(409, f"'{source.name}' already exists in destination")
    try:
        shutil.move(str(source), str(new_path))
        return {"success": True, "old_path": str(source), "new_path": str(new_path)}
    except OSError as e:
        raise HTTPException(500, f"Move failed: {e}")


@router.post("/delete")
def delete_file(req: DeleteRequest):
    target = Path(req.path)
    if not is_safe_path(target, get_allowed_roots()):
        raise HTTPException(403, "Access denied")
    if not target.exists():
        raise HTTPException(404, "File not found")
    try:
        from send2trash import send2trash
        send2trash(str(target))
        return {"success": True, "path": str(target), "method": "recycle_bin"}
    except ImportError:
        try:
            if target.is_dir():
                shutil.rmtree(str(target))
            else:
                target.unlink()
            return {"success": True, "path": str(target), "method": "permanent"}
        except OSError as e:
            raise HTTPException(500, f"Delete failed: {e}")


@router.post("/create-folder")
def create_folder(req: CreateFolderRequest):
    parent = Path(req.parent_path)
    if not is_safe_path(parent, get_allowed_roots()):
        raise HTTPException(403, "Access denied")
    if not parent.exists() or not parent.is_dir():
        raise HTTPException(404, "Parent not found")
    if "/" in req.folder_name or "\\" in req.folder_name:
        raise HTTPException(400, "Invalid folder name")
    new_dir = parent / req.folder_name
    if new_dir.exists():
        raise HTTPException(409, "Folder already exists")
    try:
        new_dir.mkdir(parents=False)
        return {"success": True, "path": str(new_dir)}
    except OSError as e:
        raise HTTPException(500, f"Create folder failed: {e}")


@router.get("/storage-stats")
def storage_stats():
    cat_paths = get_category_paths()
    stats = []
    for cat_id, cat_path in cat_paths.items():
        total_size = 0
        file_count = 0
        if cat_path.exists():
            try:
                for item in cat_path.rglob("*"):
                    if item.is_file():
                        try:
                            total_size += item.stat().st_size
                            file_count += 1
                        except OSError:
                            pass
            except PermissionError:
                pass
        stats.append({"category": cat_id, "path": str(cat_path), "total_size": total_size, "file_count": file_count})

    home = Path.home()
    try:
        disk_usage = shutil.disk_usage(str(home))
        disk_info = {"total": disk_usage.total, "used": disk_usage.used, "free": disk_usage.free}
    except OSError:
        disk_info = None
    return {"categories": stats, "disk": disk_info}
