# FILE: D:\Orb\sandbox_controller\main.py
"""
Sandbox Controller (FastAPI)

Purpose:
- Provide safe, sandbox-scoped filesystem and repo endpoints for ASTRA/Orb.
- Enforce strict allow-list roots and deny/ignore rules so scans can't escape the sandbox.

v0.5.0 (2026-02): Fixed /shell/run encoding — uses UTF-8 instead of default cp1252
v0.4.0 (2026-01): Added /fs/contents endpoint for reading file contents
v0.3.1 (2026-01): Fixed zone detection for backend/frontend
v0.3.0 (2026-01): Added /fs/tree endpoint for multi-root scanning
v0.2.1: Various safety hardening
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# -----------------------------
# Roots / limits
# -----------------------------

def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_cache_root(repo_root: Path) -> Path:
    return (repo_root / "_sandbox_cache").resolve()


_env_repo = (
    os.environ.get("ASTRA_REPO_ROOT")
    or os.environ.get("ZOMBIEORB_REPO_ROOT")
)
_env_cache = (
    os.environ.get("ASTRA_SANDBOX_CACHE_ROOT")
    or os.environ.get("ORB_SANDBOX_CACHE_ROOT")
)

REPO_ROOT = Path(_env_repo).resolve() if _env_repo else _default_repo_root().resolve()
CACHE_ROOT = Path(_env_cache).resolve() if _env_cache else _default_cache_root(REPO_ROOT)
ARTIFACT_ROOT = (CACHE_ROOT / "jobs").resolve()
SCRATCH_ROOT = (CACHE_ROOT / "scratch").resolve()
MAX_FILE_BYTES = int(os.environ.get("ORB_SANDBOX_MAX_FILE_BYTES", "512000"))


# -----------------------------
# Allow / deny rules (filesystem scanning)
# -----------------------------

# Allowed roots for /fs/tree scans
# - Everything on D:
# - ONLY C:\Users\dizzi on C:
ALLOWED_FS_ROOTS = {
    "D:\\",                    # Everything on D: inside the sandbox
    r"C:\Users\dizzi",        # ONLY this user profile folder on C:
}

# Hard-denied roots (never scan these even if someone tries)
HARD_DENY_ROOTS = {
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"C:\ProgramData",
    r"C:\Python314",
    r"C:\OneDriveTemp",
    r"C:\PerfLogs",
    r"C:\$Recycle.Bin",
    r"C:\System Volume Information",
}

# Deny patterns for repo endpoints (relative POSIX paths under REPO_ROOT)
DENY_PATTERNS = [
    r"(^|/)\.git(/|$)",
    r"(^|/)node_modules(/|$)",
    r"(^|/)\.venv(/|$)",
    r"(^|/)venv(/|$)",
    r"(^|/)__pycache__(/|$)",
    r"(^|/)dist(/|$)",
    r"(^|/)build(/|$)",
    r"(^|/)\.pytest_cache(/|$)",
    r"(^|/)\.idea(/|$)",
    r"(^|/)\.vscode(/|$)",
    r"(^|/)logs?(/|$)",
    r"(^|/)\.cache(/|$)",
    r"(^|/)cache(/|$)",
    r"(^|/)temp(/|$)",
    r"(^|/)tmp(/|$)",
    r"(^|/)\.next(/|$)",
    r"(^|/)\.nuxt(/|$)",
    r"(^|/)\.turbo(/|$)",
    r"(^|/)\.vite(/|$)",
    r"(^|/)coverage(/|$)",
    r"(^|/)\.nyc_output(/|$)",
]
_DENY_RX = [re.compile(p) for p in DENY_PATTERNS]

DENY_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".pytest_cache", ".idea", ".vscode",
    "logs", ".cache", "cache", "temp", "tmp",
    ".next", ".nuxt", ".turbo", ".vite", "coverage", ".nyc_output",
}

FS_TREE_SKIP_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__",
    "dist", "build", ".pytest_cache", ".idea", ".vscode",
    "logs", ".cache", "cache", "temp", "tmp",
    ".next", ".nuxt", ".turbo", ".vite", "coverage", ".nyc_output",
}

FS_TREE_SKIP_EXTENSIONS = {
    ".log", ".iso", ".vhd", ".vhdx", ".qcow2", ".img", ".vdi",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2",
    ".dll", ".exe", ".sys", ".msi", ".cab",
    ".pyc", ".pyo", ".class", ".o", ".obj",
}


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="sandbox_controller", version="0.5.0")
logging.basicConfig(level=os.environ.get("SANDBOX_CONTROLLER_LOG_LEVEL", "INFO"))
log = logging.getLogger("sandbox_controller")


@app.on_event("startup")
def _startup_log() -> None:
    log.info("sandbox_controller starting")
    log.info("REPO_ROOT=%s", REPO_ROOT)
    log.info("CACHE_ROOT=%s", CACHE_ROOT)
    log.info("ALLOWED_FS_ROOTS=%s", sorted(ALLOWED_FS_ROOTS))


# -----------------------------
# Models
# -----------------------------

class FsTreeRequest(BaseModel):
    roots: List[str] = Field(..., description="Root directories to scan")
    max_files: int = Field(100000, ge=1, le=500000)
    include_size: bool = False
    include_mtime: bool = False


class FsTreeEntry(BaseModel):
    path: str
    name: str
    ext: str
    zone: str
    root: str
    size_bytes: Optional[int] = None
    mtime: Optional[str] = None


class FsTreeResponse(BaseModel):
    files: List[FsTreeEntry]
    roots_scanned: List[str]
    total_files: int
    scan_time_ms: int
    truncated: bool


class FsWriteRequest(BaseModel):
    path: str = Field(..., description="Absolute path inside allowed roots")
    content: str = Field(..., description="UTF-8 text content")
    overwrite: bool = True


class FsContentsRequest(BaseModel):
    """Request to read contents of multiple files."""
    paths: List[str] = Field(..., description="List of absolute file paths to read")
    max_file_size: int = Field(500000, ge=1, le=2000000, description="Max bytes per file (default 500KB)")
    include_line_numbers: bool = Field(True, description="Add line numbers to content")


class FsFileContent(BaseModel):
    """Content of a single file."""
    path: str
    name: str
    ext: str
    zone: str
    size_bytes: int
    line_count: int
    language: str
    content: str
    content_hash: str
    error: Optional[str] = None


class FsContentsResponse(BaseModel):
    """Response containing file contents."""
    files: List[FsFileContent]
    total_files: int
    total_bytes: int
    total_lines: int
    read_time_ms: int
    errors: int


class ShellRunRequest(BaseModel):
    cmd: List[str]
    cwd: Optional[str] = None
    timeout_sec: int = 60


# -----------------------------
# Helpers
# -----------------------------

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _get_desktop_path() -> Path:
    return Path(os.environ.get("USERPROFILE", r"C:\Users\dizzi")) / "Desktop"


def _get_documents_path() -> Path:
    return Path(os.environ.get("USERPROFILE", r"C:\Users\dizzi")) / "Documents"


def _to_posix_rel(p: Path, root: Path) -> str:
    rel = p.relative_to(root)
    return rel.as_posix()


def _is_denied_rel_posix(rel_posix: str) -> bool:
    p = "/" + rel_posix.lstrip("/")
    return any(rx.search(p) for rx in _DENY_RX)


def _ensure_under_root(p: Path, root: Path) -> Path:
    try:
        rp = p.resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    try:
        rr = root.resolve()
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid root")
    if not str(rp).lower().startswith(str(rr).lower() + os.sep.lower()) and str(rp).lower() != str(rr).lower():
        raise HTTPException(status_code=403, detail="Path not under root")
    return rp


def _is_root_allowed(root_path: str) -> bool:
    """Check if a root path is in the allowed list."""
    norm = os.path.normpath(root_path).rstrip(os.sep)
    for deny in HARD_DENY_ROOTS:
        deny_norm = os.path.normpath(deny).rstrip(os.sep)
        if norm.lower().startswith(deny_norm.lower()):
            return False
    for allowed in ALLOWED_FS_ROOTS:
        allowed_norm = os.path.normpath(allowed).rstrip(os.sep)
        if norm.lower().startswith(allowed_norm.lower()):
            return True
    return False


def _determine_zone(abs_path: str) -> str:
    """Determine logical zone for architecture mapping."""
    p = abs_path.replace("/", "\\").lower()

    # Frontend: orb-desktop
    if "\\orb-desktop\\" in p or p.startswith("d:\\orb-desktop"):
        return "frontend"

    # Backend: Orb (but not orb-desktop)
    if "\\orb\\" in p or p.startswith("d:\\orb"):
        return "backend"

    # Tools
    if "\\tools\\" in p or p.startswith("d:\\tools"):
        return "tools"

    # User profile
    if p.startswith("c:\\users"):
        return "user"

    return "other"


# -----------------------------
# API
# -----------------------------

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "cache_root": str(CACHE_ROOT),
        "artifact_root": str(ARTIFACT_ROOT),
        "scratch_root": str(SCRATCH_ROOT),
        "desktop_path": str(_get_desktop_path()),
        "documents_path": str(_get_documents_path()),
        "allowed_fs_roots": sorted(ALLOWED_FS_ROOTS),
    }


@app.get("/repo/tree")
def repo_tree(
    include_hashes: bool = Query(False),
    max_files: int = Query(80000, ge=1, le=200000),
) -> List[dict]:
    if not REPO_ROOT.exists():
        raise HTTPException(status_code=404, detail=f"REPO_ROOT not found: {REPO_ROOT}")

    out: List[dict] = []
    count = 0
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in DENY_DIRS]
        for fn in files:
            if count >= max_files:
                return out
            p = Path(root) / fn
            try:
                rel = _to_posix_rel(p, REPO_ROOT)
            except Exception:
                continue
            if _is_denied_rel_posix(rel):
                continue
            try:
                st = p.stat()
            except Exception:
                continue
            entry = {"path": rel, "size_bytes": int(st.st_size)}
            if include_hashes:
                try:
                    b = p.read_bytes()
                    if len(b) <= MAX_FILE_BYTES:
                        entry["sha256"] = _sha256_bytes(b)
                    else:
                        entry["sha256"] = None
                except Exception:
                    entry["sha256"] = None
            out.append(entry)
            count += 1
    return out


@app.get("/repo/file")
def repo_file(path: str = Query(..., description="Relative POSIX path under repo root")) -> dict:
    if not path:
        raise HTTPException(status_code=400, detail="Missing path")

    rel = path.replace("\\", "/").lstrip("/")
    if ".." in rel.split("/"):
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    if _is_denied_rel_posix(rel):
        raise HTTPException(status_code=403, detail="Denied path")

    abs_path = _ensure_under_root(REPO_ROOT / rel, REPO_ROOT)
    if not abs_path.exists() or not abs_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    b = abs_path.read_bytes()
    if len(b) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (> {MAX_FILE_BYTES} bytes)")

    content = b.decode("utf-8", errors="replace")
    return {
        "path": rel,
        "sha256": _sha256_bytes(b),
        "bytes": len(b),
        "content": content,
    }


@app.post("/fs/tree", response_model=FsTreeResponse)
def fs_tree(req: FsTreeRequest) -> FsTreeResponse:
    """
    Scan multiple root directories and return file tree.
    """
    start_time = time.time()

    valid_roots: List[str] = []
    for root in req.roots:
        if not _is_root_allowed(root):
            raise HTTPException(status_code=403, detail=f"Root not allowed: {root}")
        rp = Path(root)
        if not rp.exists() or not rp.is_dir():
            raise HTTPException(status_code=404, detail=f"Root not found: {root}")
        valid_roots.append(str(rp.resolve()))

    files: List[FsTreeEntry] = []
    truncated = False

    for root_path in valid_roots:
        root_dir = Path(root_path)
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = [d for d in dirnames if d not in FS_TREE_SKIP_DIRS]

            for fname in filenames:
                if len(files) >= req.max_files:
                    truncated = True
                    break

                ext = os.path.splitext(fname)[1].lower()
                if ext in FS_TREE_SKIP_EXTENSIONS:
                    continue

                full_path = Path(dirpath) / fname
                try:
                    rel_posix = full_path.relative_to(root_dir).as_posix()
                except ValueError:
                    continue
                if _is_denied_rel_posix(rel_posix):
                    continue

                entry = FsTreeEntry(
                    path=str(full_path),
                    name=fname,
                    ext=ext,
                    zone=_determine_zone(str(full_path)),
                    root=str(root_dir),
                )

                if req.include_size or req.include_mtime:
                    try:
                        st = full_path.stat()
                        if req.include_size:
                            entry.size_bytes = int(st.st_size)
                        if req.include_mtime:
                            entry.mtime = datetime.fromtimestamp(st.st_mtime).isoformat()
                    except Exception:
                        pass

                files.append(entry)

            if truncated:
                break
        if truncated:
            break

    elapsed_ms = int((time.time() - start_time) * 1000)

    return FsTreeResponse(
        files=files,
        roots_scanned=valid_roots,
        total_files=len(files),
        scan_time_ms=elapsed_ms,
        truncated=truncated,
    )


@app.post("/fs/write")
def fs_write(req: FsWriteRequest) -> dict:
    if not req.path:
        raise HTTPException(status_code=400, detail="Missing path")
    if not _is_root_allowed(req.path):
        raise HTTPException(status_code=403, detail="Path not under allowed roots")

    p = Path(req.path)
    if p.exists() and (not req.overwrite):
        raise HTTPException(status_code=409, detail="File exists and overwrite=false")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(req.content, encoding="utf-8")
    return {"status": "ok", "path": str(p), "bytes": len(req.content.encode("utf-8"))}


# Language detection for syntax highlighting
EXTENSION_TO_LANGUAGE = {
    ".py": "python", ".pyw": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".json": "json", ".jsonc": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini", ".cfg": "ini", ".conf": "ini",
    ".md": "markdown", ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".txt": "text",
    ".html": "html", ".htm": "html",
    ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".sql": "sql",
    ".sh": "bash", ".bash": "bash", ".zsh": "zsh",
    ".ps1": "powershell", ".psm1": "powershell",
    ".bat": "batch", ".cmd": "batch",
}


def _detect_language(ext: str, filename: str = "") -> str:
    """Detect programming language from extension or filename."""
    ext_lower = (ext or "").lower()
    if ext_lower in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[ext_lower]

    filename_lower = (filename or "").lower()
    if filename_lower == "dockerfile":
        return "dockerfile"
    if filename_lower == "makefile":
        return "makefile"
    if filename_lower in (".env", ".env.example", ".env.local"):
        return "dotenv"
    if filename_lower in (".gitignore", ".dockerignore"):
        return "gitignore"

    return "text"


@app.post("/fs/contents", response_model=FsContentsResponse)
def fs_contents(req: FsContentsRequest) -> FsContentsResponse:
    """
    Read contents of multiple files.
    Returns file contents with metadata for architecture documentation.

    - Only reads files under ALLOWED_FS_ROOTS
    - Skips files larger than max_file_size
    - Adds line numbers if requested
    """
    start_time = time.time()
    files: List[FsFileContent] = []
    total_bytes = 0
    total_lines = 0
    errors = 0

    for file_path in req.paths:
        # Security check
        if not _is_root_allowed(file_path):
            files.append(FsFileContent(
                path=file_path,
                name=os.path.basename(file_path),
                ext=os.path.splitext(file_path)[1].lower(),
                zone=_determine_zone(file_path),
                size_bytes=0,
                line_count=0,
                language="text",
                content="",
                content_hash="",
                error=f"Path not allowed: {file_path}",
            ))
            errors += 1
            continue

        p = Path(file_path)
        if not p.exists() or not p.is_file():
            files.append(FsFileContent(
                path=file_path,
                name=os.path.basename(file_path),
                ext=os.path.splitext(file_path)[1].lower(),
                zone=_determine_zone(file_path),
                size_bytes=0,
                line_count=0,
                language="text",
                content="",
                content_hash="",
                error=f"File not found: {file_path}",
            ))
            errors += 1
            continue

        try:
            stat = p.stat()
            size = int(stat.st_size)

            if size > req.max_file_size:
                files.append(FsFileContent(
                    path=file_path,
                    name=p.name,
                    ext=os.path.splitext(p.name)[1].lower(),
                    zone=_determine_zone(file_path),
                    size_bytes=size,
                    line_count=0,
                    language=_detect_language(os.path.splitext(p.name)[1], p.name),
                    content="",
                    content_hash="",
                    error=f"File too large: {size} bytes (max {req.max_file_size})",
                ))
                errors += 1
                continue

            # Read content
            raw_bytes = p.read_bytes()
            content_hash = _sha256_bytes(raw_bytes)

            # Decode as UTF-8 with error handling
            try:
                content = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                content = raw_bytes.decode("utf-8", errors="replace")

            # Count lines
            lines = content.splitlines()
            line_count = len(lines)

            # Add line numbers if requested
            if req.include_line_numbers:
                width = len(str(line_count))
                numbered_lines = [f"{i+1:>{width}}: {line}" for i, line in enumerate(lines)]
                content = "\n".join(numbered_lines)

            ext = os.path.splitext(p.name)[1].lower()
            files.append(FsFileContent(
                path=file_path,
                name=p.name,
                ext=ext,
                zone=_determine_zone(file_path),
                size_bytes=size,
                line_count=line_count,
                language=_detect_language(ext, p.name),
                content=content,
                content_hash=content_hash,
            ))
            total_bytes += size
            total_lines += line_count

        except Exception as e:
            files.append(FsFileContent(
                path=file_path,
                name=os.path.basename(file_path),
                ext=os.path.splitext(file_path)[1].lower(),
                zone=_determine_zone(file_path),
                size_bytes=0,
                line_count=0,
                language="text",
                content="",
                content_hash="",
                error=str(e),
            ))
            errors += 1

    elapsed_ms = int((time.time() - start_time) * 1000)

    return FsContentsResponse(
        files=files,
        total_files=len(files),
        total_bytes=total_bytes,
        total_lines=total_lines,
        read_time_ms=elapsed_ms,
        errors=errors,
    )


@app.post("/shell/run")
def shell_run(req: ShellRunRequest) -> dict:
    """
    Execute a shell command in the sandbox.

    v0.5.0: Uses encoding='utf-8' and errors='replace' to prevent
    UnicodeDecodeError when commands output non-ASCII characters.
    Previously used text=True which defaults to the system codepage
    (cp1252 on Windows), crashing on emoji or Unicode output.
    """
    if not req.cmd:
        raise HTTPException(status_code=400, detail="Missing cmd")

    cwd = Path(req.cwd).resolve() if req.cwd else None
    try:
        cp = subprocess.run(
            req.cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=req.timeout_sec,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Command timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {e}")

    return {
        "returncode": cp.returncode,
        "stdout": cp.stdout,
        "stderr": cp.stderr,
    }
