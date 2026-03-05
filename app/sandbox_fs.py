# FILE: app/sandbox_fs.py
"""
Sandbox Filesystem Helpers — single entry point for all pipeline stages.

v3.2.1 (2026-02-28): Fixed to use actual sandbox controller API endpoints.

The sandbox controller exposes:
    POST /fs/tree     — list files under root directories
    POST /fs/contents — read file contents

Existence checks:
    - Directory: POST /fs/tree with path as root → 200 = exists, 404 = not found
    - File:      POST /fs/contents → check for "error" field in response

USAGE:
    from app.sandbox_fs import sandbox_isfile, sandbox_isdir, sandbox_exists
    from app.sandbox_fs import sandbox_read_text, sandbox_listdir
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request, error as urllib_error

logger = logging.getLogger(__name__)

_DEFAULT_URL = "http://192.168.250.2:8765"
_CONTROLLER_ADDR_FILE = r"C:\Orb\_sandbox_cache\controller_addr.txt"
_TIMEOUT = 15


def _get_controller_url() -> str:
    try:
        if os.path.isfile(_CONTROLLER_ADDR_FILE):
            with open(_CONTROLLER_ADDR_FILE, "r") as f:
                addr = f.read().strip()
                if addr:
                    return f"http://{addr}"
    except Exception:
        pass
    return _DEFAULT_URL


def _normalise_sandbox_path(path: str) -> str:
    """Normalise a path for the sandbox controller.

    The sandbox controller expects forward slashes. Windows backslash
    paths (D:\\orb-desktop\\src\\...) fail with 'File not found' while
    forward slash equivalents (D:/orb-desktop/src/...) succeed.
    """
    return path.replace("\\", "/")


def _post(endpoint: str, payload: dict) -> Tuple[Optional[int], Optional[dict], str]:
    """Generic POST to sandbox controller. Returns (status, body_dict, error)."""
    url = f"{_get_controller_url().rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib_request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            status = resp.getcode() or 200
            return status, json.loads(body) if body.strip() else {}, ""
    except urllib_error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, None, body
    except urllib_error.URLError as e:
        return None, None, f"Connection failed: {e.reason}"
    except Exception as e:
        return None, None, str(e)


# ---------------------------------------------------------------------------
# IO Tracking Helpers (Pipeline Logging Overhaul)
# ---------------------------------------------------------------------------

def _get_tracker():
    """Get the active IOTracker from the context variable, or None."""
    try:
        from app.transparency.io_tracker import get_active_tracker
        return get_active_tracker()
    except ImportError:
        return None


def _track_read(path: str, bytes_count: int = 0) -> None:
    """If an IOTracker is active, record a sandbox file read."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_read(
            path=path,
            source="sandbox",
            bytes_count=bytes_count,
        )


def _track_exists(path: str, result: bool) -> None:
    """If an IOTracker is active, record a sandbox existence check."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_exists_check(
            path=path,
            source="sandbox",
            result=result,
        )


def _track_dir_scan(path: str, file_count: int = 0) -> None:
    """If an IOTracker is active, record a sandbox directory scan."""
    tracker = _get_tracker()
    if tracker:
        tracker.record_dir_scan(
            path=path,
            source="sandbox",
            file_count=file_count,
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def sandbox_isdir(path: str) -> bool:
    """Check if path is a directory in the sandbox.

    Strategy: POST /fs/tree with path as root.
    - 200 → directory exists (even if empty)
    - 404 → "Root not found" → doesn't exist
    - 403 → relative path rejected by controller
    - None → sandbox unreachable
    """
    # Controller rejects relative paths with 403. Skip obviously relative ones.
    if not os.path.isabs(path):
        return False
    path = _normalise_sandbox_path(path)
    status, data, error = _post("/fs/tree", {"roots": [path], "max_files": 1})
    if status is None:
        logger.error("[sandbox_fs] sandbox_isdir FAILED for %s — unreachable: %s", path, error)
        return False
    if status in (404, 403):
        _track_exists(path, False)
        return False
    if status == 200:
        _track_exists(path, True)
        return True
    logger.warning("[sandbox_fs] sandbox_isdir unexpected status %s for %s: %s", status, path, error)
    return False


def sandbox_isfile(path: str) -> bool:
    """Check if path is a file in the sandbox.

    Strategy: POST /fs/contents and check for error field.
    - 200 + no error → file exists
    - 200 + error "File not found" → doesn't exist
    - Non-200 / None → sandbox unreachable
    """
    path = _normalise_sandbox_path(path)
    status, data, error = _post("/fs/contents", {
        "paths": [path], "max_file_size": 100,
    })
    if status is None:
        logger.error("[sandbox_fs] sandbox_isfile FAILED for %s — unreachable: %s", path, error)
        return False
    if status != 200 or not data:
        return False

    files = data.get("files", [])
    if not files:
        _track_exists(path, False)
        return False
    entry = files[0]
    file_error = entry.get("error", "")
    if file_error:
        # "File too large" means the file EXISTS but exceeds max_file_size.
        # That's not "file not found" — the file is real.
        if "too large" in file_error.lower():
            _track_exists(path, True)
            return True
        # "File not found" or other errors mean the file doesn't exist
        _track_exists(path, False)
        return False
    _track_exists(path, True)
    return True


def sandbox_exists(path: str) -> bool:
    """Check if path exists (file or directory) in the sandbox."""
    # Try directory first (cheaper — just a tree probe)
    if sandbox_isdir(path):
        return True
    # Then try file
    return sandbox_isfile(path)


def sandbox_read_text(path: str) -> Optional[str]:
    """Read a text file from the sandbox. Returns None if unreadable.

    If an IOTracker is active (set via context variable by the current
    pipeline stage), the read is automatically logged with source="sandbox".
    """
    path = _normalise_sandbox_path(path)
    status, data, error = _post("/fs/contents", {"paths": [path]})
    if status is None:
        logger.error("[sandbox_fs] sandbox_read_text FAILED for %s — unreachable: %s", path, error)
        return None
    if status != 200 or not data:
        logger.warning("[sandbox_fs] sandbox_read_text non-200 for %s: status=%s", path, status)
        return None

    files = data.get("files", [])
    if not files:
        return None

    entry = files[0]
    if entry.get("error"):
        logger.info("[sandbox_fs] sandbox_read_text: file error for %s: %s", path, entry["error"])
        return None

    content = entry.get("content")
    if content is not None:
        _track_read(path, len(content))
        return content
    return None


def sandbox_listdir(path: str) -> List[Dict[str, Any]]:
    """List directory contents in the sandbox.

    Returns list of dicts with at minimum 'path', 'name' keys.
    Returns empty list on failure.
    """
    path = _normalise_sandbox_path(path)
    status, data, error = _post("/fs/tree", {
        "roots": [path], "max_files": 500, "include_size": True,
    })
    if status is None or status != 200 or not data:
        logger.error("[sandbox_fs] sandbox_listdir FAILED for %s: status=%s, err=%s", path, status, error)
        return []
    result = data.get("files", [])
    _track_dir_scan(path, len(result))
    return result


def sandbox_file_size(path: str) -> Optional[int]:
    """Get file size in sandbox. Returns None if not found."""
    path = _normalise_sandbox_path(path)
    status, data, error = _post("/fs/contents", {
        "paths": [path], "max_file_size": 100,
    })
    if status != 200 or not data:
        return None
    files = data.get("files", [])
    if not files:
        return None
    entry = files[0]
    if entry.get("error"):
        return None
    return entry.get("size_bytes")
