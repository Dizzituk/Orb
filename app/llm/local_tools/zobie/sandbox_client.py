# FILE: app/llm/local_tools/zobie/sandbox_client.py
"""Sandbox controller HTTP client functions.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same HTTP call behavior.

v5.6 (2026-01-18): FIX - call_fs_write now sends {"path": ...} instead of {"target": ..., "filename": ...}
  - Remote API schema requires "path" field, was getting HTTP 422
v5.5 (2026-01): Added call_fs_write for Stage 1 deterministic file editing
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request, error as urllib_error

from .config import (
    SANDBOX_CONTROLLER_URL,
    MAX_CONTENT_FILE_SIZE,
    FS_TREE_TIMEOUT_SEC,
)


def call_fs_tree(
    roots: List[str],
    max_files: int = 100000,
    timeout: int = FS_TREE_TIMEOUT_SEC,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
    """
    Call sandbox_controller /fs/tree endpoint.
    
    Returns: (status_code, response_data, error_message)
    """
    url = f"{SANDBOX_CONTROLLER_URL.rstrip('/')}/fs/tree"
    
    payload = json.dumps({
        "roots": roots,
        "max_files": max_files,
        "include_size": True,
        "include_mtime": True,
    }).encode("utf-8")
    
    try:
        req = urllib_request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode() or 200, json.loads(body), ""
    except urllib_error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, None, body
    except urllib_error.URLError as e:
        return None, None, f"Connection failed: {e.reason}"
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON response: {e}"
    except Exception as e:
        return None, None, str(e)


def call_fs_contents(
    paths: List[str],
    max_file_size: int = MAX_CONTENT_FILE_SIZE,
    include_line_numbers: bool = False,
    timeout: int = FS_TREE_TIMEOUT_SEC,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
    """
    Call sandbox_controller /fs/contents endpoint to read file contents.
    
    Returns: (status_code, response_data, error_message)
    """
    url = f"{SANDBOX_CONTROLLER_URL.rstrip('/')}/fs/contents"
    
    payload = json.dumps({
        "paths": paths,
        "max_file_size": max_file_size,
        "include_line_numbers": include_line_numbers,
    }).encode("utf-8")
    
    try:
        req = urllib_request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode() or 200, json.loads(body), ""
    except urllib_error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, None, body
    except urllib_error.URLError as e:
        return None, None, f"Connection failed: {e.reason}"
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON response: {e}"
    except Exception as e:
        return None, None, str(e)


def call_fs_write(
    target: str,
    filename: str,
    content: str,
    subdir: Optional[str] = None,
    overwrite: bool = True,
    timeout: int = FS_TREE_TIMEOUT_SEC,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
    """
    Call sandbox_controller /fs/write endpoint to write file contents.
    
    This is the Stage 1 deterministic write primitive.
    Uses the same endpoint as app/overwatcher/sandbox_client.py.
    
    v5.6 (2026-01-18): FIX - Remote API expects {"path": ...} not {"target": ..., "filename": ...}
    
    Args:
        target: Target location (e.g., "REPO", "SCRATCH", "DESKTOP", or absolute path/directory)
        filename: File name to write
        content: File content to write
        subdir: Optional subdirectory within target
        overwrite: Allow overwriting existing file (default True for Stage 1)
        timeout: Request timeout in seconds
    
    Returns: 
        (status_code, response_data, error_message)
        
        response_data on success:
        {
            "ok": bool,
            "path": str,      # Resolved absolute path
            "bytes": int,     # Bytes written
            "sha256": str,    # Content hash
        }
    
    If endpoint returns 404, the sandbox controller doesn't support writes.
    """
    import os
    
    url = f"{SANDBOX_CONTROLLER_URL.rstrip('/')}/fs/write"
    
    # v5.6 FIX: Build full path - remote API expects "path" not "target"+"filename"
    if subdir:
        full_path = os.path.join(target, subdir, filename)
    else:
        full_path = os.path.join(target, filename)
    
    # Normalize to Windows backslashes
    full_path = full_path.replace('/', '\\')
    
    payload_dict: Dict[str, Any] = {
        "path": full_path,
        "content": content,
        "overwrite": overwrite,
    }
    
    payload = json.dumps(payload_dict).encode("utf-8")
    
    try:
        req = urllib_request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode() or 200, json.loads(body), ""
    except urllib_error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = str(e)
        return e.code, None, body
    except urllib_error.URLError as e:
        return None, None, f"Connection failed: {e.reason}"
    except json.JSONDecodeError as e:
        return None, None, f"Invalid JSON response: {e}"
    except Exception as e:
        return None, None, str(e)


def call_fs_write_absolute(
    absolute_path: str,
    content: str,
    overwrite: bool = True,
    timeout: int = FS_TREE_TIMEOUT_SEC,
) -> Tuple[Optional[int], Optional[Dict[str, Any]], str]:
    """
    Write to an absolute path via sandbox controller.
    
    This is a convenience wrapper that splits an absolute path into
    target (drive/parent) and filename for the /fs/write endpoint.
    
    Args:
        absolute_path: Full absolute path (e.g., "D:\\test file\\ME")
        content: File content to write
        overwrite: Allow overwriting existing file
        timeout: Request timeout
    
    Returns:
        (status_code, response_data, error_message)
    """
    import os
    
    # Normalize path
    norm_path = absolute_path.replace('/', '\\')
    
    # Split into directory and filename
    parent_dir = os.path.dirname(norm_path)
    filename = os.path.basename(norm_path)
    
    if not filename:
        return None, None, "Cannot write to a directory path"
    
    if not parent_dir:
        return None, None, "Cannot determine parent directory"
    
    # Use parent directory as target, no subdir needed
    # The sandbox controller should handle absolute paths
    return call_fs_write(
        target=parent_dir,
        filename=filename,
        content=content,
        subdir=None,
        overwrite=overwrite,
        timeout=timeout,
    )
