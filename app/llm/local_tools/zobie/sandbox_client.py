# FILE: app/llm/local_tools/zobie/sandbox_client.py
"""Sandbox controller HTTP client functions.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same HTTP call behavior.
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
