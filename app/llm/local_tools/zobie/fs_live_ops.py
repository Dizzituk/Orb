# FILE: app/llm/local_tools/zobie/fs_live_ops.py
# Purpose: Live filesystem operations for the filesystem query system.
# Called-by: app.llm.local_tools.zobie._fs_live_ops_utils_1, app.llm.local_tools.zobie._fs_write_ops_utils_2, app.llm.local_tools.zobie.fs_write_ops, app.llm.local_tools.zobie.streams._fs_query_utils_3 (+2 more)
# Depends-on: app.llm.local_tools.zobie._fs_live_ops_utils_1, app.llm.local_tools.zobie.config, app.llm.local_tools.zobie.fs_path_utils, app.llm.local_tools.zobie.sandbox_client (+1 more)
# Last-renovated: 2026-06-11
"""Live filesystem operations for the filesystem query system.

v10.0 (2026-03): Sandbox-only. All reads go through the sandbox controller.
No host filesystem access. No encoding fallback chains.
If the sandbox can't read it, the file doesn't exist.

v5.4 (2026-01): Added remote_agent fallback via sandbox controller for OneDrive paths
v5.3 (2026-01): Added extended-length path syntax + multiple fallback methods
"""

from __future__ import annotations

import logging
import sys
from typing import List, Optional, Tuple

from .config import (
    FILESYSTEM_READ_MAX_LINES,
    FILESYSTEM_READ_MAX_BYTES,
)
from .fs_path_utils import normalize_path, is_path_allowed
from .sandbox_client import call_fs_contents
from app.sandbox_fs import sandbox_read_text, sandbox_exists, sandbox_listdir
from app.llm.local_tools.zobie._fs_live_ops_utils_1 import (
    check_path_exists,
    live_list_directory,
    live_read_file_with_remote_fallback,
    remote_read_file,
)

logger = logging.getLogger(__name__)


def live_read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
    debug: bool = True,
) -> Tuple[Optional[str], int, int, bool, str]:
    """
    Read a file from the sandbox (v10.0 — sandbox-only, no host fallbacks).

    Args:
        path: Absolute path to file (should already be normalized)
        start_line: Start line for range (1-indexed, inclusive)
        end_line: End line for range (1-indexed, inclusive)
        head_lines: Number of lines from start (for head command)
        debug: If True, print debug info

    Returns:
        (content, total_lines, total_bytes, truncated, error_msg)
    """
    norm_path = normalize_path(path, debug=debug)

    if debug:
        print(f"[FS_LIVE] === SANDBOX READ START ===", file=sys.stderr)
        print(f"[FS_LIVE] path={repr(norm_path)}", file=sys.stderr)

    try:
        # Single sandbox read — no fallback chain
        content = sandbox_read_text(norm_path)

        if content is None:
            if debug:
                print(f"[FS_LIVE] FAILURE: file not found in sandbox", file=sys.stderr)
            return None, 0, 0, False, f"File not found in sandbox: {norm_path}"

        # Check for binary content (null bytes in first 1000 chars)
        if '\x00' in content[:1000]:
            return None, 0, len(content), False, "File appears to be binary"

        lines = content.splitlines()
        total_lines = len(lines)
        total_bytes = len(content.encode('utf-8', errors='replace'))

        truncated = False

        # Apply line range if specified
        if start_line is not None and end_line is not None:
            start_idx = max(0, start_line - 1)
            end_idx = min(total_lines, end_line)
            lines = lines[start_idx:end_idx]

        elif head_lines is not None:
            if head_lines < total_lines:
                lines = lines[:head_lines]
                truncated = True
        else:
            # Apply default limits
            if total_lines > FILESYSTEM_READ_MAX_LINES:
                lines = lines[:FILESYSTEM_READ_MAX_LINES]
                truncated = True

        result_text = '\n'.join(lines)

        # Check byte limit
        result_bytes = len(result_text.encode('utf-8', errors='replace'))
        if result_bytes > FILESYSTEM_READ_MAX_BYTES:
            result_text = result_text[:FILESYSTEM_READ_MAX_BYTES]
            last_nl = result_text.rfind('\n')
            if last_nl > FILESYSTEM_READ_MAX_BYTES // 2:
                result_text = result_text[:last_nl]
            truncated = True

        if debug:
            print(
                f"[FS_LIVE] SUCCESS: {total_lines} lines, {total_bytes} bytes, "
                f"truncated={truncated}",
                file=sys.stderr,
            )
            print(f"[FS_LIVE] === SANDBOX READ END ===", file=sys.stderr)

        return result_text, total_lines, total_bytes, truncated, ""

    except Exception as e:
        error_type = type(e).__name__
        if debug:
            print(f"[FS_LIVE] UNEXPECTED ERROR: {error_type}: {e}", file=sys.stderr)
        return None, 0, 0, False, f"Unexpected error ({error_type}): {e}"
