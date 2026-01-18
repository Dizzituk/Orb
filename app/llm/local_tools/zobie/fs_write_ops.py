# FILE: app/llm/local_tools/zobie/fs_write_ops.py
"""Stage 1 deterministic file write operations for sandbox filesystem.

This module handles surgical file edits via the sandbox controller:
- Append: Add text to end of file
- Overwrite: Replace entire file content
- Delete-Area: Remove content between marker lines (inclusive)

All operations are executed in the sandbox environment via /fs/write endpoint.
No LLM reasoning - pure deterministic tool operations.

v5.6 (2026-01-18): FIX - Treat any 2xx HTTP status as success
  - Remote agent may return 200 with empty body (no "ok" field)
  - Changed check from `status != 200 or not resp.get("ok")` to `status < 200 or status >= 300`
v5.5 (2026-01): Initial Stage 1 implementation
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

from .config import FILESYSTEM_READ_MAX_BYTES
from .fs_path_utils import normalize_path, is_path_allowed
from .fs_live_ops import live_read_file_with_remote_fallback
from .sandbox_client import call_fs_write_absolute

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Delete-area markers (case-sensitive, exact match)
DELETE_AREA_START_MARKER = "# START ASTRA_BLOCK"
DELETE_AREA_END_MARKER = "# END ASTRA_BLOCK"


# =============================================================================
# RESULT STRUCTURES
# =============================================================================

@dataclass
class WriteResult:
    """Result from a write operation."""
    status: str  # "ok" or "error"
    action: str  # "append", "overwrite", "delete_area"
    resolved_path: str
    bytes_before: int
    bytes_after: int
    lines_before: int
    lines_after: int
    preview_before: str  # Short excerpt of content before
    preview_after: str   # Short excerpt of content after
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "action": self.action,
            "resolved_path": self.resolved_path,
            "bytes_before": self.bytes_before,
            "bytes_after": self.bytes_after,
            "lines_before": self.lines_before,
            "lines_after": self.lines_after,
            "preview_before": self.preview_before,
            "preview_after": self.preview_after,
            "error": self.error,
        }
    
    def summary(self) -> str:
        """Human-readable summary of the operation."""
        if self.status == "error":
            return f"❌ {self.action.upper()} FAILED: {self.error}"
        
        delta_bytes = self.bytes_after - self.bytes_before
        delta_lines = self.lines_after - self.lines_before
        sign_bytes = "+" if delta_bytes >= 0 else ""
        sign_lines = "+" if delta_lines >= 0 else ""
        
        return (
            f"✅ {self.action.upper()} OK\n"
            f"   Path: {self.resolved_path}\n"
            f"   Size: {self.bytes_before} → {self.bytes_after} bytes ({sign_bytes}{delta_bytes})\n"
            f"   Lines: {self.lines_before} → {self.lines_after} ({sign_lines}{delta_lines})"
        )


def _get_preview(content: str, max_lines: int = 5, from_end: bool = False) -> str:
    """Get a short preview excerpt from content."""
    if not content:
        return "(empty)"
    
    lines = content.splitlines()
    if len(lines) <= max_lines:
        return content
    
    if from_end:
        excerpt = lines[-max_lines:]
        return "...\n" + "\n".join(excerpt)
    else:
        excerpt = lines[:max_lines]
        return "\n".join(excerpt) + "\n..."


def _count_lines(content: str) -> int:
    """Count lines in content."""
    if not content:
        return 0
    return len(content.splitlines())


# =============================================================================
# APPEND OPERATION
# =============================================================================

def sandbox_append_file(
    path: str,
    content_to_append: str,
    debug: bool = True,
) -> WriteResult:
    """
    Append text to end of a file via sandbox.
    
    Reads current content, appends new content (with newline if needed),
    then writes back via sandbox /fs/write endpoint.
    
    Args:
        path: Absolute path to file
        content_to_append: Text to append
        debug: Print debug info
    
    Returns:
        WriteResult with operation status and evidence
    """
    norm_path = normalize_path(path, debug=debug)
    
    if debug:
        print(f"[FS_WRITE] === APPEND START ===", file=sys.stderr)
        print(f"[FS_WRITE] path={repr(norm_path)}", file=sys.stderr)
        print(f"[FS_WRITE] content_len={len(content_to_append)}", file=sys.stderr)
    
    # Validate path
    allowed, reason = is_path_allowed(norm_path)
    if not allowed:
        return WriteResult(
            status="error",
            action="append",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"Path blocked: {reason}",
        )
    
    # Read current content
    current_content, total_lines, total_bytes, truncated, read_error, source = \
        live_read_file_with_remote_fallback(norm_path, debug=debug)
    
    if read_error and current_content is None:
        # File might not exist - that's OK for append, start with empty
        if "not found" in read_error.lower() or "FileNotFoundError" in read_error:
            current_content = ""
            total_lines = 0
            total_bytes = 0
            if debug:
                print(f"[FS_WRITE] File not found, will create new", file=sys.stderr)
        else:
            return WriteResult(
                status="error",
                action="append",
                resolved_path=norm_path,
                bytes_before=0,
                bytes_after=0,
                lines_before=0,
                lines_after=0,
                preview_before="",
                preview_after="",
                error=f"Read failed: {read_error}",
            )
    
    # Prepare new content
    # Add newline separator if current content doesn't end with newline
    if current_content and not current_content.endswith('\n'):
        new_content = current_content + '\n' + content_to_append
    else:
        new_content = (current_content or "") + content_to_append
    
    # Ensure trailing newline
    if new_content and not new_content.endswith('\n'):
        new_content += '\n'
    
    # Capture before state
    bytes_before = len((current_content or "").encode('utf-8', errors='replace'))
    lines_before = _count_lines(current_content or "")
    preview_before = _get_preview(current_content or "", from_end=True)
    
    # Write via sandbox
    status, resp, write_error = call_fs_write_absolute(
        absolute_path=norm_path,
        content=new_content,
        overwrite=True,
    )
    
    if debug:
        print(f"[FS_WRITE] write status={status}", file=sys.stderr)
        if write_error:
            print(f"[FS_WRITE] write_error={write_error}", file=sys.stderr)
    
    # Check for write failure
    if status is None:
        return WriteResult(
            status="error",
            action="append",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Sandbox connection failed: {write_error}",
        )
    
    if status == 404:
        return WriteResult(
            status="error",
            action="append",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error="Sandbox controller doesn't support writes (404 on /fs/write)",
        )
    
    # v5.6 FIX: Treat any 2xx status as success (remote may return empty body)
    if status < 200 or status >= 300:
        error_detail = write_error or (resp.get("error") if resp else "Unknown error")
        return WriteResult(
            status="error",
            action="append",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Write failed (HTTP {status}): {error_detail}",
        )
    
    # Success - capture after state
    bytes_after = len(new_content.encode('utf-8', errors='replace'))
    lines_after = _count_lines(new_content)
    preview_after = _get_preview(new_content, from_end=True)
    
    if debug:
        print(f"[FS_WRITE] APPEND SUCCESS: {bytes_before} → {bytes_after} bytes", file=sys.stderr)
        print(f"[FS_WRITE] === APPEND END ===", file=sys.stderr)
    
    return WriteResult(
        status="ok",
        action="append",
        resolved_path=resp.get("path", norm_path) if resp else norm_path,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lines_before=lines_before,
        lines_after=lines_after,
        preview_before=preview_before,
        preview_after=preview_after,
    )


# =============================================================================
# OVERWRITE OPERATION
# =============================================================================

def sandbox_overwrite_file(
    path: str,
    new_content: str,
    debug: bool = True,
) -> WriteResult:
    """
    Replace entire file content via sandbox.
    
    Args:
        path: Absolute path to file
        new_content: Complete new content
        debug: Print debug info
    
    Returns:
        WriteResult with operation status and evidence
    """
    norm_path = normalize_path(path, debug=debug)
    
    if debug:
        print(f"[FS_WRITE] === OVERWRITE START ===", file=sys.stderr)
        print(f"[FS_WRITE] path={repr(norm_path)}", file=sys.stderr)
        print(f"[FS_WRITE] new_content_len={len(new_content)}", file=sys.stderr)
    
    # Validate path
    allowed, reason = is_path_allowed(norm_path)
    if not allowed:
        return WriteResult(
            status="error",
            action="overwrite",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"Path blocked: {reason}",
        )
    
    # Read current content for evidence
    current_content, total_lines, total_bytes, truncated, read_error, source = \
        live_read_file_with_remote_fallback(norm_path, debug=debug)
    
    # Capture before state (file might not exist)
    if current_content is None:
        bytes_before = 0
        lines_before = 0
        preview_before = "(file did not exist)"
    else:
        bytes_before = len(current_content.encode('utf-8', errors='replace'))
        lines_before = _count_lines(current_content)
        preview_before = _get_preview(current_content, from_end=False)
    
    # Ensure trailing newline
    content_to_write = new_content
    if content_to_write and not content_to_write.endswith('\n'):
        content_to_write += '\n'
    
    # Write via sandbox
    status, resp, write_error = call_fs_write_absolute(
        absolute_path=norm_path,
        content=content_to_write,
        overwrite=True,
    )
    
    if debug:
        print(f"[FS_WRITE] write status={status}", file=sys.stderr)
        if write_error:
            print(f"[FS_WRITE] write_error={write_error}", file=sys.stderr)
    
    # Check for write failure
    if status is None:
        return WriteResult(
            status="error",
            action="overwrite",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Sandbox connection failed: {write_error}",
        )
    
    if status == 404:
        return WriteResult(
            status="error",
            action="overwrite",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error="Sandbox controller doesn't support writes (404 on /fs/write)",
        )
    
    # v5.6 FIX: Treat any 2xx status as success (remote may return empty body)
    if status < 200 or status >= 300:
        error_detail = write_error or (resp.get("error") if resp else "Unknown error")
        return WriteResult(
            status="error",
            action="overwrite",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Write failed (HTTP {status}): {error_detail}",
        )
    
    # Success - capture after state
    bytes_after = len(content_to_write.encode('utf-8', errors='replace'))
    lines_after = _count_lines(content_to_write)
    preview_after = _get_preview(content_to_write, from_end=False)
    
    if debug:
        print(f"[FS_WRITE] OVERWRITE SUCCESS: {bytes_before} → {bytes_after} bytes", file=sys.stderr)
        print(f"[FS_WRITE] === OVERWRITE END ===", file=sys.stderr)
    
    return WriteResult(
        status="ok",
        action="overwrite",
        resolved_path=resp.get("path", norm_path) if resp else norm_path,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lines_before=lines_before,
        lines_after=lines_after,
        preview_before=preview_before,
        preview_after=preview_after,
    )


# =============================================================================
# DELETE-AREA OPERATION
# =============================================================================

def sandbox_delete_area(
    path: str,
    start_marker: str = DELETE_AREA_START_MARKER,
    end_marker: str = DELETE_AREA_END_MARKER,
    debug: bool = True,
) -> WriteResult:
    """
    Delete content between markers (inclusive) via sandbox.
    
    Markers are matched exactly (case-sensitive).
    Both the start marker line and end marker line are removed.
    
    Args:
        path: Absolute path to file
        start_marker: Start marker text (exact match)
        end_marker: End marker text (exact match)
        debug: Print debug info
    
    Returns:
        WriteResult with operation status and evidence
    """
    norm_path = normalize_path(path, debug=debug)
    
    if debug:
        print(f"[FS_WRITE] === DELETE_AREA START ===", file=sys.stderr)
        print(f"[FS_WRITE] path={repr(norm_path)}", file=sys.stderr)
        print(f"[FS_WRITE] start_marker={repr(start_marker)}", file=sys.stderr)
        print(f"[FS_WRITE] end_marker={repr(end_marker)}", file=sys.stderr)
    
    # Validate path
    allowed, reason = is_path_allowed(norm_path)
    if not allowed:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"Path blocked: {reason}",
        )
    
    # Read current content
    current_content, total_lines, total_bytes, truncated, read_error, source = \
        live_read_file_with_remote_fallback(norm_path, debug=debug)
    
    if read_error or current_content is None:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"Read failed: {read_error or 'No content'}",
        )
    
    # Capture before state
    bytes_before = len(current_content.encode('utf-8', errors='replace'))
    lines_before = _count_lines(current_content)
    
    # Find markers
    lines = current_content.splitlines()
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        # Exact match (stripped to handle trailing whitespace)
        if line.strip() == start_marker.strip():
            start_idx = i
        elif line.strip() == end_marker.strip():
            end_idx = i
            break  # Stop at first end marker after start
    
    if debug:
        print(f"[FS_WRITE] start_idx={start_idx} end_idx={end_idx}", file=sys.stderr)
    
    # Validate markers found
    if start_idx is None:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=bytes_before,
            lines_before=lines_before,
            lines_after=lines_before,
            preview_before=_get_preview(current_content),
            preview_after="",
            error=f"Start marker not found: {repr(start_marker)}",
        )
    
    if end_idx is None:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=bytes_before,
            lines_before=lines_before,
            lines_after=lines_before,
            preview_before=_get_preview(current_content),
            preview_after="",
            error=f"End marker not found: {repr(end_marker)}",
        )
    
    if end_idx <= start_idx:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=bytes_before,
            lines_before=lines_before,
            lines_after=lines_before,
            preview_before=_get_preview(current_content),
            preview_after="",
            error=f"End marker (line {end_idx+1}) must come after start marker (line {start_idx+1})",
        )
    
    # Delete inclusive: remove lines from start_idx to end_idx (inclusive)
    deleted_lines = lines[start_idx:end_idx + 1]
    new_lines = lines[:start_idx] + lines[end_idx + 1:]
    new_content = '\n'.join(new_lines)
    
    # Ensure trailing newline
    if new_content and not new_content.endswith('\n'):
        new_content += '\n'
    
    preview_before = (
        f"Lines {start_idx+1}-{end_idx+1} to delete:\n" +
        '\n'.join(deleted_lines[:5]) +
        ('\n...' if len(deleted_lines) > 5 else '')
    )
    
    if debug:
        print(f"[FS_WRITE] Deleting {len(deleted_lines)} lines ({start_idx+1} to {end_idx+1})", file=sys.stderr)
    
    # Write via sandbox
    status, resp, write_error = call_fs_write_absolute(
        absolute_path=norm_path,
        content=new_content,
        overwrite=True,
    )
    
    if debug:
        print(f"[FS_WRITE] write status={status}", file=sys.stderr)
        if write_error:
            print(f"[FS_WRITE] write_error={write_error}", file=sys.stderr)
    
    # Check for write failure
    if status is None:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Sandbox connection failed: {write_error}",
        )
    
    if status == 404:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error="Sandbox controller doesn't support writes (404 on /fs/write)",
        )
    
    # v5.6 FIX: Treat any 2xx status as success (remote may return empty body)
    if status < 200 or status >= 300:
        error_detail = write_error or (resp.get("error") if resp else "Unknown error")
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Write failed (HTTP {status}): {error_detail}",
        )
    
    # Success
    bytes_after = len(new_content.encode('utf-8', errors='replace'))
    lines_after = _count_lines(new_content)
    preview_after = _get_preview(new_content)
    
    if debug:
        print(f"[FS_WRITE] DELETE_AREA SUCCESS: removed {len(deleted_lines)} lines", file=sys.stderr)
        print(f"[FS_WRITE] === DELETE_AREA END ===", file=sys.stderr)
    
    return WriteResult(
        status="ok",
        action="delete_area",
        resolved_path=resp.get("path", norm_path) if resp else norm_path,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lines_before=lines_before,
        lines_after=lines_after,
        preview_before=preview_before,
        preview_after=preview_after,
    )


# =============================================================================
# LINE RANGE DELETE (alternative to markers)
# =============================================================================

def sandbox_delete_lines(
    path: str,
    start_line: int,
    end_line: int,
    debug: bool = True,
) -> WriteResult:
    """
    Delete specific line range (1-indexed, inclusive) via sandbox.
    
    Args:
        path: Absolute path to file
        start_line: First line to delete (1-indexed)
        end_line: Last line to delete (1-indexed, inclusive)
        debug: Print debug info
    
    Returns:
        WriteResult with operation status and evidence
    """
    norm_path = normalize_path(path, debug=debug)
    
    if debug:
        print(f"[FS_WRITE] === DELETE_LINES START ===", file=sys.stderr)
        print(f"[FS_WRITE] path={repr(norm_path)}", file=sys.stderr)
        print(f"[FS_WRITE] range={start_line}-{end_line}", file=sys.stderr)
    
    # Validate path
    allowed, reason = is_path_allowed(norm_path)
    if not allowed:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"Path blocked: {reason}",
        )
    
    # Validate line range
    if start_line < 1:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"start_line must be >= 1 (got {start_line})",
        )
    
    if end_line < start_line:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"end_line ({end_line}) must be >= start_line ({start_line})",
        )
    
    # Read current content
    current_content, total_lines, total_bytes, truncated, read_error, source = \
        live_read_file_with_remote_fallback(norm_path, debug=debug)
    
    if read_error or current_content is None:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=0,
            bytes_after=0,
            lines_before=0,
            lines_after=0,
            preview_before="",
            preview_after="",
            error=f"Read failed: {read_error or 'No content'}",
        )
    
    lines = current_content.splitlines()
    bytes_before = len(current_content.encode('utf-8', errors='replace'))
    lines_before = len(lines)
    
    # Validate range against file
    if start_line > lines_before:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=bytes_before,
            lines_before=lines_before,
            lines_after=lines_before,
            preview_before=_get_preview(current_content),
            preview_after="",
            error=f"start_line ({start_line}) exceeds file length ({lines_before} lines)",
        )
    
    # Clamp end_line to file length
    actual_end = min(end_line, lines_before)
    
    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = actual_end - 1
    
    # Delete lines
    deleted_lines = lines[start_idx:end_idx + 1]
    new_lines = lines[:start_idx] + lines[end_idx + 1:]
    new_content = '\n'.join(new_lines)
    
    if new_content and not new_content.endswith('\n'):
        new_content += '\n'
    
    preview_before = (
        f"Lines {start_line}-{actual_end} to delete:\n" +
        '\n'.join(deleted_lines[:5]) +
        ('\n...' if len(deleted_lines) > 5 else '')
    )
    
    # Write via sandbox
    status, resp, write_error = call_fs_write_absolute(
        absolute_path=norm_path,
        content=new_content,
        overwrite=True,
    )
    
    if status is None:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Sandbox connection failed: {write_error}",
        )
    
    if status == 404:
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error="Sandbox controller doesn't support writes (404 on /fs/write)",
        )
    
    # v5.6 FIX: Treat any 2xx status as success (remote may return empty body)
    if status < 200 or status >= 300:
        error_detail = write_error or (resp.get("error") if resp else "Unknown error")
        return WriteResult(
            status="error",
            action="delete_area",
            resolved_path=norm_path,
            bytes_before=bytes_before,
            bytes_after=0,
            lines_before=lines_before,
            lines_after=0,
            preview_before=preview_before,
            preview_after="",
            error=f"Write failed (HTTP {status}): {error_detail}",
        )
    
    bytes_after = len(new_content.encode('utf-8', errors='replace'))
    lines_after = _count_lines(new_content)
    
    if debug:
        print(f"[FS_WRITE] DELETE_LINES SUCCESS: removed {len(deleted_lines)} lines", file=sys.stderr)
        print(f"[FS_WRITE] === DELETE_LINES END ===", file=sys.stderr)
    
    return WriteResult(
        status="ok",
        action="delete_area",
        resolved_path=resp.get("path", norm_path) if resp else norm_path,
        bytes_before=bytes_before,
        bytes_after=bytes_after,
        lines_before=lines_before,
        lines_after=lines_after,
        preview_before=preview_before,
        preview_after=_get_preview(new_content),
    )
