from __future__ import annotations
import sys
from .fs_live_ops import live_read_file_with_remote_fallback
from .fs_path_utils import is_path_allowed, normalize_path
from .sandbox_client import call_fs_write_absolute
from app.llm.local_tools.zobie.fs_write_ops import WriteResult, _count_lines, _get_preview


DELETE_AREA_START_MARKER = "# START ASTRA_BLOCK"

DELETE_AREA_END_MARKER = "# END ASTRA_BLOCK"

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
