from __future__ import annotations
import os
import sys
from .config import FILESYSTEM_READ_MAX_BYTES, FILESYSTEM_READ_MAX_LINES
from .fs_path_utils import is_path_allowed, normalize_path
from .sandbox_client import call_fs_contents
from pathlib import Path
from typing import List, Optional, Tuple


def remote_read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
    debug: bool = True,
) -> Tuple[Optional[str], int, int, bool, str]:
    """
    Read a file via sandbox controller's /fs/contents endpoint.
    
    This is used as a fallback when local filesystem access fails
    (e.g., for OneDrive paths that the backend process cannot see).
    
    Args:
        path: Absolute path to file (should already be normalized)
        start_line: Start line for range (1-indexed, inclusive)
        end_line: End line for range (1-indexed, inclusive)
        head_lines: Number of lines from start (for head command)
        debug: If True, print debug info
    
    Returns:
        (content, total_lines, total_bytes, truncated, error_msg)
        Same format as live_read_file for compatibility.
    """
    # Normalize path
    norm_path = normalize_path(path, debug=debug)
    
    if debug:
        print(f"[FS_REMOTE] === REMOTE READ START ===", file=sys.stderr)
        print(f"[FS_REMOTE] path={repr(norm_path)}", file=sys.stderr)
    
    try:
        # Call sandbox controller /fs/contents endpoint
        status, resp, error_msg = call_fs_contents(
            paths=[norm_path],
            max_file_size=FILESYSTEM_READ_MAX_BYTES,
            include_line_numbers=False,
            timeout=30,
        )
        
        if debug:
            print(f"[FS_REMOTE] call_fs_contents status={status}", file=sys.stderr)
        
        # Handle connection/HTTP errors
        if status is None:
            if debug:
                print(f"[FS_REMOTE] Connection failed: {error_msg}", file=sys.stderr)
            return None, 0, 0, False, f"Remote read failed (connection): {error_msg}"
        
        if status != 200:
            if debug:
                print(f"[FS_REMOTE] HTTP error {status}: {error_msg}", file=sys.stderr)
            return None, 0, 0, False, f"Remote read failed (HTTP {status}): {error_msg}"
        
        if not resp:
            return None, 0, 0, False, "Remote read failed: empty response"
        
        # Extract file content from response
        files = resp.get("files", [])
        if not files:
            if debug:
                print(f"[FS_REMOTE] No files in response", file=sys.stderr)
            return None, 0, 0, False, "Remote read failed: file not found"
        
        file_data = files[0]
        
        # Check for file-level error
        if file_data.get("error"):
            error = file_data.get("error", "Unknown error")
            if debug:
                print(f"[FS_REMOTE] File error: {error}", file=sys.stderr)
            return None, 0, 0, False, f"Remote read failed: {error}"
        
        content = file_data.get("content", "")
        if content is None:
            return None, 0, 0, False, "Remote read failed: no content"
        
        if debug:
            print(f"[FS_REMOTE] Got {len(content)} chars", file=sys.stderr)
        
        # Check for binary content
        if '\x00' in content[:1000]:
            return None, 0, len(content), False, "File appears to be binary"
        
        # Process content into lines
        lines = content.splitlines()
        total_lines = len(lines)
        total_bytes = len(content.encode('utf-8', errors='replace'))
        
        truncated = False
        
        # Apply line range if specified
        if start_line is not None and end_line is not None:
            # Convert to 0-indexed
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
            print(f"[FS_REMOTE] SUCCESS: {total_lines} lines, {total_bytes} bytes, truncated={truncated}", file=sys.stderr)
            print(f"[FS_REMOTE] === REMOTE READ END ===", file=sys.stderr)
        
        return result_text, total_lines, total_bytes, truncated, ""
        
    except Exception as e:
        error_type = type(e).__name__
        if debug:
            print(f"[FS_REMOTE] UNEXPECTED ERROR: {error_type}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None, 0, 0, False, f"Remote read error ({error_type}): {e}"

def live_read_file_with_remote_fallback(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
    debug: bool = True,
) -> Tuple[Optional[str], int, int, bool, str, str]:
    """
    Read a file with automatic remote fallback for paths not visible locally.
    
    This is the main entry point for filesystem reads. It:
    1. Tries local read first (fast, no network)
    2. If local fails with FileNotFoundError/WinError 3, tries remote via sandbox controller
    
    Args:
        path: Absolute path to file (should already be normalized)
        start_line: Start line for range (1-indexed, inclusive)
        end_line: End line for range (1-indexed, inclusive)
        head_lines: Number of lines from start (for head command)
        debug: If True, print debug info
    
    Returns:
        (content, total_lines, total_bytes, truncated, error_msg, source)
        - source: "local" if local read succeeded, "remote_agent" if fallback used
    """
    from .fs_live_ops import live_read_file
    # Normalize path
    norm_path = normalize_path(path, debug=debug)
    
    # =========================================================================
    # Stage A: Try local read first
    # =========================================================================
    content, total_lines, total_bytes, truncated, error = live_read_file(
        norm_path, start_line, end_line, head_lines, debug=debug
    )
    
    if content is not None:
        # Local read succeeded
        if debug:
            print(f"[FS_LIVE] live_read=True source=local", file=sys.stderr)
        return content, total_lines, total_bytes, truncated, "", "local"
    
    # =========================================================================
    # Stage B: Remote fallback (if local failed with FileNotFoundError)
    # =========================================================================
    # Only attempt remote fallback for FileNotFoundError / WinError 3 type errors
    # Other errors like PermissionError should not trigger remote fallback
    is_not_found_error = (
        error and (
            "FileNotFoundError" in error or
            "WinError 3" in error or
            "File not found" in error or
            "Path not found" in error
        )
    )
    
    if not is_not_found_error:
        # Not a "file not found" type error - don't try remote
        if debug:
            print(f"[FS_LIVE] local_failed_non_recoverable error={error}", file=sys.stderr)
        return None, 0, 0, False, error, "local"
    
    # Check allowlist before remote attempt (security)
    allowed, reason = is_path_allowed(norm_path)
    if not allowed:
        # Path is blocked - don't try remote
        if debug:
            print(f"[FS_LIVE] path_blocked reason={reason}", file=sys.stderr)
        return None, 0, 0, False, f"Path blocked: {reason}", "local"
    
    # Try remote read
    if debug:
        print(f"[FS_QUERY] local_read_failed -> trying remote_agent_read", file=sys.stderr)
    
    content, total_lines, total_bytes, truncated, remote_error = remote_read_file(
        norm_path, start_line, end_line, head_lines, debug=debug
    )
    
    if content is not None:
        # Remote read succeeded
        if debug:
            print(f"[FS_QUERY] live_read=True source=remote_agent", file=sys.stderr)
        return content, total_lines, total_bytes, truncated, "", "remote_agent"
    
    # Both local and remote failed
    combined_error = f"Local: {error}\nRemote: {remote_error}"
    if debug:
        print(f"[FS_QUERY] both_reads_failed", file=sys.stderr)
    return None, 0, 0, False, combined_error, "failed"

def live_list_directory(
    path: str,
    debug: bool = True,
) -> Tuple[List[str], List[str], str]:
    """
    List directory contents directly from disk.
    
    Uses stat() for proper error diagnostics.
    
    Args:
        path: Absolute path to directory (should already be normalized)
        debug: If True, print debug info
    
    Returns:
        (folders: List[str], files: List[str], error_msg: str)
        - folders: List of folder paths (sorted)
        - files: List of file paths (sorted)
        - error_msg: Empty string on success, error description on failure
    """
    from .fs_live_ops import _get_extended_path
    # Normalize path
    norm_path = normalize_path(path, debug=debug)
    
    if debug:
        print(f"[FS_LIVE] list raw_path repr={repr(path)}", file=sys.stderr)
        print(f"[FS_LIVE] list norm_path repr={repr(norm_path)}", file=sys.stderr)
    
    try:
        # Try normal path first, then extended
        paths_to_try = [norm_path, _get_extended_path(norm_path)]
        
        for try_path in paths_to_try:
            try:
                stat_result = os.stat(try_path)
                norm_path = try_path  # Use the working path
                if debug:
                    print(f"[FS_LIVE] list stat OK for {repr(try_path)}", file=sys.stderr)
                break
            except FileNotFoundError:
                continue
            except PermissionError:
                return [], [], f"Permission denied: {norm_path}"
            except OSError as e:
                continue
        else:
            return [], [], f"Directory not found: {norm_path}"
        
        dir_path = Path(norm_path)
        
        # Check if it's actually a directory
        if not dir_path.is_dir():
            return [], [], f"Path is not a directory: {norm_path}"
        
        folders = []
        files = []
        
        try:
            for item in dir_path.iterdir():
                try:
                    if item.is_dir():
                        folders.append(str(item))
                    else:
                        files.append(str(item))
                except PermissionError:
                    # Skip items we can't access
                    continue
                except OSError:
                    # Skip items with other OS errors
                    continue
        except PermissionError:
            return [], [], f"Permission denied listing directory: {norm_path}"
        
        folders.sort(key=str.lower)
        files.sort(key=str.lower)
        
        if debug:
            print(f"[FS_LIVE] list SUCCESS: {len(folders)} folders, {len(files)} files", file=sys.stderr)
        
        return folders, files, ""
        
    except Exception as e:
        error_type = type(e).__name__
        if debug:
            print(f"[FS_LIVE] list UNEXPECTED ERROR: {error_type}: {e}", file=sys.stderr)
        return [], [], f"Unexpected error ({error_type}): {e}"

def check_path_exists(path: str) -> Tuple[bool, str, str]:
    """
    Check if a path exists and get its type.
    
    Uses stat() for proper error diagnostics.
    
    Args:
        path: Path to check (should already be normalized)
    
    Returns:
        (exists: bool, path_type: str, error_msg: str)
        - exists: True if path exists
        - path_type: "file", "directory", or "unknown"
        - error_msg: Empty on success/exists, error description on failure
    """
    from .fs_live_ops import _get_extended_path
    norm_path = normalize_path(path)
    
    try:
        # Try normal and extended paths
        for try_path in [norm_path, _get_extended_path(norm_path)]:
            try:
                os.stat(try_path)
                p = Path(try_path)
                if p.is_file():
                    return True, "file", ""
                elif p.is_dir():
                    return True, "directory", ""
                else:
                    return True, "unknown", ""
            except FileNotFoundError:
                continue
            except PermissionError:
                return False, "", "Permission denied"
            except OSError as e:
                continue
        
        return False, "", "Path not found"
            
    except Exception as e:
        return False, "", f"Error checking path: {e}"
