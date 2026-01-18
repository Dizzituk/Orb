# FILE: app/llm/local_tools/zobie/fs_live_ops.py
"""Live filesystem operations for the filesystem query system.

This module handles surgical live reads directly from disk:
- Reading file content (full, head, line range)
- Listing directory contents

Uses multiple fallback methods for OneDrive/cloud-synced paths.
Includes remote/agent fallback via sandbox controller for paths not visible
to the local backend process.

v5.4 (2026-01): Added remote_agent fallback via sandbox controller for OneDrive paths
v5.3 (2026-01): Added extended-length path syntax + multiple fallback methods
v5.2 (2026-01): Fixed file existence checks to use stat() for better OneDrive support
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .config import (
    FILESYSTEM_READ_MAX_LINES,
    FILESYSTEM_READ_MAX_BYTES,
)
from .fs_path_utils import normalize_path, is_path_allowed
from .sandbox_client import call_fs_contents

logger = logging.getLogger(__name__)


def _get_extended_path(path: str) -> str:
    """
    Convert a Windows path to extended-length path syntax.
    
    The \\\\?\\ prefix allows Windows to handle paths up to 32,767 characters
    and can help with OneDrive/cloud storage paths that have virtualization issues.
    """
    # Only apply to absolute Windows paths
    if len(path) >= 2 and path[1] == ':':
        # Already has extended prefix
        if path.startswith('\\\\?\\'):
            return path
        # Convert to extended syntax
        return '\\\\?\\' + path
    return path


def live_read_file(
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
    debug: bool = True,
) -> Tuple[Optional[str], int, int, bool, str]:
    """
    Read a file directly from disk (surgical live read).
    
    This function uses multiple fallback methods to access files,
    which is critical for OneDrive and cloud-synced paths.
    
    Args:
        path: Absolute path to file (should already be normalized)
        start_line: Start line for range (1-indexed, inclusive)
        end_line: End line for range (1-indexed, inclusive)
        head_lines: Number of lines from start (for head command)
        debug: If True, print debug info including path repr()
    
    Returns:
        (content, total_lines, total_bytes, truncated, error_msg)
        - content: File content (possibly truncated), or None on error
        - total_lines: Total lines in file
        - total_bytes: Total bytes in file  
        - truncated: True if content was truncated
        - error_msg: Empty string on success, error description on failure
    """
    # Normalize path (in case caller didn't)
    norm_path = normalize_path(path, debug=debug)
    
    # DEBUG: Print to stderr for immediate visibility
    if debug:
        print(f"[FS_LIVE] === LIVE READ START ===", file=sys.stderr)
        print(f"[FS_LIVE] raw_path repr={repr(path)}", file=sys.stderr)
        print(f"[FS_LIVE] norm_path repr={repr(norm_path)}", file=sys.stderr)
        # Show byte-level representation to catch hidden characters
        print(f"[FS_LIVE] norm_path hex bytes={norm_path.encode('utf-8').hex()}", file=sys.stderr)
    
    # Paths to try (in order of preference)
    paths_to_try = [
        ("normal", norm_path),
        ("extended", _get_extended_path(norm_path)),
    ]
    
    # Also try with trailing space stripped (common OneDrive issue)
    stripped_path = norm_path.rstrip()
    if stripped_path != norm_path:
        paths_to_try.append(("stripped", stripped_path))
    
    try:
        file_exists = False
        file_size = 0
        working_path = None
        access_method = "none"
        all_errors = []
        
        # Try each path variant
        for path_type, try_path in paths_to_try:
            if debug:
                print(f"[FS_LIVE] Trying {path_type}: {repr(try_path)}", file=sys.stderr)
            
            # Method 1: os.path.exists() - quick check
            exists_check = os.path.exists(try_path)
            if debug:
                print(f"[FS_LIVE]   os.path.exists({path_type})={exists_check}", file=sys.stderr)
            
            # Method 2: os.stat() - most direct syscall
            try:
                stat_result = os.stat(try_path)
                file_exists = True
                file_size = stat_result.st_size
                working_path = try_path
                access_method = f"os.stat({path_type})"
                if debug:
                    print(f"[FS_LIVE]   os.stat({path_type}) SUCCESS: size={file_size}", file=sys.stderr)
                break  # Found a working path
            except FileNotFoundError as e:
                all_errors.append(f"os.stat({path_type}) FileNotFoundError: {e}")
                if debug:
                    print(f"[FS_LIVE]   os.stat({path_type}) FileNotFoundError: {e}", file=sys.stderr)
            except PermissionError as e:
                all_errors.append(f"os.stat({path_type}) PermissionError: {e}")
                if debug:
                    print(f"[FS_LIVE]   os.stat({path_type}) PermissionError: {e}", file=sys.stderr)
            except OSError as e:
                all_errors.append(f"os.stat({path_type}) OSError[{e.errno}]: {e}")
                if debug:
                    print(f"[FS_LIVE]   os.stat({path_type}) OSError[{e.errno}]: {type(e).__name__}: {e}", file=sys.stderr)
        
        # Method 3: Try direct open() if stat failed (sometimes works on OneDrive)
        if not file_exists:
            for path_type, try_path in paths_to_try:
                try:
                    if debug:
                        print(f"[FS_LIVE] Trying direct open({path_type})", file=sys.stderr)
                    with open(try_path, 'r', encoding='utf-8') as f:
                        # Try to read first byte to verify access
                        test_read = f.read(1)
                        file_exists = True
                        working_path = try_path
                        access_method = f"direct_open({path_type})"
                        if debug:
                            print(f"[FS_LIVE]   direct open({path_type}) SUCCESS", file=sys.stderr)
                        break
                except FileNotFoundError as e:
                    all_errors.append(f"open({path_type}) FileNotFoundError: {e}")
                    if debug:
                        print(f"[FS_LIVE]   open({path_type}) FileNotFoundError: {e}", file=sys.stderr)
                except PermissionError as e:
                    all_errors.append(f"open({path_type}) PermissionError: {e}")
                    if debug:
                        print(f"[FS_LIVE]   open({path_type}) PermissionError: {e}", file=sys.stderr)
                except OSError as e:
                    all_errors.append(f"open({path_type}) OSError[{e.errno}]: {e}")
                    if debug:
                        print(f"[FS_LIVE]   open({path_type}) OSError[{e.errno}]: {type(e).__name__}: {e}", file=sys.stderr)
                except UnicodeDecodeError:
                    # File exists but has encoding issues - we'll handle that later
                    file_exists = True
                    working_path = try_path
                    access_method = f"direct_open({path_type})+encoding_fallback"
                    if debug:
                        print(f"[FS_LIVE]   open({path_type}) encoding error - file exists", file=sys.stderr)
                    break
        
        if not file_exists:
            # Build detailed error message
            error_msg = f"File not found: {norm_path}"
            if all_errors:
                # Show the most relevant error (usually the first one)
                error_msg = f"{error_msg}\nDetails: {all_errors[0]}"
            if debug:
                print(f"[FS_LIVE] FAILURE: file not accessible", file=sys.stderr)
                for err in all_errors:
                    print(f"[FS_LIVE]   {err}", file=sys.stderr)
            return None, 0, 0, False, error_msg
        
        if debug:
            print(f"[FS_LIVE] File found via {access_method}, working_path={repr(working_path)}", file=sys.stderr)
        
        # Check if it's actually a file (not a directory)
        if access_method.startswith("os.stat"):
            p = Path(working_path)
            if not p.is_file():
                return None, 0, 0, False, f"Path is not a file: {norm_path}"
        
        # Read the file content
        content = None
        read_error = None
        
        # Try UTF-8 first
        try:
            with open(working_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if debug:
                print(f"[FS_LIVE] Read UTF-8 SUCCESS: {len(content)} chars", file=sys.stderr)
        except UnicodeDecodeError:
            # Fallback to latin-1
            try:
                with open(working_path, 'r', encoding='latin-1', errors='replace') as f:
                    content = f.read()
                if debug:
                    print(f"[FS_LIVE] Read latin-1 SUCCESS: {len(content)} chars", file=sys.stderr)
            except Exception as e:
                read_error = f"Failed to decode file: {e}"
        except PermissionError as e:
            read_error = f"Permission denied reading file: {e}"
        except OSError as e:
            read_error = f"Error reading file (OSError[{e.errno}]): {e}"
        
        if content is None:
            if debug:
                print(f"[FS_LIVE] Read FAILURE: {read_error}", file=sys.stderr)
            return None, 0, 0, False, read_error or "Failed to read file content"
        
        # Check for binary content (null bytes in first 1000 chars)
        if '\x00' in content[:1000]:
            return None, 0, file_size or len(content), False, "File appears to be binary"
        
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
            # Truncate to byte limit
            result_text = result_text[:FILESYSTEM_READ_MAX_BYTES]
            # Find last newline to avoid cutting mid-line
            last_nl = result_text.rfind('\n')
            if last_nl > FILESYSTEM_READ_MAX_BYTES // 2:
                result_text = result_text[:last_nl]
            truncated = True
        
        if debug:
            print(f"[FS_LIVE] SUCCESS: {total_lines} lines, {total_bytes} bytes, truncated={truncated}", file=sys.stderr)
            print(f"[FS_LIVE] === LIVE READ END ===", file=sys.stderr)
        
        return result_text, total_lines, total_bytes, truncated, ""
        
    except Exception as e:
        # Catch-all for unexpected errors
        error_type = type(e).__name__
        if debug:
            print(f"[FS_LIVE] UNEXPECTED ERROR: {error_type}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
        return None, 0, 0, False, f"Unexpected error ({error_type}): {e}"


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
