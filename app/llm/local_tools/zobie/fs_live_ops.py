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
from app.llm.local_tools.zobie._fs_live_ops_utils import check_path_exists, live_list_directory, live_read_file_with_remote_fallback, remote_read_file
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
