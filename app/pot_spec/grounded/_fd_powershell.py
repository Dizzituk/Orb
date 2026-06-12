# FILE: app/pot_spec/grounded/_fd_powershell.py
# Purpose: File Discovery — PowerShell command builders and output parsers.
# Called-by: app.pot_spec.grounded.file_discovery
# Depends-on: app.pot_spec.grounded._fd_models
# Last-renovated: 2026-06-11
"""
File Discovery — PowerShell command builders and output parsers.

Extracted from file_discovery.py to reduce file size.
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

from ._fd_models import (
    FileMatch, LineMatch, MatchBucket,
    _should_skip_line, _classify_match_mechanical,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Local PowerShell fallback (v2.3)
# =============================================================================

def run_powershell_local(
    command: str,
    timeout_seconds: int = 120,
) -> Tuple[bool, str, str, int]:
    """v2.3: Run PowerShell command locally when sandbox unavailable."""
    start_time = time.time()
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True, text=True,
            timeout=timeout_seconds, encoding='utf-8', errors='replace',
        )
        duration_ms = int((time.time() - start_time) * 1000)
        return True, result.stdout or '', result.stderr or '', duration_ms
    except subprocess.TimeoutExpired:
        return False, '', f'Command timed out after {timeout_seconds}s', int((time.time() - start_time) * 1000)
    except Exception as e:
        return False, '', str(e), int((time.time() - start_time) * 1000)


# =============================================================================
# PowerShell Command Builders
# =============================================================================

def build_select_string_command(
    pattern: str,
    roots: List[str],
    exclusions: List[str],
    file_filter: Optional[str],
    case_sensitive: bool,
) -> str:
    """Build PowerShell Select-String command."""
    escaped_pattern = pattern.replace("'", "''")
    exclusion_parts = [exc.replace(".", r"\.").replace("*", ".*") for exc in exclusions]
    exclusion_regex = "|".join(exclusion_parts) if exclusion_parts else ""

    if file_filter and ',' in file_filter:
        filter_parts = [f.strip() for f in file_filter.split(',')]
        include_filter = "','".join(filter_parts)
    else:
        include_filter = file_filter or "*.*"

    case_flag = "" if case_sensitive else "-CaseSensitive:$false"
    roots_joined = "', '".join(roots)

    cmd_parts = [f"Get-ChildItem -Path '{roots_joined}' -Recurse -File -Include '{include_filter}' -ErrorAction SilentlyContinue"]
    if exclusion_regex:
        cmd_parts.append(f"| Where-Object {{ $_.FullName -notmatch '{exclusion_regex}' }}")
    cmd_parts.append(f"| Select-String -Pattern '{escaped_pattern}' {case_flag} -ErrorAction SilentlyContinue")
    cmd_parts.append("| ForEach-Object { \"$($_.Path)|$($_.LineNumber)|$($_.Line)\" }")
    return " ".join(cmd_parts)


def build_extension_search_command(
    extension: str,
    roots: List[str],
    exclusions: List[str],
) -> str:
    """Build PowerShell Get-ChildItem command for extension search."""
    exclusion_parts = [exc.replace(".", r"\.").replace("*", ".*") for exc in exclusions]
    exclusion_regex = "|".join(exclusion_parts) if exclusion_parts else ""
    roots_joined = "', '".join(roots)

    cmd_parts = [f"Get-ChildItem -Path '{roots_joined}' -Recurse -File -Filter '*.{extension}' -ErrorAction SilentlyContinue"]
    if exclusion_regex:
        cmd_parts.append(f"| Where-Object {{ $_.FullName -notmatch '{exclusion_regex}' }}")
    cmd_parts.append("| ForEach-Object { $_.FullName }")
    return " ".join(cmd_parts)


# =============================================================================
# Output Parsers
# =============================================================================

def parse_select_string_output_v21(
    stdout: str,
    max_results: int,
    max_samples_per_file: int,
) -> Tuple[List[FileMatch], int, bool, int]:
    """v2.1: Parse Select-String output with filtering and classification."""
    if not stdout.strip():
        return [], 0, False, 0

    file_matches: Dict[str, FileMatch] = {}
    total_occurrences = 0
    truncated = False
    lines_filtered = 0

    for line in stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        path = parts[0].strip()
        try:
            line_num = int(parts[1].strip())
        except ValueError:
            continue
        line_content = parts[2] if len(parts) > 2 else ""

        if _should_skip_line(line_content, path):
            lines_filtered += 1
            continue

        total_occurrences += 1
        if path not in file_matches:
            if len(file_matches) >= max_results:
                truncated = True
                continue
            file_matches[path] = FileMatch(path=path, occurrence_count=0, line_matches=[])

        fm = file_matches[path]
        fm.occurrence_count += 1
        if len(fm.line_matches) < max_samples_per_file:
            bucket = _classify_match_mechanical(line_content, path)
            fm.line_matches.append(LineMatch(line_number=line_num, line_content=line_content.strip(), bucket=bucket))

    return list(file_matches.values()), total_occurrences, truncated, lines_filtered


def parse_file_list_output(
    stdout: str,
    max_results: int,
) -> Tuple[List[FileMatch], bool]:
    """Parse Get-ChildItem file list output."""
    if not stdout.strip():
        return [], False
    files: List[FileMatch] = []
    truncated = False
    for line in stdout.strip().split("\n"):
        path = line.strip()
        if not path:
            continue
        if len(files) >= max_results:
            truncated = True
            break
        files.append(FileMatch(path=path, occurrence_count=1, line_matches=[]))
    return files, truncated


__all__ = [
    "run_powershell_local",
    "build_select_string_command", "build_extension_search_command",
    "parse_select_string_output_v21", "parse_file_list_output",
]
