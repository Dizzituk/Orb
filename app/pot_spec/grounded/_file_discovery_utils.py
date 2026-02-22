from __future__ import annotations
import logging
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)

def _get_match_bucket():
    from app.pot_spec.grounded.file_discovery import MatchBucket
    return MatchBucket


FILE_DISCOVERY_BUILD_ID = "2026-02-02-v2.3-local-fallback"

def _run_powershell_local(
    command: str,
    timeout_seconds: int = 120,
) -> Tuple[bool, str, str, int]:
    """
    v2.3: Run PowerShell command locally (no sandbox).
    
    Used as fallback when sandbox is unavailable.
    
    Args:
        command: PowerShell command to run
        timeout_seconds: Command timeout
        
    Returns:
        (success, stdout, stderr, duration_ms)
    """
    start_time = time.time()
    
    try:
        logger.info("[file_discovery] v2.3 LOCAL FALLBACK: Running PowerShell locally")
        print("[file_discovery] v2.3 LOCAL FALLBACK: Sandbox unavailable, using local PowerShell")
        
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            encoding='utf-8',
            errors='replace',
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            "[file_discovery] v2.3 LOCAL result: exit=%d, stdout=%d chars, stderr=%d chars",
            result.returncode, len(result.stdout or ''), len(result.stderr or '')
        )
        
        return True, result.stdout or '', result.stderr or '', duration_ms
        
    except subprocess.TimeoutExpired:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("[file_discovery] v2.3 LOCAL timeout after %ds", timeout_seconds)
        return False, '', f'Command timed out after {timeout_seconds}s', duration_ms
        
    except Exception as e:
        duration_ms = int((time.time() - start_time) * 1000)
        logger.error("[file_discovery] v2.3 LOCAL exception: %s", e)
        return False, '', str(e), duration_ms

# Lazy — MatchBucket lives in parent, avoid circular import at module level
_MUST_REVIEW_BUCKETS = None

def _get_must_review_buckets():
    global _MUST_REVIEW_BUCKETS
    if _MUST_REVIEW_BUCKETS is None:
        MB = _get_match_bucket()
        _MUST_REVIEW_BUCKETS = frozenset({MB.ENV_VAR_KEY, MB.DATABASE_ARTIFACT, MB.FILE_FOLDER_NAME, MB.HISTORICAL_DATA})
    return _MUST_REVIEW_BUCKETS

# Keep original name accessible for importers
class _LazyBuckets:
    def __contains__(self, item): return item in _get_must_review_buckets()
    def __iter__(self): return iter(_get_must_review_buckets())
    def __len__(self): return len(_get_must_review_buckets())
MUST_REVIEW_BUCKETS = _LazyBuckets()

def _should_skip_line(line: str, file_path: str = "") -> bool:
    """
    v2.1: Determine if a line should be filtered from evidence.
    
    Filters:
    - Base64-encoded content (50+ chars of base64 alphabet)
    - Encrypted values (ENC:, ENCRYPTED:, etc.)
    - Embedding vectors (sequences of floats)
    - Binary/non-printable content
    - Excessively long lines (>1000 chars, likely minified)
    
    Args:
        line: Line content to check
        file_path: File path for filetype-aware filtering
        
    Returns:
        True if line should be skipped, False if it should be included
    """
    if not line:
        return True
    
    # v2.1: Get file extension for filetype-aware filtering
    path_lower = file_path.lower()
    is_minified = '.min.' in path_lower or '/dist/' in path_lower or '\\dist\\' in path_lower or '/build/' in path_lower or '\\build\\' in path_lower
    is_config = any(ext in path_lower for ext in ['.env', '.yaml', '.yml', '.json', '.toml', '.ini'])
    
    # Encrypted content markers
    if re.search(r'\bENC[:=]', line, re.IGNORECASE):
        logger.debug("[file_discovery] v2.1 Skipping encrypted: %s...", line[:50])
        return True
    if re.search(r'\bENCRYPTED[:=]', line, re.IGNORECASE):
        return True
    
    # Base64-encoded content (50+ chars that look like base64)
    # Be more conservative for config files (could be JWTs, certs)
    base64_threshold = 100 if is_config else 50
    if re.search(rf'[A-Za-z0-9+/]{{{base64_threshold},}}={"{0,2}"}', line):
        logger.debug("[file_discovery] v2.1 Skipping base64: %s...", line[:50])
        return True
    
    # Embedding vectors (sequences of floats like 0.123, -0.456, 0.789)
    if re.search(r'(-?\d+\.\d+,?\s*){10,}', line):
        logger.debug("[file_discovery] v2.1 Skipping embedding: %s...", line[:50])
        return True
    
    # Binary/non-printable characters
    if re.search(r'[\x00-\x08\x0e-\x1f\x7f-\xff]', line):
        return True
    
    # Excessively long lines (likely minified code)
    # Be aggressive for minified files, cautious for config
    max_length = 200 if is_minified else (2000 if is_config else 1000)
    if len(line) > max_length:
        logger.debug("[file_discovery] v2.1 Skipping long line (%d chars): %s...", len(line), line[:50])
        return True
    
    return False

def _build_select_string_command(
    pattern: str,
    roots: List[str],
    exclusions: List[str],
    file_filter: Optional[str],
    case_sensitive: bool,
) -> str:
    """Build PowerShell Select-String command."""
    escaped_pattern = pattern.replace("'", "''")
    
    exclusion_parts = []
    for exc in exclusions:
        exc_regex = exc.replace(".", r"\.").replace("*", ".*")
        exclusion_parts.append(exc_regex)
    exclusion_regex = "|".join(exclusion_parts) if exclusion_parts else ""
    
    # v2.2: Handle comma-separated file filters properly
    # PowerShell -Include accepts array: -Include '*.tsx','*.jsx'
    if file_filter and ',' in file_filter:
        # Convert "*.tsx,*.jsx" to "'*.tsx','*.jsx'"
        filter_parts = [f.strip() for f in file_filter.split(',')]
        include_filter = "','" .join(filter_parts)
    else:
        include_filter = file_filter or "*.*"
    
    case_flag = "" if case_sensitive else "-CaseSensitive:$false"
    
    roots_joined = "', '".join(roots)
    
    cmd_parts = [
        f"Get-ChildItem -Path '{roots_joined}' -Recurse -File -Include '{include_filter}' -ErrorAction SilentlyContinue",
    ]
    
    if exclusion_regex:
        cmd_parts.append(f"| Where-Object {{ $_.FullName -notmatch '{exclusion_regex}' }}")
    
    cmd_parts.append(f"| Select-String -Pattern '{escaped_pattern}' {case_flag} -ErrorAction SilentlyContinue")
    cmd_parts.append("| ForEach-Object { \"$($_.Path)|$($_.LineNumber)|$($_.Line)\" }")
    
    return " ".join(cmd_parts)

def _build_extension_search_command(
    extension: str,
    roots: List[str],
    exclusions: List[str],
) -> str:
    """Build PowerShell Get-ChildItem command for extension search."""
    exclusion_parts = []
    for exc in exclusions:
        exc_regex = exc.replace(".", r"\.").replace("*", ".*")
        exclusion_parts.append(exc_regex)
    exclusion_regex = "|".join(exclusion_parts) if exclusion_parts else ""
    
    roots_joined = "', '".join(roots)
    
    cmd_parts = [
        f"Get-ChildItem -Path '{roots_joined}' -Recurse -File -Filter '*.{extension}' -ErrorAction SilentlyContinue",
    ]
    
    if exclusion_regex:
        cmd_parts.append(f"| Where-Object {{ $_.FullName -notmatch '{exclusion_regex}' }}")
    
    cmd_parts.append("| ForEach-Object { $_.FullName }")
    
    return " ".join(cmd_parts)

def _parse_select_string_output_v21(
    stdout: str,
    max_results: int,
    max_samples_per_file: int,
) -> Tuple[List[FileMatch], int, bool, int]:
    """
    v2.1: Parse Select-String output with filtering and classification.
    
    Returns: (files, total_occurrences, truncated, lines_filtered)
    """
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
        
        # v2.1: Filter garbage lines
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
            # v2.1: Classify mechanically
            bucket = _classify_match_mechanical(line_content, path)
            fm.line_matches.append(LineMatch(
                line_number=line_num,
                line_content=line_content.strip(),
                bucket=bucket,
            ))
    
    return list(file_matches.values()), total_occurrences, truncated, lines_filtered

def _parse_file_list_output(
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
        
        files.append(FileMatch(
            path=path,
            occurrence_count=1,
            line_matches=[],
        ))
    
    return files, truncated


# Auto-generated re-exports for symbols in numbered _utils files
# Circular-safe re-exports. MatchBucket/DiscoveryResult live in file_discovery.py
# which imports from us, so _utils_3 (which re-defines MatchBucket) would cause a loop.
# Non-circular symbols can go through _utils_3 directly.
_REEXPORT_MAP = {
    "FileMatch": "_file_discovery_utils_1",
    "LineMatch": "_file_discovery_utils_1",
    "_classify_match_mechanical": "_file_discovery_utils_1",
    "discover_files": "_file_discovery_utils_2",
    "discover_files_by_extension": "_file_discovery_utils_2",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.pot_spec.grounded.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    if name == "MatchBucket":
        return _get_match_bucket()
    if name == "DiscoveryResult":
        from app.pot_spec.grounded.file_discovery import DiscoveryResult
        return DiscoveryResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
