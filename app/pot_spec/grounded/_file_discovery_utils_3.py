from __future__ import annotations
import logging
import re
from app.pot_spec.grounded._file_discovery_utils import _build_extension_search_command, _build_select_string_command, _parse_file_list_output, _parse_select_string_output_v21, _run_powershell_local
from app.pot_spec.grounded.file_discovery import DiscoveryResult, logger
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


class MatchBucket(str, Enum):
    """Categories for match classification."""
    CODE_IDENTIFIER = "code_identifier"
    IMPORT_PATH = "import_path"
    MODULE_PACKAGE = "module_package"
    ENV_VAR_KEY = "env_var_key"
    CONFIG_KEY = "config_key"
    API_ROUTE = "api_route"
    FILE_FOLDER_NAME = "file_folder_name"
    DATABASE_ARTIFACT = "database_artifact"
    HISTORICAL_DATA = "historical_data"
    DOCUMENTATION = "documentation"
    UI_LABEL = "ui_label"
    TEST_ASSERTION = "test_assertion"
    GARBAGE = "garbage"  # v2.1: Filtered out
    UNKNOWN = "unknown"

def _classify_match_mechanical(line: str, file_path: str) -> MatchBucket:
    """
    v2.1: Mechanically classify a match based on path and content patterns.
    
    This is deterministic, not LLM-based. Used for:
    - Bucketing before LLM sees the data
    - Must-review flagging
    - Summary generation
    
    Args:
        line: Line content
        file_path: Full file path
        
    Returns:
        MatchBucket classification
    """
    path_lower = file_path.lower()
    line_lower = line.lower()
    
    # File/folder name matches (path contains pattern, not just content)
    if '\\' in line or '/' in line:
        return MatchBucket.FILE_FOLDER_NAME
    
    # Environment variable keys
    if re.search(r'^[A-Z][A-Z0-9_]*\s*[=:]', line):
        return MatchBucket.ENV_VAR_KEY
    if '.env' in path_lower:
        return MatchBucket.ENV_VAR_KEY
    
    # Database artifacts
    if any(x in path_lower for x in ['.db', '.sqlite', 'database', 'migration']):
        return MatchBucket.DATABASE_ARTIFACT
    if re.search(r'(CREATE|ALTER|INSERT|UPDATE|DELETE)\s+', line, re.IGNORECASE):
        return MatchBucket.DATABASE_ARTIFACT
    
    # Historical data (job outputs, logs)
    if any(x in path_lower for x in ['jobs/', 'jobs\\', 'history/', 'history\\', 'output/', 'output\\']):
        return MatchBucket.HISTORICAL_DATA
    
    # Import paths
    if re.search(r'^(from|import)\s+', line):
        return MatchBucket.IMPORT_PATH
    if re.search(r'require\s*\(', line):
        return MatchBucket.IMPORT_PATH
    
    # Test assertions
    if 'test' in path_lower or '_test.' in path_lower:
        return MatchBucket.TEST_ASSERTION
    if re.search(r'(assert|expect|should)\s*[.(]', line_lower):
        return MatchBucket.TEST_ASSERTION
    
    # API routes
    if re.search(r'@(app|router)\.(get|post|put|delete|patch)', line_lower):
        return MatchBucket.API_ROUTE
    if re.search(r'(path|route)\s*[=:]\s*["\']/', line_lower):
        return MatchBucket.API_ROUTE
    
    # Documentation
    if any(ext in path_lower for ext in ['.md', '.rst', '.txt', 'readme', 'doc']):
        return MatchBucket.DOCUMENTATION
    if re.search(r'^#+\s+', line):  # Markdown headers
        return MatchBucket.DOCUMENTATION
    
    # Config keys
    if any(ext in path_lower for ext in ['.yaml', '.yml', '.json', '.toml', 'config']):
        return MatchBucket.CONFIG_KEY
    
    # UI labels (strings in JSX/TSX)
    if any(ext in path_lower for ext in ['.tsx', '.jsx']):
        if re.search(r'["\'][^"\']{2,}["\']', line):
            return MatchBucket.UI_LABEL
    
    # Code identifiers (default for .py, .js, .ts)
    if any(ext in path_lower for ext in ['.py', '.js', '.ts', '.tsx', '.jsx']):
        return MatchBucket.CODE_IDENTIFIER
    
    return MatchBucket.UNKNOWN

@dataclass
class LineMatch:
    """Single line match within a file."""
    line_number: int
    line_content: str
    bucket: MatchBucket = MatchBucket.UNKNOWN  # v2.1: Classification
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_number": self.line_number,
            "line_content": self.line_content,
            "bucket": self.bucket.value,
        }

@dataclass
class FileMatch:
    """Single file with matches."""
    path: str
    occurrence_count: int
    line_matches: List[LineMatch] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "occurrence_count": self.occurrence_count,
            "line_matches": [m.to_dict() for m in self.line_matches],
        }

def discover_files(
    search_pattern: str,
    sandbox_client: Any,
    roots: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    file_filter: Optional[str] = None,
    case_sensitive: bool = False,
    max_results: int = DEFAULT_MAX_RESULTS,
    max_samples_per_file: int = DEFAULT_MAX_SAMPLES_PER_FILE,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> DiscoveryResult:
    """
    Discover files containing a search pattern.
    
    v2.3: Now with local PowerShell fallback when sandbox unavailable.
    v2.1: Now filters garbage lines and classifies matches mechanically.
    """
    roots = roots or DEFAULT_ROOTS
    exclusions = exclusions or DEFAULT_EXCLUSIONS
    
    logger.info(f"[file_discovery] v2.3 Pattern search: {search_pattern}, roots={roots}")
    
    ps_command = _build_select_string_command(
        pattern=search_pattern,
        roots=roots,
        exclusions=exclusions,
        file_filter=file_filter,
        case_sensitive=case_sensitive,
    )
    
    logger.debug(f"[file_discovery] PowerShell command: {ps_command[:200]}...")
    
    # v2.3: Try sandbox first, fall back to local
    stdout = ''
    stderr = ''
    duration_ms = 0
    use_local = False
    
    if sandbox_client:
        try:
            shell_result = sandbox_client.shell_run(
                command=ps_command,
                cwd_target="REPO",
                timeout_seconds=timeout_seconds,
            )
            
            stdout = getattr(shell_result, 'stdout', '') or ''
            stderr = getattr(shell_result, 'stderr', '') or ''
            duration_ms = getattr(shell_result, 'duration_ms', 0)
            
            logger.debug(f"[file_discovery] Sandbox result: stdout_len={len(stdout)}")
            
        except Exception as e:
            # v2.3: Sandbox failed, use local fallback
            logger.warning(f"[file_discovery] v2.3 Sandbox failed: {e}")
            print(f"[file_discovery] v2.3 Sandbox failed, trying local fallback...")
            use_local = True
    else:
        # No sandbox client provided
        use_local = True
    
    # v2.3: Local fallback
    if use_local or (not stdout.strip() and not stderr.strip()):
        logger.info("[file_discovery] v2.3 Using LOCAL PowerShell fallback")
        success, stdout, stderr, duration_ms = _run_powershell_local(
            command=ps_command,
            timeout_seconds=timeout_seconds,
        )
        
        if not success:
            return DiscoveryResult(
                success=False,
                search_pattern=search_pattern,
                total_files=0,
                total_occurrences=0,
                error_message=f"Local PowerShell failed: {stderr}",
                duration_ms=duration_ms,
                roots_searched=roots,
            )
    
    # Parse results
    if stdout.strip():
        files, total_occurrences, truncated, lines_filtered = _parse_select_string_output_v21(
            stdout=stdout,
            max_results=max_results,
            max_samples_per_file=max_samples_per_file,
        )
        
        logger.info(f"[file_discovery] v2.3 Found {len(files)} files, {total_occurrences} occurrences")
        print(f"[file_discovery] v2.3 SCAN COMPLETE: {total_occurrences} matches in {len(files)} files")
        
        return DiscoveryResult(
            success=True,
            search_pattern=search_pattern,
            total_files=len(files),
            total_occurrences=total_occurrences,
            files=files,
            truncated=truncated,
            duration_ms=duration_ms,
            roots_searched=roots,
            lines_filtered=lines_filtered,
        )
    
    # No stdout
    if stderr.strip():
        error_msg = f"PowerShell error: {stderr[:500]}"
        logger.warning(f"[file_discovery] {error_msg}")
        return DiscoveryResult(
            success=False,
            search_pattern=search_pattern,
            total_files=0,
            total_occurrences=0,
            error_message=error_msg,
            duration_ms=duration_ms,
            roots_searched=roots,
        )
    
    # No matches found
    logger.info(f"[file_discovery] No matches found for pattern: {search_pattern}")
    return DiscoveryResult(
        success=True,
        search_pattern=search_pattern,
        total_files=0,
        total_occurrences=0,
        files=[],
        truncated=False,
        duration_ms=duration_ms,
        roots_searched=roots,
    )

def discover_files_by_extension(
    extension: str,
    sandbox_client: Any,
    roots: Optional[List[str]] = None,
    exclusions: Optional[List[str]] = None,
    max_results: int = DEFAULT_MAX_RESULTS,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> DiscoveryResult:
    """Discover all files with a specific extension."""
    roots = roots or DEFAULT_ROOTS
    exclusions = exclusions or DEFAULT_EXCLUSIONS
    
    ext = extension.lstrip("*").lstrip(".")
    
    logger.info(f"[file_discovery] Extension search: .{ext}, roots={roots}")
    
    ps_command = _build_extension_search_command(
        extension=ext,
        roots=roots,
        exclusions=exclusions,
    )
    
    try:
        shell_result = sandbox_client.shell_run(
            command=ps_command,
            cwd_target="REPO",
            timeout_seconds=timeout_seconds,
        )
        
        if not shell_result.ok and shell_result.exit_code != 0:
            return DiscoveryResult(
                success=False,
                search_pattern=f"*.{ext}",
                total_files=0,
                total_occurrences=0,
                error_message=f"PowerShell error: {shell_result.stderr[:500]}",
                duration_ms=shell_result.duration_ms,
                roots_searched=roots,
            )
        
        files, truncated = _parse_file_list_output(
            stdout=shell_result.stdout,
            max_results=max_results,
        )
        
        logger.info(f"[file_discovery] Found {len(files)} .{ext} files")
        
        return DiscoveryResult(
            success=True,
            search_pattern=f"*.{ext}",
            total_files=len(files),
            total_occurrences=len(files),
            files=files,
            truncated=truncated,
            duration_ms=shell_result.duration_ms,
            roots_searched=roots,
        )
        
    except Exception as e:
        logger.error(f"[file_discovery] Exception: {e}")
        return DiscoveryResult(
            success=False,
            search_pattern=f"*.{ext}",
            total_files=0,
            total_occurrences=0,
            error_message=str(e),
            roots_searched=roots,
        )
