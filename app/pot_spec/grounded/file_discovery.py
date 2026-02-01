# FILE: app/pot_spec/grounded/file_discovery.py
"""File Discovery System (v2.1)

Discovers files matching search patterns across the codebase using PowerShell
Select-String via the sandbox controller. Used by SpecGate to build file lists
for multi-file operations.

Architecture:
    SpecGate → file_discovery.py → SandboxClient.shell_run() → PowerShell Select-String

v2.1 (2026-02-01): TWO-LAYER EVIDENCE ARCHITECTURE
    - get_summary_report() returns 5-50KB summary for LLM prompts
    - get_full_evidence_json() returns complete evidence for grounding_data
    - _should_skip_line() filters garbage (base64, encrypted, embeddings)
    - Filetype-aware filtering (aggressive for minified, cautious for .env)
    - NEVER dumps 10MB into a prompt again

v1.43 (2026-01-31): CRITICAL FIX - Remove truncation for grounded truth evidence
v1.2 (2026-01-31): CRITICAL FIX - Prioritize stdout over exit codes
v1.1 (2026-01-31): BUGFIX - Handle sandbox controller exit codes
v1.0 (2026-01-28): Initial implementation
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ROOTS: List[str] = [
    r"D:\Orb",
    r"D:\Orb Desktop",  # v2.2: Fixed - actual folder name has space, not hyphen
    r"D:\orb-desktop",  # Legacy: Keep for backwards compatibility
]

DEFAULT_EXCLUSIONS: List[str] = [
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "*.egg-info",
    ".next",
    "coverage",
    ".coverage",
    "htmlcov",
]

DEFAULT_FILE_EXTENSIONS: List[str] = [
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".md",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".html",
    ".css",
    ".sql",
]

DEFAULT_TIMEOUT_SECONDS: int = 120
DEFAULT_MAX_RESULTS: int = 2000
DEFAULT_MAX_SAMPLES_PER_FILE: int = 50

# v2.1: Summary report limits
SUMMARY_MAX_FILES: int = 20  # Top N files by occurrence count
SUMMARY_MAX_SAMPLES_PER_BUCKET: int = 3  # Sample matches per category
SUMMARY_MAX_CHARS: int = 50000  # ~50KB limit for summary


# =============================================================================
# v2.1: Line Filtering (Skip Garbage)
# =============================================================================

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


# v2.1: Must-review buckets that should NEVER be auto-changed
MUST_REVIEW_BUCKETS = frozenset({
    MatchBucket.ENV_VAR_KEY,
    MatchBucket.DATABASE_ARTIFACT,
    MatchBucket.FILE_FOLDER_NAME,
    MatchBucket.HISTORICAL_DATA,
})


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


# =============================================================================
# Data Models
# =============================================================================

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


@dataclass
class DiscoveryResult:
    """Complete discovery results."""
    success: bool
    search_pattern: str
    total_files: int
    total_occurrences: int
    files: List[FileMatch] = field(default_factory=list)
    truncated: bool = False
    error_message: Optional[str] = None
    duration_ms: int = 0
    roots_searched: List[str] = field(default_factory=list)
    
    # v2.1: Filtering stats
    lines_filtered: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "search_pattern": self.search_pattern,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "files": [f.to_dict() for f in self.files],
            "truncated": self.truncated,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "roots_searched": self.roots_searched,
            "lines_filtered": self.lines_filtered,
        }
    
    def get_summary_report(self) -> str:
        """
        v2.1: Generate compact summary report for LLM prompts (5-50KB).
        
        This is Layer 1 of the two-layer evidence architecture.
        Goes into the SPoT markdown/prompt.
        
        Contains:
        - Totals and bucket counts
        - Must-review items highlighted
        - Sample matches per category (max 3 each)
        - Top 20 files by occurrence count
        - Risk assessment
        
        Returns:
            Compact summary suitable for LLM context (~5-50KB)
        """
        lines = [
            "# Discovery Summary",
            "",
            f"**Pattern:** `{self.search_pattern}`",
            f"**Total Files:** {self.total_files}",
            f"**Total Occurrences:** {self.total_occurrences}",
            f"**Lines Filtered (garbage):** {self.lines_filtered}",
            f"**Duration:** {self.duration_ms}ms",
            "",
        ]
        
        # Bucket all matches
        buckets: Dict[MatchBucket, List[Tuple[str, LineMatch]]] = {}
        must_review_items: List[Tuple[str, LineMatch]] = []
        
        for fm in self.files:
            for lm in fm.line_matches:
                bucket = lm.bucket if lm.bucket != MatchBucket.UNKNOWN else _classify_match_mechanical(lm.line_content, fm.path)
                if bucket not in buckets:
                    buckets[bucket] = []
                buckets[bucket].append((fm.path, lm))
                
                if bucket in MUST_REVIEW_BUCKETS:
                    must_review_items.append((fm.path, lm))
        
        # Must-review section (CRITICAL - always show these)
        if must_review_items:
            lines.append("## ⚠️ MUST-REVIEW ITEMS (Do NOT auto-change)")
            lines.append("")
            for path, lm in must_review_items[:20]:  # Cap at 20
                lines.append(f"- `{path}` L{lm.line_number}: `{lm.line_content[:100]}`")
            if len(must_review_items) > 20:
                lines.append(f"- ... and {len(must_review_items) - 20} more must-review items")
            lines.append("")
        
        # Bucket summary with samples
        lines.append("## Matches by Category")
        lines.append("")
        
        for bucket in MatchBucket:
            if bucket == MatchBucket.GARBAGE:
                continue
            items = buckets.get(bucket, [])
            if not items:
                continue
            
            is_must_review = bucket in MUST_REVIEW_BUCKETS
            marker = "🔴" if is_must_review else "🔵"
            lines.append(f"### {marker} {bucket.value} ({len(items)} matches)")
            
            # Show samples
            for path, lm in items[:SUMMARY_MAX_SAMPLES_PER_BUCKET]:
                content_preview = lm.line_content[:80] + "..." if len(lm.line_content) > 80 else lm.line_content
                lines.append(f"  - `{path}` L{lm.line_number}: `{content_preview}`")
            if len(items) > SUMMARY_MAX_SAMPLES_PER_BUCKET:
                lines.append(f"  - ... and {len(items) - SUMMARY_MAX_SAMPLES_PER_BUCKET} more")
            lines.append("")
        
        # Top files by occurrence count
        lines.append("## Top Files (by occurrence count)")
        lines.append("")
        
        sorted_files = sorted(self.files, key=lambda f: f.occurrence_count, reverse=True)
        for fm in sorted_files[:SUMMARY_MAX_FILES]:
            lines.append(f"- `{fm.path}` ({fm.occurrence_count} matches)")
        if len(sorted_files) > SUMMARY_MAX_FILES:
            lines.append(f"- ... and {len(sorted_files) - SUMMARY_MAX_FILES} more files")
        lines.append("")
        
        # Risk analysis
        lines.append("## Risk Assessment")
        dep_analysis = self._analyze_dependencies()
        for category, risks in dep_analysis.items():
            if risks:
                lines.append(f"### {category}")
                for risk in risks[:3]:  # Cap at 3 per category
                    lines.append(f"- {risk}")
                lines.append("")
        
        result = "\n".join(lines)
        
        # Enforce size limit
        if len(result) > SUMMARY_MAX_CHARS:
            result = result[:SUMMARY_MAX_CHARS - 100] + "\n\n... [TRUNCATED - see full evidence in grounding_data]"
        
        return result
    
    def get_full_evidence_json(self) -> Dict[str, Any]:
        """
        v2.1: Get complete evidence as structured JSON for grounding_data.
        
        This is Layer 2 of the two-layer evidence architecture.
        Stored in grounding_data, NOT in the prompt.
        
        Returns:
            Complete evidence dict (can be large)
        """
        return {
            "search_pattern": self.search_pattern,
            "total_files": self.total_files,
            "total_occurrences": self.total_occurrences,
            "lines_filtered": self.lines_filtered,
            "duration_ms": self.duration_ms,
            "roots_searched": self.roots_searched,
            "files": [f.to_dict() for f in self.files],
        }
    
    def get_file_preview(self, max_files: int = 10) -> str:
        """
        Generate human-readable preview (legacy method, uses summary now).
        """
        return self.get_summary_report()
    
    def get_full_evidence_report(self) -> str:
        """
        v2.1: DEPRECATED - Use get_summary_report() for prompts.
        
        This method still exists for backward compatibility but now
        returns the summary report instead of the full dump.
        Full evidence should be accessed via get_full_evidence_json().
        """
        logger.warning("[file_discovery] v2.1 get_full_evidence_report() is deprecated, use get_summary_report()")
        return self.get_summary_report()
    
    def _categorize_files(self) -> Dict[str, List["FileMatch"]]:
        """Categorize files by component type for impact analysis."""
        categories = {
            "🔴 CRITICAL - Core System": [],
            "🟡 HIGH - API & Data Layer": [],
            "🟢 MEDIUM - Services & Utilities": [],
            "⚪ LOW - Tests & Documentation": [],
            "📁 OTHER": [],
        }
        
        critical_patterns = [
            "encryption", "crypto", "auth", "routing", "stream_router",
            "overwatcher", "translation", "middleware", "security",
            "token", "session", "master_key", "sandbox_client"
        ]
        
        high_patterns = [
            "api", "endpoint", "database", "db", "config", "settings",
            "provider", "registry", "llm", "model", "service"
        ]
        
        medium_patterns = [
            "service", "util", "helper", "tool", "parser", "builder",
            "handler", "processor", "manager"
        ]
        
        low_patterns = [
            "test", "_test", "tests", "spec", "docs", "readme",
            "__init__", "example", "sample", "mock"
        ]
        
        for fm in self.files:
            path_lower = fm.path.lower()
            
            if any(p in path_lower for p in critical_patterns):
                categories["🔴 CRITICAL - Core System"].append(fm)
            elif any(p in path_lower for p in high_patterns):
                categories["🟡 HIGH - API & Data Layer"].append(fm)
            elif any(p in path_lower for p in medium_patterns):
                categories["🟢 MEDIUM - Services & Utilities"].append(fm)
            elif any(p in path_lower for p in low_patterns):
                categories["⚪ LOW - Tests & Documentation"].append(fm)
            else:
                categories["📁 OTHER"].append(fm)
        
        return {k: v for k, v in categories.items() if v}
    
    def _analyze_dependencies(self) -> Dict[str, List[str]]:
        """Analyze dependency impacts of the refactor."""
        analysis = {
            "🔴 Critical Risks": [],
            "🟡 High Risks": [],
            "🟢 Manageable Risks": [],
        }
        
        encryption_files = [f for f in self.files if "encrypt" in f.path.lower() or "crypto" in f.path.lower()]
        if encryption_files:
            analysis["🔴 Critical Risks"].append(
                f"Encryption layer affected ({len(encryption_files)} files)"
            )
        
        auth_files = [f for f in self.files if "auth" in f.path.lower() or "session" in f.path.lower()]
        if auth_files:
            analysis["🔴 Critical Risks"].append(
                f"Authentication system affected ({len(auth_files)} files)"
            )
        
        db_files = [f for f in self.files if "db" in f.path.lower() or "database" in f.path.lower()]
        if db_files:
            analysis["🟡 High Risks"].append(
                f"Database layer affected ({len(db_files)} files)"
            )
        
        config_files = [f for f in self.files if "config" in f.path.lower() or ".env" in f.path.lower()]
        if config_files:
            analysis["🟡 High Risks"].append(
                f"Configuration affected ({len(config_files)} files)"
            )
        
        test_files = [f for f in self.files if "test" in f.path.lower()]
        if test_files:
            analysis["🟢 Manageable Risks"].append(
                f"Test files affected ({len(test_files)} files)"
            )
        
        return {k: v for k, v in analysis.items() if v}
    
    def get_file_list_for_implementation(self) -> List[Dict[str, Any]]:
        """Get structured file list for Implementer stage."""
        return [
            {
                "path": fm.path,
                "occurrence_count": fm.occurrence_count,
                "line_numbers": [lm.line_number for lm in fm.line_matches],
            }
            for fm in self.files
        ]


# =============================================================================
# Discovery Functions
# =============================================================================

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
    
    v2.1: Now filters garbage lines and classifies matches mechanically.
    """
    roots = roots or DEFAULT_ROOTS
    exclusions = exclusions or DEFAULT_EXCLUSIONS
    
    logger.info(f"[file_discovery] v2.1 Pattern search: {search_pattern}, roots={roots}")
    
    ps_command = _build_select_string_command(
        pattern=search_pattern,
        roots=roots,
        exclusions=exclusions,
        file_filter=file_filter,
        case_sensitive=case_sensitive,
    )
    
    logger.debug(f"[file_discovery] PowerShell command: {ps_command[:200]}...")
    
    try:
        shell_result = sandbox_client.shell_run(
            command=ps_command,
            cwd_target="REPO",
            timeout_seconds=timeout_seconds,
        )
        
        stdout = getattr(shell_result, 'stdout', '') or ''
        stderr = getattr(shell_result, 'stderr', '') or ''
        ok = getattr(shell_result, 'ok', None)
        exit_code = getattr(shell_result, 'exit_code', None)
        
        logger.debug(f"[file_discovery] Result: ok={ok}, exit_code={exit_code}, stdout_len={len(stdout)}")
        
        # v1.2: Prioritize stdout over exit codes
        if stdout.strip():
            files, total_occurrences, truncated, lines_filtered = _parse_select_string_output_v21(
                stdout=stdout,
                max_results=max_results,
                max_samples_per_file=max_samples_per_file,
            )
            
            logger.info(f"[file_discovery] v2.1 Found {len(files)} files, {total_occurrences} occurrences, {lines_filtered} lines filtered")
            
            return DiscoveryResult(
                success=True,
                search_pattern=search_pattern,
                total_files=len(files),
                total_occurrences=total_occurrences,
                files=files,
                truncated=truncated,
                duration_ms=getattr(shell_result, 'duration_ms', 0),
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
                duration_ms=getattr(shell_result, 'duration_ms', 0),
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
            duration_ms=getattr(shell_result, 'duration_ms', 0),
            roots_searched=roots,
        )
        
    except Exception as e:
        logger.error(f"[file_discovery] Exception: {e}")
        return DiscoveryResult(
            success=False,
            search_pattern=search_pattern,
            total_files=0,
            total_occurrences=0,
            error_message=str(e),
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


# =============================================================================
# PowerShell Command Builders
# =============================================================================

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


# =============================================================================
# Output Parsers
# =============================================================================

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


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "LineMatch",
    "FileMatch",
    "DiscoveryResult",
    "MatchBucket",
    "MUST_REVIEW_BUCKETS",
    # Functions
    "discover_files",
    "discover_files_by_extension",
    "_should_skip_line",
    "_classify_match_mechanical",
    # Config
    "DEFAULT_ROOTS",
    "DEFAULT_EXCLUSIONS",
    "DEFAULT_FILE_EXTENSIONS",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RESULTS",
]
