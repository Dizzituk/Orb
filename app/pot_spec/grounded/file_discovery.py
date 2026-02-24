# FILE: app/pot_spec/grounded/file_discovery.py
"""File Discovery System (v2.4)

v2.4 (2026-02-24): Refactored — models/config to _fd_models.py, PS commands to _fd_powershell.py
v2.3: Local PowerShell fallback
v2.1: Two-layer evidence architecture
v1.0: Initial implementation
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ._fd_models import (
    MatchBucket, MUST_REVIEW_BUCKETS, LineMatch, FileMatch,
    _should_skip_line, _classify_match_mechanical,
    DEFAULT_ROOTS, DEFAULT_EXCLUSIONS, DEFAULT_FILE_EXTENSIONS,
    DEFAULT_TIMEOUT_SECONDS, DEFAULT_MAX_RESULTS, DEFAULT_MAX_SAMPLES_PER_FILE,
    SUMMARY_MAX_FILES, SUMMARY_MAX_SAMPLES_PER_BUCKET, SUMMARY_MAX_CHARS,
)
from ._fd_powershell import (
    run_powershell_local,
    build_select_string_command,
    build_extension_search_command,
    parse_select_string_output_v21,
    parse_file_list_output,
)

FILE_DISCOVERY_BUILD_ID = "2026-02-24-v2.4-refactored"
print(f"[FILE_DISCOVERY_LOADED] BUILD_ID={FILE_DISCOVERY_BUILD_ID}")

logger = logging.getLogger(__name__)


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
    lines_filtered: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success, "search_pattern": self.search_pattern,
            "total_files": self.total_files, "total_occurrences": self.total_occurrences,
            "files": [f.to_dict() for f in self.files], "truncated": self.truncated,
            "error_message": self.error_message, "duration_ms": self.duration_ms,
            "roots_searched": self.roots_searched, "lines_filtered": self.lines_filtered,
        }

    def get_summary_report(self) -> str:
        """v2.1: Compact summary report for LLM prompts (5-50KB)."""
        lines = [
            "# Discovery Summary", "",
            f"**Pattern:** `{self.search_pattern}`",
            f"**Total Files:** {self.total_files}",
            f"**Total Occurrences:** {self.total_occurrences}",
            f"**Lines Filtered (garbage):** {self.lines_filtered}",
            f"**Duration:** {self.duration_ms}ms", "",
        ]

        buckets: Dict[MatchBucket, List[Tuple[str, LineMatch]]] = {}
        must_review_items: List[Tuple[str, LineMatch]] = []
        for fm in self.files:
            for lm in fm.line_matches:
                bucket = lm.bucket if lm.bucket != MatchBucket.UNKNOWN else _classify_match_mechanical(lm.line_content, fm.path)
                buckets.setdefault(bucket, []).append((fm.path, lm))
                if bucket in MUST_REVIEW_BUCKETS:
                    must_review_items.append((fm.path, lm))

        if must_review_items:
            lines.append("## ⚠️ MUST-REVIEW ITEMS (Do NOT auto-change)")
            lines.append("")
            for path, lm in must_review_items[:20]:
                lines.append(f"- `{path}` L{lm.line_number}: `{lm.line_content[:100]}`")
            if len(must_review_items) > 20:
                lines.append(f"- ... and {len(must_review_items) - 20} more")
            lines.append("")

        lines.append("## Matches by Category")
        lines.append("")
        for bucket in MatchBucket:
            if bucket == MatchBucket.GARBAGE:
                continue
            items = buckets.get(bucket, [])
            if not items:
                continue
            marker = "🔴" if bucket in MUST_REVIEW_BUCKETS else "🔵"
            lines.append(f"### {marker} {bucket.value} ({len(items)} matches)")
            for path, lm in items[:SUMMARY_MAX_SAMPLES_PER_BUCKET]:
                preview = lm.line_content[:80] + "..." if len(lm.line_content) > 80 else lm.line_content
                lines.append(f"  - `{path}` L{lm.line_number}: `{preview}`")
            if len(items) > SUMMARY_MAX_SAMPLES_PER_BUCKET:
                lines.append(f"  - ... and {len(items) - SUMMARY_MAX_SAMPLES_PER_BUCKET} more")
            lines.append("")

        lines.append("## Top Files (by occurrence count)")
        lines.append("")
        sorted_files = sorted(self.files, key=lambda f: f.occurrence_count, reverse=True)
        for fm in sorted_files[:SUMMARY_MAX_FILES]:
            lines.append(f"- `{fm.path}` ({fm.occurrence_count} matches)")
        if len(sorted_files) > SUMMARY_MAX_FILES:
            lines.append(f"- ... and {len(sorted_files) - SUMMARY_MAX_FILES} more files")
        lines.append("")

        result = "\n".join(lines)
        if len(result) > SUMMARY_MAX_CHARS:
            result = result[:SUMMARY_MAX_CHARS - 100] + "\n\n... [TRUNCATED]"
        return result

    def get_full_evidence_json(self) -> Dict[str, Any]:
        return {
            "search_pattern": self.search_pattern, "total_files": self.total_files,
            "total_occurrences": self.total_occurrences, "lines_filtered": self.lines_filtered,
            "duration_ms": self.duration_ms, "roots_searched": self.roots_searched,
            "files": [f.to_dict() for f in self.files],
        }

    def get_file_preview(self, max_files: int = 10) -> str:
        return self.get_summary_report()

    def get_full_evidence_report(self) -> str:
        logger.warning("[file_discovery] get_full_evidence_report() deprecated, use get_summary_report()")
        return self.get_summary_report()

    def get_file_list_for_implementation(self) -> List[Dict[str, Any]]:
        return [
            {"path": fm.path, "occurrence_count": fm.occurrence_count, "line_numbers": [lm.line_number for lm in fm.line_matches]}
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
    """Discover files containing a search pattern. v2.3: with local fallback."""
    roots = roots or DEFAULT_ROOTS
    exclusions = exclusions or DEFAULT_EXCLUSIONS

    ps_command = build_select_string_command(
        pattern=search_pattern, roots=roots, exclusions=exclusions,
        file_filter=file_filter, case_sensitive=case_sensitive,
    )

    stdout, stderr, duration_ms = '', '', 0
    use_local = False

    if sandbox_client:
        try:
            shell_result = sandbox_client.shell_run(command=ps_command, cwd_target="REPO", timeout_seconds=timeout_seconds)
            stdout = getattr(shell_result, 'stdout', '') or ''
            stderr = getattr(shell_result, 'stderr', '') or ''
            duration_ms = getattr(shell_result, 'duration_ms', 0)
        except Exception:
            use_local = True
    else:
        use_local = True

    if use_local or (not stdout.strip() and not stderr.strip()):
        success, stdout, stderr, duration_ms = run_powershell_local(command=ps_command, timeout_seconds=timeout_seconds)
        if not success:
            return DiscoveryResult(success=False, search_pattern=search_pattern, total_files=0, total_occurrences=0, error_message=f"Local PowerShell failed: {stderr}", duration_ms=duration_ms, roots_searched=roots)

    if stdout.strip():
        files, total_occ, truncated, filtered = parse_select_string_output_v21(stdout=stdout, max_results=max_results, max_samples_per_file=max_samples_per_file)
        return DiscoveryResult(success=True, search_pattern=search_pattern, total_files=len(files), total_occurrences=total_occ, files=files, truncated=truncated, duration_ms=duration_ms, roots_searched=roots, lines_filtered=filtered)

    if stderr.strip():
        return DiscoveryResult(success=False, search_pattern=search_pattern, total_files=0, total_occurrences=0, error_message=f"PowerShell error: {stderr[:500]}", duration_ms=duration_ms, roots_searched=roots)

    return DiscoveryResult(success=True, search_pattern=search_pattern, total_files=0, total_occurrences=0, files=[], duration_ms=duration_ms, roots_searched=roots)


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
    ps_command = build_extension_search_command(extension=ext, roots=roots, exclusions=exclusions)

    try:
        shell_result = sandbox_client.shell_run(command=ps_command, cwd_target="REPO", timeout_seconds=timeout_seconds)
        if not shell_result.ok and shell_result.exit_code != 0:
            return DiscoveryResult(success=False, search_pattern=f"*.{ext}", total_files=0, total_occurrences=0, error_message=f"PowerShell error: {shell_result.stderr[:500]}", duration_ms=shell_result.duration_ms, roots_searched=roots)
        files, truncated = parse_file_list_output(stdout=shell_result.stdout, max_results=max_results)
        return DiscoveryResult(success=True, search_pattern=f"*.{ext}", total_files=len(files), total_occurrences=len(files), files=files, truncated=truncated, duration_ms=shell_result.duration_ms, roots_searched=roots)
    except Exception as e:
        return DiscoveryResult(success=False, search_pattern=f"*.{ext}", total_files=0, total_occurrences=0, error_message=str(e), roots_searched=roots)


__all__ = [
    "LineMatch", "FileMatch", "DiscoveryResult", "MatchBucket", "MUST_REVIEW_BUCKETS",
    "discover_files", "discover_files_by_extension",
    "_should_skip_line", "_classify_match_mechanical",
    "DEFAULT_ROOTS", "DEFAULT_EXCLUSIONS", "DEFAULT_FILE_EXTENSIONS",
    "DEFAULT_TIMEOUT_SECONDS", "DEFAULT_MAX_RESULTS",
]
