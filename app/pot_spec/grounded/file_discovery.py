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
from app.pot_spec.grounded._file_discovery_utils import FILE_DISCOVERY_BUILD_ID, MUST_REVIEW_BUCKETS, _build_extension_search_command, _build_select_string_command, _parse_file_list_output, _parse_select_string_output_v21, _run_powershell_local, _should_skip_line
from app.pot_spec.grounded._file_discovery_utils import FileMatch, LineMatch, MatchBucket, _classify_match_mechanical, discover_files, discover_files_by_extension

# Build ID for verification
print(f"[FILE_DISCOVERY_LOADED] BUILD_ID={FILE_DISCOVERY_BUILD_ID}")

logger = logging.getLogger(__name__)


# =============================================================================
# v2.3: LOCAL POWERSHELL FALLBACK
# =============================================================================

import subprocess
import time


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


# v2.1: Must-review buckets that should NEVER be auto-changed


# =============================================================================
# Data Models
# =============================================================================


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


# =============================================================================
# PowerShell Command Builders
# =============================================================================


# =============================================================================
# Output Parsers
# =============================================================================


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
