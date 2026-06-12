# FILE: app/pot_spec/grounded/_fd_models.py
# Purpose: File Discovery — data models and configuration constants.
# Called-by: app.pot_spec.grounded._fd_powershell, app.pot_spec.grounded.file_discovery
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
File Discovery — data models and configuration constants.

Extracted from file_discovery.py to reduce file size.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_ROOTS: List[str] = [
    r"D:\Orb",
    r"D:\Orb Desktop",
    r"D:\orb-desktop",
]

DEFAULT_EXCLUSIONS: List[str] = [
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    "*.egg-info", ".next", "coverage", ".coverage", "htmlcov",
]

DEFAULT_FILE_EXTENSIONS: List[str] = [
    ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".md",
    ".yaml", ".yml", ".toml", ".txt", ".html", ".css", ".sql",
]

DEFAULT_TIMEOUT_SECONDS: int = 120
DEFAULT_MAX_RESULTS: int = 2000
DEFAULT_MAX_SAMPLES_PER_FILE: int = 50

# v2.1: Summary report limits
SUMMARY_MAX_FILES: int = 20
SUMMARY_MAX_SAMPLES_PER_BUCKET: int = 3
SUMMARY_MAX_CHARS: int = 50000


# =============================================================================
# Match Classification
# =============================================================================

class MatchBucket(str, Enum):
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
    GARBAGE = "garbage"
    UNKNOWN = "unknown"


MUST_REVIEW_BUCKETS = frozenset({
    MatchBucket.ENV_VAR_KEY,
    MatchBucket.DATABASE_ARTIFACT,
    MatchBucket.FILE_FOLDER_NAME,
    MatchBucket.HISTORICAL_DATA,
})


def _should_skip_line(line: str, file_path: str = "") -> bool:
    """v2.1: Determine if a line should be filtered from evidence."""
    if not line:
        return True
    path_lower = file_path.lower()
    is_minified = '.min.' in path_lower or '/dist/' in path_lower or '\\dist\\' in path_lower
    is_config = any(ext in path_lower for ext in ['.env', '.yaml', '.yml', '.json', '.toml', '.ini'])

    if re.search(r'\bENC[:=]', line, re.IGNORECASE):
        return True
    if re.search(r'\bENCRYPTED[:=]', line, re.IGNORECASE):
        return True
    base64_threshold = 100 if is_config else 50
    if re.search(rf'[A-Za-z0-9+/]{{{base64_threshold},}}={{0,2}}', line):
        return True
    if re.search(r'(-?\d+\.\d+,?\s*){10,}', line):
        return True
    if re.search(r'[\x00-\x08\x0e-\x1f\x7f-\xff]', line):
        return True
    max_length = 200 if is_minified else (2000 if is_config else 1000)
    if len(line) > max_length:
        return True
    return False


def _classify_match_mechanical(line: str, file_path: str) -> MatchBucket:
    """v2.1: Mechanically classify a match based on path and content patterns."""
    path_lower = file_path.lower()
    line_lower = line.lower()

    if '\\' in line or '/' in line:
        return MatchBucket.FILE_FOLDER_NAME
    if re.search(r'^[A-Z][A-Z0-9_]*\s*[=:]', line) or '.env' in path_lower:
        return MatchBucket.ENV_VAR_KEY
    if any(x in path_lower for x in ['.db', '.sqlite', 'database', 'migration']):
        return MatchBucket.DATABASE_ARTIFACT
    if re.search(r'(CREATE|ALTER|INSERT|UPDATE|DELETE)\s+', line, re.IGNORECASE):
        return MatchBucket.DATABASE_ARTIFACT
    if any(x in path_lower for x in ['jobs/', 'jobs\\', 'history/', 'history\\', 'output/', 'output\\']):
        return MatchBucket.HISTORICAL_DATA
    if re.search(r'^(from|import)\s+', line) or re.search(r'require\s*\(', line):
        return MatchBucket.IMPORT_PATH
    if 'test' in path_lower or '_test.' in path_lower:
        return MatchBucket.TEST_ASSERTION
    if re.search(r'(assert|expect|should)\s*[.(]', line_lower):
        return MatchBucket.TEST_ASSERTION
    if re.search(r'@(app|router)\.(get|post|put|delete|patch)', line_lower):
        return MatchBucket.API_ROUTE
    if any(ext in path_lower for ext in ['.md', '.rst', '.txt', 'readme', 'doc']):
        return MatchBucket.DOCUMENTATION
    if any(ext in path_lower for ext in ['.yaml', '.yml', '.json', '.toml', 'config']):
        return MatchBucket.CONFIG_KEY
    if any(ext in path_lower for ext in ['.tsx', '.jsx']) and re.search(r'["\'][^"\']{2,}["\']', line):
        return MatchBucket.UI_LABEL
    if any(ext in path_lower for ext in ['.py', '.js', '.ts', '.tsx', '.jsx']):
        return MatchBucket.CODE_IDENTIFIER
    return MatchBucket.UNKNOWN


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class LineMatch:
    line_number: int
    line_content: str
    bucket: MatchBucket = MatchBucket.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        return {"line_number": self.line_number, "line_content": self.line_content, "bucket": self.bucket.value}


@dataclass
class FileMatch:
    path: str
    occurrence_count: int
    line_matches: List[LineMatch] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "occurrence_count": self.occurrence_count, "line_matches": [m.to_dict() for m in self.line_matches]}


__all__ = [
    "MatchBucket", "MUST_REVIEW_BUCKETS", "LineMatch", "FileMatch",
    "_should_skip_line", "_classify_match_mechanical",
    "DEFAULT_ROOTS", "DEFAULT_EXCLUSIONS", "DEFAULT_FILE_EXTENSIONS",
    "DEFAULT_TIMEOUT_SECONDS", "DEFAULT_MAX_RESULTS", "DEFAULT_MAX_SAMPLES_PER_FILE",
    "SUMMARY_MAX_FILES", "SUMMARY_MAX_SAMPLES_PER_BUCKET", "SUMMARY_MAX_CHARS",
]
