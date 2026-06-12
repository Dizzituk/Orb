# Purpose: RAG security utilities.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""RAG security utilities."""
from .sensitive_files import (
    is_sensitive_file,
    should_skip_directory,
    SENSITIVE_EXACT_NAMES,
    SENSITIVE_PATTERNS,
)
