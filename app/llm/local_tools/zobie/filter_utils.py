# FILE: app/llm/local_tools/zobie/filter_utils.py
"""Scan result filtering utilities.

Extracted from zobie_tools.py for modularity.
No logic changes - exact same filtering behavior.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .config import is_excluded_path, is_excluded_extension


def filter_scan_results(files_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Filter scan results to exclude junk directories and file types.
    
    Returns: (filtered_files, excluded_count)
    """
    filtered = []
    excluded = 0
    
    for f in files_data:
        path = f.get("path", "")
        
        # Check directory exclusions
        if is_excluded_path(path):
            excluded += 1
            continue
        
        # Check extension exclusions
        if is_excluded_extension(path):
            excluded += 1
            continue
        
        # Check for hidden files (except allowed ones)
        name = os.path.basename(path)
        if name.startswith(".") and name not in {".env.example", ".gitignore", ".gitattributes", ".dockerignore"}:
            excluded += 1
            continue
        
        filtered.append(f)
    
    return filtered, excluded
