# Purpose: sandbox inspector utils 4
# Called-by: app.llm.local_tools.zobie._sandbox_inspector_utils_5, app.llm.local_tools.zobie.sandbox_inspector
# Depends-on: app.llm.local_tools.zobie.sandbox_client
# Last-renovated: 2026-06-11
from __future__ import annotations
import os
from .sandbox_client import call_fs_contents, call_fs_tree
from typing import Optional


MAX_FILE_CHARS = 8000           # Prevent ingesting megabyte logs

MAX_CLASSIFICATION_CHARS = 800  # Snippet for classification (fast)

MAX_CANDIDATE_FILES = 20        # Don't read more than this

D_DRIVE_ROOT = "D:\\"

SANDBOX_DESKTOP_ROOTS = [
    r"C:\Users\WDAGUtilityAccount\Desktop",
]

MAX_SEARCH_DEPTH = 3

def _read_snippet(file_path: str, max_chars: int) -> Optional[str]:
    """Read a small snippet for classification."""
    status, data, error = call_fs_contents([file_path])
    if status == 200 and data:
        files = data.get("files", [])
        if files:
            content = files[0].get("content")
            if content:
                return content[:max_chars]
    return None

def file_exists_in_sandbox(file_path: str) -> bool:
    """Check if file exists in sandbox (by listing parent folder)."""
    parent = os.path.dirname(file_path)
    target = os.path.basename(file_path).lower()
    
    if not parent or not target:
        return False
    
    status, data, error = call_fs_tree([parent], max_files=200)
    if status != 200 or not data:
        return False
    
    return any(
        (f.get("name") or "").lower() == target
        for f in data.get("files", [])
    )
