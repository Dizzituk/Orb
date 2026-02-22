from __future__ import annotations
import os
from typing import List, Optional


SEGMENTATION_BUILD_ID = "2026-02-18-v1.6-monolith-replacement-facade-detection"

FILE_COUNT_THRESHOLD = 15

MIN_FILES_PER_SEGMENT = 2

MAX_FILES_PER_SEGMENT = 15

BACKEND_PATH_INDICATORS = {
    "app/", "app\\",
    "routers/", "routers\\",
    "services/", "services\\",
    "endpoints/", "endpoints\\",
    "models/", "models\\",
    "main.py",
}

FRONTEND_PATH_INDICATORS = {
    "src/components/", "src\\components\\",
    "src/hooks/", "src\\hooks\\",
    "src/services/", "src\\services\\",
    "src/types/", "src\\types\\",
    "src/styles/", "src\\styles\\",
    ".tsx", ".jsx",
    "main.js",
    "main.tsx",
    "App.tsx",
}

def _generate_segment_id(index: int, layer: str) -> str:
    """Generate a segment ID like 'seg-01-backend-services'."""
    return f"seg-{index:02d}-{layer}"

def _resolve_to_absolute(path: str, known_paths: List[str]) -> Optional[str]:
    """
    Resolve a relative path to its absolute form using the architecture index.
    
    If the path is already absolute and exists, return it directly.
    Otherwise, search known_paths for a match on the relative suffix.
    """
    # Already absolute?
    if os.path.isabs(path):
        return path if os.path.exists(path) else path  # Return as-is even if missing
    
    # Normalise for matching
    path_norm = path.replace('/', '\\').lower()
    
    for known in known_paths:
        known_norm = known.replace('/', '\\').lower()
        if known_norm.endswith(path_norm):
            return known
    
    return None
