from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Set


CODEBASE_REPORT_OUTPUT_DIR = Path(r"D:\Orb.architecture")

FULL_MAX_MATCHES_PER_FILE = 50

ABSOLUTE_PATH_PATTERNS = [
    re.compile(r'[A-Za-z]:\\(?:[^\s"\'<>|*?\n]+)'),  # Windows: C:\path\to\file
    re.compile(r'\\\\[A-Za-z0-9._-]+\\[^\s"\'<>|*?\n]+'),  # UNC: \\server\share
    re.compile(r'/(?:mnt|home)/[^\s"\'<>|*?\n]+'),  # Unix: /mnt/ or /home/
]

def _should_exclude_folder(folder_name: str) -> bool:
    """Check if folder should be excluded."""
    from .codebase_report import EXCLUDE_FOLDER_NAMES
    return folder_name.lower() in {n.lower() for n in EXCLUDE_FOLDER_NAMES}

def _should_exclude_file(path: Path) -> bool:
    """Check if file should be excluded."""
    from .codebase_report import EXCLUDE_FILE_EXTENSIONS
    return path.suffix.lower() in EXCLUDE_FILE_EXTENSIONS

def _count_lines_fast(path: Path, max_bytes: int = 1_000_000) -> Optional[int]:
    """Count lines in a text file (fast, with byte limit)."""
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return None  # Too large for fast counting
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def _detect_floating_files(root: Path, expected: Set[str]) -> List[str]:
    """Detect unexpected files/folders at root level."""
    floating = []
    if not root.exists():
        return floating
    
    expected_lower = {e.lower() for e in expected}
    
    for item in root.iterdir():
        if item.name.lower() not in expected_lower:
            # Exclude common generated files
            if not item.name.endswith((".backup", ".bak", ".log")):
                floating.append(item.name)
    
    return floating

def _detect_duplicate_filenames(files: List[FileEntry], threshold: int = 6) -> Dict[str, List[str]]:
    """Detect filenames that appear too many times."""
    from .codebase_report import FileEntry
    name_to_paths: Dict[str, List[str]] = {}
    
    for f in files:
        name = f.path.name.lower()
        if name not in name_to_paths:
            name_to_paths[name] = []
        name_to_paths[name].append(f.relative_path)
    
    # Filter to those over threshold
    return {name: paths for name, paths in name_to_paths.items() if len(paths) >= threshold}
