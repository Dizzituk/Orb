"""
Scan root configuration for RAG memory system.

Scope:
- D:/ full scan (sandbox)
- C:/Users/<current_user>/ filtered
- System paths always excluded
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# =============================================================================
# ENVIRONMENT CONFIGURATION
# =============================================================================

_USER_HOME = Path(os.getenv("ASTRA_USER_HOME", str(Path.home())))
_D_DRIVE = os.getenv("ASTRA_D_DRIVE", "D:\\")

# =============================================================================
# ZOBIE MAPPER OUTPUT LOCATION
# =============================================================================

ZOBIE_OUTPUT_DIR = os.getenv(
    "ASTRA_ZOBIE_OUTPUT_DIR",
    os.path.join(_D_DRIVE, "Tools", "arch_query_service", "scans", "orb")
)

# =============================================================================
# SCAN ROOTS
# =============================================================================

INDEX_ROOTS: List[str] = [
    _D_DRIVE,
    str(_USER_HOME),
]

# =============================================================================
# USER HOME FILTERING
# =============================================================================

USER_HOME_ALLOWED_SUBDIRS: Set[str] = {
    "Desktop", "Documents", "Projects", "Downloads",
    "Source", "dev", "code", "repos",
}

# =============================================================================
# SYSTEM PATH EXCLUSIONS
# =============================================================================

SYSTEM_PATH_EXCLUDES: Set[str] = {
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\ProgramData",
    "C:\\$Recycle.Bin",
    "C:\\System Volume Information",
}

# =============================================================================
# DIRECTORY SKIP LIST
# =============================================================================

SKIP_DIRECTORY_NAMES: Set[str] = {
    ".git", ".svn", ".hg",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "venv", ".venv", "env",
    "dist", "build", ".next", ".nuxt",
    ".idea", ".vscode",
    "AppData", "$RECYCLE.BIN",
}

# =============================================================================
# FILE SKIP LIST
# =============================================================================

SKIP_FILE_EXTENSIONS: Set[str] = {
    ".iso", ".vmdk", ".vdi", ".vhd",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".exe", ".dll", ".msi", ".sys",
    ".mp4", ".mkv", ".avi", ".mov",
    ".bak", ".db-journal",
}

LARGE_FILE_THRESHOLD_MB = 50

# =============================================================================
# ROOT ALIASES
# =============================================================================

ROOT_ALIASES: Dict[str, Dict] = {
    "d-drive": {
        "abs_path": _D_DRIVE,
        "root_kind": "sandbox",
        "default_zone": "projects",
        "full_scan": True,
    },
    "user-home": {
        "abs_path": str(_USER_HOME),
        "root_kind": "user",
        "default_zone": "user",
        "full_scan": False,
    },
}

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_scan_targets() -> List[str]:
    """Get scan targets that exist."""
    return [r for r in INDEX_ROOTS if os.path.isdir(r)]


def is_system_path(abs_path: str) -> bool:
    """Check if path is system (never scan)."""
    abs_lower = abs_path.lower()
    for sys_path in SYSTEM_PATH_EXCLUDES:
        if abs_lower.startswith(sys_path.lower()):
            return True
    return False


def is_allowed_user_subdir(abs_path: str) -> bool:
    """Check if user home path is in allowed subdirs."""
    abs_path = os.path.normpath(abs_path)
    user_home = os.path.normpath(str(_USER_HOME))
    
    if not abs_path.lower().startswith(user_home.lower()):
        return True  # Not under user home
    
    relative = abs_path[len(user_home):].lstrip(os.sep)
    if not relative:
        return True
    
    first_component = relative.split(os.sep)[0]
    return first_component in USER_HOME_ALLOWED_SUBDIRS


def should_skip_directory(dirname: str) -> bool:
    """Check if directory should be skipped."""
    return dirname.lower() in {d.lower() for d in SKIP_DIRECTORY_NAMES}


def should_skip_file(filename: str, size_bytes: int = 0) -> Tuple[bool, Optional[str]]:
    """Check if file should be skipped."""
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in SKIP_FILE_EXTENSIONS:
        return True, f"Extension: {ext}"
    
    if size_bytes > LARGE_FILE_THRESHOLD_MB * 1024 * 1024:
        return True, f"Size: {size_bytes // (1024*1024)}MB"
    
    return False, None


def get_root_alias(abs_path: str) -> Optional[Tuple[str, str, str]]:
    """Find root alias for path. Returns (alias, kind, zone) or None."""
    abs_path = os.path.normpath(abs_path).lower()
    
    best_match = None
    best_length = 0
    
    for alias, config in ROOT_ALIASES.items():
        alias_path = os.path.normpath(config["abs_path"]).lower()
        
        if abs_path.startswith(alias_path):
            remainder = abs_path[len(alias_path):]
            # Valid match if:
            # 1. Exact match (remainder is empty)
            # 2. Remainder starts with separator (alias doesn't end with one)
            # 3. Alias ends with separator (like D:\) - any remainder is valid
            if remainder == "" or remainder.startswith(os.sep) or alias_path.endswith(os.sep):
                if len(alias_path) > best_length:
                    best_length = len(alias_path)
                    best_match = (alias, config["root_kind"], config["default_zone"])
    
    return best_match


def is_scannable_path(abs_path: str) -> Tuple[bool, Optional[str]]:
    """Full check if path should be scanned."""
    if is_system_path(abs_path):
        return False, "System path"
    
    if not is_allowed_user_subdir(abs_path):
        return False, "Not in allowed user subdirs"
    
    for component in abs_path.split(os.sep):
        if should_skip_directory(component):
            return False, f"Skip dir: {component}"
    
    return True, None


def get_zobie_output_dir() -> str:
    """Get the zobie mapper output directory."""
    return ZOBIE_OUTPUT_DIR


def get_latest_zobie_file(prefix: str) -> Optional[str]:
    """
    Get the most recent zobie output file matching prefix.
    
    Args:
        prefix: File prefix like "SIGNATURES_" or "INDEX_"
        
    Returns:
        Full path to most recent file, or None if not found
    """
    import glob
    
    pattern = os.path.join(ZOBIE_OUTPUT_DIR, f"{prefix}*.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by filename (timestamps are sortable: YYYY-MM-DD_HHMM)
    files.sort(reverse=True)
    return files[0]
