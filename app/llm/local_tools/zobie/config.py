# FILE: app/llm/local_tools/zobie/config.py
"""Configuration constants and exclusion patterns for zobie tools.

Extracted from zobie_tools.py - no logic changes.
"""

from __future__ import annotations

import os
import re
from typing import List, Set

from app.llm.local_tools.archmap_helpers import (
    default_controller_base_url,
    default_zobie_mapper_out_dir,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Sandbox controller URL (inside Windows Sandbox)
SANDBOX_CONTROLLER_URL = os.getenv("ORB_SANDBOX_CONTROLLER_URL", "http://192.168.250.2:8765")

# Legacy zobie mapper settings
ZOBIE_CONTROLLER_URL = os.getenv("ORB_ZOBIE_CONTROLLER_URL") or default_controller_base_url(__file__)
ZOBIE_MAPPER_OUT_DIR = os.getenv("ORB_ZOBIE_MAPPER_OUT_DIR") or default_zobie_mapper_out_dir(__file__)
ZOBIE_MAPPER_ARGS_RAW = os.getenv("ORB_ZOBIE_MAPPER_ARGS", "200000 0 60 120000").strip()
ZOBIE_MAPPER_ARGS: List[str] = [a for a in ZOBIE_MAPPER_ARGS_RAW.split() if a]

# Scan roots - must match sandbox_controller ALLOWED_FS_ROOTS
# CODE: D:\Orb + D:\orb-desktop only
# SANDBOX: D:\ (full drive) + C:\Users\<user> 
CODE_SCAN_ROOTS = [r"D:\Orb", r"D:\orb-desktop"]  # Code repos only
SANDBOX_SCAN_ROOTS = ["D:\\", r"C:\Users\dizzi"]  # Full D: drive + user folder

# Output directory for CREATE ARCHITECTURE MAP (ALL CAPS)
# This is where INDEX.json, CODEBASE.md and ARCHITECTURE_MAP.md go
FULL_ARCHMAP_OUTPUT_DIR = r"D:\Orb\.architecture"
FULL_ARCHMAP_OUTPUT_FILE = "ARCHITECTURE_MAP.md"
FULL_CODEBASE_OUTPUT_FILE = "CODEBASE.md"

# Max file size for content capture (500KB)
MAX_CONTENT_FILE_SIZE = 500 * 1024

# v4.3: Sandbox scan content fetch settings
# Smaller batch size for sandbox (larger scale than code repos)
SANDBOX_CONTENT_BATCH_SIZE = int(os.getenv("ORB_SANDBOX_CONTENT_BATCH", "25"))
# Hard size cap for sandbox content fetch (1MB)
SANDBOX_MAX_CONTENT_SIZE = int(os.getenv("ORB_SANDBOX_MAX_CONTENT_SIZE", str(1_000_000)))

# Timeouts
FS_TREE_TIMEOUT_SEC = int(os.getenv("ORB_FS_TREE_TIMEOUT_SEC", "120"))


# =============================================================================
# EXCLUSION PATTERNS (v4.2 - from host_fs_scanner.py)
# =============================================================================

# Directory patterns to exclude (regex, matched against full path)
EXCLUDE_DIR_PATTERNS = [
    r"\.git$",
    r"\.git[/\\]",
    r"node_modules$",
    r"node_modules[/\\]",
    r"dist$",
    r"build$",
    r"\.next$",
    r"\.vite$",
    r"\.venv$",
    r"venv$",
    r"__pycache__$",
    r"__pycache__[/\\]",
    r"\.pytest_cache$",
    r"\.mypy_cache$",
    r"\.ruff_cache$",
    r"\.idea$",
    r"\.vscode$",
    r"\.tox$",
    r"\.nox$",
    r"\.eggs$",
    r"\.egg-info$",
    r"htmlcov$",
    r"\.coverage$",
    r"orb-electron-data$",
    # Windows caches and temp
    r"Code Cache$",
    r"GPUCache$",
    r"Cache$",
    r"CachedData$",
    r"CachedExtensions$",
    r"AppData[/\\]Local[/\\]Temp",
    r"AppData[/\\]Local[/\\]Microsoft",
    r"AppData[/\\]Local[/\\]Google[/\\]Chrome",
    r"AppData[/\\]Local[/\\]Mozilla",
    r"AppData[/\\]LocalLow",
    r"NTUSER\.DAT",
    # v4.6: Windows Store app caches
    r"AppData[/\\]Local[/\\]Packages",  # All Windows Store app data
    r"EBWebView",                        # Edge WebView2 caches
    r"Cache_Data",                       # Chromium cache folders
    r"ShaderCache",                      # GPU shader caches
    r"Cookies$",                         # Browser cookies folders
    # System folders
    r"\$Recycle\.Bin",
    r"System Volume Information",
]

# File extensions to exclude
EXCLUDE_FILE_EXTENSIONS: Set[str] = {
    # Logs and temp
    ".log",
    # Archives
    ".iso", ".vhd", ".vhdx", ".qcow2", ".img",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
    # Databases (metadata ok but don't scan content)
    ".sqlite", ".sqlite3", ".db", ".wal", ".shm",
    # Binaries
    ".dll", ".exe", ".msi", ".sys", ".bin", ".dat",
    ".pdb", ".obj", ".o", ".a", ".so", ".dylib",
    ".pyc", ".pyo", ".class", ".jar", ".war",
    # Large media
    ".mp4", ".mkv", ".avi", ".mov", ".wmv",
    ".mp3", ".wav", ".flac", ".aac", ".ogg",
    # Large images
    ".psd", ".xcf", ".raw", ".cr2", ".nef",
}

# Compile exclusion patterns once
_EXCLUDE_DIR_RX = [re.compile(p, re.IGNORECASE) for p in EXCLUDE_DIR_PATTERNS]


# =============================================================================
# SANDBOX SCAN CONTENT SETTINGS (v4.3)
# =============================================================================

# Content extensions for sandbox scan (same as generate_full_architecture_map_stream)
# IMPORTANT: Do NOT broaden this - sandbox scans D:\ + C:\Users which is huge
SANDBOX_CONTENT_EXTENSIONS: Set[str] = {
    ".py", ".pyw", ".pyi",
    ".js", ".mjs", ".cjs", ".jsx",
    ".ts", ".tsx", ".mts", ".cts",
    ".json", ".jsonc",
    ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".sql", ".sh", ".bash", ".zsh", ".ps1", ".psm1", ".bat", ".cmd",
    ".md", ".markdown", ".rst", ".txt",
    "",  # Files without extension (like Dockerfile, Makefile)
}

# Files to NEVER capture (secrets, credentials, keys)
SANDBOX_SKIP_FILENAMES: Set[str] = {
    ".env", ".env.local", ".env.production", ".env.development",
    ".env.example",
    "secrets.json", "credentials.json", "config.secret.json",
    ".npmrc", ".pypirc",
    "id_rsa", "id_ed25519", "id_ecdsa",
    ".pem", ".key", ".crt", ".p12", ".pfx",
}

# Patterns in filename to skip
SANDBOX_SKIP_PATTERNS: Set[str] = {"secret", "credential", "password", "token", "apikey", "api_key"}


# =============================================================================
# FILESYSTEM QUERY SETTINGS (v5.0 - surgical live read)
# =============================================================================

# v5.8: Expanded allowlist for full sandbox D: drive access
# Allowed roots for list/read/head/lines/write commands
# NOTE: D:\ allows ALL of D: drive - safe because this runs in sandbox only
FILESYSTEM_QUERY_ALLOWED_ROOTS = [
    "D:\\",           # Full D: drive (sandbox root) - can't use raw string here
    r"D:\Orb",        # Legacy: still explicit for clarity
    r"D:\orb-desktop",
    r"C:\Users\dizzi\OneDrive\Desktop",
    r"C:\Users\dizzi\OneDrive\Documents",
    r"C:\Users\dizzi\OneDrive\Downloads",
    r"C:\Users\dizzi\OneDrive\Pictures",
    r"C:\Users\dizzi\OneDrive",
    r"C:\Users\dizzi\Desktop",
    r"C:\Users\dizzi\Documents",
    r"C:\Users\dizzi\Downloads",
    r"C:\Users\dizzi\Pictures",
]

# v5.0: Explicitly blocked paths (even if under allowed roots)
FILESYSTEM_QUERY_BLOCKED_PATHS = [
    r"C:\Users\dizzi\AppData",
    r"C:\Users\dizzi\.vscode",
    r"C:\Users\dizzi\.venv",
    r"C:\Users\dizzi\Contacts",
    r"C:\Users\dizzi\Searches",
    r"C:\Users\dizzi\Saved Games",
    r"C:\Windows",
    r"C:\Program Files",
    r"C:\Program Files (x86)",
    r"Microsoft\Protect",
    r"Microsoft\Credentials",
]

# Max entries to return (hard cap)
FILESYSTEM_QUERY_MAX_ENTRIES = 200

# v5.0: Read file content limits (increased for surgical reads)
FILESYSTEM_READ_MAX_LINES = 400
FILESYSTEM_READ_MAX_BYTES = 64 * 1024  # 64KB limit

# Known folder mappings (for queries like "What's in my Desktop")
KNOWN_FOLDER_PATHS = {
    "desktop": r"C:\Users\dizzi\OneDrive\Desktop",
    "onedrive": r"C:\Users\dizzi\OneDrive",
    "documents": r"C:\Users\dizzi\OneDrive\Documents",
    "downloads": r"C:\Users\dizzi\OneDrive\Downloads",
    "pictures": r"C:\Users\dizzi\OneDrive\Pictures",
    "videos": r"C:\Users\dizzi\Videos",
    "music": r"C:\Users\dizzi\Music",
}


# =============================================================================
# EXCLUSION FILTER FUNCTIONS
# =============================================================================

def is_excluded_path(path: str) -> bool:
    """Check if path should be excluded based on directory patterns."""
    path_norm = path.replace("\\", "/")
    for rx in _EXCLUDE_DIR_RX:
        if rx.search(path_norm):
            return True
    return False


def is_excluded_extension(path: str) -> bool:
    """Check if file extension should be excluded."""
    import os
    ext = os.path.splitext(path.lower())[1]
    return ext in EXCLUDE_FILE_EXTENSIONS
