# FILE: app/pot_spec/grounded/evidence_gathering.py
"""
Evidence-First Filesystem Validation (v1.27)

Core principle: Evidence gathered BEFORE LLM call, not discovered by LLM.

Access control (non-negotiable):
- SpecGate has RAG hints ✅
- SpecGate has Live FS read ✅ (via sandbox client)
- SpecGate has Live FS write ❌ (NEVER)

Version Notes:
-------------
v1.29 (2026-01-29): CRITICAL FIX - format_multi_target_reply() now generates LLM responses
    - Made format_multi_target_reply() async and call generate_reply_from_content() for EACH file
    - Now generates conversational LLM replies for each file's content, not just raw dumps
    - Added provider_id, model_id, llm_call_func parameters
    - Each file gets: Content excerpt + LLM-generated response
    - Fallback to acknowledgment if qa_processing unavailable
v1.28 (2026-01-29): CRITICAL BUG FIX - sandbox_read_file() now actually reads files!
    - Fixed sandbox_read_file() to use data.get('files', []) instead of data.get('contents', {})
    - This matches sandbox_inspector's working implementation
    - Previous version couldn't read ANY files despite correct path resolution
    - Removed max_file_size parameter from call_fs_contents (not supported)
v1.27 (2026-01-29): Multi-target file read support (Level 2.5)
    - Added gather_multi_target_evidence() for reading N specific files
    - Added drive-letter path resolution (D:, C:, E:, etc.)
    - Imports extract_file_targets from sandbox_discovery
    - Updated ANCHOR_RESOLUTION_MAP with drive-letter support
    - gather_filesystem_evidence() now handles multi-target requests
v1.26.1 (2026-01-27): Case-insensitive path matching on Windows
    - sandbox_path_exists() now handles case variations (Test vs test)
    - sandbox_read_file() uses actual path from sandbox_path_exists()
    - resolve_path_enhanced() returns actual path from filesystem
v1.26 (2026-01-27): CRITICAL FIX - Use sandbox client instead of os.path.exists()
    - Evidence gathering now uses call_fs_tree/call_fs_contents to check SANDBOX filesystem
    - Added sandbox_path_exists() and sandbox_read_file() helper functions
    - Fixed subfolder handling: "folder called Test" + "file called Test" -> Desktop/Test/Test.txt
    - All filesystem checks now go through sandbox controller at 192.168.250.2:8765
v1.25.1 (2026-01-27): Bug fixes for path resolution
    - Fixed extract_path_references() extracting "with" and other stopwords
    - Added extension fallback: "Test" -> tries "Test.txt", "Test.md", etc.
v1.25 (2026-01): Initial Evidence-First architecture
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# v1.29 BUILD VERIFICATION
# =============================================================================
EVIDENCE_GATHERING_BUILD_ID = "2026-01-30-v1.32-scan-roots-fix"
print(f"[EVIDENCE_GATHERING_LOADED] BUILD_ID={EVIDENCE_GATHERING_BUILD_ID}")
logger.info(f"[evidence_gathering] Module loaded: BUILD_ID={EVIDENCE_GATHERING_BUILD_ID}")


# =============================================================================
# SANDBOX CLIENT IMPORTS (v1.26)
# =============================================================================

try:
    from app.llm.local_tools.zobie.sandbox_client import (
        call_fs_tree,
        call_fs_contents,
    )
    _SANDBOX_CLIENT_AVAILABLE = True
    logger.info("[evidence_gathering] v1.26 Sandbox client loaded successfully")
except ImportError as e:
    _SANDBOX_CLIENT_AVAILABLE = False
    logger.warning("[evidence_gathering] v1.26 Sandbox client not available: %s", e)
    call_fs_tree = None
    call_fs_contents = None


# =============================================================================
# v1.27: MULTI-TARGET EXTRACTION IMPORT
# =============================================================================

try:
    from .sandbox_discovery import (
        extract_file_targets,
        is_multi_target_request,
        is_system_wide_scan_request,
        get_system_scan_targets,
        extract_scan_file_names,
        SYSTEM_SCAN_ROOTS,
    )
    _MULTI_TARGET_AVAILABLE = True
    _SYSTEM_SCAN_AVAILABLE = True
    logger.info("[evidence_gathering] v1.31 Multi-target + system scan extraction loaded")
except ImportError as e:
    _MULTI_TARGET_AVAILABLE = False
    _SYSTEM_SCAN_AVAILABLE = False
    logger.warning("[evidence_gathering] v1.31 Multi-target/system scan extraction not available: %s", e)
    extract_file_targets = None
    is_multi_target_request = None
    is_system_wide_scan_request = None
    get_system_scan_targets = None
    extract_scan_file_names = None
    SYSTEM_SCAN_ROOTS = []


# =============================================================================
# CONSTANTS
# =============================================================================

# Import sandbox configuration from authoritative source
try:
    from app.llm.local_tools.zobie.config import (
        KNOWN_FOLDER_PATHS,
        FILESYSTEM_QUERY_ALLOWED_ROOTS,
        FILESYSTEM_QUERY_BLOCKED_PATHS,
    )
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    KNOWN_FOLDER_PATHS = {}
    FILESYSTEM_QUERY_ALLOWED_ROOTS = []
    FILESYSTEM_QUERY_BLOCKED_PATHS = []

# Allowed roots for evidence gathering (fallback if config unavailable)
EVIDENCE_ALLOWED_ROOTS = FILESYSTEM_QUERY_ALLOWED_ROOTS if _CONFIG_AVAILABLE else [
    "D:\\",
    "D:\\Orb",
    "D:\\orb-desktop",
    "C:\\Users\\dizzi\\OneDrive",
]

# Forbidden paths (never resolve to these)
EVIDENCE_FORBIDDEN_PATHS = FILESYSTEM_QUERY_BLOCKED_PATHS if _CONFIG_AVAILABLE else [
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
    "C:\\Users\\dizzi\\AppData",
]

# Known anchor locations for Desktop/Documents resolution
# v1.27: Added drive-letter anchors
ANCHOR_RESOLUTION_MAP = {
    "desktop": [
        "C:\\Users\\dizzi\\OneDrive\\Desktop",
        "C:\\Users\\dizzi\\Desktop",
        "C:\\Users\\Public\\Desktop",
    ],
    "documents": [
        "C:\\Users\\dizzi\\OneDrive\\Documents",
        "C:\\Users\\dizzi\\Documents",
    ],
    "downloads": [
        "C:\\Users\\dizzi\\OneDrive\\Downloads",
        "C:\\Users\\dizzi\\Downloads",
    ],
    "pictures": [
        "C:\\Users\\dizzi\\OneDrive\\Pictures",
        "C:\\Users\\dizzi\\Pictures",
    ],
    "onedrive": [
        "C:\\Users\\dizzi\\OneDrive",
    ],
    # v1.27: Drive-letter anchors - these are root paths
    "D:": ["D:\\"],
    "C:": ["C:\\"],
    "E:": ["E:\\"],
    "F:": ["F:\\"],
    "G:": ["G:\\"],
}

# Common file extensions to try when a filename has no extension
COMMON_FILE_EXTENSIONS = ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.js', '.ts', '.html', '.css']

# Stopwords that should never be extracted as file/folder names
PATH_REFERENCE_STOPWORDS = {
    'the', 'a', 'an', 'my', 'your', 'this', 'that', 'it',
    'with', 'from', 'to', 'in', 'on', 'at', 'of', 'for', 'by',
    'and', 'or', 'but', 'not', 'is', 'are', 'was', 'were',
    'can', 'will', 'should', 'could', 'would', 'may', 'might',
    'read', 'write', 'delete', 'create', 'make', 'get', 'set',
    'overwrite', 'replace', 'modify', 'change', 'update',
    'only', 'just', 'also', 'here', 'there', 'where', 'when',
}


# =============================================================================
# EVIDENCE TYPES
# =============================================================================

class FilesystemEvidenceSource(str, Enum):
    """Source of filesystem evidence."""
    RAG_HINT = "rag_hint"           # Found via RAG query (may be stale)
    LIVE_FS = "live_fs"             # Confirmed via live filesystem check
    SANDBOX_FS = "sandbox_fs"       # v1.26: Confirmed via sandbox client
    RAG_CONFIRMED = "rag_confirmed" # RAG hint validated by live filesystem
    SEARCH_FOUND = "search_found"   # Discovered via live filesystem search
    NOT_FOUND = "not_found"         # Path was checked but does not exist
    USER_PROVIDED = "user_provided" # Explicitly provided by user


@dataclass
class FileEvidence:
    """Evidence about a single file."""
    original_reference: str           # What user said ("test.txt", "Test folder")
    resolved_path: Optional[str]      # Full filesystem path or None
    source: FilesystemEvidenceSource  # How we found/validated it
    exists: bool                      # Does it exist on filesystem?
    readable: bool                    # Can we read it?
    writable: bool = False            # SpecGate NEVER has write access
    size_bytes: Optional[int] = None
    mtime: Optional[str] = None
    content_preview: Optional[str] = None  # First ~500 chars
    content_hash: Optional[str] = None     # For change detection
    detected_structure: Optional[str] = None  # qa_format, python, json, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # v1.27: Full content for multi-target read operations
    full_content: Optional[str] = None
    
    def to_evidence_line(self) -> str:
        """Format as single-line evidence for LLM prompt."""
        status = "✓ EXISTS" if self.exists else "✗ NOT_FOUND"
        readable_str = " (readable)" if self.readable else " (not readable)"
        size_str = f" [{self.size_bytes} bytes]" if self.size_bytes else ""
        structure_str = f" [{self.detected_structure}]" if self.detected_structure else ""
        
        return f"  - {self.resolved_path or self.original_reference}: {status}{readable_str}{size_str}{structure_str}"


@dataclass
class EvidencePackage:
    """Complete evidence package for a task."""
    task_type: str                     # qa_file_task, code_analysis, file_operation, multi_target_read, unresolved
    target_files: List[FileEvidence] = field(default_factory=list)
    rag_hints_used: List[dict] = field(default_factory=list)
    search_queries_run: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ground_truth_timestamp: str = ""
    # v1.27: Multi-target read tracking
    is_multi_target: bool = False
    multi_target_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def has_valid_targets(self) -> bool:
        """Check if we have at least one valid target file."""
        return any(fe.exists and fe.readable for fe in self.target_files)
    
    def get_primary_target(self) -> Optional[FileEvidence]:
        """Get the primary (first valid) target file."""
        for fe in self.target_files:
            if fe.exists and fe.readable:
                return fe
        return None
    
    def get_all_valid_targets(self) -> List[FileEvidence]:
        """v1.27: Get all valid (existing and readable) target files."""
        return [fe for fe in self.target_files if fe.exists and fe.readable]
    
    def to_summary(self) -> str:
        """Generate summary string."""
        valid = sum(1 for fe in self.target_files if fe.exists)
        total = len(self.target_files)
        multi_str = " [MULTI-TARGET]" if self.is_multi_target else ""
        return f"EvidencePackage(task={self.task_type}{multi_str}, files={valid}/{total} valid, errors={len(self.validation_errors)})"


# =============================================================================
# SANDBOX FILESYSTEM HELPERS (v1.26)
# =============================================================================

def sandbox_path_exists(path: str) -> Tuple[bool, Optional[Dict]]:
    """
    v1.26: Check if a path exists in the SANDBOX filesystem.
    
    Uses call_fs_tree to check existence via sandbox controller.
    v1.26.1: Case-insensitive matching on Windows.
    
    Returns:
        (exists: bool, file_info: Optional[Dict])
        file_info contains 'size', 'mtime', 'actual_path' if file exists
    """
    if not _SANDBOX_CLIENT_AVAILABLE or not call_fs_tree:
        logger.warning("[evidence_gathering] v1.26 sandbox_path_exists: sandbox client not available, falling back to os.path.exists")
        exists = os.path.exists(path)
        return exists, None
    
    try:
        # Check if parent directory exists and contains this file/folder
        parent_dir = os.path.dirname(path)
        target_name = os.path.basename(path)
        
        logger.info("[evidence_gathering] v1.26.1 sandbox_path_exists: checking %s", path)
        
        status, data, error = call_fs_tree([parent_dir], max_files=100)
        
        if status != 200 or not data:
            # v1.26.1: If parent doesn't exist, try case variations
            # Try common case variations for Desktop/Documents/Test folders
            parent_variations = [
                parent_dir,
                parent_dir.replace('\\Test', '\\test'),
                parent_dir.replace('\\test', '\\Test'),
                parent_dir.replace('Desktop\\Test', 'Desktop\\test'),
                parent_dir.replace('Desktop\\test', 'Desktop\\Test'),
            ]
            
            for parent_var in parent_variations[1:]:  # Skip first, already tried
                if parent_var == parent_dir:
                    continue
                status, data, error = call_fs_tree([parent_var], max_files=100)
                if status == 200 and data:
                    parent_dir = parent_var
                    logger.info(
                        "[evidence_gathering] v1.26.1 Found parent with case variation: %s",
                        parent_var
                    )
                    break
            
            if status != 200 or not data:
                logger.info(
                    "[evidence_gathering] v1.26.1 sandbox_path_exists: parent dir check failed for %s (status=%s, error=%s)",
                    parent_dir, status, error
                )
                return False, None
        
        files = data.get("files", [])
        
        # Look for match (case-insensitive on Windows)
        for f in files:
            f_path = f.get("path", "") if isinstance(f, dict) else str(f)
            f_name = os.path.basename(f_path)
            
            # v1.26.1: Case-insensitive comparison
            if f_name.lower() == target_name.lower():
                file_info = {
                    "path": f_path,  # Return ACTUAL path from filesystem
                    "actual_path": f_path,
                    "size": f.get("size") if isinstance(f, dict) else None,
                    "mtime": f.get("mtime") if isinstance(f, dict) else None,
                    "is_dir": f.get("is_dir", False) if isinstance(f, dict) else False,
                }
                logger.info(
                    "[evidence_gathering] v1.26.1 sandbox_path_exists: FOUND %s (actual: %s, size=%s)",
                    path, f_path, file_info.get("size")
                )
                return True, file_info
        
        logger.info(
            "[evidence_gathering] v1.26.1 sandbox_path_exists: NOT FOUND %s in %d files",
            target_name, len(files)
        )
        return False, None
        
    except Exception as e:
        logger.warning(
            "[evidence_gathering] v1.26 sandbox_path_exists: exception checking %s: %s",
            path, e
        )
        return False, None


def sandbox_read_file(path: str, max_chars: int = 8000) -> Tuple[bool, Optional[str]]:
    """
    v1.26.1: Read file content from SANDBOX filesystem.
    
    Uses call_fs_contents to read via sandbox controller.
    v1.26.1: Case-insensitive path resolution.
    
    Returns:
        (success: bool, content: Optional[str])
    """
    if not _SANDBOX_CLIENT_AVAILABLE or not call_fs_contents:
        logger.warning("[evidence_gathering] v1.26 sandbox_read_file: sandbox client not available")
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(max_chars)
            return True, content
        except Exception as e:
            logger.warning("[evidence_gathering] v1.26 sandbox_read_file fallback failed: %s", e)
            return False, None
    
    try:
        # v1.26.1: First resolve the actual path (case-insensitive)
        exists, file_info = sandbox_path_exists(path)
        if exists and file_info:
            actual_path = file_info.get("actual_path", path)
        else:
            actual_path = path
        
        logger.info("[evidence_gathering] v1.26.1 sandbox_read_file: reading %s (actual: %s)", path, actual_path)
        
        status, data, error = call_fs_contents([actual_path])
        
        if status != 200 or not data:
            logger.warning(
                "[evidence_gathering] v1.28 sandbox_read_file: failed for %s (status=%s, error=%s)",
                actual_path, status, error
            )
            return False, None
        
        # v1.28 FIX: Use 'files' array like sandbox_inspector (not 'contents' dict)
        files = data.get("files", [])
        
        if not files:
            logger.warning(
                "[evidence_gathering] v1.28 sandbox_read_file: no files in response for %s",
                actual_path
            )
            return False, None
        
        # Get content from first file
        content = files[0].get("content")
        if not content:
            logger.warning(
                "[evidence_gathering] v1.28 sandbox_read_file: no content in file object for %s",
                actual_path
            )
            return False, None
        
        # Truncate if too long
        if len(content) > max_chars:
            logger.info(
                "[evidence_gathering] v1.28 sandbox_read_file: truncating %s from %d to %d chars",
                actual_path, len(content), max_chars
            )
            content = content[:max_chars]
        
        logger.info(
            "[evidence_gathering] v1.28 sandbox_read_file: SUCCESS %s (%d chars)",
            actual_path, len(content)
        )
        return True, content
        
    except Exception as e:
        logger.warning(
            "[evidence_gathering] v1.26 sandbox_read_file: exception reading %s: %s",
            path, e
        )
        return False, None


def sandbox_list_directory(path: str) -> Tuple[bool, List[Dict]]:
    """
    v1.26: List directory contents from SANDBOX filesystem.
    
    Returns:
        (success: bool, files: List[Dict])
        Each file dict contains 'path', 'name', 'size', 'is_dir'
    """
    if not _SANDBOX_CLIENT_AVAILABLE or not call_fs_tree:
        logger.warning("[evidence_gathering] v1.26 sandbox_list_directory: sandbox client not available")
        return False, []
    
    try:
        logger.info("[evidence_gathering] v1.26 sandbox_list_directory: listing %s", path)
        
        status, data, error = call_fs_tree([path], max_files=100)
        
        if status != 200 or not data:
            logger.warning(
                "[evidence_gathering] v1.26 sandbox_list_directory: failed for %s (status=%s, error=%s)",
                path, status, error
            )
            return False, []
        
        files = data.get("files", [])
        
        result = []
        for f in files:
            if isinstance(f, dict):
                result.append({
                    "path": f.get("path", ""),
                    "name": os.path.basename(f.get("path", "")),
                    "size": f.get("size"),
                    "is_dir": f.get("is_dir", False),
                })
            else:
                result.append({
                    "path": str(f),
                    "name": os.path.basename(str(f)),
                    "size": None,
                    "is_dir": False,
                })
        
        logger.info(
            "[evidence_gathering] v1.26 sandbox_list_directory: found %d items in %s",
            len(result), path
        )
        return True, result
        
    except Exception as e:
        logger.warning(
            "[evidence_gathering] v1.26 sandbox_list_directory: exception listing %s: %s",
            path, e
        )
        return False, []


# =============================================================================
# PATH RESOLUTION FUNCTIONS (v1.27 - MULTI-TARGET AWARE)
# =============================================================================

def resolve_path_enhanced(
    user_reference: str,
    anchor: Optional[str] = None,
    subfolder: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    v1.27: Enhanced path resolution using SANDBOX filesystem.
    
    Now supports drive-letter anchors (D:, C:, etc.).
    
    Args:
        user_reference: What user said ("test.txt", "Test folder")
        anchor: Detected anchor point ("desktop", "documents", "D:", "C:", etc.)
        subfolder: Detected subfolder name ("Test", "reports", etc.)
        
    Returns:
        Tuple of (resolved_path, resolution_method)
    """
    if not user_reference:
        return None, "empty_reference"
    
    ref_normalized = user_reference.strip().replace('/', '\\')
    
    logger.info(
        "[evidence_gathering] v1.27 resolve_path_enhanced: ref='%s', anchor='%s', subfolder='%s'",
        user_reference, anchor, subfolder
    )
    
    # Check if already absolute path
    if len(ref_normalized) > 2 and ref_normalized[1] == ':':
        for forbidden in EVIDENCE_FORBIDDEN_PATHS:
            if ref_normalized.lower().startswith(forbidden.lower()):
                logger.warning(
                    "[evidence_gathering] v1.27 resolve_path_enhanced: FORBIDDEN path: %s",
                    ref_normalized
                )
                return None, "forbidden_path"
        
        exists, file_info = sandbox_path_exists(ref_normalized)
        if exists:
            actual_path = file_info.get("actual_path", ref_normalized) if file_info else ref_normalized
            return actual_path, "direct_absolute"
        return None, "absolute_not_found"
    
    # Check if reference already has an extension
    has_extension = '.' in ref_normalized and ref_normalized.rsplit('.', 1)[-1].lower() in [
        'txt', 'md', 'py', 'json', 'yaml', 'yml', 'js', 'ts', 'html', 'css'
    ]
    
    # Generate name variants to try (original + extensions if no extension)
    name_variants = [ref_normalized]
    if not has_extension:
        for ext in COMMON_FILE_EXTENSIONS:
            name_variants.append(ref_normalized + ext)
    
    # v1.27: Check if anchor is a drive letter
    if anchor and len(anchor) == 2 and anchor[1] == ':':
        # Drive-letter anchor (D:, C:, etc.)
        drive_letter = anchor.upper()
        base_paths = [f"{drive_letter}\\"]
        
        logger.info("[evidence_gathering] v1.27 Using drive-letter anchor: %s", drive_letter)
        
        for base_path in base_paths:
            if subfolder:
                subfolder_path = os.path.join(base_path, subfolder)
                subfolder_exists, subfolder_info = sandbox_path_exists(subfolder_path)
                
                if subfolder_exists:
                    actual_subfolder_path = subfolder_info.get("actual_path", subfolder_path) if subfolder_info else subfolder_path
                    
                    for variant in name_variants:
                        candidate = os.path.join(actual_subfolder_path, variant)
                        exists, file_info = sandbox_path_exists(candidate)
                        if exists:
                            actual_path = file_info.get("actual_path", candidate) if file_info else candidate
                            method = f"drive_{drive_letter}_subfolder"
                            if variant != ref_normalized:
                                method += "_ext_fallback"
                            logger.info("[evidence_gathering] v1.27 FOUND via %s: %s", method, actual_path)
                            return actual_path, method
            
            # Try directly in drive root
            for variant in name_variants:
                candidate = os.path.join(base_path, variant)
                exists, file_info = sandbox_path_exists(candidate)
                if exists:
                    actual_path = file_info.get("actual_path", candidate) if file_info else candidate
                    method = f"drive_{drive_letter}"
                    if variant != ref_normalized:
                        method += "_ext_fallback"
                    logger.info("[evidence_gathering] v1.27 FOUND via %s: %s", method, actual_path)
                    return actual_path, method
        
        logger.warning("[evidence_gathering] v1.27 NOT FOUND on drive %s: %s", drive_letter, ref_normalized)
        return None, f"drive_{drive_letter}_not_found"
    
    # Standard anchor-based resolution (desktop, documents, etc.)
    if anchor:
        anchor_lower = anchor.lower()
        anchor_paths = ANCHOR_RESOLUTION_MAP.get(anchor_lower, [])
        
        for base_path in anchor_paths:
            if subfolder:
                # v1.26: First check if subfolder exists
                subfolder_path = os.path.join(base_path, subfolder)
                subfolder_exists, subfolder_info = sandbox_path_exists(subfolder_path)
                
                if subfolder_exists:
                    # v1.26.1: Use actual path from filesystem if available
                    actual_subfolder_path = subfolder_info.get("actual_path", subfolder_path) if subfolder_info else subfolder_path
                    
                    logger.info(
                        "[evidence_gathering] v1.26.1 Subfolder exists: %s (actual: %s)",
                        subfolder_path, actual_subfolder_path
                    )
                    
                    # Try with subfolder - look for file inside
                    for variant in name_variants:
                        candidate = os.path.join(actual_subfolder_path, variant)
                        exists, file_info = sandbox_path_exists(candidate)
                        if exists:
                            # v1.26.1: Use actual path from filesystem
                            actual_path = file_info.get("actual_path", candidate) if file_info else candidate
                            method = f"anchor_{anchor_lower}_subfolder"
                            if variant != ref_normalized:
                                method += "_ext_fallback"
                            logger.info(
                                "[evidence_gathering] v1.26.1 FOUND via %s: %s (actual: %s)",
                                method, candidate, actual_path
                            )
                            return actual_path, method
                    
                    # Also try the subfolder itself as a direct match (if looking for folder)
                    return actual_subfolder_path, f"anchor_{anchor_lower}_subfolder_direct"
                else:
                    logger.info(
                        "[evidence_gathering] v1.26 Subfolder NOT exists: %s",
                        subfolder_path
                    )
            
            # Try without subfolder
            for variant in name_variants:
                candidate = os.path.join(base_path, variant)
                exists, _ = sandbox_path_exists(candidate)
                if exists:
                    method = f"anchor_{anchor_lower}"
                    if variant != ref_normalized:
                        method += "_ext_fallback"
                    logger.info(
                        "[evidence_gathering] v1.26 FOUND via %s: %s",
                        method, candidate
                    )
                    return candidate, method
    
    # Sandbox roots fallback
    for root in EVIDENCE_ALLOWED_ROOTS:
        for variant in name_variants:
            candidate = os.path.join(root, variant)
            exists, _ = sandbox_path_exists(candidate)
            if exists:
                method = "sandbox_root"
                if variant != ref_normalized:
                    method += "_ext_fallback"
                logger.info(
                    "[evidence_gathering] v1.26 FOUND via %s: %s",
                    method, candidate
                )
                return candidate, method
    
    logger.warning(
        "[evidence_gathering] v1.27 Path NOT FOUND: ref='%s', anchor='%s', subfolder='%s' (tried %d variants)",
        user_reference, anchor, subfolder, len(name_variants)
    )
    return None, "not_found"


def extract_path_references(text: str) -> List[str]:
    """
    v1.26: Extract path references from user text.
    
    Detects patterns like:
    - "file called test.txt"
    - "file called Test" (no extension)
    - "Test folder"
    - "D:\\Orb\\file.py"
    - "Desktop/Test/test.txt"
    """
    if not text:
        return []
    
    references = []
    
    def is_valid_reference(match: str) -> bool:
        """Check if a match is a valid file/folder reference."""
        if not match or len(match) < 2:
            return False
        if match.lower() in PATH_REFERENCE_STOPWORDS:
            return False
        return True
    
    # File mentions - "file called X" or "file named X"
    file_called_pattern = r'file\s+(?:called|named)\s+["\']?([\w\.\-]+)["\']?'
    matches = re.findall(file_called_pattern, text, re.IGNORECASE)
    for match in matches:
        if is_valid_reference(match) and match not in references:
            references.append(match)
    
    # Standalone filename with extension
    extension_pattern = r'["\']?([\w\-]+\.(?:txt|md|py|json|yaml|yml|js|ts|html|css))["\']?'
    matches = re.findall(extension_pattern, text, re.IGNORECASE)
    for match in matches:
        if is_valid_reference(match) and match not in references:
            references.append(match)
    
    # Folder mentions - "folder called X" or "X folder"
    folder_called_pattern = r'folder\s+(?:called|named)\s+["\']?([\w\-]+)["\']?'
    matches = re.findall(folder_called_pattern, text, re.IGNORECASE)
    for match in matches:
        if is_valid_reference(match) and match not in references:
            references.append(match)
    
    # "in the X folder" pattern
    in_folder_pattern = r'in\s+(?:the\s+)?["\']?([\w\-]+)["\']?\s+folder'
    matches = re.findall(in_folder_pattern, text, re.IGNORECASE)
    for match in matches:
        if is_valid_reference(match) and match not in references:
            references.append(match)
    
    # Explicit paths
    path_patterns = [
        r'([A-Za-z]:\\[\w\\\.\-]+)',
        r'(?:desktop|documents)[/\\]([\w/\\\.\-]+)',
    ]
    
    for pattern in path_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if is_valid_reference(match) and match not in references:
                references.append(match)
    
    logger.info(
        "[evidence_gathering] v1.26 extract_path_references: Found %d references: %s",
        len(references), references[:5]
    )
    
    return references


def detect_file_structure(content: str, filename: Optional[str] = None) -> Optional[str]:
    """
    v1.26: Detect file structure/format from content and filename.
    
    Returns: qa_format, python, javascript, json, markdown, plain_text, code
    """
    if not content:
        return None
    
    content_preview = content[:500] if len(content) > 500 else content
    
    # Extension-based detection
    if filename:
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        ext_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'md': 'markdown',
            'html': 'html',
            'css': 'css',
            'txt': 'plain_text',
        }
        if ext in ext_map:
            return ext_map[ext]
    
    # Content-based detection
    if re.search(r'Question\s*\d+', content_preview, re.IGNORECASE):
        return "qa_format"
    
    if 'def ' in content_preview or 'import ' in content_preview or 'class ' in content_preview:
        return "python"
    
    if content_preview.strip().startswith('{') or content_preview.strip().startswith('['):
        return "json"
    
    if content_preview.startswith('#') or '## ' in content_preview:
        return "markdown"
    
    return "plain_text"


# =============================================================================
# MAIN EVIDENCE GATHERING (v1.27 - MULTI-TARGET SUPPORT)
# =============================================================================

def resolve_and_validate_path(
    reference: str,
    anchor: Optional[str] = None,
    subfolder: Optional[str] = None,
    read_full_content: bool = False,
) -> FileEvidence:
    """
    v1.27: Resolve a single path reference and validate via SANDBOX filesystem.
    
    Args:
        reference: File reference to resolve
        anchor: Location anchor (desktop, documents, D:, etc.)
        subfolder: Subfolder within anchor
        read_full_content: If True, store full content (for multi-target reads)
    """
    resolved_path, resolution_method = resolve_path_enhanced(reference, anchor, subfolder)
    
    if resolved_path:
        source = FilesystemEvidenceSource.SANDBOX_FS
    elif resolution_method == "forbidden_path":
        source = FilesystemEvidenceSource.NOT_FOUND
    else:
        source = FilesystemEvidenceSource.NOT_FOUND
    
    evidence = FileEvidence(
        original_reference=reference,
        resolved_path=resolved_path,
        source=source,
        exists=False,
        readable=False,
        writable=False,
        metadata={"resolution_method": resolution_method, "anchor": anchor, "subfolder": subfolder},
    )
    
    if resolved_path:
        # v1.26: Use sandbox client to check existence and read content
        exists, file_info = sandbox_path_exists(resolved_path)
        evidence.exists = exists
        
        if exists and file_info:
            evidence.size_bytes = file_info.get("size")
            evidence.mtime = file_info.get("mtime")
            
            # Only read if it's a file (not directory)
            is_dir = file_info.get("is_dir", False)
            if not is_dir:
                # v1.27: Read more content for multi-target reads
                max_chars = 50000 if read_full_content else 8000
                success, content = sandbox_read_file(resolved_path, max_chars=max_chars)
                if success and content:
                    evidence.readable = True
                    evidence.content_preview = content[:500] if len(content) > 500 else content
                    if read_full_content:
                        evidence.full_content = content
                    evidence.detected_structure = detect_file_structure(
                        content,
                        os.path.basename(resolved_path)
                    )
                    evidence.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    logger.info(
        "[evidence_gathering] v1.27 resolve_and_validate_path: ref='%s' -> path='%s', exists=%s, readable=%s",
        reference, resolved_path, evidence.exists, evidence.readable
    )
    
    return evidence


def gather_multi_target_evidence(
    combined_text: str,
    rag_hints: Optional[List[dict]] = None,
) -> EvidencePackage:
    """
    v1.27: Gather evidence for multiple file targets from different locations.
    
    Handles patterns like "read test on desktop and test2 on D drive".
    Each target is resolved independently with its own anchor.
    
    Args:
        combined_text: User request text
        rag_hints: Optional RAG hints
        
    Returns:
        EvidencePackage with all targets resolved
    """
    logger.info("[evidence_gathering] v1.27 gather_multi_target_evidence: starting")
    
    package = EvidencePackage(
        task_type="unresolved",
        ground_truth_timestamp=datetime.now(timezone.utc).isoformat(),
        is_multi_target=True,
    )
    
    if rag_hints:
        package.rag_hints_used = rag_hints
    
    # Check if multi-target extraction is available
    if not _MULTI_TARGET_AVAILABLE or not extract_file_targets:
        package.validation_errors.append("Multi-target extraction not available")
        logger.error("[evidence_gathering] v1.27 extract_file_targets not available")
        return package
    
    # Extract file targets with their individual anchors
    targets = extract_file_targets(combined_text)
    
    if not targets:
        package.validation_errors.append("No file targets found in text")
        logger.warning("[evidence_gathering] v1.27 No targets extracted from: %s", combined_text[:100])
        return package
    
    logger.info("[evidence_gathering] v1.27 Extracted %d targets: %s", len(targets), targets)
    
    # Resolve each target independently
    for target in targets:
        name = target.get("name", "")
        anchor = target.get("anchor")
        subfolder = target.get("subfolder")
        explicit_path = target.get("explicit_path")
        
        if explicit_path:
            # If we have an explicit path, use it directly
            evidence = resolve_and_validate_path(
                explicit_path,
                anchor=None,
                subfolder=None,
                read_full_content=True,
            )
        else:
            # Resolve using anchor
            evidence = resolve_and_validate_path(
                name,
                anchor=anchor,
                subfolder=subfolder,
                read_full_content=True,
            )
        
        # Track the original target info
        evidence.metadata["target_info"] = target
        package.target_files.append(evidence)
        
        # Track results for summary
        package.multi_target_results.append({
            "name": name,
            "anchor": anchor,
            "explicit_path": explicit_path,
            "resolved_path": evidence.resolved_path,
            "found": evidence.exists and evidence.readable,
            "error": None if evidence.exists else f"Not found: {name}",
        })
    
    # Determine task type
    valid_count = sum(1 for fe in package.target_files if fe.exists and fe.readable)
    total_count = len(package.target_files)
    
    if valid_count == 0:
        package.task_type = "unresolved"
        package.validation_errors.append(
            f"No valid files found. Tried {total_count} targets."
        )
    elif valid_count < total_count:
        package.task_type = "multi_target_read"
        missing = [t["name"] for t in package.multi_target_results if not t["found"]]
        package.warnings.append(
            f"Found {valid_count}/{total_count} files. Missing: {missing}"
        )
    else:
        package.task_type = "multi_target_read"
    
    logger.info(
        "[evidence_gathering] v1.27 gather_multi_target_evidence COMPLETE: %s",
        package.to_summary()
    )
    
    return package


# =============================================================================
# SYSTEM-WIDE SCAN (v1.31 - Level 2.5+)
# =============================================================================

# Roots to scan for system-wide file searches
USER_SCAN_ROOTS = [
    # User Desktop locations
    "C:\\Users\\dizzi\\OneDrive\\Desktop",
    "C:\\Users\\dizzi\\Desktop",
    # User Documents
    "C:\\Users\\dizzi\\OneDrive\\Documents",
    "C:\\Users\\dizzi\\Documents",
    # User Downloads
    "C:\\Users\\dizzi\\Downloads",
    # Common project directories (v1.32)
    "D:\\Orb",
    "D:\\orb-desktop",
    # Drive roots (shallow scan)
    "D:\\",
    "C:\\",
]


def scan_root_for_file(root: str, filename: str, max_depth: int = 2) -> Optional[str]:
    """
    v1.31: Scan a root directory for a file by name.
    
    Scans shallowly (max_depth levels) to avoid scanning entire drives.
    Tries with and without common extensions.
    
    Args:
        root: Root directory to scan
        filename: File name to search for (with or without extension)
        max_depth: Maximum directory depth to scan (default 2)
        
    Returns:
        Full path if found, None otherwise
    """
    if not _SANDBOX_CLIENT_AVAILABLE or not call_fs_tree:
        logger.warning("[evidence_gathering] v1.31 scan_root_for_file: sandbox client not available")
        return None
    
    logger.info(
        "[evidence_gathering] v1.31 scan_root_for_file: searching for '%s' in '%s' (depth=%d)",
        filename, root, max_depth
    )
    
    # Build list of name variants to look for
    filename_lower = filename.lower()
    has_extension = '.' in filename and len(filename.split('.')[-1]) <= 4
    
    variants = [filename_lower]
    if not has_extension:
        for ext in COMMON_FILE_EXTENSIONS:
            variants.append(filename_lower + ext)
    
    try:
        # List files in root
        status, data, error = call_fs_tree([root], max_files=200)
        
        if status != 200 or not data:
            logger.info(
                "[evidence_gathering] v1.31 scan_root_for_file: failed to list %s (status=%s, error=%s)",
                root, status, error
            )
            return None
        
        files = data.get("files", [])
        
        for f in files:
            f_path = f.get("path", "") if isinstance(f, dict) else str(f)
            f_name = os.path.basename(f_path).lower()
            is_dir = f.get("is_dir", False) if isinstance(f, dict) else False
            
            # Check if this file matches
            if not is_dir and f_name in variants:
                logger.info(
                    "[evidence_gathering] v1.31 scan_root_for_file: FOUND '%s' at '%s'",
                    filename, f_path
                )
                return f_path
            
            # Recurse into subdirectories if we haven't hit max depth
            if is_dir and max_depth > 1:
                # Don't recurse into system directories
                dir_name = os.path.basename(f_path).lower()
                skip_dirs = {'windows', 'program files', 'program files (x86)', 'appdata', 
                             '$recycle.bin', 'system volume information', 'programdata',
                             '.git', 'node_modules', '__pycache__', '.venv'}
                if dir_name in skip_dirs:
                    continue
                
                found = scan_root_for_file(f_path, filename, max_depth - 1)
                if found:
                    return found
        
        return None
        
    except Exception as e:
        logger.warning(
            "[evidence_gathering] v1.31 scan_root_for_file: exception scanning %s: %s",
            root, e
        )
        return None


def gather_system_wide_scan_evidence(
    combined_text: str,
    rag_hints: Optional[List[dict]] = None,
) -> EvidencePackage:
    """
    v1.31: Gather evidence by scanning the entire system for named files.
    
    Handles patterns like "Find all files called test1, test2, test3, test4 on my system".
    Scans USER_SCAN_ROOTS for each target file.
    
    Args:
        combined_text: User request text  
        rag_hints: Optional RAG hints
        
    Returns:
        EvidencePackage with files found across the system
    """
    logger.info("[evidence_gathering] v1.31 gather_system_wide_scan_evidence: starting")
    
    package = EvidencePackage(
        task_type="unresolved",
        ground_truth_timestamp=datetime.now(timezone.utc).isoformat(),
        is_multi_target=True,
    )
    
    if rag_hints:
        package.rag_hints_used = rag_hints
    
    # Check if system scan extraction is available
    if not _SYSTEM_SCAN_AVAILABLE or not extract_scan_file_names:
        package.validation_errors.append("System scan extraction not available")
        logger.error("[evidence_gathering] v1.31 extract_scan_file_names not available")
        return package
    
    # Extract file names to search for
    file_names = extract_scan_file_names(combined_text)
    
    if not file_names:
        package.validation_errors.append("No file names found to scan for")
        logger.warning("[evidence_gathering] v1.31 No file names extracted from: %s", combined_text[:100])
        return package
    
    logger.info("[evidence_gathering] v1.31 System scan for %d files: %s", len(file_names), file_names)
    
    # Scan for each file across all roots
    for filename in file_names:
        found_path = None
        searched_roots = []
        
        # Search in each root
        for root in USER_SCAN_ROOTS:
            searched_roots.append(root)
            found_path = scan_root_for_file(root, filename, max_depth=2)
            if found_path:
                break
        
        if found_path:
            # Found! Now gather full evidence
            evidence = resolve_and_validate_path(
                found_path,
                anchor=None,
                subfolder=None,
                read_full_content=True,
            )
            evidence.metadata["target_info"] = {
                "name": filename,
                "scan_all_roots": True,
                "searched_roots": searched_roots,
            }
            evidence.metadata["discovery_method"] = "system_wide_scan"
            package.target_files.append(evidence)
            
            package.multi_target_results.append({
                "name": filename,
                "anchor": None,
                "explicit_path": None,
                "resolved_path": evidence.resolved_path,
                "found": evidence.exists and evidence.readable,
                "searched_roots": searched_roots,
                "error": None,
            })
            
            logger.info(
                "[evidence_gathering] v1.31 FOUND '%s' via system scan at: %s",
                filename, found_path
            )
        else:
            # Not found anywhere
            evidence = FileEvidence(
                original_reference=filename,
                resolved_path=None,
                source=FilesystemEvidenceSource.NOT_FOUND,
                exists=False,
                readable=False,
                metadata={
                    "target_info": {
                        "name": filename,
                        "scan_all_roots": True,
                        "searched_roots": searched_roots,
                    },
                    "discovery_method": "system_wide_scan",
                },
            )
            package.target_files.append(evidence)
            
            package.multi_target_results.append({
                "name": filename,
                "anchor": None,
                "explicit_path": None,
                "resolved_path": None,
                "found": False,
                "searched_roots": searched_roots,
                "error": f"Not found in {len(searched_roots)} locations",
            })
            
            logger.warning(
                "[evidence_gathering] v1.31 NOT FOUND '%s' after scanning %d roots",
                filename, len(searched_roots)
            )
    
    # Determine task type
    valid_count = sum(1 for fe in package.target_files if fe.exists and fe.readable)
    total_count = len(package.target_files)
    
    if valid_count == 0:
        package.task_type = "unresolved"
        package.validation_errors.append(
            f"No files found. Searched for {total_count} files across {len(USER_SCAN_ROOTS)} locations."
        )
    elif valid_count < total_count:
        package.task_type = "multi_target_read"
        missing = [t["name"] for t in package.multi_target_results if not t["found"]]
        package.warnings.append(
            f"Found {valid_count}/{total_count} files. Missing: {missing}"
        )
    else:
        package.task_type = "multi_target_read"
    
    logger.info(
        "[evidence_gathering] v1.31 gather_system_wide_scan_evidence COMPLETE: %s",
        package.to_summary()
    )
    
    return package


def gather_filesystem_evidence(
    combined_text: str,
    anchor: Optional[str] = None,
    subfolder: Optional[str] = None,
    rag_hints: Optional[List[dict]] = None,
) -> EvidencePackage:
    """
    v1.31: Main entry point for Evidence-First filesystem validation.
    
    Gathers filesystem evidence BEFORE any LLM calls using SANDBOX filesystem.
    Now automatically detects and handles:
    - v1.31: System-wide scan requests ("find files on my system")
    - v1.27: Multi-target requests with explicit anchors
    - Standard single-target requests
    """
    logger.info(
        "[evidence_gathering] v1.31 gather_filesystem_evidence: anchor='%s', subfolder='%s', rag_hints=%d",
        anchor, subfolder, len(rag_hints) if rag_hints else 0
    )
    
    # v1.31: Check if this is a system-wide scan request FIRST
    if _SYSTEM_SCAN_AVAILABLE and is_system_wide_scan_request and is_system_wide_scan_request(combined_text):
        logger.info("[evidence_gathering] v1.31 Detected SYSTEM-WIDE SCAN request, delegating...")
        return gather_system_wide_scan_evidence(combined_text, rag_hints)
    
    # v1.27: Check if this is a multi-target request (with explicit anchors)
    if _MULTI_TARGET_AVAILABLE and is_multi_target_request and is_multi_target_request(combined_text):
        logger.info("[evidence_gathering] v1.27 Detected MULTI-TARGET request, delegating...")
        return gather_multi_target_evidence(combined_text, rag_hints)
    
    # Standard single-target handling
    package = EvidencePackage(
        task_type="unresolved",
        ground_truth_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    
    if rag_hints:
        package.rag_hints_used = rag_hints
    
    # Extract path references from text
    references = extract_path_references(combined_text)
    
    if not references and not anchor:
        package.validation_errors.append("No path references found in text")
        logger.warning("[evidence_gathering] v1.27 No path references found")
        return package
    
    # v1.26: If we have anchor + subfolder + file reference, prioritize that
    # E.g., "On the desktop, there is a folder called Test. In the test folder, there is a file called Test"
    # Should resolve to: Desktop/Test/Test.txt
    
    if anchor and subfolder:
        # Look for file references that might be inside the subfolder
        for ref in references:
            # Skip the subfolder name itself
            if ref.lower() == subfolder.lower():
                continue
            
            evidence = resolve_and_validate_path(ref, anchor, subfolder)
            package.target_files.append(evidence)
            
            if evidence.exists and evidence.readable:
                package.task_type = "file_operation"
                logger.info(
                    "[evidence_gathering] v1.27 Found valid target in subfolder: %s",
                    evidence.resolved_path
                )
    
    # If no valid targets found yet, try each reference without subfolder constraint
    if not package.has_valid_targets():
        for ref in references:
            # Skip if we already tried this reference
            if any(fe.original_reference == ref for fe in package.target_files):
                continue
            
            evidence = resolve_and_validate_path(ref, anchor, None)
            package.target_files.append(evidence)
    
    # Determine task type based on what we found
    if package.has_valid_targets():
        primary = package.get_primary_target()
        if primary and primary.detected_structure == "qa_format":
            package.task_type = "qa_file_task"
        elif primary and primary.detected_structure in ("python", "javascript", "typescript"):
            package.task_type = "code_analysis"
        else:
            package.task_type = "file_operation"
    else:
        package.task_type = "unresolved"
        package.validation_errors.append(
            f"No valid target files found. References checked: {references}"
        )
    
    logger.info(
        "[evidence_gathering] v1.27 gather_filesystem_evidence COMPLETE: %s",
        package.to_summary()
    )
    
    return package


def format_evidence_for_prompt(package: EvidencePackage) -> str:
    """
    v1.27: Format evidence package for injection into LLM prompt.
    
    Now handles multi-target read packages with full content.
    """
    lines = [
        "## Filesystem Evidence (Ground Truth)",
        f"Timestamp: {package.ground_truth_timestamp}",
        f"Task Type: {package.task_type}",
    ]
    
    if package.is_multi_target:
        lines.append("Mode: MULTI-TARGET READ")
    
    lines.append("")
    lines.append("### Target Files")
    
    if package.target_files:
        for fe in package.target_files:
            lines.append(fe.to_evidence_line())
            
            # v1.27: Include full content for multi-target reads
            if package.is_multi_target and fe.full_content:
                lines.append("")
                lines.append(f"**Content of {fe.resolved_path}:**")
                lines.append("```")
                lines.append(fe.full_content)
                lines.append("```")
                lines.append("")
    else:
        lines.append("  (No files resolved)")
    
    if package.validation_errors:
        lines.append("")
        lines.append("### Validation Errors")
        for err in package.validation_errors:
            lines.append(f"  - ⚠️ {err}")
    
    if package.warnings:
        lines.append("")
        lines.append("### Warnings")
        for warn in package.warnings:
            lines.append(f"  - {warn}")
    
    return "\n".join(lines)


async def format_multi_target_reply(
    package: EvidencePackage,
    provider_id: str = "openai",
    model_id: str = "gpt-5-mini",
    llm_call_func = None,
    user_request: Optional[str] = None,
) -> str:
    """
    v1.30: Format a SYNTHESIZED reply for multi-target read operations.
    
    CRITICAL FIX v1.30: Now generates ONE SYNTHESIZED reply from ALL files!
    Previous versions generated separate replies per file, missing the conceptual link.
    
    Example:
        File 1: "My name is Astra"
        File 2: "I'm going to be an assistant"
        File 3: "I'm going to help you every day"
        File 4: "Where shall we begin?"
        
        Synthesized output: "The files together describe an AI assistant named Astra
                            that will help the user daily and is ready to start."
    
    Args:
        package: EvidencePackage with multi-target files
        provider_id: LLM provider for reply generation
        model_id: LLM model for reply generation
        llm_call_func: Optional LLM call function
        user_request: The user's original request for context
    """
    if not package.is_multi_target:
        return ""
    
    valid_targets = package.get_all_valid_targets()
    
    if not valid_targets:
        lines = ["I couldn't find any of the requested files."]
        for result in package.multi_target_results:
            lines.append(f"- {result['name']}: Not found")
        return "\n".join(lines)
    
    # Build list of files for synthesis
    total = len(package.target_files)
    found = len(valid_targets)
    
    # v1.30: Collect all file contents for synthesis
    file_contents = []
    for fe in valid_targets:
        content = fe.full_content if fe.full_content else fe.content_preview
        if content:
            file_contents.append({
                'path': fe.resolved_path,
                'name': os.path.basename(fe.resolved_path) if fe.resolved_path else fe.original_reference,
                'content': content,
                'content_type': fe.detected_structure,
            })
    
    if not file_contents:
        return "Found files but all were empty or unreadable."
    
    # v1.30: Generate SYNTHESIZED reply using qa_processing
    try:
        from .qa_processing import generate_synthesized_reply_from_files
        _SYNTHESIS_AVAILABLE = True
    except ImportError:
        logger.warning("[evidence_gathering] v1.30 generate_synthesized_reply_from_files not available")
        _SYNTHESIS_AVAILABLE = False
    
    # Build header with file status
    header_lines = []
    if found < total:
        missing = [r["name"] for r in package.multi_target_results if not r["found"]]
        header_lines.append(f"Found {found} of {total} files. Missing: {', '.join(missing)}")
        header_lines.append("")
    
    # v1.30: CRITICAL - Call synthesis function for COMBINED understanding
    if _SYNTHESIS_AVAILABLE:
        logger.info(
            "[evidence_gathering] v1.30 MULTI-FILE SYNTHESIS: Calling generate_synthesized_reply_from_files for %d files",
            len(file_contents)
        )
        try:
            synthesis = await generate_synthesized_reply_from_files(
                file_contents=file_contents,
                provider_id=provider_id,
                model_id=model_id,
                llm_call_func=llm_call_func,
                user_request=user_request,
            )
            if synthesis:
                header_lines.append("**Synthesized Understanding:**")
                header_lines.append(synthesis)
                logger.info(
                    "[evidence_gathering] v1.30 MULTI-FILE SYNTHESIS SUCCESS: %d chars",
                    len(synthesis)
                )
            else:
                header_lines.append("(Synthesis returned empty - see file contents below)")
                logger.warning("[evidence_gathering] v1.30 Synthesis returned empty")
        except Exception as e:
            logger.error(
                "[evidence_gathering] v1.30 MULTI-FILE SYNTHESIS FAILED: %s", e
            )
            header_lines.append(f"(Synthesis error: {str(e)[:100]})")
    else:
        # Fallback: just list file contents
        logger.warning("[evidence_gathering] v1.30 Synthesis unavailable, using fallback")
        header_lines.append("**Files read:**")
        for i, fc in enumerate(file_contents, 1):
            name = fc.get('name', f'File {i}')
            content = fc['content'][:200] if len(fc['content']) > 200 else fc['content']
            header_lines.append(f"")
            header_lines.append(f"File {i} ({name}):")
            header_lines.append(content)
    
    return "\n".join(header_lines)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "FilesystemEvidenceSource",
    "FileEvidence",
    "EvidencePackage",
    # Sandbox helpers (v1.26)
    "sandbox_path_exists",
    "sandbox_read_file",
    "sandbox_list_directory",
    # Path resolution
    "resolve_path_enhanced",
    "extract_path_references",
    "detect_file_structure",
    # Main entry points
    "resolve_and_validate_path",
    "gather_filesystem_evidence",
    "gather_multi_target_evidence",
    "gather_system_wide_scan_evidence",  # v1.31
    "scan_root_for_file",  # v1.31
    "format_evidence_for_prompt",
    "format_multi_target_reply",
    # Constants
    "ANCHOR_RESOLUTION_MAP",
    "COMMON_FILE_EXTENSIONS",
    "PATH_REFERENCE_STOPWORDS",
    "USER_SCAN_ROOTS",  # v1.31
]
