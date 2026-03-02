# FILE: app/pot_spec/grounded/evidence_gathering.py
"""
Evidence-First Filesystem Validation (v2.1)

Core principle: Evidence gathered BEFORE LLM call, not discovered by LLM.

Access control (non-negotiable):
- SpecGate has RAG hints ✅
- SpecGate has Live FS read ✅ (via sandbox client)
- SpecGate has Live FS write ❌ (NEVER)

Version Notes:
-------------
v2.1 (2026-02-06): HARD READ FAIL + Evidence Gap Detection
    - sandbox_read_file() now logs [ERROR] when file exists but BOTH
      sandbox and direct read return no content (not silently swallowed)
    - Added EvidencePackage.check_critical_evidence_gaps() method
    - Detects exists=True + readable=False (hard gap)
    - Detects exists=True + readable=True but no content (silent failure)
    - Surfaces gaps as validation_errors on the EvidencePackage
    - Pipeline can check validation_errors before proceeding with LLM calls
v1.34 (2026-01-30): CRITICAL FIX - Exclude CREATE targets from ALL code paths
    - v1.33 only excluded CREATE targets from system_wide_scan
    - But requests with anchors ("desktop") went through gather_filesystem_evidence()
    - Now gather_filesystem_evidence() ALSO excludes CREATE targets
    - Filters create_names from extract_path_references() results
    - Fixes: "create a new file on desktop called reply" no longer searches for reply.txt
    - Stores create_targets in package.metadata for Critical Pipeline / Implementer
v1.33 (2026-01-30): CRITICAL FIX - Exclude CREATE targets from system scan
    - Added import for extract_create_targets from sandbox_discovery
    - Modified gather_system_wide_scan_evidence() to exclude CREATE targets
    - CREATE targets (files to be created) are no longer searched for
    - Fixes "reply: Not found" error when user wants to CREATE a reply file
    - Stores create_targets in package.metadata for downstream use
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
from app.pot_spec.grounded._evidence_gathering_utils_8 import EVIDENCE_ALLOWED_ROOTS, EVIDENCE_FORBIDDEN_PATHS, EVIDENCE_GATHERING_BUILD_ID, _resolve_via_index_json, format_evidence_for_prompt, format_multi_target_reply, read_interface_signatures, sandbox_list_directory
from app.pot_spec.grounded._evidence_gathering_utils_9 import ANCHOR_RESOLUTION_MAP, COMMON_FILE_EXTENSIONS, PATH_REFERENCE_STOPWORDS, USER_SCAN_ROOTS, detect_file_structure, extract_path_references, sandbox_read_file, scan_root_for_file
from app.pot_spec.grounded._evidence_gathering_utils_10 import gather_filesystem_evidence, resolve_path_enhanced
from app.pot_spec.grounded._evidence_gathering_utils_11 import gather_multi_target_evidence, gather_system_wide_scan_evidence, sandbox_path_exists

logger = logging.getLogger(__name__)

# =============================================================================
# v1.34 BUILD VERIFICATION
# =============================================================================
print(f"[EVIDENCE_GATHERING_LOADED] BUILD_ID={EVIDENCE_GATHERING_BUILD_ID}")
logger.info(f"[evidence_gathering] Module loaded: BUILD_ID={EVIDENCE_GATHERING_BUILD_ID}")


# =============================================================================
# SANDBOX CLIENT IMPORTS (v1.26)
# =============================================================================

# v3.2-fix: The old zobie sandbox_client import was never wired through
# to the utility modules, so evidence gathering always fell back to host
# os.path.  All utility modules now import directly from app.sandbox_fs.
# This block is kept only for backward compat with any code that checks
# _SANDBOX_CLIENT_AVAILABLE.
try:
    from app.sandbox_fs import sandbox_exists, sandbox_isfile, sandbox_read_text
    _SANDBOX_CLIENT_AVAILABLE = True
    logger.info("[evidence_gathering] v3.2 sandbox_fs loaded successfully")
except ImportError as e:
    _SANDBOX_CLIENT_AVAILABLE = False
    logger.error("[evidence_gathering] v3.2 CRITICAL: sandbox_fs not available: %s", e)


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
        extract_create_targets,  # v1.33: CREATE target detection
        SYSTEM_SCAN_ROOTS,
    )
    _MULTI_TARGET_AVAILABLE = True
    _SYSTEM_SCAN_AVAILABLE = True
    _CREATE_TARGET_AVAILABLE = True
    logger.info("[evidence_gathering] v1.33 Multi-target + system scan + CREATE target extraction loaded")
except ImportError as e:
    _MULTI_TARGET_AVAILABLE = False
    _SYSTEM_SCAN_AVAILABLE = False
    _CREATE_TARGET_AVAILABLE = False
    logger.warning("[evidence_gathering] v1.33 Multi-target/system scan/CREATE target extraction not available: %s", e)
    extract_file_targets = None
    is_multi_target_request = None
    is_system_wide_scan_request = None
    get_system_scan_targets = None
    extract_scan_file_names = None
    extract_create_targets = None
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

# Forbidden paths (never resolve to these)

# Known anchor locations for Desktop/Documents resolution
# v1.27: Added drive-letter anchors

# Common file extensions to try when a filename has no extension

# Stopwords that should never be extracted as file/folder names


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
    task_type: str                     # qa_file_task, code_analysis, file_operation, multi_target_read, unresolved, create_only
    target_files: List[FileEvidence] = field(default_factory=list)
    rag_hints_used: List[dict] = field(default_factory=list)
    search_queries_run: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ground_truth_timestamp: str = ""
    # v1.27: Multi-target read tracking
    is_multi_target: bool = False
    multi_target_results: List[Dict[str, Any]] = field(default_factory=list)
    # v1.34: Package-level metadata (includes create_targets for downstream use)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    
    def check_critical_evidence_gaps(self) -> List[str]:
        """v2.1: Check for files that exist but have no content (evidence gap).
        
        Returns list of error messages for files where exists=True but
        readable=False or content_preview is None. These represent
        hard evidence failures where the pipeline SHOULD NOT proceed
        with LLM calls that depend on this grounding.
        """
        gaps = []
        for fe in self.target_files:
            if fe.exists and not fe.readable:
                msg = (
                    f"EVIDENCE GAP: '{fe.original_reference}' exists at {fe.resolved_path} "
                    f"but is NOT readable. Grounding data missing for this file."
                )
                gaps.append(msg)
                print(f"[ERROR] [evidence_gathering] v2.1 {msg}")
            elif fe.exists and fe.readable and fe.content_preview is None and fe.full_content is None:
                msg = (
                    f"EVIDENCE GAP: '{fe.original_reference}' at {fe.resolved_path} "
                    f"is marked readable but has NO content. Possible silent read failure."
                )
                gaps.append(msg)
                print(f"[WARN] [evidence_gathering] v2.1 {msg}")
        if gaps:
            self.validation_errors.extend(gaps)
        return gaps


# =============================================================================
# SANDBOX FILESYSTEM HELPERS (v1.26)
# =============================================================================


# =============================================================================
# INTERFACE SIGNATURE READING (v2.2 - Segmentation support)
# =============================================================================


# =============================================================================
# PATH RESOLUTION FUNCTIONS (v1.27 - MULTI-TARGET AWARE)
# =============================================================================

# v2.2: INDEX.json cache lives in _evidence_gathering_utils_8.py
# (where _resolve_via_index_json() uses it via `global` declaration)


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
                # v1.35 (Job 9): Check pattern cache before sandbox read
                # Skip cache for full_content reads — cached content is
                # truncated to 10KB which is too short for 50KB reads.
                _ev_cached = False
                if not read_full_content:
                    try:
                        from app.pot_spec.pattern_cache import get_pattern_cache
                        _ev_cache = get_pattern_cache()
                        _ev_hit = _ev_cache.get(resolved_path)
                        if _ev_hit:
                            content = _ev_hit.content
                            success = True
                            _ev_cached = True
                            logger.debug(
                                "[evidence_gathering] v1.35 Cache HIT: %s",
                                resolved_path,
                            )
                    except Exception:
                        _ev_cached = False

                if not _ev_cached:
                    # v1.27: Read more content for multi-target reads
                    max_chars = 50000 if read_full_content else 8000
                    success, content = sandbox_read_file(resolved_path, max_chars=max_chars)

                    # v1.35: Cache the read for next time
                    if success and content and len(content) >= 50:
                        try:
                            _ev_cache = get_pattern_cache()
                            _ev_cache.put(resolved_path, content, pattern_type="evidence")
                        except Exception:
                            pass

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


# =============================================================================
# SYSTEM-WIDE SCAN (v1.31 - Level 2.5+)
# =============================================================================

# Roots to scan for system-wide file searches


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
