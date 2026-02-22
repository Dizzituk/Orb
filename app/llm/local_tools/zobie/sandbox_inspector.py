# FILE: app/llm/local_tools/zobie/sandbox_inspector.py
"""
Shared Sandbox Inspection Capability for Evidence Stages.

READ-ONLY. No writes allowed.

Used by:
- SpecGate (first full discovery)
- Critical Pipeline (re-check / fill gaps)
- Critique (verify alignment)
- Overwatcher (verify execution results)

RULES:
- NEVER import call_fs_write
- ONE canonical discovery function
- Content-aware file selection
- Progressive reads (snippet for classification, full read for winner only)

v1.0 (2026-01): Initial implementation for SpecGate sandbox discovery
v1.1 (2026-01): Desktop alias expansion (human-style resolution)
              - "Desktop" is now an alias, not a single hardcoded path
              - Tries OneDrive Desktop first (most common redirect), then normal Desktop, then sandbox
              - If "SANDBOX" explicitly mentioned in intent, sandbox roots tried first
              - Added _get_ordered_roots() for context-aware root ordering
              - Added _bounded_folder_search() for depth-limited fallback search
              - Added _identify_root_type() for reporting (host_onedrive_desktop, sandbox_desktop, etc.)
              - All discovery functions now track roots_tried for debugging
              - Returns discovery_method in results (exact, bounded_search, folder_discovery)
v1.2 (2026-02): Project root resolution (D: drive support)
              - anchor now accepts project names ("orb", "orb-desktop") and absolute paths
              - PROJECT_ROOTS dict maps known project names to D: paths
              - Dynamic D: discovery: unknown anchors checked against D: subdirectories
              - _identify_root_type() recognises project roots ("project_orb", "project_orb-desktop")
              - Unknown anchors fall back to D: root scan instead of returning empty
              - Evidence contract prompt updated to tell LLMs about project anchors
              - Subfolder paths normalised to OS separators (forward slashes converted)
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from .sandbox_client import call_fs_tree, call_fs_contents
from .content_classifier import classify_content, score_file_for_intent, ContentType
from app.llm.local_tools.zobie._sandbox_inspector_utils import D_DRIVE_ROOT, MAX_CANDIDATE_FILES, MAX_CLASSIFICATION_CHARS, MAX_FILE_CHARS, MAX_SEARCH_DEPTH, SANDBOX_DESKTOP_ROOTS, _read_snippet, file_exists_in_sandbox
from app.llm.local_tools.zobie._sandbox_inspector_utils import HOST_DESKTOP_ROOTS, HOST_DOCUMENTS_ROOTS, SANDBOX_DOCUMENTS_ROOTS, _bounded_folder_search, _discover_folder_without_hint, _select_file_from_folder, read_sandbox_file

logger = logging.getLogger(__name__)


# =============================================================================
# SAFETY LIMITS
# =============================================================================
MAX_FILES_PER_FOLDER = 50       # Don't list more than this

# Sandbox root paths (Windows Sandbox + host user)
# v1.1: Added OneDrive Desktop as primary host location
# v1.2: Added D:\ project roots for direct project directory access
# Priority order depends on whether "SANDBOX" is mentioned in intent
SANDBOX_ROOTS: Dict[str, List[str]] = {
    "desktop": [
        # Host candidates (OneDrive first - most common redirect on this machine)
        r"C:\Users\dizzi\OneDrive\Desktop",
        r"C:\Users\dizzi\Desktop",
        # Sandbox candidates
        r"C:\Users\WDAGUtilityAccount\Desktop",
    ],
    "documents": [
        # Host candidates (OneDrive first)
        r"C:\Users\dizzi\OneDrive\Documents",
        r"C:\Users\dizzi\Documents",
        # Sandbox candidates
        r"C:\Users\WDAGUtilityAccount\Documents",
    ],
}

# v1.2: Known project roots on D:\ drive
# These are resolved directly — no anchor alias needed
# The sandbox controller already allows D:\ in ALLOWED_FS_ROOTS
PROJECT_ROOTS: Dict[str, str] = {
    "orb": r"D:\Orb",
    "orb-desktop": r"D:\orb-desktop",
    "orb-electron-data": r"D:\orb-electron-data",
    "sandbox_controller": r"D:\sandbox_controller",
}

# v1.2: D:\ drive root for dynamic project discovery

# Separate lists for intelligent ordering

# Candidate file extensions for discovery
CANDIDATE_EXTENSIONS = [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".log", ".csv"]

# Maximum search depth for bounded fallback


# =============================================================================
# ROOT ORDERING (v1.1 - Context-Aware)
# =============================================================================

def _get_ordered_roots(anchor: str, job_intent: Optional[str] = None) -> List[str]:
    """
    Get roots in priority order based on context.
    
    v1.1: If "SANDBOX" is explicitly mentioned in intent, try sandbox roots first.
    Otherwise, try host OneDrive roots first (most likely real location).
    
    v1.2: Added project root resolution.
    - If anchor matches a known project name ("orb", "orb-desktop", etc.),
      return the project's D: path directly.
    - If anchor is a full absolute path (e.g. "D:\\orb-desktop"), return it directly.
    - If anchor doesn't match anything known, try to find it on D: dynamically.
    
    Args:
        anchor: "desktop", "documents", project name, or absolute path
        job_intent: User's intent text (optional)
        
    Returns:
        List of root paths in priority order
    """
    anchor_lower = anchor.lower().strip().rstrip("\\/")
    intent_lower = (job_intent or "").lower()
    
    # ── v1.2: Check if anchor is a full absolute path ──
    if len(anchor) >= 2 and anchor[1] == ":":
        # It's already an absolute path like D:\orb-desktop or D:\Orb\app
        logger.info("[sandbox_inspector] v1.2: Absolute path anchor: %s", anchor)
        return [anchor]
    
    # ── v1.2: Check known project names ──
    if anchor_lower in PROJECT_ROOTS:
        project_path = PROJECT_ROOTS[anchor_lower]
        logger.info("[sandbox_inspector] v1.2: Project anchor '%s' -> %s", anchor, project_path)
        return [project_path]
    
    # ── v1.2: Check if anchor is a D:\ subdirectory name (dynamic discovery) ──
    # e.g. anchor="orb-desktop" might not be in PROJECT_ROOTS but exists on D:\
    candidate_d_path = os.path.join(D_DRIVE_ROOT, anchor)
    if os.path.isdir(candidate_d_path):
        logger.info("[sandbox_inspector] v1.2: Dynamic D:\\ discovery: %s -> %s", anchor, candidate_d_path)
        return [candidate_d_path]
    
    # ── v1.1: Original desktop/documents logic ──
    sandbox_explicit = "sandbox" in intent_lower
    
    if anchor_lower == "desktop":
        if sandbox_explicit:
            roots = SANDBOX_DESKTOP_ROOTS + HOST_DESKTOP_ROOTS
            logger.info("[sandbox_inspector] v1.1: SANDBOX explicit - sandbox roots first")
        else:
            roots = HOST_DESKTOP_ROOTS + SANDBOX_DESKTOP_ROOTS
            logger.info("[sandbox_inspector] v1.1: Host roots first (OneDrive prioritized)")
    elif anchor_lower == "documents":
        if sandbox_explicit:
            roots = SANDBOX_DOCUMENTS_ROOTS + HOST_DOCUMENTS_ROOTS
            logger.info("[sandbox_inspector] v1.1: SANDBOX explicit - sandbox roots first")
        else:
            roots = HOST_DOCUMENTS_ROOTS + SANDBOX_DOCUMENTS_ROOTS
            logger.info("[sandbox_inspector] v1.1: Host roots first (OneDrive prioritized)")
    else:
        # Fallback to original SANDBOX_ROOTS
        roots = SANDBOX_ROOTS.get(anchor_lower, [])
        if not roots:
            # v1.2: Last resort - try D:\ root itself so bounded search can find it
            logger.warning("[sandbox_inspector] v1.2: Unknown anchor '%s', falling back to D:\\ scan", anchor)
            roots = [D_DRIVE_ROOT]
        else:
            logger.warning("[sandbox_inspector] Unknown anchor '%s', using default roots", anchor)
    
    logger.info("[sandbox_inspector] Ordered roots: %s", roots)
    return roots


def _identify_root_type(root_path: str) -> str:
    """
    Identify the type of root for reporting purposes.
    
    v1.2: Added project root identification.
    Returns: "project_orb", "project_orb-desktop", "host_onedrive_desktop", etc.
    """
    path_lower = root_path.lower().rstrip("\\/")
    
    # v1.2: Check if it's a known project root
    for name, project_path in PROJECT_ROOTS.items():
        if path_lower == project_path.lower().rstrip("\\/") or path_lower.startswith(project_path.lower().rstrip("\\/") + "\\"):
            return f"project_{name}"
    
    # v1.2: Check if it's on D:\ drive (unknown project)
    if path_lower.startswith("d:"):
        return "project_d_drive"
    
    if "wdagutilityaccount" in path_lower:
        if "desktop" in path_lower:
            return "sandbox_desktop"
        elif "documents" in path_lower:
            return "sandbox_documents"
        return "sandbox_unknown"
    elif "onedrive" in path_lower:
        if "desktop" in path_lower:
            return "host_onedrive_desktop"
        elif "documents" in path_lower:
            return "host_onedrive_documents"
        return "host_onedrive_unknown"
    else:
        if "desktop" in path_lower:
            return "host_desktop"
        elif "documents" in path_lower:
            return "host_documents"
        return "host_unknown"


# =============================================================================
# MAIN DISCOVERY FUNCTION (ONE Canonical Entry Point)
# =============================================================================

def run_sandbox_discovery_chain(
    anchor: str,
    subfolder: Optional[str] = None,
    job_intent: Optional[str] = None,
) -> Dict[str, Any]:
    """
    ONE canonical discovery function. Supports two modes:
    
    Mode A (subfolder provided): Go straight to Desktop\\<subfolder>
    Mode B (subfolder missing): List Desktop, pick folder or ask
    
    v1.1: Desktop alias expansion
    - "Desktop" is treated as an alias, not a single path
    - Tries multiple candidate roots in intelligent order
    - If "SANDBOX" in intent, sandbox roots first; otherwise host OneDrive first
    - Includes bounded search fallback if exact path not found
    - Reports what was tried for debugging
    
    Args:
        anchor: "desktop", "documents", etc.
        subfolder: Target folder name (optional)
        job_intent: User's intent for content-aware selection and root ordering
    
    Returns:
        {
            "found": bool,
            "root": str,                    # Which root was used
            "root_type": str,               # v1.1: "host_onedrive_desktop", "sandbox_desktop", etc.
            "path": str,                    # Full folder path
            "files": [...],                 # Files in folder
            "file_contents": {...},         # path -> content (winner only)
            "classifications": {...},       # path -> {type, confidence, intent_score}
            "selected_file": {...} or None, # Winner details
            "ambiguous": bool,              # True if couldn't decide
            "ambiguous_candidates": [...],  # Close-scoring files
            "question": str or None,        # Question to ask user
            "roots_tried": [...],           # v1.1: List of roots that were checked
            "discovery_method": str,        # v1.1: "exact", "bounded_search", "folder_search"
        }
    """
    result: Dict[str, Any] = {
        "found": False,
        "root": None,
        "root_type": None,
        "path": None,
        "files": [],
        "file_contents": {},
        "classifications": {},
        "selected_file": None,
        "ambiguous": False,
        "ambiguous_candidates": [],
        "question": None,
        "roots_tried": [],
        "discovery_method": None,
    }
    
    # v1.1: Get ordered roots based on context
    ordered_roots = _get_ordered_roots(anchor, job_intent)
    
    # ===== MODE A: Subfolder provided =====
    if subfolder:
        folder_result = _discover_specific_folder(anchor, subfolder, job_intent)
        result["roots_tried"] = folder_result.get("roots_tried", [])
        
        if not folder_result["found"]:
            logger.warning(
                "[sandbox_inspector] Folder not found: %s\\%s (tried %d roots)",
                anchor, subfolder, len(result["roots_tried"])
            )
            return result
        
        result["found"] = True
        result["root"] = folder_result["root"]
        result["root_type"] = folder_result.get("root_type")
        result["path"] = folder_result["path"]
        result["files"] = folder_result["files"]
        result["discovery_method"] = folder_result.get("discovery_method", "exact")
    
    # ===== MODE B: Subfolder missing =====
    else:
        folder_result = _discover_folder_without_hint(anchor, job_intent)
        result["roots_tried"] = folder_result.get("roots_tried", [])
        
        if folder_result.get("ambiguous"):
            # Multiple folders with candidates - ask once
            result["ambiguous"] = True
            result["root"] = folder_result.get("root")
            result["root_type"] = folder_result.get("root_type")
            result["question"] = folder_result["question"]
            result["ambiguous_candidates"] = folder_result["candidates"]
            logger.info(
                "[sandbox_inspector] Folder ambiguity: %d candidates",
                len(folder_result["candidates"])
            )
            return result
        
        if not folder_result["found"]:
            logger.warning("[sandbox_inspector] No folder found for anchor: %s", anchor)
            return result
        
        result["found"] = True
        result["root"] = folder_result.get("root")
        result["root_type"] = folder_result.get("root_type")
        result["path"] = folder_result["path"]
        result["files"] = folder_result.get("files", [])
        result["discovery_method"] = folder_result.get("discovery_method", "folder_search")
    
    # ===== FILE SELECTION (both modes) =====
    return _select_file_from_folder(result, job_intent)


# =============================================================================
# FOLDER DISCOVERY
# =============================================================================

def _discover_specific_folder(anchor: str, subfolder: str, job_intent: Optional[str] = None) -> Dict[str, Any]:
    """
    Go straight to a specific subfolder (Mode A).
    
    v1.1: Enhanced with:
    - Intelligent root ordering based on job_intent
    - Bounded search fallback if exact path not found
    - Tracking of roots tried
    
    v1.2: Normalise subfolder path separators (forward slashes to backslashes)
    """
    # v1.2: Normalise subfolder separators — LLMs often send forward slashes
    subfolder = subfolder.replace("/", "\\")
    
    # v1.1: Use intelligent ordering
    roots = _get_ordered_roots(anchor, job_intent)
    roots_tried = []
    
    # ===== STEP A: Try exact path at each root =====
    for root in roots:
        root_type = _identify_root_type(root)
        search_path = os.path.join(root, subfolder)
        roots_tried.append({
            "root": root,
            "root_type": root_type,
            "exact_path": search_path,
            "found": False,
            "method": "exact",
        })
        
        logger.info("[sandbox_inspector] v1.1 Trying exact path: %s", search_path)
        
        status, data, error = call_fs_tree([search_path], max_files=MAX_FILES_PER_FOLDER)
        
        if status == 200 and data and data.get("files"):
            roots_tried[-1]["found"] = True
            logger.info(
                "[sandbox_inspector] v1.1 FOUND (exact): %s (%d files) [%s]",
                search_path, len(data.get("files", [])), root_type
            )
            return {
                "found": True,
                "root": root,
                "root_type": root_type,
                "path": search_path,
                "files": data.get("files", []),
                "roots_tried": roots_tried,
                "discovery_method": "exact",
            }
        else:
            logger.debug(
                "[sandbox_inspector] v1.1 Not found at %s (status=%s, error=%s)",
                search_path, status, error
            )
    
    # ===== STEP B: Bounded search fallback =====
    # If exact path not found, search for subfolder within each root (depth-limited)
    logger.info(
        "[sandbox_inspector] v1.1 Exact path not found, trying bounded search for '%s'",
        subfolder
    )
    
    for root in roots:
        root_type = _identify_root_type(root)
        
        # First check if the root itself exists
        status, data, error = call_fs_tree([root], max_files=MAX_FILES_PER_FOLDER)
        if status != 200 or not data:
            logger.debug("[sandbox_inspector] v1.1 Root not accessible: %s", root)
            continue
        
        # Search for folder named subfolder within root (depth-limited)
        found_folder = _bounded_folder_search(root, subfolder, max_depth=MAX_SEARCH_DEPTH)
        
        if found_folder:
            roots_tried.append({
                "root": root,
                "root_type": root_type,
                "search_path": root,
                "found_at": found_folder["path"],
                "found": True,
                "method": "bounded_search",
            })
            
            logger.info(
                "[sandbox_inspector] v1.1 FOUND (bounded search): %s (%d files) [%s]",
                found_folder["path"], len(found_folder.get("files", [])), root_type
            )
            return {
                "found": True,
                "root": root,
                "root_type": root_type,
                "path": found_folder["path"],
                "files": found_folder.get("files", []),
                "roots_tried": roots_tried,
                "discovery_method": "bounded_search",
            }
    
    logger.warning(
        "[sandbox_inspector] v1.1 Folder '%s' not found after trying %d roots and bounded search",
        subfolder, len(roots)
    )
    return {"found": False, "roots_tried": roots_tried}


# =============================================================================
# FILE SELECTION (Content-Aware)
# =============================================================================


# =============================================================================
# FILE READING UTILITIES
# =============================================================================


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main discovery
    "run_sandbox_discovery_chain",
    # Utilities
    "read_sandbox_file",
    "file_exists_in_sandbox",
    # v1.1: Root ordering and identification
    "_get_ordered_roots",
    "_identify_root_type",
    "_bounded_folder_search",
    # Constants
    "SANDBOX_ROOTS",
    "PROJECT_ROOTS",
    "D_DRIVE_ROOT",
    "HOST_DESKTOP_ROOTS",
    "SANDBOX_DESKTOP_ROOTS",
    "HOST_DOCUMENTS_ROOTS",
    "SANDBOX_DOCUMENTS_ROOTS",
    "CANDIDATE_EXTENSIONS",
    "MAX_FILE_CHARS",
    "MAX_SEARCH_DEPTH",
]
