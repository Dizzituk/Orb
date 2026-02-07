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

logger = logging.getLogger(__name__)


# =============================================================================
# SAFETY LIMITS
# =============================================================================

MAX_FILE_CHARS = 8000           # Prevent ingesting megabyte logs
MAX_CLASSIFICATION_CHARS = 800  # Snippet for classification (fast)
MAX_CANDIDATE_FILES = 20        # Don't read more than this
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
D_DRIVE_ROOT = "D:\\"

# Separate lists for intelligent ordering
HOST_DESKTOP_ROOTS = [
    r"C:\Users\dizzi\OneDrive\Desktop",
    r"C:\Users\dizzi\Desktop",
]
SANDBOX_DESKTOP_ROOTS = [
    r"C:\Users\WDAGUtilityAccount\Desktop",
]
HOST_DOCUMENTS_ROOTS = [
    r"C:\Users\dizzi\OneDrive\Documents",
    r"C:\Users\dizzi\Documents",
]
SANDBOX_DOCUMENTS_ROOTS = [
    r"C:\Users\WDAGUtilityAccount\Documents",
]

# Candidate file extensions for discovery
CANDIDATE_EXTENSIONS = [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".log", ".csv"]

# Maximum search depth for bounded fallback
MAX_SEARCH_DEPTH = 3


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


def _bounded_folder_search(
    root: str,
    target_folder: str,
    max_depth: int = 3,
    current_depth: int = 0,
) -> Optional[Dict[str, Any]]:
    """
    v1.1: Search for a folder by name within root, up to max_depth levels.
    
    This is a bounded search - it will NOT scan the entire disk.
    
    Args:
        root: Starting directory
        target_folder: Name of folder to find (case-insensitive)
        max_depth: Maximum depth to search
        current_depth: Current recursion depth
        
    Returns:
        {"path": str, "files": [...]} if found, None otherwise
    """
    if current_depth >= max_depth:
        return None
    
    target_lower = target_folder.lower()
    
    # List contents of current directory
    status, data, error = call_fs_tree([root], max_files=MAX_FILES_PER_FOLDER)
    if status != 200 or not data:
        return None
    
    items = data.get("files", [])
    
    # Check each subdirectory
    for item in items:
        if not item.get("is_dir"):
            continue
        
        folder_name = (item.get("name") or "").lower()
        folder_path = item.get("path")
        
        if not folder_path:
            continue
        
        # Check if this is the target folder
        if folder_name == target_lower:
            # Found it! Get its contents
            status2, data2, _ = call_fs_tree([folder_path], max_files=MAX_FILES_PER_FOLDER)
            if status2 == 200 and data2:
                logger.info(
                    "[sandbox_inspector] v1.1 Bounded search found folder: %s",
                    folder_path
                )
                return {
                    "path": folder_path,
                    "files": data2.get("files", []),
                }
        
        # Recurse into subdirectory (depth-limited)
        if current_depth + 1 < max_depth:
            sub_result = _bounded_folder_search(
                folder_path,
                target_folder,
                max_depth,
                current_depth + 1,
            )
            if sub_result:
                return sub_result
    
    return None


def _discover_folder_without_hint(anchor: str, job_intent: Optional[str] = None) -> Dict[str, Any]:
    """
    Mode B: No subfolder provided.
    Rule: First folder with readable candidates wins. Multiple = ask once.
    
    v1.1: Uses intelligent root ordering based on job_intent.
    Now properly tracks roots_tried and root_type.
    """
    # v1.1: Use intelligent ordering
    roots = _get_ordered_roots(anchor, job_intent)
    roots_tried = []
    
    for root in roots:
        root_type = _identify_root_type(root)
        roots_tried.append({
            "root": root,
            "root_type": root_type,
            "accessible": False,
            "folders_found": 0,
            "folders_with_candidates": 0,
        })
        
        status, data, error = call_fs_tree([root], max_files=MAX_FILES_PER_FOLDER)
        if status != 200:
            logger.debug("[sandbox_inspector] v1.1 Root not accessible: %s (status=%s)", root, status)
            continue
        
        roots_tried[-1]["accessible"] = True
        
        folders = [f for f in data.get("files", []) if f.get("is_dir")]
        roots_tried[-1]["folders_found"] = len(folders)
        
        if not folders:
            continue
        
        # Which folders have ANY readable files?
        folders_with_candidates: List[Dict[str, Any]] = []
        for folder in folders:
            peek_status, peek_data, _ = call_fs_tree([folder["path"]], max_files=10)
            if peek_status != 200:
                continue
            
            files = [f for f in peek_data.get("files", []) if not f.get("is_dir")]
            has_readable = any(
                (f.get("name") or "").lower().endswith(tuple(CANDIDATE_EXTENSIONS))
                for f in files
            )
            
            if has_readable:
                folders_with_candidates.append({
                    "folder": folder,
                    "files": peek_data.get("files", []),
                })
        
        roots_tried[-1]["folders_with_candidates"] = len(folders_with_candidates)
        
        if not folders_with_candidates:
            continue
        
        if len(folders_with_candidates) == 1:
            # Clear winner - use it
            logger.info(
                "[sandbox_inspector] v1.1 FOUND (folder discovery): %s [%s]",
                folders_with_candidates[0]["folder"]["path"], root_type
            )
            return {
                "found": True,
                "root": root,
                "root_type": root_type,
                "path": folders_with_candidates[0]["folder"]["path"],
                "files": folders_with_candidates[0]["files"],
                "roots_tried": roots_tried,
                "discovery_method": "folder_discovery",
            }
        
        # Multiple folders - ask ONE question
        names = [f["folder"]["name"] for f in folders_with_candidates[:5]]
        logger.info(
            "[sandbox_inspector] v1.1 AMBIGUOUS: %d folders with candidates at %s [%s]",
            len(folders_with_candidates), root, root_type
        )
        return {
            "found": False,
            "root": root,
            "root_type": root_type,
            "ambiguous": True,
            "candidates": names,
            "question": f"I found these folders with files: {', '.join(names)} — which one?",
            "roots_tried": roots_tried,
        }
    
    logger.warning(
        "[sandbox_inspector] v1.1 No folders with candidates found after trying %d roots",
        len(roots)
    )
    return {"found": False, "roots_tried": roots_tried}


# =============================================================================
# FILE SELECTION (Content-Aware)
# =============================================================================

def _select_file_from_folder(result: Dict[str, Any], job_intent: Optional[str]) -> Dict[str, Any]:
    """
    Select the best file using content classification + intent scoring.
    
    Uses progressive reads: snippet for classification, full read only for winner.
    """
    candidates: List[Dict[str, Any]] = []
    
    for f in result["files"]:
        name = (f.get("name") or "").lower()
        path = f.get("path")
        is_dir = bool(f.get("is_dir"))
        
        if not path or is_dir:
            continue
        
        # Extension filter
        if not any(name.endswith(ext) for ext in CANDIDATE_EXTENSIONS):
            continue
        
        candidates.append({"name": f.get("name", name), "path": path})
        
        # Cap candidates
        if len(candidates) >= MAX_CANDIDATE_FILES:
            logger.warning("[sandbox_inspector] Capped at %d candidates", MAX_CANDIDATE_FILES)
            break
    
    if not candidates:
        logger.info("[sandbox_inspector] No candidate files found in folder")
        return result
    
    # ===== PHASE 1: Read snippets for classification =====
    scored_candidates: List[Dict[str, Any]] = []
    
    for c in candidates:
        snippet = _read_snippet(c["path"], MAX_CLASSIFICATION_CHARS)
        if not snippet:
            continue
        
        # Classify content
        content_type, confidence = classify_content(snippet)
        
        # Score against intent (if provided)
        if job_intent:
            intent_score = score_file_for_intent(content_type, job_intent)
        else:
            # No intent - use confidence only
            intent_score = 0.5
        
        final_score = confidence * intent_score
        
        result["classifications"][c["path"]] = {
            "type": content_type.value,
            "confidence": round(confidence, 3),
            "intent_score": round(intent_score, 3),
            "final_score": round(final_score, 3),
        }
        
        scored_candidates.append({
            **c,
            "content_type": content_type,
            "confidence": confidence,
            "intent_score": intent_score,
            "final_score": final_score,
        })
        
        logger.info(
            "[sandbox_inspector] %s: %s (conf=%.2f, intent=%.2f, final=%.2f)",
            c["name"], content_type.value, confidence, intent_score, final_score
        )
    
    if not scored_candidates:
        logger.warning("[sandbox_inspector] No candidates could be classified")
        return result
    
    # Sort by final score
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    best = scored_candidates[0]
    
    # ===== CHECK FOR AMBIGUITY =====
    if len(scored_candidates) > 1:
        second = scored_candidates[1]
        same_type = best["content_type"] == second["content_type"]
        close_score = best["final_score"] - second["final_score"] < 0.15
        both_viable = best["final_score"] > 0.4 and second["final_score"] > 0.4
        
        if same_type and close_score and both_viable:
            # MUST ASK - do not guess
            result["ambiguous"] = True
            result["ambiguous_candidates"] = [
                {"name": c["name"], "type": c["content_type"].value, "path": c["path"]}
                for c in scored_candidates
                if c["content_type"] == best["content_type"] and c["final_score"] > 0.4
            ]
            type_name = best["content_type"].value
            names = [c["name"] for c in result["ambiguous_candidates"]]
            result["question"] = f"I found {len(names)} {type_name} files: {', '.join(names)} — which one?"
            
            logger.warning(
                "[sandbox_inspector] AMBIGUOUS: %d %s files with close scores",
                len(names), type_name
            )
            return result
    
    # ===== PHASE 2: Full read of winner only =====
    full_content = read_sandbox_file(best["path"])
    
    result["selected_file"] = {
        "path": best["path"],
        "name": best["name"],
        "content_type": best["content_type"].value,
        "confidence": best["confidence"],
        "intent_score": best["intent_score"],
        "final_score": best["final_score"],
        "content": full_content,
    }
    result["file_contents"][best["path"]] = full_content
    
    logger.info(
        "[sandbox_inspector] Selected: %s (%s, score=%.2f)",
        best["name"], best["content_type"].value, best["final_score"]
    )
    
    return result


# =============================================================================
# FILE READING UTILITIES
# =============================================================================

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


def read_sandbox_file(file_path: str) -> Optional[str]:
    """Read full file content (capped at MAX_FILE_CHARS)."""
    status, data, error = call_fs_contents([file_path])
    if status == 200 and data:
        files = data.get("files", [])
        if files:
            content = files[0].get("content")
            if content:
                if len(content) > MAX_FILE_CHARS:
                    logger.warning(
                        "[sandbox_inspector] Truncating %s from %d to %d chars",
                        file_path, len(content), MAX_FILE_CHARS
                    )
                    return content[:MAX_FILE_CHARS]
                return content
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
