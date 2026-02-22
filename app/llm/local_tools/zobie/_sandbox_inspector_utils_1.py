from __future__ import annotations
import logging
from .content_classifier import classify_content, score_file_for_intent
from .sandbox_client import call_fs_contents, call_fs_tree
from app.llm.local_tools.zobie._sandbox_inspector_utils import MAX_CANDIDATE_FILES, MAX_CLASSIFICATION_CHARS, MAX_FILE_CHARS, _read_snippet
from typing import Any, Dict, List, Optional
from .sandbox_inspector import _get_ordered_roots, _identify_root_type
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
MAX_FILES_PER_FOLDER = 50       # Don't list more than this
CANDIDATE_EXTENSIONS = [".txt", ".md", ".py", ".json", ".yaml", ".yml", ".log", ".csv"]


HOST_DESKTOP_ROOTS = [
    r"C:\Users\dizzi\OneDrive\Desktop",
    r"C:\Users\dizzi\Desktop",
]

HOST_DOCUMENTS_ROOTS = [
    r"C:\Users\dizzi\OneDrive\Documents",
    r"C:\Users\dizzi\Documents",
]

SANDBOX_DOCUMENTS_ROOTS = [
    r"C:\Users\WDAGUtilityAccount\Documents",
]

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
