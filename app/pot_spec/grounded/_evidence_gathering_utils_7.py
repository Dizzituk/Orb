from __future__ import annotations
import logging
import os
from app.pot_spec.grounded._evidence_gathering_utils import USER_SCAN_ROOTS, scan_root_for_file
from app.pot_spec.grounded.evidence_gathering import EvidencePackage, FileEvidence, FilesystemEvidenceSource, logger, resolve_and_validate_path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_SANDBOX_CLIENT_AVAILABLE = True
call_fs_tree = None
_MULTI_TARGET_AVAILABLE = True
_SYSTEM_SCAN_AVAILABLE = True
_CREATE_TARGET_AVAILABLE = True
extract_file_targets = None
extract_scan_file_names = None
extract_create_targets = None


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

def gather_system_wide_scan_evidence(
    combined_text: str,
    rag_hints: Optional[List[dict]] = None,
) -> EvidencePackage:
    """
    v1.33: Gather evidence by scanning the entire system for named files.
    
    Handles patterns like "Find all files called test1, test2, test3, test4 on my system".
    Scans USER_SCAN_ROOTS for each target file.
    
    v1.33 FIX: Now excludes CREATE targets from scan list.
    
    Args:
        combined_text: User request text  
        rag_hints: Optional RAG hints
        
    Returns:
        EvidencePackage with files found across the system
    """
    logger.info("[evidence_gathering] v1.33 gather_system_wide_scan_evidence: starting")
    
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
        logger.error("[evidence_gathering] v1.33 extract_scan_file_names not available")
        return package
    
    # Extract file names to search for
    file_names = extract_scan_file_names(combined_text)
    
    # v1.33 NEW: Extract CREATE targets and exclude them from scan list
    # These are files the user wants to CREATE, not FIND
    create_names = set()
    if _CREATE_TARGET_AVAILABLE and extract_create_targets:
        create_targets = extract_create_targets(combined_text)
        create_names = {t["name"].lower() for t in create_targets}
        
        if create_names:
            original_count = len(file_names)
            file_names = [f for f in file_names if f.lower() not in create_names]
            excluded_count = original_count - len(file_names)
            
            if excluded_count > 0:
                logger.info(
                    "[evidence_gathering] v1.33 Excluded %d CREATE targets from scan: %s",
                    excluded_count, list(create_names)
                )
                # Store create targets in package metadata for downstream use
                package.metadata = package.metadata if hasattr(package, 'metadata') else {}
                if not hasattr(package, 'metadata') or package.metadata is None:
                    package.metadata = {}
                package.metadata["create_targets"] = create_targets
    
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
