from __future__ import annotations
import logging
import os
from app.pot_spec.grounded._evidence_gathering_utils_9 import USER_SCAN_ROOTS, scan_root_for_file
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_MULTI_TARGET_AVAILABLE = True
_SYSTEM_SCAN_AVAILABLE = True
_CREATE_TARGET_AVAILABLE = True
extract_file_targets = None
extract_scan_file_names = None
extract_create_targets = None

# v3.2-fix: Import centralised sandbox filesystem helpers.
# These replace the broken call_fs_tree wiring that was always None.
try:
    from app.sandbox_fs import (
        sandbox_exists as _sf_exists,
        sandbox_isfile as _sf_isfile,
        sandbox_listdir as _sf_listdir,
        sandbox_file_size as _sf_file_size,
    )
    _SANDBOX_FS_AVAILABLE = True
except ImportError:
    _SANDBOX_FS_AVAILABLE = False
    logger.error("[evidence_gathering] v3.2 CRITICAL: app.sandbox_fs not available")


def sandbox_path_exists(path: str) -> Tuple[bool, Optional[Dict]]:
    """
    v1.26 / v3.2-fix: Check if a path exists in the SANDBOX filesystem.

    v3.2-fix: Now uses centralised app.sandbox_fs helpers instead of
    the broken call_fs_tree wiring (which was always None, causing
    every call to fall back to host os.path.exists).

    Returns:
        (exists: bool, file_info: Optional[Dict])
        file_info contains 'size', 'mtime', 'actual_path' if file exists
    """
    if not _SANDBOX_FS_AVAILABLE:
        logger.error("[evidence_gathering] v3.2 sandbox_path_exists: sandbox_fs not available — REFUSING to fall back to host")
        return False, None
    
    try:
        logger.info("[evidence_gathering] v3.2 sandbox_path_exists: checking %s", path)

        # v3.2-fix: Use centralised sandbox_fs helpers (always hits sandbox,
        # never falls back to host os.path).
        found = _sf_exists(path)
        if not found:
            # Case-insensitive retry for common variations
            variations = [
                path.replace('\\Test', '\\test'),
                path.replace('\\test', '\\Test'),
            ]
            for var in variations:
                if var != path and _sf_exists(var):
                    path = var
                    found = True
                    break

        if not found:
            logger.info(
                "[evidence_gathering] v3.2 sandbox_path_exists: NOT FOUND %s",
                path,
            )
            return False, None

        # Build file_info from sandbox
        size = _sf_file_size(path)
        file_info = {
            "path": path,
            "actual_path": path,
            "size": size,
            "mtime": None,
            "is_dir": not _sf_isfile(path),
        }
        logger.info(
            "[evidence_gathering] v3.2 sandbox_path_exists: FOUND %s (size=%s)",
            path, size,
        )
        return True, file_info

    except Exception as e:
        logger.error(
            "[evidence_gathering] v3.2 sandbox_path_exists: exception checking %s: %s",
            path, e,
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
    from .evidence_gathering import EvidencePackage, resolve_and_validate_path
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
    from .evidence_gathering import EvidencePackage, FileEvidence, FilesystemEvidenceSource, resolve_and_validate_path
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
