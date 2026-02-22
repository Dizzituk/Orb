import logging
import os
from app.pot_spec.grounded._evidence_gathering_utils import ANCHOR_RESOLUTION_MAP, COMMON_FILE_EXTENSIONS, extract_path_references
from app.pot_spec.grounded._evidence_gathering_utils import EVIDENCE_ALLOWED_ROOTS, EVIDENCE_FORBIDDEN_PATHS, _resolve_via_index_json
from datetime import datetime, timezone
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_MULTI_TARGET_AVAILABLE = True
_SYSTEM_SCAN_AVAILABLE = True
_CREATE_TARGET_AVAILABLE = True
is_multi_target_request = None
is_system_wide_scan_request = None
extract_create_targets = None


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

    # v2.2: INDEX.json fallback — search the codebase index for bare filenames.
    # When LLM requests just "architecture_executor.py" (no path), the root-level
    # search above fails. INDEX.json has every project file with full absolute paths.
    resolved_via_index = _resolve_via_index_json(ref_normalized)
    if resolved_via_index:
        logger.info(
            "[evidence_gathering] v2.2 FOUND via INDEX.json: ref='%s' -> '%s'",
            user_reference, resolved_via_index
        )
        return resolved_via_index, "index_json_lookup"

    logger.warning(
        "[evidence_gathering] v1.27 Path NOT FOUND: ref='%s', anchor='%s', subfolder='%s' (tried %d variants)",
        user_reference, anchor, subfolder, len(name_variants)
    )
    return None, "not_found"

def gather_filesystem_evidence(
    combined_text: str,
    anchor: Optional[str] = None,
    subfolder: Optional[str] = None,
    rag_hints: Optional[List[dict]] = None,
) -> EvidencePackage:
    """
    v1.34: Main entry point for Evidence-First filesystem validation.
    
    Gathers filesystem evidence BEFORE any LLM calls using SANDBOX filesystem.
    Now automatically detects and handles:
    - v1.31: System-wide scan requests ("find files on my system")
    - v1.27: Multi-target requests with explicit anchors
    - Standard single-target requests
    
    v1.34: Now excludes CREATE targets from ALL code paths.
    """
    logger.info(
        "[evidence_gathering] v1.34 gather_filesystem_evidence: anchor='%s', subfolder='%s', rag_hints=%d",
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
    
    # v1.34 CRITICAL FIX: Extract CREATE targets and exclude them from search
    # CREATE targets are files the user wants to CREATE (output), not READ (input)
    # SpecGate is READ-ONLY; Implementer handles CREATE operations
    create_names = set()
    create_targets = []
    if _CREATE_TARGET_AVAILABLE and extract_create_targets:
        create_targets = extract_create_targets(combined_text)
        create_names = {t["name"].lower() for t in create_targets}
        
        if create_names:
            original_count = len(references)
            references = [r for r in references if r.lower() not in create_names]
            excluded_count = original_count - len(references)
            
            if excluded_count > 0:
                logger.info(
                    "[evidence_gathering] v1.34 Excluded %d CREATE targets from references: %s",
                    excluded_count, list(create_names)
                )
            
            # Store create targets in package metadata for downstream use (Critical Pipeline / Implementer)
            package.metadata["create_targets"] = create_targets
            logger.info(
                "[evidence_gathering] v1.34 Stored %d CREATE targets in metadata: %s",
                len(create_targets), [t["name"] for t in create_targets]
            )
    
    if not references and not anchor:
        # v1.34: If we ONLY have CREATE targets, that's valid - not an error
        if create_targets:
            package.task_type = "create_only"
            logger.info("[evidence_gathering] v1.34 CREATE-only task detected, no READ targets")
            return package
        package.validation_errors.append("No path references found in text")
        logger.warning("[evidence_gathering] v1.34 No path references found")
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
        "[evidence_gathering] v1.34 gather_filesystem_evidence COMPLETE: %s",
        package.to_summary()
    )
    
    return package
