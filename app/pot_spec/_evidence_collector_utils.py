from __future__ import annotations
import logging
import re
from app.pot_spec.evidence_collector import EvidenceBundle, WRITE_REFUSED_ERROR, logger, read_file, search_repo
from pathlib import Path
from typing import Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
WRITE_REFUSED_ERROR = "SpecGate runtime is read-only. Write actions are not permitted."


EVIDENCE_PRIORITY = [
    "architecture_map",      # Highest priority
    "codebase_report",
    "file_read",
    "repo_search",
    "arch_query_fallback",  # Lowest priority - fallback only
]

DEFAULT_MAX_BYTES = 100_000

def refuse_write_operation(operation: str) -> str:
    """
    Called when any write operation is attempted.
    
    Args:
        operation: Description of the attempted operation
        
    Returns:
        Always returns the hard error message
        
    Raises:
        RuntimeError: Always raises to prevent any write
    """
    logger.error("[evidence_collector] WRITE REFUSED: %s", operation)
    raise RuntimeError(WRITE_REFUSED_ERROR)

def add_file_read_to_bundle(
    bundle: EvidenceBundle,
    path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    head_lines: Optional[int] = None,
) -> Optional[str]:
    """
    Add a file read to an existing evidence bundle.
    
    Returns:
        File content if successful, None otherwise
    """
    content, source = read_file(
        path=path,
        start_line=start_line,
        end_line=end_line,
        head_lines=head_lines,
    )
    
    bundle.add_source(source)
    
    if content:
        repo_path = source.path or path
        bundle.file_reads[repo_path] = content
    
    return content

def add_search_to_bundle(
    bundle: EvidenceBundle,
    query: str,
    limit: int = 20,
) -> List[Dict]:
    """
    Add a search result to an existing evidence bundle.
    
    NOTE: This uses arch_query as fallback. Prefer direct file reads.
    
    Returns:
        Search results
    """
    results, source = search_repo(query=query, limit=limit)
    
    bundle.add_source(source)
    bundle.arch_query_used = source.is_fallback and source.found
    
    if results:
        bundle.search_results[query] = results
    
    return results

def _to_absolute_path(path: str) -> str:
    """Convert relative path to absolute path in repo."""
    if Path(path).is_absolute():
        return path
    
    # Assume relative to D:\Orb
    return str(Path(r"D:\Orb") / path)

def find_in_evidence(
    bundle: EvidenceBundle,
    pattern: str,
    source_type: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Search for a pattern in loaded evidence.
    
    Args:
        bundle: Evidence bundle to search
        pattern: Regex pattern to find
        source_type: Limit to specific source type (architecture_map, codebase_report, etc.)
        
    Returns:
        List of (source_name, matched_content) tuples
    """
    results = []
    regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    
    # Search architecture map
    if (source_type is None or source_type == "architecture_map") and bundle.arch_map_content:
        matches = regex.findall(bundle.arch_map_content)
        for match in matches:
            results.append(("architecture_map", match))
    
    # Search codebase report
    if (source_type is None or source_type == "codebase_report") and bundle.codebase_report_content:
        matches = regex.findall(bundle.codebase_report_content)
        for match in matches:
            results.append(("codebase_report", match))
    
    # Search file reads
    if source_type is None or source_type == "file_read":
        for path, content in bundle.file_reads.items():
            matches = regex.findall(content)
            for match in matches:
                results.append((f"file:{path}", match))
    
    return results

def verify_path_exists(
    bundle: EvidenceBundle,
    path: str,
) -> Tuple[bool, Optional[str]]:
    """
    Verify a path exists using evidence (no additional reads).
    
    Args:
        bundle: Evidence bundle
        path: Path to verify (repo-relative)
        
    Returns:
        (exists, source) - where source is which evidence confirmed it
    """
    # Normalize path for comparison
    path_normalized = path.replace("\\", "/").lower()
    
    # Check architecture map
    if bundle.arch_map_content:
        if path_normalized in bundle.arch_map_content.lower():
            return True, "architecture_map"
    
    # Check codebase report
    if bundle.codebase_report_content:
        if path_normalized in bundle.codebase_report_content.lower():
            return True, "codebase_report"
    
    # Check if we've already read this file
    for read_path in bundle.file_reads.keys():
        if read_path.replace("\\", "/").lower() == path_normalized:
            return True, f"file_read:{read_path}"
    
    return False, None
