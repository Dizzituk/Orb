from __future__ import annotations
import logging
import os
import re
from app.pot_spec.grounded.evidence_gathering import logger, sandbox_path_exists
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
_SANDBOX_CLIENT_AVAILABLE = True
call_fs_tree = None
call_fs_contents = None


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

COMMON_FILE_EXTENSIONS = ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.js', '.ts', '.html', '.css']

PATH_REFERENCE_STOPWORDS = {
    'the', 'a', 'an', 'my', 'your', 'this', 'that', 'it',
    'with', 'from', 'to', 'in', 'on', 'at', 'of', 'for', 'by',
    'and', 'or', 'but', 'not', 'is', 'are', 'was', 'were',
    'can', 'will', 'should', 'could', 'would', 'may', 'might',
    'read', 'write', 'delete', 'create', 'make', 'get', 'set',
    'overwrite', 'replace', 'modify', 'change', 'update',
    'only', 'just', 'also', 'here', 'there', 'where', 'when',
}

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
        elif not exists:
            # v2.3 FIX: If sandbox_path_exists says file doesn't exist, don't
            # attempt reads that will fail and produce misleading HARD READ FAIL
            # errors. This commonly happens when GPT's evidence loop requests reads
            # of files that are CREATE targets (not yet implemented).
            logger.info(
                "[evidence_gathering] v2.3 sandbox_read_file: path does not exist, skipping: %s",
                path,
            )
            return False, None
        else:
            actual_path = path
        
        logger.info("[evidence_gathering] v1.26.1 sandbox_read_file: reading %s (actual: %s)", path, actual_path)
        
        status, data, error = call_fs_contents([actual_path])
        
        if status != 200 or not data:
            logger.warning(
                "[evidence_gathering] v1.28 sandbox_read_file: failed for %s (status=%s, error=%s) — trying host fallback",
                actual_path, status, error
            )
            # v2.2 FIX: Fall back to host filesystem when sandbox returns non-200
            # This handles paths that exist on the host but not in the sandbox
            # (e.g., D:\orb-desktop frontend files when sandbox only has D:\Orb backend)
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    fallback_content = f.read(max_chars)
                logger.info(
                    "[evidence_gathering] v2.2 sandbox_read_file: HOST FALLBACK SUCCESS %s (%d chars)",
                    path, len(fallback_content)
                )
                return True, fallback_content
            except Exception as host_err:
                logger.warning(
                    "[evidence_gathering] v2.2 sandbox_read_file: host fallback also failed for %s: %s",
                    path, host_err
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
                "[evidence_gathering] v1.35 sandbox_read_file: no content in file object for %s — trying direct read fallback",
                actual_path
            )
            # v1.35 FIX: Fallback to direct open() when sandbox returns empty content
            # This handles cases where the sandbox controller finds the file but
            # returns no content (encoding issues, empty files reported wrong, etc.)
            try:
                with open(actual_path, 'r', encoding='utf-8', errors='replace') as f:
                    fallback_content = f.read(max_chars)
                if fallback_content is not None:
                    # Distinguish genuinely empty files from read failures
                    if len(fallback_content) == 0:
                        logger.info(
                            "[evidence_gathering] v1.35 sandbox_read_file: file IS genuinely empty: %s",
                            actual_path
                        )
                        return True, ""  # File exists but is empty — that's valid evidence
                    logger.info(
                        "[evidence_gathering] v1.35 sandbox_read_file: FALLBACK SUCCESS %s (%d chars)",
                        actual_path, len(fallback_content)
                    )
                    return True, fallback_content
            except Exception as fallback_err:
                logger.warning(
                    "[evidence_gathering] v1.35 sandbox_read_file: direct read fallback also failed for %s: %s",
                    actual_path, fallback_err
                )
            # v2.1: HARD FAIL — file exists but BOTH read methods returned nothing.
            # This is NOT a normal "file not found" — the sandbox confirmed the file
            # exists but we cannot read its content. This MUST surface as an error,
            # not be silently swallowed.
            print(
                f"[ERROR] [evidence_gathering] v2.1 HARD READ FAIL: File exists at "
                f"{actual_path} but BOTH sandbox and direct read returned no content. "
                f"Evidence grounding is BROKEN for this file."
            )
            logger.error(
                "[evidence_gathering] v2.1 HARD READ FAIL: %s exists but unreadable by both methods",
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
