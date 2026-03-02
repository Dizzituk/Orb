from __future__ import annotations
import logging
import os
import re
from typing import List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
# v3.2-fix: Import centralised sandbox filesystem helpers.
try:
    from app.sandbox_fs import (
        sandbox_exists as _sf_exists,
        sandbox_isfile as _sf_isfile,
        sandbox_isdir as _sf_isdir,
        sandbox_read_text as _sf_read_text,
        sandbox_listdir as _sf_listdir,
    )
    _SANDBOX_FS_AVAILABLE = True
except ImportError:
    _SANDBOX_FS_AVAILABLE = False
    logger.error("[evidence_gathering] v3.2 CRITICAL: app.sandbox_fs not available")


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

COMMON_FILE_EXTENSIONS = ['.txt', '.md', '.py', '.json', '.yaml', '.yml', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', '.svg', '.mjs']

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
    v3.2-fix: Read file content from SANDBOX filesystem.

    Uses centralised app.sandbox_fs helpers. NO host fallback.
    If the sandbox is unreachable, returns (False, None) so the pipeline
    knows evidence is missing rather than silently using stale host data.

    Returns:
        (success: bool, content: Optional[str])
    """
    if not _SANDBOX_FS_AVAILABLE:
        logger.error("[evidence_gathering] v3.2 sandbox_read_file: sandbox_fs not available")
        return False, None

    try:
        # Check existence via sandbox first
        if not _sf_exists(path):
            # Case-insensitive retry
            found = False
            for var in [
                path.replace('\\Test', '\\test'),
                path.replace('\\test', '\\Test'),
            ]:
                if var != path and _sf_exists(var):
                    path = var
                    found = True
                    break
            if not found:
                logger.info(
                    "[evidence_gathering] v3.2 sandbox_read_file: path does not exist: %s",
                    path,
                )
                return False, None

        logger.info("[evidence_gathering] v3.2 sandbox_read_file: reading %s", path)

        content = _sf_read_text(path)
        if content is None:
            logger.error(
                "[evidence_gathering] v3.2 HARD READ FAIL: %s exists in sandbox but unreadable",
                path,
            )
            return False, None

        # Truncate if too long
        if len(content) > max_chars:
            logger.info(
                "[evidence_gathering] v3.2 sandbox_read_file: truncating %s from %d to %d chars",
                path, len(content), max_chars,
            )
            content = content[:max_chars]

        logger.info(
            "[evidence_gathering] v3.2 sandbox_read_file: SUCCESS %s (%d chars)",
            path, len(content),
        )
        return True, content

    except Exception as e:
        logger.error(
            "[evidence_gathering] v3.2 sandbox_read_file: exception reading %s: %s",
            path, e,
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
    extension_pattern = r'["\']?([\w\-]+\.(?:txt|md|py|json|yaml|yml|jsx?|tsx?|html|css|scss|svg|mjs))["\']?'
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
    v3.2-fix: Scan a root directory for a file by name via SANDBOX.

    Scans shallowly (max_depth levels) to avoid scanning entire drives.
    Tries with and without common extensions.

    Args:
        root: Root directory to scan
        filename: File name to search for (with or without extension)
        max_depth: Maximum directory depth to scan (default 2)

    Returns:
        Full path if found, None otherwise
    """
    if not _SANDBOX_FS_AVAILABLE:
        logger.error("[evidence_gathering] v3.2 scan_root_for_file: sandbox_fs not available")
        return None

    logger.info(
        "[evidence_gathering] v3.2 scan_root_for_file: searching for '%s' in '%s' (depth=%d)",
        filename, root, max_depth,
    )

    # Build list of name variants to look for
    filename_lower = filename.lower()
    has_extension = '.' in filename and len(filename.split('.')[-1]) <= 4

    variants = [filename_lower]
    if not has_extension:
        for ext in COMMON_FILE_EXTENSIONS:
            variants.append(filename_lower + ext)

    try:
        # List files in root via sandbox
        entries = _sf_listdir(root)

        for f in entries:
            if isinstance(f, dict):
                f_path = f.get("path", "")
                f_name = os.path.basename(f_path).lower()
                is_dir = f.get("is_dir", False)
            else:
                f_path = str(f)
                f_name = os.path.basename(f_path).lower()
                is_dir = False

            # Check if this file matches
            if not is_dir and f_name in variants:
                logger.info(
                    "[evidence_gathering] v3.2 scan_root_for_file: FOUND '%s' at '%s'",
                    filename, f_path,
                )
                return f_path

            # Recurse into subdirectories if we haven't hit max depth
            if is_dir and max_depth > 1:
                dir_name = os.path.basename(f_path).lower()
                skip_dirs = {
                    'windows', 'program files', 'program files (x86)',
                    'appdata', '$recycle.bin', 'system volume information',
                    'programdata', '.git', 'node_modules', '__pycache__', '.venv',
                }
                if dir_name in skip_dirs:
                    continue

                found = scan_root_for_file(f_path, filename, max_depth - 1)
                if found:
                    return found

        return None

    except Exception as e:
        logger.error(
            "[evidence_gathering] v3.2 scan_root_for_file: exception scanning %s: %s",
            root, e,
        )
        return None
