"""
Architecture document parsing utilities.

This module provides pure text parsing functions for extracting file inventory
and section content from architecture markdown documents. It is stdlib-only
and has no dependencies on other package modules to maintain independent
importability.

Extracted from app/overwatcher/architecture_executor.py as part of the
architecture executor decomposition.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# v5.22: Invalid filesystem characters and placeholder patterns
_INVALID_PATH_CHARS = set('*?<>|"')
_PLACEHOLDER_PATTERNS = {'none', 'n/a', 'na', 'tbd', 'placeholder', 'empty', 'nil', 'null'}


def _is_valid_file_path(path: str) -> bool:
    """Check if a parsed path is a real file path, not a placeholder or invalid.

    Rejects:
      - Paths containing invalid filesystem chars: * ? < > | "
      - Markdown formatting artifacts: *(none)*, _(n/a)_
      - Common placeholder words: none, n/a, tbd, empty
      - Paths with no file extension (bare words)
    """
    if not path:
        return False
    # Strip markdown emphasis wrappers
    stripped = path.strip('*_()').lower()
    if stripped in _PLACEHOLDER_PATTERNS:
        return False
    # Any invalid filesystem character → reject
    if _INVALID_PATH_CHARS & set(path):
        logger.debug("[parsing] v5.22 Skipped invalid path: %s", path)
        return False
    # Must contain at least one dot (file extension) or slash (directory)
    if '.' not in path and '/' not in path and '\\' not in path:
        return False
    return True


def parse_file_inventory(architecture: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Parse the File Inventory section of an architecture document.

    Returns:
        Tuple of (new_files, modified_files), where each is a list of dicts
        with keys "path" and "description".
    """
    new_files: List[Dict[str, str]] = []
    modified_files: List[Dict[str, str]] = []

    # Prefer isolating the File Inventory section using multiline anchored regex
    inventory_pattern = re.compile(
        r'^##\s+File Inventory.*?(?=^##\s+|\Z)',
        re.DOTALL | re.IGNORECASE | re.MULTILINE
    )
    inventory_match = inventory_pattern.search(architecture)
    if inventory_match:
        inventory_section = inventory_match.group(0)
    else:
        # Fallback: use entire architecture if no File Inventory section found
        inventory_section = architecture

    # Parse "New Files" table
    new_table_pattern = re.compile(
        r'###?\s+New Files\s*\n.*?\n((?:\|[^\n]+\|\s*\n)+)',
        re.DOTALL | re.IGNORECASE
    )
    new_table_match = new_table_pattern.search(inventory_section)
    if new_table_match:
        table_text = new_table_match.group(1)
        for line in table_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('|---') or line.startswith('| File') or line.startswith('| Path'):
                continue
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                path_part = parts[0].strip('`').strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ""
                if path_part and _is_valid_file_path(path_part):
                    new_files.append({"path": path_part, "description": desc_part})

    # Parse "Modified Files" table
    mod_table_pattern = re.compile(
        r'###?\s+Modified Files\s*\n.*?\n((?:\|[^\n]+\|\s*\n)+)',
        re.DOTALL | re.IGNORECASE
    )
    mod_table_match = mod_table_pattern.search(inventory_section)
    if mod_table_match:
        table_text = mod_table_match.group(1)
        for line in table_text.split('\n'):
            line = line.strip()
            if not line or line.startswith('|---') or line.startswith('| File') or line.startswith('| Path'):
                continue
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                path_part = parts[0].strip('`').strip()
                desc_part = parts[1].strip() if len(parts) > 1 else ""
                if path_part and _is_valid_file_path(path_part):
                    modified_files.append({"path": path_part, "description": desc_part})

    # Fallback: parse heading-based lists if tables are not found
    if not new_files:
        new_heading_pattern = re.compile(
            r'(?:^|\n)(?:###?\s+)?New File:\s*[`"]?([^\n`"]+)[`"]?',
            re.IGNORECASE | re.MULTILINE
        )
        for match in new_heading_pattern.finditer(inventory_section):
            path = match.group(1).strip()
            if path:
                new_files.append({"path": path, "description": ""})

    if not modified_files:
        mod_heading_pattern = re.compile(
            r'(?:^|\n)(?:###?\s+)?Modifications to\s*[`"]?([^\n`"]+)[`"]?',
            re.IGNORECASE | re.MULTILINE
        )
        for match in mod_heading_pattern.finditer(inventory_section):
            path = match.group(1).strip()
            if path:
                modified_files.append({"path": path, "description": ""})

    # Additional fallback: header-based listing with heuristic for facade->modified
    if not new_files and not modified_files:
        header_pattern = re.compile(
            r'^#{2,6}\s+[`"]?([^\n`"]+\.\w+)[`"]?',
            re.MULTILINE
        )
        for match in header_pattern.finditer(inventory_section):
            path = match.group(1).strip()
            if path:
                # Heuristic: if path contains "facade" or starts with "Modified", treat as modified
                if "facade" in path.lower() or path.lower().startswith("modified"):
                    modified_files.append({"path": path, "description": ""})
                else:
                    new_files.append({"path": path, "description": ""})

    # Print info if using heading fallback (preserve original behavior)
    if (new_files or modified_files) and not (new_table_match or mod_table_match):
        print(f"[ARCH_EXEC] File inventory parsed using heading fallback: {len(new_files)} new, {len(modified_files)} modified")

    return new_files, modified_files


def extract_section_for_file(architecture: str, file_path: str) -> str:
    """
    Extract the relevant section from an architecture document for a given file path.

    Returns:
        String containing the section text, or empty string if not found.
    """
    # Normalize path for comparison
    normalized_path = file_path.replace("\\", "/")
    filename_only = Path(file_path).name
    path_variants = [
        normalized_path,
        file_path.replace("/", "\\"),
        filename_only
    ]

    lines = architecture.split('\n')
    relevant_section = []
    in_relevant_section = False
    relevant_section_depth = 0

    for i, line in enumerate(lines):
        # Check if line is a header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            depth = len(header_match.group(1))
            header_text = header_match.group(2).strip()

            # Check if this header mentions any path variant
            mentions_file = any(variant in header_text for variant in path_variants)

            if mentions_file:
                # Start capturing this section
                if not in_relevant_section:
                    in_relevant_section = True
                    relevant_section_depth = depth
                    relevant_section.append(line)
                elif depth <= relevant_section_depth:
                    # New section at same or higher level, restart capture
                    relevant_section = [line]
                    relevant_section_depth = depth
                else:
                    # Sub-section within relevant section
                    relevant_section.append(line)
            elif in_relevant_section:
                # Check if this header ends the relevant section
                if depth <= relevant_section_depth:
                    # End of relevant section
                    break
                else:
                    # Sub-section within relevant section
                    relevant_section.append(line)
        elif in_relevant_section:
            # Regular line within relevant section
            relevant_section.append(line)

    if relevant_section:
        return '\n'.join(relevant_section)

    # Fallback: look for paragraphs mentioning the file
    paragraphs = architecture.split('\n\n')
    matching_paragraphs = []
    for para in paragraphs:
        if any(variant in para for variant in path_variants):
            matching_paragraphs.append(para)
            if len(matching_paragraphs) >= 5:
                break

    if matching_paragraphs:
        return '\n\n'.join(matching_paragraphs)

    return ""


def _extract_verbatim_code_from_architecture(file_context: str, rel_path: str) -> Optional[str]:
    """
    Extract verbatim code blocks from architecture context.

    This helper implements the "one large block" heuristic:
    - Extract all fenced code blocks
    - If one block is >500 chars and matches file extension, use it
    - If multiple large blocks exist, only combine when context indicates "complete file"

    Args:
        file_context: Architecture section text for the file
        rel_path: Relative file path (used for extension checking)

    Returns:
        Combined code string if extraction succeeds, None otherwise
    """
    # Extract all fenced code blocks
    code_pattern = re.compile(r'```(?:\w+)?\s*\n(.*?)```', re.DOTALL)
    code_blocks = code_pattern.findall(file_context)

    if not code_blocks:
        return None

    # v6.1 FIX 24d: Separate import blocks from content blocks.
    # The imports block is often small (<500 chars) but MUST be included
    # and placed FIRST, otherwise the file fails with NameError at runtime.
    import_blocks = []
    content_blocks = []
    for block in code_blocks:
        stripped = block.strip()
        if not stripped:
            continue
        # Detect import blocks: majority of lines are import/from statements
        lines = [l for l in stripped.split('\n') if l.strip()]
        import_lines = sum(1 for l in lines
                          if l.strip().startswith(('import ', 'from ')))
        if lines and import_lines / len(lines) > 0.5:
            import_blocks.append(stripped)
        elif len(stripped) > 500:
            content_blocks.append(stripped)

    large_blocks = content_blocks  # back-compat name for logic below

    if not large_blocks and not import_blocks:
        return None

    # Check file extension for sanity
    ext = Path(rel_path).suffix.lower()
    valid_extensions = {'.py', '.ts', '.tsx', '.js', '.jsx', '.json', '.yaml', '.yml', '.toml', '.md', '.txt'}

    if ext not in valid_extensions:
        logger.debug(f"[ARCH_EXEC] Verbatim extraction: unrecognized extension {ext}, skipping")
        return None

    # v6.1 FIX 24d: Helper to prepend import blocks
    def _prepend_imports(code: str) -> str:
        if not import_blocks:
            return code
        return "\n".join(import_blocks) + "\n\n" + code

    # If single large block, use it
    if len(large_blocks) == 1:
        return _prepend_imports(large_blocks[0].strip())

    # No large content blocks but imports exist — not enough for verbatim
    if not large_blocks:
        return None

    # Multiple large blocks: check context for "complete file" indicators
    combine_indicators = [
        "complete file",
        "full content",
        "verbatim",
        "extract the following",
        "entire file",
        "whole file"
    ]

    context_lower = file_context.lower()
    should_combine = any(indicator in context_lower for indicator in combine_indicators)

    if should_combine:
        # Combine blocks with double newline separator
        combined = "\n\n".join(block.strip() for block in large_blocks)
        return _prepend_imports(combined)

    # Default: use first large block
    return _prepend_imports(large_blocks[0].strip())