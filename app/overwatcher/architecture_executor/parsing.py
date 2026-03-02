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

# v3.2: Phrases in a MODIFY section body that indicate "no actual changes"
_NO_CHANGE_PHRASES = [
    'no functional change',
    'no changes required',
    'no change required',
    'no modifications required',
    'no modification required',
    'keep unchanged',
    'retain as stable',
    'leave unchanged',
    'unchanged unless',
    'no change needed',
    'kept as-is',
    'keep as-is',
    'no edits needed',
    'no edits required',
]


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
    # Any invalid filesystem character -> reject
    if _INVALID_PATH_CHARS & set(path):
        logger.debug("[parsing] v5.22 Skipped invalid path: %s", path)
        return False
    # Must contain at least one dot (file extension) or slash (directory)
    if '.' not in path and '/' not in path and '\\' not in path:
        return False
    return True


def _section_says_no_change(architecture: str, file_path: str) -> bool:
    """v3.2: Check if the architecture says 'no changes needed' for a file.

    Uses two strategies:
    1. Find the section headed with the file path (## N) `path` (MODIFY))
       and check its body for no-change phrases.
    2. Scan the File Inventory table for no-change descriptions.

    This catches the contradiction where a file appears in the MODIFY
    inventory but the architecture explicitly says leave it alone.
    """
    filename = Path(file_path).name
    path_normalised = file_path.replace('\\', '/')
    arch_lower = architecture.lower()

    # Strategy 1: Find the MODIFY header section for this specific file
    # Pattern: ## N) `path` (MODIFY) ... until next ## header
    escaped_path = re.escape(path_normalised)
    section_pattern = re.compile(
        r'^##\s+\d+\)\s+[`"]?' + escaped_path + r'[`"]?\s*\(MODIFY\)'
        r'(.*?)(?=^##\s+\d+\)|\Z)',
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    section_match = section_pattern.search(architecture)
    if section_match:
        body = section_match.group(1).lower()
        for phrase in _NO_CHANGE_PHRASES:
            if phrase in body:
                logger.info(
                    "[parsing] v3.2 Stripped no-change MODIFY file: %s "
                    "(section match: '%s')", file_path, phrase,
                )
                return True

    # Strategy 2: Check File Inventory table row description
    # Pattern: | `path` | No functional change... |
    for line in architecture.split('\n'):
        if path_normalised in line or filename in line:
            line_lower = line.lower()
            for phrase in _NO_CHANGE_PHRASES:
                if phrase in line_lower:
                    logger.info(
                        "[parsing] v3.2 Stripped no-change MODIFY file: %s "
                        "(inventory row match: '%s')", file_path, phrase,
                    )
                    return True

    return False


def parse_file_inventory(architecture: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Parse the File Inventory section of an architecture document.

    v3.2: Modified files whose section body contains "no change" phrases
    are automatically stripped from the inventory to prevent unnecessary
    and potentially destructive rewrites.

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

    # v3.2: Strip modified files whose section body says "no changes needed".
    # This prevents the implementer from rewriting files the architecture
    # explicitly said to leave alone (e.g. models.py with existing enums).
    if modified_files:
        before_count = len(modified_files)
        modified_files = [
            f for f in modified_files
            if not _section_says_no_change(architecture, f["path"])
        ]
        stripped = before_count - len(modified_files)
        if stripped:
            print(
                f"[ARCH_EXEC] v3.2 Stripped {stripped} no-change MODIFY file(s) "
                f"from inventory ({len(modified_files)} remain)"
            )

    return new_files, modified_files


def extract_section_for_file(architecture: str, file_path: str) -> str:
    """
    Extract the relevant section from an architecture document for a given file path.

    v3.2-fix: Tracks fenced code blocks (``` markers) so that lines inside
    code fences are never misidentified as markdown headers. This prevents
    Python comments like ``# app/education/models.py`` from being treated
    as depth-1 headers, which previously caused section boundaries to break.

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
    in_code_fence = False

    for i, line in enumerate(lines):
        # v3.2-fix: Track fenced code blocks -- toggle on ``` boundary lines.
        # Lines inside code fences must never be treated as markdown headers,
        # because Python comments like "# filename.py" and "## Imports" look
        # like markdown headers to the regex but are actually code.
        if line.startswith('```'):
            in_code_fence = not in_code_fence
            if in_relevant_section:
                relevant_section.append(line)
            continue

        # Inside a code fence: capture if in section, but never parse as header
        if in_code_fence:
            if in_relevant_section:
                relevant_section.append(line)
            continue

        # Check if line is a header (only outside code fences)
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


# =============================================================================
# v1.3: SSE PREAMBLE STRIPPING
# =============================================================================

# Known SSE progress markers streamed by the Critical Pipeline before
# the actual architecture content.  These must never appear in code.
_SSE_MARKERS = frozenset([
    '\u2699\ufe0f',   # ⚙️
    '\U0001f4cb',     # 📋
    '\u2705',         # ✅
    '\U0001f9e9',     # 🧩
    '\U0001f9e0',     # 🧠
    '\U0001f3f7\ufe0f', # 🏷️
    '\U0001f3d7\ufe0f', # 🏗️
    '\U0001f4c1',     # 📁
    '\U0001f4da',     # 📚
    '\U0001f527',     # 🔧
    '\U0001f50d',     # 🔍
    '\U0001f4e6',     # 📦
    '\u26a0\ufe0f',   # ⚠️
])

# Valid code opening patterns — the first line matching one of these
# marks the start of real content.  Language-agnostic.
_CODE_START_RE = re.compile(
    r'^\s*('
    r'import\s|from\s|export\s|const\s|let\s|var\s|'
    r'class\s|def\s|async\s+def\s|async\s+function\s|function\s|'
    r'interface\s|type\s|enum\s|'
    r'#!|#\s|//\s|/\*|"""|\'\'\'|'
    r'@|'
    r'[A-Z_][A-Z_0-9]*\s*='
    r')',
)


def _strip_sse_preamble(content: str, rel_path: str = "") -> str:
    """Remove SSE stream headers that contaminate verbatim extraction.

    v1.3 (2026-03-01): The Critical Pipeline streams emoji-laden progress
    messages (⚙️ **Critical Pipeline**, 📋 **Loading validated spec...**,
    ✅ **Pipeline Complete**) before the actual architecture content.
    When verbatim extraction pulls code from these sections, the SSE
    preamble gets baked into the source file.

    This function scans for the first line matching a valid code opening
    pattern and discards everything above it.  Known SSE markers and
    markdown horizontal rules are explicitly skipped.

    Args:
        content: Extracted code that may contain SSE preamble.
        rel_path: For logging context.

    Returns:
        The content with any SSE preamble removed.
    """
    if not content:
        return content

    lines = content.split('\n')
    preamble_end = 0
    found_code = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Blank lines before code are fine
        if not stripped:
            continue

        # Known SSE marker — skip
        if any(stripped.startswith(marker) for marker in _SSE_MARKERS):
            continue

        # Markdown horizontal rule — skip
        if stripped.startswith('---') or stripped.startswith('***'):
            cleaned = stripped.replace('-', '').replace('*', '').strip()
            if not cleaned:
                continue

        # Markdown bold/heading — skip (architecture prose, not code)
        if re.match(r'^#{1,6}\s', stripped) or re.match(r'^\*\*', stripped):
            continue

        # FILE: header comment sometimes precedes the SSE block
        if stripped.startswith('// FILE:') or stripped.startswith('# FILE:'):
            continue

        # Check if this looks like valid code
        if _CODE_START_RE.match(stripped):
            found_code = True
            preamble_end = i
            break

        # Unknown line before any code — could be SSE text without emoji
        # (e.g. "Loading validated spec...").  Keep scanning.
        continue

    if not found_code:
        # No recognisable code start found — return unchanged and let
        # downstream sanitisation / syntax checks handle it.
        return content

    if preamble_end > 0:
        stripped_lines = lines[:preamble_end]
        stripped_text = '\n'.join(l.strip() for l in stripped_lines if l.strip())
        if stripped_text:
            logger.warning(
                "[parsing] v1.3 SSE_STRIP: Removed %d lines of SSE preamble "
                "from %s: %s",
                preamble_end, rel_path, stripped_text[:150],
            )
        return '\n'.join(lines[preamble_end:])

    return content


def _extract_verbatim_code_from_architecture(file_context: str, rel_path: str) -> Optional[str]:
    """
    Extract verbatim code blocks from architecture context.

    This helper implements the "one large block" heuristic:
    - Extract all fenced code blocks
    - If one block is >500 chars and matches file extension, use it
    - If multiple large blocks exist, only combine when context indicates "complete file"

    v1.3: Applies SSE preamble stripping to all extracted content.

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
        block_lines = [l for l in stripped.split('\n') if l.strip()]
        import_lines = sum(1 for l in block_lines
                          if l.strip().startswith(('import ', 'from ')))
        if block_lines and import_lines / len(block_lines) > 0.5:
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

    # v1.3: Apply SSE strip to the final result before returning.
    def _finalise(code: str) -> str:
        return _strip_sse_preamble(_prepend_imports(code), rel_path)

    # If single large block, use it
    if len(large_blocks) == 1:
        return _finalise(large_blocks[0].strip())

    # No large content blocks but imports exist -- not enough for verbatim
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
        return _finalise(combined)

    # Default: use first large block
    return _finalise(large_blocks[0].strip())
