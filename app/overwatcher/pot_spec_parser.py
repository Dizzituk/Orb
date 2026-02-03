"""POT Spec Parser: Parse POT (Plan of Tasks) markdown into atomic tasks.

POT specs contain grounded file paths, line numbers, and content changes
in a structured markdown format:

## Replace (case-preserving)
- `Orb` → `Astra`
- `ORB` → `ASTRA`

## Change (N matches)
### `FILE_PATH`
- L<line_number>: `<content>`

## Skip (N matches)
CATEGORY (N matches)
- FILE L<line_number>: `<content>`

This parser extracts Change items into executable atomic tasks for Overwatcher.

v1.0 (2026-02-02): Initial implementation
v2.0 (2026-02-03): Fixed three parsing bugs:
  - BUG 1: File paths formatted as '### `path`' were skipped because
    parse_change_section() excluded all lines starting with '#'.
    FIX: Detect '###' headers and extract path from them.
  - BUG 2: Replace section uses backtick-wrapped terms (`Orb` → `Astra`)
    but extract_refactor_terms() only handled quotes and bare words.
    FIX: Added backtick patterns and dedicated ## Replace section parser.
  - BUG 3: Line content wrapped in backticks (e.g. `<title>Orb</title>`)
    was stored with backticks included.
    FIX: Strip enclosing backticks from parsed content.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Build verification
POT_PARSER_BUILD_ID = "2026-02-03-v2.0-format-fix"
print(f"[POT_PARSER_LOADED] BUILD_ID={POT_PARSER_BUILD_ID}")


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class POTAtomicTask:
    """Single atomic task extracted from POT spec Change section.
    
    Represents one line change in one file.
    """
    file_path: str           # Full path: "D:\orb-desktop\src\components\Header.tsx"
    line_number: int         # Line number: 38
    original_content: str    # Content at that line: '<h1 className="app-title">Orb</h1>'
    search_term: str         # What to search for: "Orb"
    replace_term: str        # What to replace with: "Astra"
    task_id: str            # Unique ID for this task
    
    def __str__(self) -> str:
        return f"POTAtomicTask({self.file_path}:{self.line_number} | {self.search_term}→{self.replace_term})"
    
    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "original_content": self.original_content,
            "search_term": self.search_term,
            "replace_term": self.replace_term,
            "task_id": self.task_id,
        }


@dataclass
class POTParseResult:
    """Result of parsing a POT spec."""
    tasks: List[POTAtomicTask]
    change_count: int
    skip_count: int
    search_term: Optional[str] = None
    replace_term: Optional[str] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def is_valid(self) -> bool:
        """Check if parse was successful."""
        return len(self.tasks) > 0 and len(self.errors) == 0
    
    def to_dict(self) -> dict:
        return {
            "task_count": len(self.tasks),
            "change_count": self.change_count,
            "skip_count": self.skip_count,
            "search_term": self.search_term,
            "replace_term": self.replace_term,
            "is_valid": self.is_valid,
            "errors": self.errors,
        }


# =============================================================================
# Utility: Strip enclosing backticks
# =============================================================================

def strip_backticks(text: str) -> str:
    """Strip enclosing backticks from a string.
    
    Examples:
        '`Orb`'                     -> 'Orb'
        '`<title>Orb</title>`'      -> '<title>Orb</title>'
        'plain text'                -> 'plain text'
        '`D:\\orb-desktop\\main.js`' -> 'D:\\orb-desktop\\main.js'
    """
    text = text.strip()
    if text.startswith('`') and text.endswith('`') and len(text) >= 2:
        return text[1:-1]
    return text


# =============================================================================
# Search/Replace Term Extraction
# =============================================================================

def extract_replace_section_terms(markdown: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract search/replace terms from the ## Replace section.
    
    v2.0: This is the most reliable source for refactor terms because
    the Replace section is explicitly formatted:
        ## Replace (case-preserving)
        - `Orb` → `Astra`
        - `ORB` → `ASTRA`
        - `orb` → `astra`
    
    We take the FIRST entry as the canonical search/replace pair.
    
    Args:
        markdown: Full POT spec markdown
    
    Returns:
        Tuple of (search_term, replace_term) or (None, None)
    """
    replace_section = extract_section(markdown, "## Replace")
    if not replace_section:
        logger.debug("[extract_replace_section] No ## Replace section found")
        return None, None
    
    logger.info(f"[extract_replace_section] Found Replace section ({len(replace_section)} chars)")
    
    # Pattern: - `Orb` → `Astra`  or  - `Orb` -> `Astra`
    backtick_arrow = re.compile(r'-\s+`([^`]+)`\s*(?:→|->)\s*`([^`]+)`')
    
    for line in replace_section.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = backtick_arrow.search(line)
        if match:
            search_term = match.group(1).strip()
            replace_term = match.group(2).strip()
            logger.info(
                f"[extract_replace_section] Found: '{search_term}' → '{replace_term}'"
            )
            print(f"[POT_PARSER] v2.0 Replace section: '{search_term}' → '{replace_term}'")
            return search_term, replace_term
    
    logger.warning("[extract_replace_section] Replace section exists but no parseable entries")
    return None, None


def extract_refactor_terms(spec_content: str, markdown: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract search and replace terms from spec content.
    
    v2.0: Tries strategies in this priority order:
    1. Parse the ## Replace section in markdown (most reliable)
    2. Look for backtick-wrapped arrow patterns: `X` → `Y`
    3. Look for quoted arrow patterns: "X" → "Y"
    4. Look for bare arrow patterns: X → Y
    5. Look for title/summary refactor pattern: "Rename X to Y"
    6. Infer search term from common words in Change section
    
    Args:
        spec_content: Full spec content (JSON or markdown)
        markdown: POT markdown section
    
    Returns:
        Tuple of (search_term, replace_term) or (None, None)
    """
    # Strategy 0 (v2.0): Parse the ## Replace section directly
    search_term, replace_term = extract_replace_section_terms(markdown)
    if search_term and replace_term:
        return search_term, replace_term
    
    # Strategy 1: Look for explicit arrow pattern in spec_content
    # v2.0: Added backtick patterns first since POT specs use backticks
    arrow_patterns = [
        r'`([^`]+)`\s*(?:→|->)\s*`([^`]+)`',               # `Orb` → `Astra`
        r'["\']([^"\']+)["\']\s*→\s*["\']([^"\']+)["\']',   # "Orb" → "Astra"
        r'["\']([^"\']+)["\']\s*->\s*["\']([^"\']+)["\']',  # "Orb" -> "Astra"
        r'(\w+)\s*→\s*(\w+)',                                # Orb → Astra
        r'(\w+)\s*->\s*(\w+)',                               # Orb -> Astra
    ]
    
    # Search both spec_content and markdown
    for source_name, source_text in [("spec_content", spec_content), ("markdown", markdown)]:
        if not source_text:
            continue
        for pattern in arrow_patterns:
            match = re.search(pattern, source_text)
            if match:
                search_term = match.group(1).strip()
                replace_term = match.group(2).strip()
                logger.info(
                    f"[extract_terms] Found via arrow pattern in {source_name}: "
                    f"'{search_term}' → '{replace_term}'"
                )
                return search_term, replace_term
    
    # Strategy 2: Look in title/summary
    title_patterns = [
        r'(?:rename|replace|refactor|change)\s+[`"\']?(\w+)[`"\']?\s+(?:to|with|→|->)\s+[`"\']?(\w+)[`"\']?',
    ]
    
    for source_text in [spec_content, markdown]:
        if not source_text:
            continue
        for pattern in title_patterns:
            match = re.search(pattern, source_text, re.IGNORECASE)
            if match:
                search_term = match.group(1).strip()
                replace_term = match.group(2).strip()
                logger.info(f"[extract_terms] Found via title pattern: '{search_term}' → '{replace_term}'")
                return search_term, replace_term
    
    # Strategy 3: Infer from markdown content
    search_term = infer_search_term_from_changes(markdown)
    if search_term:
        logger.info(f"[extract_terms] Inferred search term: '{search_term}' (replace term unknown)")
        return search_term, None
    
    logger.warning("[extract_terms] Could not extract search/replace terms")
    return None, None


def infer_search_term_from_changes(markdown: str) -> Optional[str]:
    """
    Infer the search term by finding common substring in Change section.
    
    This is a fallback when explicit terms aren't provided.
    """
    # Extract all line contents from Change section
    change_section = extract_section(markdown, "## Change")
    if not change_section:
        return None
    
    # Find all line contents (with or without backticks)
    line_pattern = r'-\s+L\d+:\s+(.+)'
    lines = re.findall(line_pattern, change_section)
    
    if len(lines) < 2:
        return None
    
    # Find common words (simple approach - look for word that appears in all lines)
    from collections import Counter
    
    word_counts = Counter()
    for line in lines:
        # Strip backticks before extracting words
        line = strip_backticks(line)
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', line)
        for word in words:
            word_counts[word] += 1
    
    # Find word that appears in most lines
    if word_counts:
        most_common = word_counts.most_common(1)[0]
        if most_common[1] >= len(lines) * 0.5:  # Appears in at least 50% of lines
            logger.info(f"[infer_search_term] Common word: '{most_common[0]}' (appears {most_common[1]}/{len(lines)} times)")
            return most_common[0]
    
    return None


# =============================================================================
# Markdown Parsing
# =============================================================================

def extract_section(markdown: str, header: str) -> str:
    """
    Extract content between header and next ## header or end of string.
    
    Uses '^## ' (exactly two hashes + space) to detect section boundaries,
    which correctly ignores ### sub-headers within sections.
    
    Args:
        markdown: Full markdown content
        header: Section header to extract (e.g., "## Change")
    
    Returns:
        Section content (without the header line)
    """
    # Find the header
    header_pattern = re.escape(header) + r'.*?$'
    header_match = re.search(header_pattern, markdown, re.MULTILINE)
    
    if not header_match:
        return ""
    
    # Find start position (after the header line)
    start_pos = header_match.end()
    
    # Find next ## header (exactly 2 hashes + space, NOT ###) or end of string
    # Using negative lookahead to ensure it's exactly ## and not ###
    next_header_pattern = r'^##(?!#)\s+'
    next_match = re.search(next_header_pattern, markdown[start_pos:], re.MULTILINE)
    
    if next_match:
        end_pos = start_pos + next_match.start()
        return markdown[start_pos:end_pos].strip()
    else:
        return markdown[start_pos:].strip()


def parse_pot_spec_markdown(
    markdown: str,
    spec_content: Optional[str] = None,
    search_term: Optional[str] = None,
    replace_term: Optional[str] = None,
) -> POTParseResult:
    """
    Parse POT spec markdown into atomic tasks.
    
    Args:
        markdown: POT spec markdown content
        spec_content: Full spec content for extracting search/replace terms
        search_term: Explicit search term (overrides extraction)
        replace_term: Explicit replace term (overrides extraction)
    
    Returns:
        POTParseResult with tasks and metadata
    """
    tasks: List[POTAtomicTask] = []
    errors: List[str] = []
    
    logger.info("[parse_pot_markdown] v2.0 Starting parse...")
    print(f"[POT_PARSER] v2.0 parse_pot_spec_markdown called: markdown={len(markdown)} chars")
    
    # Extract search/replace terms if not provided
    if not search_term or not replace_term:
        extracted_search, extracted_replace = extract_refactor_terms(
            spec_content or markdown,
            markdown
        )
        search_term = search_term or extracted_search
        replace_term = replace_term or extracted_replace
    
    if not search_term:
        errors.append("Could not extract search term from spec")
        logger.warning("[parse_pot_markdown] No search term - tasks will have empty search_term")
    
    if not replace_term:
        # v2.0: This is now an error, not just a warning, since we have the Replace section parser
        errors.append("Could not extract replace term from spec")
        logger.error("[parse_pot_markdown] No replace term found - check ## Replace section")
        print(f"[POT_PARSER] ERROR: No replace term found!")
    
    print(f"[POT_PARSER] v2.0 Terms: search='{search_term}', replace='{replace_term}'")
    
    # Extract Change section
    change_section = extract_section(markdown, "## Change")
    if not change_section:
        errors.append("No '## Change' section found in markdown")
        logger.error("[parse_pot_markdown] No Change section found")
        print(f"[POT_PARSER] ERROR: No ## Change section found!")
        return POTParseResult(
            tasks=[],
            change_count=0,
            skip_count=0,
            search_term=search_term,
            replace_term=replace_term,
            errors=errors,
        )
    
    print(f"[POT_PARSER] v2.0 Change section: {len(change_section)} chars")
    logger.info(f"[parse_pot_markdown] Change section preview: {change_section[:300]}...")
    
    # Extract Skip section for counting
    skip_section = extract_section(markdown, "## Skip")
    skip_count = len(re.findall(r'-\s+', skip_section)) if skip_section else 0
    
    # Parse Change section into tasks
    tasks = parse_change_section(
        change_section,
        search_term or "",
        replace_term or "",
    )
    
    print(f"[POT_PARSER] v2.0 RESULT: {len(tasks)} tasks, {skip_count} skips")
    logger.info(
        f"[parse_pot_markdown] Parsed {len(tasks)} tasks, {skip_count} skips, "
        f"search='{search_term}', replace='{replace_term}'"
    )
    
    return POTParseResult(
        tasks=tasks,
        change_count=len(tasks),
        skip_count=skip_count,
        search_term=search_term,
        replace_term=replace_term,
        errors=errors if not tasks else [],  # Clear errors if we got tasks
    )


def parse_change_section(
    change_section: str,
    search_term: str,
    replace_term: str,
) -> List[POTAtomicTask]:
    """
    Parse Change section into atomic tasks.
    
    v2.0: Handles two markdown formats:
    
    Format A (### headers with backtick-wrapped paths and content):
        ### `FILE_PATH`
        - L<line_number>: `<content>`
    
    Format B (bare file paths):
        FILE_PATH
        - L<line_number>: <content>
    
    Args:
        change_section: Content of ## Change section
        search_term: What to search for
        replace_term: What to replace with
    
    Returns:
        List of POTAtomicTask objects
    """
    tasks: List[POTAtomicTask] = []
    current_file: Optional[str] = None
    task_counter = 0
    
    for line in change_section.split('\n'):
        line = line.strip()
        
        if not line:
            continue
        
        # v2.0 FIX (BUG 1): Handle ### `filepath` format
        # The actual POT markdown uses level-3 headers for file paths:
        #   ### `D:\orb-desktop\index.html`
        # The v1.0 parser skipped these because they start with '#'
        if line.startswith('###'):
            # Extract file path: strip ### prefix, whitespace, and backticks
            path = line.lstrip('#').strip()
            path = strip_backticks(path)
            if path:
                current_file = path
                logger.info(f"[parse_change] v2.0 File path (### header): {current_file}")
                print(f"[POT_PARSER] File: {current_file}")
            continue
        
        # Original format: bare file path (not starting with '-' or '#')
        if not line.startswith('-') and not line.startswith('#'):
            current_file = strip_backticks(line.strip())
            logger.info(f"[parse_change] File path (bare): {current_file}")
            continue
        
        # Check if this is a content line: "- L<number>: <content>" or "- L<number>: `<content>`"
        if line.startswith('-') and current_file:
            # Parse: "- L38: `<h1 className="app-title">Orb</h1>`"
            # or:    "- L38: <h1 className="app-title">Orb</h1>"
            match = re.match(r'-\s+L(\d+):\s+(.+)', line)
            if match:
                line_number = int(match.group(1))
                # v2.0 FIX (BUG 3): Strip enclosing backticks from content
                original_content = strip_backticks(match.group(2).strip())
                
                task_counter += 1
                task_id = f"pot-task-{task_counter:04d}"
                
                task = POTAtomicTask(
                    file_path=current_file,
                    line_number=line_number,
                    original_content=original_content,
                    search_term=search_term,
                    replace_term=replace_term,
                    task_id=task_id,
                )
                
                tasks.append(task)
                logger.info(f"[parse_change] v2.0 Task: {task}")
                print(f"[POT_PARSER]   Task {task_id}: L{line_number} ({current_file})")
            else:
                logger.warning(f"[parse_change] Could not parse line: {line}")
    
    logger.info(f"[parse_change] v2.0 Parsed {len(tasks)} tasks from {change_section.count(chr(10))+1} lines")
    print(f"[POT_PARSER] v2.0 parse_change_section: {len(tasks)} tasks extracted")
    return tasks


# =============================================================================
# Detection
# =============================================================================

def is_pot_spec_format(content: str) -> bool:
    """
    Detect if content is in POT spec markdown format.
    
    POT specs have distinctive markdown headers:
    - ## Change (N matches)
    - ## Skip (N matches)
    
    Args:
        content: Spec content to check
    
    Returns:
        True if this looks like a POT spec
    """
    if not content or not isinstance(content, str):
        return False
    
    # Check for POT markdown headers
    has_change = '## Change' in content
    has_skip = '## Skip' in content
    
    # Must have at least the Change section
    is_pot = has_change
    
    if is_pot:
        logger.info(f"[is_pot_spec] Detected POT format (has_change={has_change}, has_skip={has_skip})")
    
    return is_pot


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "POTAtomicTask",
    "POTParseResult",
    "parse_pot_spec_markdown",
    "is_pot_spec_format",
    "extract_refactor_terms",
    "extract_replace_section_terms",
]
