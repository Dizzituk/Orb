"""
Architecture Code Block Extractor.

Parses architecture markdown documents and extracts complete, production-ready
code blocks mapped to their target file paths. This replaces the limited
heuristic in parsing._extract_verbatim_code_from_architecture with a
comprehensive extraction that handles:

  - Multiple code blocks per file (imports + body)
  - File path detection from headers, comments, and file scope sections
  - SSE preamble stripping (delegates to parsing._strip_sse_preamble)
  - Small but complete files (no 500-char minimum)

v1.0 (2026-03-02): Initial implementation for code block extraction system.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .parsing import extract_section_for_file, _strip_sse_preamble

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedCodeBlock:
    """A single fenced code block extracted from an architecture document."""
    language: str
    content: str
    char_count: int
    source_file_hint: Optional[str] = None
    is_import_block: bool = False


@dataclass
class FileExtraction:
    """All extracted code for a single target file."""
    file_path: str
    blocks: List[ExtractedCodeBlock] = field(default_factory=list)
    merged_content: Optional[str] = None
    confidence: float = 0.0

    @property
    def total_chars(self) -> int:
        return sum(b.char_count for b in self.blocks)

    @property
    def has_content(self) -> bool:
        return bool(self.merged_content and len(self.merged_content.strip()) > 10)


@dataclass
class ExtractionResult:
    """Result of extracting code blocks from an architecture document."""
    extractions: Dict[str, FileExtraction] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def file_count(self) -> int:
        return sum(1 for e in self.extractions.values() if e.has_content)

    def get_content_for_file(self, file_path: str) -> Optional[str]:
        """Get merged content for a file path, trying normalised variants."""
        norm = file_path.replace("\\", "/")
        ext = self.extractions.get(norm)
        if ext and ext.has_content:
            return ext.merged_content
        # Try without leading prefix (orb-desktop/)
        if "/" in norm:
            short = "/".join(norm.split("/")[1:])
            ext = self.extractions.get(short)
            if ext and ext.has_content:
                return ext.merged_content
        return None


# ---------------------------------------------------------------------------
# Language-to-extension mapping
# ---------------------------------------------------------------------------

_LANG_EXTENSION_MAP: Dict[str, set] = {
    "typescript": {".ts", ".tsx"},
    "tsx": {".tsx"},
    "ts": {".ts", ".tsx"},
    "javascript": {".js", ".jsx"},
    "jsx": {".jsx"},
    "js": {".js", ".jsx"},
    "python": {".py"},
    "py": {".py"},
    "css": {".css"},
    "scss": {".scss"},
    "json": {".json"},
    "yaml": {".yaml", ".yml"},
    "yml": {".yaml", ".yml"},
    "toml": {".toml"},
    "html": {".html"},
    "markdown": {".md"},
    "md": {".md"},
    "sql": {".sql"},
}


def _lang_matches_extension(lang: str, file_path: str) -> bool:
    """Check if a code fence language tag is compatible with a file extension."""
    if not lang:
        return True  # No language tag = could be anything
    ext = Path(file_path).suffix.lower()
    valid_exts = _LANG_EXTENSION_MAP.get(lang.lower(), set())
    if not valid_exts:
        return True  # Unknown language tag = don't reject
    return ext in valid_exts


# ---------------------------------------------------------------------------
# Code block parsing
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(
    r'```(\w+)?\s*\n(.*?)```',
    re.DOTALL,
)

_FILE_COMMENT_PATTERNS = [
    # // src/components/education/EducationView.tsx
    re.compile(r'^\s*//\s*(.+\.\w+)\s*$', re.MULTILINE),
    # # app/orchestrator/scaffold/integration.py
    re.compile(r'^\s*#\s*(.+\.\w+)\s*$', re.MULTILINE),
    # /* FILE: src/styles/main.css */
    re.compile(r'/\*\s*FILE:\s*(.+\.\w+)\s*\*/', re.MULTILINE),
    # // FILE: src/components/Foo.tsx
    re.compile(r'//\s*FILE:\s*(.+\.\w+)', re.MULTILINE),
    # # FILE: app/models.py
    re.compile(r'#\s*FILE:\s*(.+\.\w+)', re.MULTILINE),
]


def _extract_file_hint_from_code(code: str) -> Optional[str]:
    """Try to find a file path hint in the first few lines of a code block."""
    # Only check first 5 lines for path comments
    first_lines = "\n".join(code.split("\n")[:5])
    for pattern in _FILE_COMMENT_PATTERNS:
        m = pattern.search(first_lines)
        if m:
            path = m.group(1).strip()
            # Sanity: must have a dot (extension)
            if "." in path and "/" in path:
                return path.replace("\\", "/")
    return None


def _classify_block(block_content: str) -> bool:
    """Determine if a code block is primarily an import block."""
    lines = [l for l in block_content.strip().split("\n") if l.strip()]
    if not lines:
        return False
    import_count = sum(
        1 for l in lines
        if l.strip().startswith(("import ", "from ", "export "))
        or l.strip().startswith("} from ")
    )
    # If majority of lines are imports, classify as import block
    return len(lines) > 0 and (import_count / len(lines)) > 0.5


def _parse_all_code_blocks(text: str) -> List[ExtractedCodeBlock]:
    """Extract all fenced code blocks from markdown text."""
    blocks = []
    for m in _CODE_FENCE_RE.finditer(text):
        lang = m.group(1) or ""
        content = m.group(2)
        if not content or not content.strip():
            continue
        stripped = content.strip()
        hint = _extract_file_hint_from_code(stripped)
        is_import = _classify_block(stripped)
        blocks.append(ExtractedCodeBlock(
            language=lang.lower(),
            content=stripped,
            char_count=len(stripped),
            source_file_hint=hint,
            is_import_block=is_import,
        ))
    return blocks


# ---------------------------------------------------------------------------
# Section-based extraction (primary strategy)
# ---------------------------------------------------------------------------

def _extract_for_file_from_section(
    architecture_content: str,
    file_path: str,
) -> Optional[FileExtraction]:
    """Extract code blocks from the architecture section for a specific file.

    This is the primary extraction strategy: find the section for the file
    in the arch doc, pull all code blocks, filter by language compatibility,
    and merge them in order (imports first, then content blocks).
    """
    section = extract_section_for_file(architecture_content, file_path)
    if not section:
        return None

    blocks = _parse_all_code_blocks(section)
    if not blocks:
        return None

    # Filter blocks by language compatibility
    compatible = [
        b for b in blocks
        if _lang_matches_extension(b.language, file_path)
    ]
    if not compatible:
        # Fall back to all blocks if none match language
        compatible = blocks

    extraction = FileExtraction(file_path=file_path.replace("\\", "/"))
    extraction.blocks = compatible

    # Merge: imports first, then content blocks in order
    import_blocks = [b for b in compatible if b.is_import_block]
    content_blocks = [b for b in compatible if not b.is_import_block]

    parts = []
    for b in import_blocks:
        parts.append(b.content)
    for b in content_blocks:
        parts.append(b.content)

    if parts:
        merged = "\n\n".join(parts)
        # Apply SSE preamble stripping
        merged = _strip_sse_preamble(merged, file_path)
        extraction.merged_content = merged

        # Confidence scoring
        extraction.confidence = _compute_confidence(extraction, section)

    return extraction


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

# Minimum chars for high confidence (complete file likely)
_MIN_COMPLETE_FILE_CHARS = 50

# Indicators in arch section that code is the complete file
_COMPLETE_FILE_INDICATORS = [
    "complete file",
    "full content",
    "verbatim",
    "entire file",
    "whole file",
    "code structure",
    "complete implementation",
]


def _compute_confidence(extraction: FileExtraction, section: str) -> float:
    """Score how confident we are the extracted code is the complete file.

    Returns 0.0 to 1.0:
      - 1.0: Very high confidence (single large block, complete file indicators)
      - 0.7: Good confidence (large block, language match)
      - 0.5: Moderate (multiple blocks merged)
      - 0.3: Low (small blocks, no indicators)
    """
    if not extraction.has_content:
        return 0.0

    score = 0.3  # Base

    content_blocks = [b for b in extraction.blocks if not b.is_import_block]
    total_chars = extraction.total_chars

    # Size boost
    if total_chars > 2000:
        score += 0.3
    elif total_chars > 500:
        score += 0.2
    elif total_chars > _MIN_COMPLETE_FILE_CHARS:
        score += 0.1

    # Single content block boost (less ambiguity)
    if len(content_blocks) == 1:
        score += 0.1

    # Complete file indicators in section text
    section_lower = section.lower()
    if any(ind in section_lower for ind in _COMPLETE_FILE_INDICATORS):
        score += 0.2

    # File hint in code matches target path
    for b in extraction.blocks:
        if b.source_file_hint:
            norm_hint = b.source_file_hint.replace("\\", "/")
            norm_path = extraction.file_path.replace("\\", "/")
            if norm_hint in norm_path or norm_path in norm_hint:
                score += 0.1
                break

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Minimum confidence to use extracted code directly (skip LLM)
DIRECT_USE_THRESHOLD = 0.6

# Minimum confidence to use as pre-fill (LLM verifies)
PREFILL_THRESHOLD = 0.3


def extract_code_for_files(
    architecture_content: str,
    file_paths: List[str],
) -> ExtractionResult:
    """Extract code blocks from an architecture document for multiple files.

    This is the main entry point. For each file path, it:
    1. Finds the relevant section in the architecture document
    2. Extracts all fenced code blocks from that section
    3. Merges them (imports first, then content)
    4. Scores confidence

    Args:
        architecture_content: Full architecture markdown document.
        file_paths: List of file paths to extract code for.

    Returns:
        ExtractionResult with extractions keyed by normalised file path.
    """
    result = ExtractionResult()

    for file_path in file_paths:
        norm_path = file_path.replace("\\", "/")
        try:
            extraction = _extract_for_file_from_section(
                architecture_content, file_path,
            )
            if extraction and extraction.has_content:
                result.extractions[norm_path] = extraction
                logger.info(
                    "[arch_extractor] Extracted %d chars for %s "
                    "(confidence=%.2f, blocks=%d)",
                    extraction.total_chars, norm_path,
                    extraction.confidence, len(extraction.blocks),
                )
            else:
                logger.debug(
                    "[arch_extractor] No extractable code for %s", norm_path,
                )
        except Exception as e:
            warning = f"Extraction failed for {norm_path}: {e}"
            result.warnings.append(warning)
            logger.warning("[arch_extractor] %s", warning)

    logger.info(
        "[arch_extractor] Extraction complete: %d/%d files have code",
        result.file_count, len(file_paths),
    )
    return result


def get_extraction_for_task(
    extraction_result: Optional[ExtractionResult],
    file_path: str,
) -> Tuple[Optional[str], float]:
    """Look up extracted code for a specific file task.

    Returns (content, confidence) or (None, 0.0) if no extraction exists.
    """
    if extraction_result is None:
        return None, 0.0

    norm = file_path.replace("\\", "/")
    ext = extraction_result.extractions.get(norm)
    if ext and ext.has_content:
        return ext.merged_content, ext.confidence

    # Try content lookup which handles prefix variants
    content = extraction_result.get_content_for_file(file_path)
    if content:
        return content, 0.5  # Lower confidence for prefix-variant match

    return None, 0.0