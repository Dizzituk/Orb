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
# Post-code tail stripping
# ---------------------------------------------------------------------------

# Patterns that indicate arch-doc prose leaked into extracted code.
# These appear after the actual code ends.
_PROSE_TAIL_INDICATORS = [
    re.compile(r'^\s*[-\u2022]\s+\.\w[\w-]*\s+', re.MULTILINE),   # CSS class descriptions: "- .class-name ..."
    re.compile(r'^\s*###?\s+', re.MULTILINE),                       # Markdown headings
    re.compile(r'^\s*\*\*[A-Z]', re.MULTILINE),                     # Bold prose: "**Note:"
    re.compile(r'^\s*>\s+', re.MULTILINE),                          # Blockquotes
    re.compile(r'^\s*\|\s+', re.MULTILINE),                         # Markdown table rows
]


def _strip_post_code_tail(content: str, file_path: str) -> str:
    """Strip architecture prose that leaked after the actual code ends.

    The architecture extractor sometimes includes descriptive sections
    (CSS class descriptions, implementation notes, markdown prose) that
    appear after the last valid code statement. These cause parse errors
    when the implementer writes them to disk.

    Strategy per file type:
    - TSX/JSX/TS/JS: find last top-level closing brace or export default,
      truncate everything after
    - Python: find last def/class/return at indent 0, keep to end of that
      block
    - CSS: find last closing brace, truncate after
    - Other: check for prose indicators in trailing lines
    """
    if not content or len(content) < 20:
        return content

    ext = Path(file_path).suffix.lower()
    lines = content.split("\n")

    if ext in (".tsx", ".jsx", ".ts", ".js"):
        return _strip_tail_js(lines)
    elif ext == ".py":
        return _strip_tail_python(lines)
    elif ext in (".css", ".scss"):
        return _strip_tail_css(lines)
    else:
        return _strip_tail_generic(lines)


def _strip_tail_js(lines: List[str]) -> str:
    """For JS/TS/TSX/JSX: truncate after last top-level closing brace or export."""
    last_code_line = len(lines) - 1

    # Walk backwards to find last meaningful code line
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        # Valid code endings: }, };, );, export default, ]
        if stripped in ("}", "};", ");", "];", "]") or \
           stripped.startswith("export default") or \
           stripped.startswith("export {"):
            last_code_line = i
            break
        # If we hit prose indicators, this line is junk
        if any(p.match(lines[i]) for p in _PROSE_TAIL_INDICATORS):
            last_code_line = i - 1
            break
        # Otherwise it's probably code — keep it
        last_code_line = i
        break

    result = "\n".join(lines[:last_code_line + 1])
    trimmed = len(lines) - (last_code_line + 1)
    if trimmed > 0:
        logger.info(
            "[arch_extractor] Stripped %d tail lines from JS/TS file", trimmed,
        )
    return result


def _strip_tail_python(lines: List[str]) -> str:
    """For Python: truncate after last non-prose line."""
    last_code_line = len(lines) - 1

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if any(p.match(lines[i]) for p in _PROSE_TAIL_INDICATORS):
            last_code_line = i - 1
            continue
        # Found real code
        last_code_line = i
        break

    result = "\n".join(lines[:last_code_line + 1])
    trimmed = len(lines) - (last_code_line + 1)
    if trimmed > 0:
        logger.info(
            "[arch_extractor] Stripped %d tail lines from Python file", trimmed,
        )
    return result


def _strip_tail_css(lines: List[str]) -> str:
    """For CSS/SCSS: truncate after last closing brace."""
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            result = "\n".join(lines[:i + 1])
            trimmed = len(lines) - (i + 1)
            if trimmed > 0:
                logger.info(
                    "[arch_extractor] Stripped %d tail lines from CSS file", trimmed,
                )
            return result
    return "\n".join(lines)


def _strip_tail_generic(lines: List[str]) -> str:
    """For other file types: strip trailing prose indicators."""
    last_code_line = len(lines) - 1

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if any(p.match(lines[i]) for p in _PROSE_TAIL_INDICATORS):
            last_code_line = i - 1
            continue
        last_code_line = i
        break

    return "\n".join(lines[:last_code_line + 1])


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
        # Strip post-code tail content (CSS descriptions, prose after closing brace)
        merged = _strip_post_code_tail(merged, file_path)
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


def _fill_from_file_hint_scan(
    architecture_content: str,
    missing_paths: List[str],
    result: ExtractionResult,
) -> None:
    """v1.1 Fallback: scan all code blocks for FILE: hints matching missing paths.

    This catches code blocks that the section-based strategy missed because:
    - The LLM put code under a different heading structure
    - The fill was spliced into a template section the extractor didn't find
    - The code block exists but outside a recognised markdown section
    """
    all_blocks = _parse_all_code_blocks(architecture_content)
    if not all_blocks:
        return

    # Build a lookup: normalised hint -> block
    hint_blocks: Dict[str, List[ExtractedCodeBlock]] = {}
    for block in all_blocks:
        hint = block.source_file_hint
        if hint:
            norm_hint = hint.replace("\\", "/")
            # Strip common prefixes: D:/Orb/, app/, orb-desktop/
            for prefix in ("D:/Orb/", "d:/orb/", "D:\\Orb\\"):
                if norm_hint.startswith(prefix):
                    norm_hint = norm_hint[len(prefix):]
            hint_blocks.setdefault(norm_hint, []).append(block)

    for target_path in missing_paths:
        norm_target = target_path.replace("\\", "/")
        # Try exact match, then suffix match
        matched_blocks = hint_blocks.get(norm_target)
        if not matched_blocks:
            # Try stripping known prefixes from target too
            short_target = norm_target
            for prefix in ("app/", "src/", "orb-desktop/src/", "orb-desktop/"):
                if short_target.startswith(prefix):
                    short_target_alt = short_target[len(prefix):]
                    matched_blocks = hint_blocks.get(short_target_alt)
                    if matched_blocks:
                        break
        if not matched_blocks:
            # Suffix match: find any hint that ends with the target's filename
            target_filename = target_path.replace("\\", "/").rsplit("/", 1)[-1]
            for hint_path, blocks in hint_blocks.items():
                if hint_path.endswith("/" + target_filename) or hint_path == target_filename:
                    matched_blocks = blocks
                    break
        if not matched_blocks:
            continue

        # Filter by language compatibility
        compatible = [
            b for b in matched_blocks
            if _lang_matches_extension(b.language, target_path)
        ]
        if not compatible:
            compatible = matched_blocks

        extraction = FileExtraction(file_path=norm_target)
        extraction.blocks = compatible

        import_blocks = [b for b in compatible if b.is_import_block]
        content_blocks = [b for b in compatible if not b.is_import_block]

        parts = [b.content for b in import_blocks] + [b.content for b in content_blocks]
        if parts:
            merged = "\n\n".join(parts)
            merged = _strip_sse_preamble(merged, target_path)
            merged = _strip_post_code_tail(merged, target_path)
            extraction.merged_content = merged
            extraction.confidence = 0.7  # Hint-matched, slightly lower than section-based

            result.extractions[norm_target] = extraction
            logger.info(
                "[arch_extractor] v1.1 FALLBACK extracted %d chars for %s (hint match, blocks=%d)",
                len(merged), norm_target, len(compatible),
            )


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

    # --- Primary strategy: section-based extraction ---
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
        except Exception as e:
            warning = f"Extraction failed for {norm_path}: {e}"
            result.warnings.append(warning)
            logger.warning("[arch_extractor] %s", warning)

    # --- v1.1 Fallback: scan ALL code blocks for # FILE: / // FILE: hints ---
    # If section-based extraction missed files, scan the entire document for
    # code blocks that have a FILE: comment matching a target path.
    missing = [fp.replace("\\", "/") for fp in file_paths if fp.replace("\\", "/") not in result.extractions]
    if missing:
        logger.info("[arch_extractor] v1.1 Fallback: scanning for %d missing file(s) via FILE: hints", len(missing))
        _fill_from_file_hint_scan(architecture_content, missing, result)

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