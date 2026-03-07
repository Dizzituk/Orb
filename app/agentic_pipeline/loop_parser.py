# FILE: app/agentic_pipeline/loop_parser.py
"""
Agentic Loop Output Parser.

Parses the multi-segment output from the agentic loop model.
Each segment's architecture doc is delimited by:

  === SEGMENT: seg-01-backend-debug-project-data-mod ===
  [ARCHITECTURE DOC WITH COMPLETE CODE BLOCKS]
  === END SEGMENT: seg-01-backend-debug-project-data-mod ===

Also extracts code blocks from individual architecture docs
for the deterministic check runner and extraction stage.

v1.0 (2026-03-05): Initial implementation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SEGMENT_START_RE = re.compile(r"^===\s*SEGMENT:\s*(\S+)\s*===\s*$", re.MULTILINE)
_SEGMENT_END_RE = re.compile(r"^===\s*END\s+SEGMENT:\s*(\S+)\s*===\s*$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)

_FILE_COMMENT_PATTERNS = [
    re.compile(r"^\s*//\s*((?:src|app|orb-desktop)[/\\][\w/\\._-]+\.\w+)", re.MULTILINE),
    re.compile(r"^\s*#\s*((?:src|app|orb-desktop)[/\\][\w/\\._-]+\.\w+)", re.MULTILINE),
    re.compile(r"//\s*FILE:\s*(.+\.\w+)", re.MULTILINE),
    re.compile(r"#\s*FILE:\s*(.+\.\w+)", re.MULTILINE),
    re.compile(r"/\*\s*FILE:\s*(.+\.\w+)\s*\*/", re.MULTILINE),
]

_INVENTORY_ROW_RE = re.compile(r"\|\s*`([^`]+\.\w+)`\s*\|")


@dataclass
class ParsedSegment:
    """A single parsed segment from the agentic loop output."""
    segment_id: str
    content: str
    code_blocks: Dict[str, str] = field(default_factory=dict)
    file_inventory: List[str] = field(default_factory=list)
    is_no_changes: bool = False


@dataclass
class ParseResult:
    """Complete parse of the agentic loop output."""
    segments: Dict[str, ParsedSegment] = field(default_factory=dict)
    unparsed_content: str = ""
    parse_errors: List[str] = field(default_factory=list)

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def get_all_arch_docs(self) -> Dict[str, str]:
        return {sid: seg.content for sid, seg in self.segments.items()}

    def get_all_code_blocks(self) -> Dict[str, Dict[str, str]]:
        return {sid: seg.code_blocks for sid, seg in self.segments.items()}


def parse_agentic_output(
    raw_output: str,
    expected_segment_ids: Optional[List[str]] = None,
) -> ParseResult:
    """Parse the multi-segment output from the agentic loop."""
    result = ParseResult()

    if not raw_output or not raw_output.strip():
        result.parse_errors.append("Empty output from agentic loop")
        return result

    starts = list(_SEGMENT_START_RE.finditer(raw_output))

    if not starts:
        result.unparsed_content = raw_output
        result.parse_errors.append(
            "No segment delimiters found. Expected === SEGMENT: <id> === markers."
        )
        return result

    for i, start_match in enumerate(starts):
        seg_id = start_match.group(1)
        content_start = start_match.end()

        end_pattern = re.compile(
            rf"^===\s*END\s+SEGMENT:\s*{re.escape(seg_id)}\s*===\s*$",
            re.MULTILINE,
        )
        end_match = end_pattern.search(raw_output, content_start)

        if end_match:
            content = raw_output[content_start:end_match.start()].strip()
        else:
            if i + 1 < len(starts):
                content = raw_output[content_start:starts[i + 1].start()].strip()
            else:
                content = raw_output[content_start:].strip()
            result.parse_errors.append(f"Missing end delimiter for segment '{seg_id}'")

        parsed = _parse_single_segment(seg_id, content)
        result.segments[seg_id] = parsed

    if expected_segment_ids:
        missing = set(expected_segment_ids) - set(result.segments.keys())
        if missing:
            result.parse_errors.append(f"Missing segments in output: {sorted(missing)}")

    logger.info("[loop_parser] Parsed %d segments, %d errors", result.segment_count, len(result.parse_errors))
    return result


def _parse_single_segment(segment_id: str, content: str) -> ParsedSegment:
    parsed = ParsedSegment(segment_id=segment_id, content=content)
    if "NO_CHANGES_NEEDED" in content:
        parsed.is_no_changes = True
        return parsed
    parsed.file_inventory = _extract_file_inventory(content)
    parsed.code_blocks = extract_code_blocks_from_arch(content)
    return parsed


def extract_code_blocks_from_arch(arch_content: str) -> Dict[str, str]:
    """Extract code blocks from an architecture doc, mapped to file paths."""
    blocks: Dict[str, List[str]] = {}

    header_positions: List[tuple] = []
    for m in re.finditer(r"^###?\s+`?([^`\n]+\.\w+)`?\s*$", arch_content, re.MULTILINE):
        path = m.group(1).strip().replace("\\", "/")
        header_positions.append((m.start(), path))

    for fence_match in _CODE_FENCE_RE.finditer(arch_content):
        code = fence_match.group(2).strip()
        if not code or len(code) < 10:
            continue

        file_path = _extract_file_hint(code)
        if not file_path:
            file_path = _nearest_header(header_positions, fence_match.start())
        if not file_path:
            continue

        norm = file_path.replace("\\", "/")
        blocks.setdefault(norm, []).append(code)

    return {path: "\n\n".join(code_list) for path, code_list in blocks.items()}


def _extract_file_hint(code: str) -> Optional[str]:
    first_lines = "\n".join(code.split("\n")[:5])
    for pattern in _FILE_COMMENT_PATTERNS:
        m = pattern.search(first_lines)
        if m:
            path = m.group(1).strip()
            if "." in path and ("/" in path or "\\" in path):
                return path.replace("\\", "/")
    return None


def _nearest_header(header_positions: List[tuple], fence_pos: int) -> Optional[str]:
    best_path = None
    best_dist = float("inf")
    for pos, path in header_positions:
        if pos < fence_pos:
            dist = fence_pos - pos
            if dist < best_dist:
                best_dist = dist
                best_path = path
    if best_path and best_dist < 2000:
        return best_path
    return None


def _extract_file_inventory(arch_content: str) -> List[str]:
    paths: List[str] = []
    inv_match = re.search(r"(?:^|\n)#+\s*File Inventory", arch_content)
    if not inv_match:
        return paths
    inv_start = inv_match.start()
    inv_end_match = re.search(r"\n(?:##[^#]|---)", arch_content[inv_start + 20:])
    if inv_end_match:
        section = arch_content[inv_start:inv_start + 20 + inv_end_match.start()]
    else:
        section = arch_content[inv_start:inv_start + 3000]
    for m in _INVENTORY_ROW_RE.finditer(section):
        path = m.group(1).strip().replace("\\", "/")
        if path not in paths:
            paths.append(path)
    return paths
