# FILE: app/pot_spec/grounded/typescript_parser.py
"""
Size Analyzer — TypeScript/JS Structure Parser.

Regex-based parsing of TypeScript and JavaScript files into SourceBlock
structures. Less precise than AST but sufficient for size estimation.

v1.0 (2026-02-14): Extracted from size_analyzer.py for module cap compliance.
"""

from __future__ import annotations

import re
from typing import List

from .size_models import SourceBlock


def parse_typescript_structure(source_code: str) -> List[SourceBlock]:
    """
    Parse a TypeScript/JS file into its exported blocks using regex.

    Identifies: export functions, export classes, export interfaces,
    and large unexported blocks.
    """
    blocks: List[SourceBlock] = []
    lines = source_code.split("\n")

    block_patterns = [
        (r"^\s*export\s+(?:async\s+)?function\s+(\w+)", "function"),
        (r"^\s*export\s+(?:default\s+)?class\s+(\w+)", "class"),
        (r"^\s*export\s+(?:default\s+)?interface\s+(\w+)", "interface"),
        (r"^\s*(?:async\s+)?function\s+(\w+)", "function"),
        (r"^\s*class\s+(\w+)", "class"),
    ]

    i = 0
    while i < len(lines):
        for pattern, kind in block_patterns:
            match = re.match(pattern, lines[i])
            if match:
                name = match.group(1)
                start = i + 1  # 1-indexed
                end = _find_block_end_braces(lines, i)
                blocks.append(SourceBlock(
                    name=name,
                    kind=kind,
                    start_line=start,
                    end_line=end + 1,
                    line_count=(end + 1) - start + 1,
                ))
                i = end + 1
                break
        else:
            i += 1

    return blocks


def _find_block_end_braces(lines: List[str], start_idx: int) -> int:
    """Find the closing brace that matches the first opening brace."""
    depth = 0
    found_open = False
    for i in range(start_idx, len(lines)):
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                found_open = True
            elif ch == "}":
                depth -= 1
                if found_open and depth == 0:
                    return i
    return len(lines) - 1
