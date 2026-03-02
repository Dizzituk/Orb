# FILE: app/llm/pipeline/critique_parts/size_checks.py
"""
Deterministic Critique — Size Estimation Check.

Check 7: Size estimation check
    Estimated output size from file count × average lines won't
    exceed the 30KB file limit. Uses architecture's File Inventory
    and code block sizes to estimate.

Zero LLM calls. Pure arithmetic.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SIZE_CHECKS_BUILD_ID = "2026-02-27-v1.0-size-estimation"

# Defaults for estimation
DEFAULT_AVG_LINES_PER_FILE = 150
BYTES_PER_LINE_ESTIMATE = 45  # Average Python line ~45 bytes
MAX_FILE_SIZE_KB = 30.0
TARGET_FILE_SIZE_KB = 20.0


# =========================================================================
# CHECK 7: Size Estimation
# =========================================================================

def _count_arch_files(arch_content: str) -> int:
    """Count files listed in architecture File Inventory."""
    inv_match = re.search(r'(?:^|\n)#+\s*File Inventory', arch_content)
    if not inv_match:
        return 0

    inv_start = inv_match.start()
    inv_end_match = re.search(r'\n(?:##[^#]|---)', arch_content[inv_start + 20:])
    if inv_end_match:
        inv_section = arch_content[inv_start:inv_start + 20 + inv_end_match.start()]
    else:
        inv_section = arch_content[inv_start:inv_start + 3000]

    count = 0
    for line in inv_section.split("\n"):
        if not line.strip().startswith("|") or line.strip().startswith("|---"):
            continue
        # Check for a file path pattern
        if re.search(r'`[\w/\\._-]+\.(?:py|ts|tsx|js|jsx|json)`', line):
            count += 1

    return count


def _estimate_from_code_blocks(arch_content: str) -> Dict[str, int]:
    """
    Estimate per-file sizes from code blocks in architecture.

    Returns {file_hint: estimated_bytes} for each code block.
    """
    estimates: Dict[str, int] = {}

    # Pattern: ```python\n# FILE: path\n...\n```
    for m in re.finditer(
        r'```(?:python)?\s*\n#\s*FILE:\s*(.+?)\n(.*?)```',
        arch_content, re.DOTALL,
    ):
        path = m.group(1).strip()
        content = m.group(2)
        if path and content.strip():
            estimates[path] = len(content.encode("utf-8"))

    # Pattern: **File: path** ... ```python\n...\n```
    for m in re.finditer(
        r'\*\*File:\s*(.+?)\*\*.*?```(?:python)?\s*\n(.*?)```',
        arch_content, re.DOTALL,
    ):
        path = m.group(1).strip()
        content = m.group(2)
        if path and content.strip() and path not in estimates:
            estimates[path] = len(content.encode("utf-8"))

    return estimates


def check_size_estimation(
    arch_content: str,
    max_size_kb: float = MAX_FILE_SIZE_KB,
    target_size_kb: float = TARGET_FILE_SIZE_KB,
) -> List[Dict[str, Any]]:
    """
    Estimate output file sizes from architecture and flag potential
    oversize files.

    Uses two methods:
    1. If architecture contains code blocks with # FILE: headers,
       measure their actual size
    2. Otherwise, estimate from file count × average line length

    Args:
        arch_content: Architecture markdown document
        max_size_kb: Hard limit per file in KB (blocker)
        target_size_kb: Target limit per file in KB (warning)

    Returns:
        List of issue dicts
    """
    issues: List[Dict[str, Any]] = []

    # Method 1: Measure actual code blocks
    block_estimates = _estimate_from_code_blocks(arch_content)

    for file_path, size_bytes in block_estimates.items():
        size_kb = size_bytes / 1024
        if size_kb > max_size_kb:
            issues.append({
                "rule_id": "DET-SIZE-OVERSIZE",
                "severity": "blocking",
                "file": file_path,
                "spec_ref": "modularity_constraints",
                "arch_ref": f"Code block for {file_path}",
                "description": (
                    f"Code block for '{file_path}' is {size_kb:.1f}KB, "
                    f"exceeding the {max_size_kb}KB maximum. The final "
                    f"file will likely be even larger with full implementation."
                ),
                "suggested_fix": (
                    f"Split '{file_path}' into smaller modules. Target "
                    f"{target_size_kb}KB per file."
                ),
            })
        elif size_kb > target_size_kb:
            issues.append({
                "rule_id": "DET-SIZE-WARNING",
                "severity": "warning",
                "file": file_path,
                "spec_ref": "modularity_constraints",
                "arch_ref": f"Code block for {file_path}",
                "description": (
                    f"Code block for '{file_path}' is {size_kb:.1f}KB, "
                    f"exceeding the {target_size_kb}KB target. Full "
                    f"implementation may exceed {max_size_kb}KB."
                ),
                "suggested_fix": (
                    f"Consider splitting '{file_path}' proactively."
                ),
            })

    # Method 2: Overall estimation if no code blocks
    if not block_estimates:
        file_count = _count_arch_files(arch_content)
        if file_count == 0:
            return issues

        # Estimate: if there's only 1 file and it has many sections,
        # the output might be too large
        # Count function definitions as a proxy for complexity
        func_count = len(re.findall(
            r'(?:async\s+)?def\s+\w+\s*\(', arch_content
        ))
        class_count = len(re.findall(r'class\s+\w+\s*[(:[]', arch_content))

        if file_count == 1 and (func_count + class_count) > 15:
            est_lines = (func_count * 25) + (class_count * 50)
            est_kb = (est_lines * BYTES_PER_LINE_ESTIMATE) / 1024
            if est_kb > max_size_kb:
                issues.append({
                    "rule_id": "DET-SIZE-ESTIMATE",
                    "severity": "warning",
                    "file": "single output file",
                    "spec_ref": "modularity_constraints",
                    "arch_ref": "File Inventory (1 file)",
                    "description": (
                        f"Architecture defines {func_count} functions and "
                        f"{class_count} classes in 1 file. Estimated size "
                        f"~{est_kb:.0f}KB may exceed {max_size_kb}KB limit."
                    ),
                    "suggested_fix": (
                        f"Consider splitting into multiple files to stay "
                        f"under {target_size_kb}KB per file."
                    ),
                })

    if issues:
        logger.info(
            "[det_critique] Size estimation: %d issues",
            len(issues),
        )

    return issues


__all__ = [
    "check_size_estimation",
    "SIZE_CHECKS_BUILD_ID",
]
