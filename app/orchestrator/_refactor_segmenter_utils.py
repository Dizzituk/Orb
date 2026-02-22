import json
import logging
import os
import re
from app.orchestrator.refactor_segmenter_models import FileNode, RefactorBuildPlan, Symbol, SymbolKind
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


REFACTOR_SEGMENTER_BUILD_ID = "2026-02-21-v1.1-fix18-source-prefix-segment-names"

_DATA_CONSTANT_PATTERNS = [
    re.compile(r"^[A-Z][A-Z_]*_PROMPT$"),      # DIAGNOSTIC_SYSTEM_PROMPT etc
    re.compile(r"^[A-Z][A-Z_]*_PATTERNS?$"),    # FILE_PATH_PATTERNS, ERROR_TYPE_PATTERNS
    re.compile(r"^[A-Z][A-Z_]*_COMMANDS?$"),    # ALLOWED_FIX_COMMANDS
    re.compile(r"^[A-Z][A-Z_]*_TEMPLATE$"),     # prompt templates
]

def _parse_line_range(line_range: Any) -> Tuple[int, int]:
    """Parse a line range like '444-585' or [444, 585] into (start, end)."""
    if not line_range:
        return 0, 0
    if isinstance(line_range, (list, tuple)) and len(line_range) == 2:
        return int(line_range[0]), int(line_range[1])
    if isinstance(line_range, str) and "-" in line_range:
        parts = line_range.split("-")
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except (ValueError, IndexError):
            return 0, 0
    return 0, 0

def _match_by_architecture_mention(
    symbol_name: str,
    nodes: Dict[str, FileNode],
    architecture_text: str,
) -> Optional[str]:
    """
    Check if the architecture explicitly mentions a symbol in a file's section.

    Looks for patterns like:
      ## _detection.py
      ... detect_project_from_path ...
    """
    # Find file sections in architecture
    current_file: Optional[str] = None
    for line in architecture_text.split("\n"):
        # Check for file section headers
        for fp, node in nodes.items():
            basename = os.path.basename(fp)
            if basename in line and re.match(r'#{1,4}\s', line.strip()):
                current_file = fp
                break

        if current_file and symbol_name in line:
            # Verify it's a real mention (not just substring)
            pattern = r'\b' + re.escape(symbol_name) + r'\b'
            if re.search(pattern, line):
                return current_file

    return None

def _match_by_reference_clustering(
    symbol: Symbol,
    assigned_names: Dict[str, str],
    nodes: Dict[str, FileNode],
) -> Optional[str]:
    """
    Assign symbol to the file where most of its references live.
    """
    if not symbol.references:
        return None

    file_ref_counts: Dict[str, int] = {}
    for ref_name in symbol.references:
        if ref_name in assigned_names:
            fp = assigned_names[ref_name]
            file_ref_counts[fp] = file_ref_counts.get(fp, 0) + 1

    if not file_ref_counts:
        return None

    # Pick file with most references, break ties by file with fewer symbols
    best_fp = max(
        file_ref_counts,
        key=lambda fp: (file_ref_counts[fp], -len(nodes[fp].symbols)),
    )

    # Only assign if there are at least 2 references in the same file
    if file_ref_counts[best_fp] >= 2:
        return best_fp

    return None

def _assign_remainder(
    symbol: Symbol,
    nodes: Dict[str, FileNode],
) -> Optional[str]:
    """
    Last resort: assign to the smallest non-facade file.

    Constants and data structures → prefer config/constants file.
    Functions → prefer the file with the fewest symbols.
    """
    candidates = [
        (fp, node) for fp, node in nodes.items()
        if not node.is_facade
    ]
    if not candidates:
        return None

    # Constants prefer config
    if symbol.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE):
        for fp, node in candidates:
            stem = node.file_stem.lstrip("_").lower()
            if "config" in stem or "constant" in stem:
                return fp

    # Otherwise → smallest file
    candidates.sort(key=lambda x: len(x[1].symbols))
    return candidates[0][0]

def _find_common_prefix(names: List[str]) -> str:
    """Find the longest common prefix among a list of names."""
    if not names:
        return ""
    prefix = names[0]
    for name in names[1:]:
        while not name.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix

def save_build_plan(plan: RefactorBuildPlan, job_dir: str) -> str:
    """Persist build plan to disk for observability."""
    out_dir = os.path.join(job_dir, "segments")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "refactor_build_plan.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)

    logger.info("[refactor_segmenter] Build plan saved: %s", path)
    return path
