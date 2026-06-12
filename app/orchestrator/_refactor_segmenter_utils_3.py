# Purpose: refactor segmenter utils 3
# Called-by: app.orchestrator.refactor_segmenter
# Depends-on: app.orchestrator.refactor_segmenter_models
# Last-renovated: 2026-06-11
from __future__ import annotations
import logging
import os
import re
from app.orchestrator.refactor_segmenter_models import FileNode, SegmentPlan, Symbol, SymbolKind
from typing import Dict, List, Optional, Set
logger = logging.getLogger(__name__)


MAX_SEGMENT_ESTIMATED_LINES = 1500

MAX_SEGMENT_FILES = 6

_BUILTIN_NAMES = frozenset({
    "None", "True", "False", "self", "cls", "args", "kwargs",
    "str", "int", "float", "bool", "list", "dict", "set", "tuple",
    "len", "range", "print", "type", "isinstance", "hasattr", "getattr",
    "setattr", "enumerate", "zip", "map", "filter", "sorted", "reversed",
    "any", "all", "min", "max", "sum", "abs", "round",
    "open", "super", "property", "staticmethod", "classmethod",
    "Exception", "ValueError", "TypeError", "KeyError", "AttributeError",
    "RuntimeError", "OSError", "IOError", "FileNotFoundError",
    "NotImplementedError", "StopIteration", "IndexError",
    "logger", "logging", "os", "re", "json", "time", "datetime",
    "timezone", "Path", "Optional", "List", "Dict", "Set", "Tuple",
    "Any", "Callable", "Union", "Literal",
})

def _find_references(body: str, all_names: Set[str], own_name: str) -> List[str]:
    """
    Find references to other symbols within a function/class/constant body.

    Uses word-boundary regex matching against known symbol names.
    Filters out builtins and the symbol's own name.
    """
    if not body:
        return []

    refs: Set[str] = set()
    for name in all_names:
        if name == own_name:
            continue
        if name in _BUILTIN_NAMES:
            continue
        if len(name) < 2:
            continue
        # Word boundary match — symbol name must appear as a whole word
        # Use re.escape in case name has special chars
        pattern = r'\b' + re.escape(name) + r'\b'
        if re.search(pattern, body):
            refs.add(name)

    return sorted(refs)

def _build_file_keyword_index(
    nodes: Dict[str, FileNode],
    architecture_text: str,
) -> Dict[str, Set[str]]:
    """
    Build a keyword set for each file from its stem and architecture description.

    Returns: {file_path: {keyword1, keyword2, ...}}
    """
    index: Dict[str, Set[str]] = {}

    for fp, node in nodes.items():
        keywords: Set[str] = set()

        # From stem: "_detection" → {"detect", "detection"}
        stem_clean = node.file_stem.lstrip("_").lower()
        keywords.add(stem_clean)
        # Add root forms
        if stem_clean.endswith("ion"):
            keywords.add(stem_clean[:-3] + "e")  # detection → detecte... not great
            keywords.add(stem_clean[:-3])         # detection → detect
        if stem_clean.endswith("ing"):
            keywords.add(stem_clean[:-3])         # parsing → pars
            keywords.add(stem_clean[:-3] + "e")   # parsing → parse
        if stem_clean.endswith("tion"):
            keywords.add(stem_clean[:-4])          # execution → execu
            keywords.add(stem_clean[:-4] + "te")   # execution → execute
        if stem_clean.endswith("ation"):
            keywords.add(stem_clean[:-5] + "e")    # validation → validate
            keywords.add(stem_clean[:-5])           # validation → valid

        # From description
        desc_words = set(re.findall(r'[a-z]{3,}', node.description.lower()))
        keywords.update(desc_words)

        index[fp] = keywords

    return index

def _match_by_semantic_affinity(
    symbol: Symbol,
    nodes: Dict[str, FileNode],
    file_keywords: Dict[str, Set[str]],
) -> Optional[str]:
    """
    Match a symbol to a file based on name/kind affinity with file keywords.

    Scoring:
      - Exact stem word match in symbol name: +3
      - Root form match: +2
      - Kind match (constants → config, classes → models): +2
      - Description word overlap: +1
    """
    name_lower = symbol.name.lower()
    name_words = set(re.findall(r'[a-z]{3,}', name_lower))

    best_score = 0
    best_file: Optional[str] = None

    for fp, node in nodes.items():
        if node.is_facade:
            continue

        score = 0
        keywords = file_keywords.get(fp, set())
        stem_clean = node.file_stem.lstrip("_").lower()

        # Exact stem in symbol name
        if stem_clean in name_lower:
            score += 3

        # Word overlap
        overlap = name_words & keywords
        score += len(overlap) * 2

        # Kind-based affinity
        if symbol.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE):
            if "config" in stem_clean or "constant" in stem_clean:
                score += 2
        if symbol.kind == SymbolKind.CLASS:
            if "model" in stem_clean or "schema" in stem_clean:
                score += 2

        # Description contains symbol name
        if symbol.name.lower() in node.description.lower():
            score += 3

        if score > best_score:
            best_score = score
            best_file = fp

    # Only assign if we have meaningful affinity
    if best_score >= 2:
        return best_file
    return None

def _make_segment(
    index: int,
    file_paths: List[str],
    tiers_included: List[int],
    nodes: Dict[str, FileNode],
    existing_segments: List[SegmentPlan],
    source_file: str,
    is_facade: bool = False,
) -> SegmentPlan:
    """Create a SegmentPlan with correct dependencies."""
    # Find which earlier segments this one depends on
    deps: Set[str] = set()
    for fp in file_paths:
        for dep_fp in nodes[fp].depends_on:
            # Which segment owns dep_fp?
            for seg in existing_segments:
                if dep_fp in seg.file_paths:
                    deps.add(seg.segment_id)

    # If facade, depend on all prior segments
    if is_facade:
        for seg in existing_segments:
            deps.add(seg.segment_id)

    # Generate readable segment ID
    # v6.1 FIX 18: Prepend source file stem to prevent name collisions
    # when multiple source files produce segments with similar structure
    # (e.g. both conduct_policy.py and sandbox_build_validator.py producing
    # seg-01-models-and-core). The source_file is always available.
    _src_stem = os.path.basename(source_file).replace(".py", "")
    # Abbreviate long stems (>20 chars) to keep segment IDs manageable
    if len(_src_stem) > 20:
        _src_prefix = _src_stem[:18]
    else:
        _src_prefix = _src_stem

    if is_facade:
        slug = "facade"
    else:
        stems = [nodes[fp].file_stem.lstrip("_") for fp in file_paths if not nodes[fp].is_facade]
        slug = "-and-".join(stems[:3])
        if len(stems) > 3:
            slug += f"-etc-{len(stems)}files"

    seg_id = f"seg-{index + 1:02d}-{_src_prefix}-{slug}"

    # Estimate total lines
    est_lines = sum(nodes[fp].estimated_lines for fp in file_paths)

    # Build title
    if is_facade:
        title = f"Facade re-exports — {len(file_paths)} file(s)"
    else:
        title = f"{', '.join(os.path.basename(fp) for fp in file_paths[:4])}"
        if len(file_paths) > 4:
            title += f" (+{len(file_paths) - 4} more)"

    return SegmentPlan(
        segment_index=index,
        segment_id=seg_id,
        title=title,
        file_paths=list(file_paths),
        tiers_included=tiers_included,
        dependencies=sorted(deps),
        estimated_lines=est_lines,
        estimated_files=len(file_paths),
        is_facade_segment=is_facade,
    )
