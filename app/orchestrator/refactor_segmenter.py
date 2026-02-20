# FILE: app/orchestrator/refactor_segmenter.py
"""
Deterministic Refactor Segmenter.

Replaces LLM-driven segmentation for refactor jobs with a scan-based
approach that:
  1. Parses enrichment data to extract symbols and their references
  2. Maps symbols to target files using architecture file descriptions
  3. Builds a file-level dependency graph
  4. Topologically sorts files into build tiers (leaves first)
  5. Groups tiers into segments that produce a standard manifest

Zero LLM calls. The entire segmentation and ordering is deterministic.

BUILD_ID: 2026-02-20-v1.0-deterministic-refactor-segmenter
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.orchestrator.refactor_segmenter_models import (
    DependencyGraph,
    FileNode,
    RefactorBuildPlan,
    SegmentPlan,
    Symbol,
    SymbolKind,
)

logger = logging.getLogger(__name__)

REFACTOR_SEGMENTER_BUILD_ID = "2026-02-20-v1.0-deterministic-refactor-segmenter"
print(f"[REFACTOR_SEGMENTER_LOADED] BUILD_ID={REFACTOR_SEGMENTER_BUILD_ID}")


# =============================================================================
# CONSTANTS
# =============================================================================

# Segment size targets
MAX_SEGMENT_ESTIMATED_LINES = 1500
MAX_SEGMENT_FILES = 6

# Symbols that are never "references" — stdlib builtins, common names
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

# Data-heavy constant patterns — these contain data, not logic
_DATA_CONSTANT_PATTERNS = [
    re.compile(r"^[A-Z][A-Z_]*_PROMPT$"),      # DIAGNOSTIC_SYSTEM_PROMPT etc
    re.compile(r"^[A-Z][A-Z_]*_PATTERNS?$"),    # FILE_PATH_PATTERNS, ERROR_TYPE_PATTERNS
    re.compile(r"^[A-Z][A-Z_]*_COMMANDS?$"),    # ALLOWED_FIX_COMMANDS
    re.compile(r"^[A-Z][A-Z_]*_TEMPLATE$"),     # prompt templates
]


# =============================================================================
# STEP 1: EXTRACT SYMBOLS FROM ENRICHMENT
# =============================================================================

def extract_symbols(enrichment: Dict[str, Any]) -> List[Symbol]:
    """
    Extract all symbols from enrichment data.

    Parses functions, classes, and constants into Symbol objects
    with their internal references detected.
    """
    if not enrichment:
        return []

    symbols: List[Symbol] = []
    source_extract = enrichment.get("source_extract", {})
    all_symbol_names: Set[str] = set()

    # Collect all names first (needed for reference detection)
    for func in enrichment.get("functions", []):
        all_symbol_names.add(func.get("name", ""))
    for cls in enrichment.get("classes", []):
        all_symbol_names.add(cls.get("name", ""))
    for const in enrichment.get("constants", []):
        all_symbol_names.add(const.get("name", ""))
    # Also add source_extract keys that might be constants
    for key in source_extract:
        all_symbol_names.add(key)
    all_symbol_names.discard("")

    # --- Functions ---
    for func in enrichment.get("functions", []):
        name = func.get("name", "")
        if not name:
            continue

        body = func.get("body", "")
        # Also get from source_extract if body is empty/stub
        if (not body or body.strip() in ("...", "pass")) and name in source_extract:
            body = source_extract[name]

        line_range = func.get("line_range", "")
        line_start, line_end = _parse_line_range(line_range)

        refs = _find_references(body, all_symbol_names, name)

        kind = SymbolKind.ASYNC_FUNCTION if func.get("is_async") else SymbolKind.FUNCTION

        symbols.append(Symbol(
            name=name,
            kind=kind,
            line_start=line_start,
            line_end=line_end,
            char_count=len(body),
            references=refs,
            is_private=name.startswith("_") and not name.startswith("__"),
            is_dunder=name.startswith("__") and name.endswith("__"),
        ))

    # --- Classes ---
    for cls in enrichment.get("classes", []):
        name = cls.get("name", "")
        if not name:
            continue

        body = ""
        if name in source_extract:
            body = source_extract[name]

        refs = _find_references(body, all_symbol_names, name) if body else []

        symbols.append(Symbol(
            name=name,
            kind=SymbolKind.CLASS,
            char_count=len(body),
            references=refs,
        ))

    # --- Constants ---
    for const in enrichment.get("constants", []):
        name = const.get("name", "")
        if not name:
            continue

        value_str = const.get("value", "")
        body = ""
        if name in source_extract:
            body = source_extract[name]

        # Classify: is this a simple constant or a data structure?
        kind = SymbolKind.CONSTANT
        if any(p.match(name) for p in _DATA_CONSTANT_PATTERNS):
            kind = SymbolKind.DATA_STRUCTURE
        elif value_str and len(value_str) > 200:
            kind = SymbolKind.DATA_STRUCTURE

        refs = _find_references(body or value_str, all_symbol_names, name)

        line_number = const.get("line_number", 0)
        # Estimate end line from value length
        est_lines = max(1, len(value_str) // 80) if value_str else 1

        symbols.append(Symbol(
            name=name,
            kind=kind,
            line_start=line_number,
            line_end=line_number + est_lines - 1,
            char_count=len(body or value_str),
            references=refs,
        ))

    logger.info(
        "[refactor_segmenter] Extracted %d symbols (%d functions, %d classes, %d constants)",
        len(symbols),
        sum(1 for s in symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)),
        sum(1 for s in symbols if s.kind == SymbolKind.CLASS),
        sum(1 for s in symbols if s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE)),
    )
    return symbols


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


# =============================================================================
# STEP 2: BUILD FILE NODES FROM ARCHITECTURE
# =============================================================================

def build_file_nodes(
    architecture_text: str,
    target_package: str,
) -> Dict[str, FileNode]:
    """
    Parse the architecture's File Inventory to create FileNode objects.

    Each row in the File Inventory becomes a FileNode with its
    file_path, stem, and description.
    """
    nodes: Dict[str, FileNode] = {}

    # Parse File Inventory markdown table
    in_inventory = False
    past_header = False

    for line in architecture_text.split("\n"):
        stripped = line.strip()

        # Detect section start
        if re.match(r'#{1,4}\s*.*[Ff]ile\s*[Ii]nventory', stripped):
            in_inventory = True
            past_header = False
            continue

        # Detect section end
        if in_inventory and stripped.startswith('#') and past_header:
            in_inventory = False
            continue

        if not in_inventory or not stripped.startswith('|'):
            continue

        # Skip separator and header rows
        if re.match(r'\|[-\s|]+\|', stripped):
            past_header = True
            continue
        if 'File' in stripped and 'Purpose' in stripped:
            continue

        # Extract file path and description
        match = re.search(r'\|\s*`([^`]+)`\s*\|\s*([^|]+)', stripped)
        if not match:
            continue

        file_path = match.group(1).strip()
        description = match.group(2).strip()

        if not file_path or file_path.lower() == 'file':
            continue

        # Extract stem
        basename = os.path.basename(file_path)
        stem = os.path.splitext(basename)[0]

        is_facade = basename == "__init__.py"

        nodes[file_path] = FileNode(
            file_path=file_path,
            file_stem=stem,
            description=description,
            is_facade=is_facade,
        )

    logger.info(
        "[refactor_segmenter] Built %d file nodes from architecture",
        len(nodes),
    )
    return nodes


# =============================================================================
# STEP 3: ASSIGN SYMBOLS TO FILES
# =============================================================================

def assign_symbols_to_files(
    symbols: List[Symbol],
    nodes: Dict[str, FileNode],
    architecture_text: str,
) -> List[Symbol]:
    """
    Assign each symbol to its best-matching target file.

    Uses a multi-pass strategy:
      Pass 1: Architecture description matching — check if the architecture
              explicitly mentions a symbol name in a file's design section
      Pass 2: Semantic affinity — match symbol kind + name patterns to
              file descriptions (e.g. "detection" functions → _detection.py)
      Pass 3: Reference clustering — symbols that reference each other
              should be in the same file when possible
      Pass 4: Remainder → assign to the most appropriate non-facade file

    Returns list of symbols that couldn't be assigned.
    """
    unassigned: List[Symbol] = []
    assigned_names: Dict[str, str] = {}  # symbol_name → file_path

    # Build description keyword index for each file
    file_keywords = _build_file_keyword_index(nodes, architecture_text)

    # --- Pass 1: Architecture mentions ---
    # Check if the architecture explicitly names a symbol in a file's section
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _match_by_architecture_mention(
            symbol.name, nodes, architecture_text,
        )
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file

    pass1_count = len(assigned_names)
    logger.info("[refactor_segmenter] Pass 1 (arch mention): assigned %d symbols", pass1_count)

    # --- Pass 2: Semantic affinity ---
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _match_by_semantic_affinity(symbol, nodes, file_keywords)
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file

    pass2_count = len(assigned_names) - pass1_count
    logger.info("[refactor_segmenter] Pass 2 (semantic): assigned %d symbols", pass2_count)

    # --- Pass 3: Reference clustering ---
    # Symbols that reference already-assigned symbols go to the same file
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _match_by_reference_clustering(symbol, assigned_names, nodes)
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file

    pass3_count = len(assigned_names) - pass1_count - pass2_count
    logger.info("[refactor_segmenter] Pass 3 (reference clustering): assigned %d symbols", pass3_count)

    # --- Pass 4: Remainder ---
    for symbol in symbols:
        if symbol.name in assigned_names:
            continue

        best_file = _assign_remainder(symbol, nodes)
        if best_file:
            nodes[best_file].symbols.append(symbol)
            assigned_names[symbol.name] = best_file
        else:
            unassigned.append(symbol)

    pass4_count = len(assigned_names) - pass1_count - pass2_count - pass3_count
    logger.info("[refactor_segmenter] Pass 4 (remainder): assigned %d, unassigned %d",
                pass4_count, len(unassigned))

    return unassigned


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


# =============================================================================
# STEP 4: BUILD DEPENDENCY EDGES
# =============================================================================

def build_dependency_edges(
    nodes: Dict[str, FileNode],
) -> None:
    """
    Build file-level dependency edges from symbol references.

    If file A contains a symbol that references a symbol in file B,
    then A depends on B.

    Mutates nodes in place — sets depends_on and depended_by.
    """
    # Build symbol → file_path lookup
    symbol_to_file: Dict[str, str] = {}
    for fp, node in nodes.items():
        for sym in node.symbols:
            symbol_to_file[sym.name] = fp

    # Build edges
    for fp, node in nodes.items():
        for sym in node.symbols:
            for ref_name in sym.references:
                ref_file = symbol_to_file.get(ref_name)
                if ref_file and ref_file != fp:
                    node.depends_on.add(ref_file)
                    nodes[ref_file].depended_by.add(fp)

    # Log edge summary
    total_edges = sum(len(n.depends_on) for n in nodes.values())
    logger.info(
        "[refactor_segmenter] Built %d dependency edges across %d files",
        total_edges, len(nodes),
    )


# =============================================================================
# STEP 5: TOPOLOGICAL SORT INTO TIERS
# =============================================================================

def sort_into_tiers(nodes: Dict[str, FileNode]) -> Tuple[List[List[str]], bool]:
    """
    Topologically sort files into build tiers.

    Tier 0: files with no internal dependencies (leaves)
    Tier 1: files that only depend on tier 0
    Tier N: files that depend on tiers 0..N-1

    Returns (tiers, cycle_detected).
    """
    # Build edges dict for Kahn's algorithm
    remaining = dict(nodes)
    tiers: List[List[str]] = []
    assigned: Set[str] = set()

    max_iterations = len(nodes) + 1
    iteration = 0

    while remaining and iteration < max_iterations:
        iteration += 1

        # Find files whose dependencies are all already assigned
        current_tier: List[str] = []
        for fp, node in remaining.items():
            unmet = node.depends_on - assigned
            if not unmet:
                current_tier.append(fp)

        if not current_tier:
            # Cycle detected — remaining nodes all have unmet deps
            logger.error(
                "[refactor_segmenter] Cycle detected! Remaining: %s",
                list(remaining.keys()),
            )
            # Break cycle: add the node with fewest unmet deps
            best = min(remaining, key=lambda fp: len(remaining[fp].depends_on - assigned))
            current_tier = [best]
            logger.warning(
                "[refactor_segmenter] Breaking cycle by forcing %s into tier %d",
                best, len(tiers),
            )

        current_tier.sort()  # Deterministic ordering
        tiers.append(current_tier)

        for fp in current_tier:
            assigned.add(fp)
            nodes[fp].tier = len(tiers) - 1
            del remaining[fp]

    cycle_detected = iteration >= max_iterations
    if cycle_detected:
        logger.error("[refactor_segmenter] Topological sort did not converge")

    # Log tier summary
    for i, tier in enumerate(tiers):
        tier_files = ", ".join(os.path.basename(fp) for fp in tier)
        logger.info("[refactor_segmenter] Tier %d: %s", i, tier_files)

    return tiers, cycle_detected


# =============================================================================
# STEP 6: GROUP TIERS INTO SEGMENTS
# =============================================================================

def group_tiers_into_segments(
    tiers: List[List[str]],
    nodes: Dict[str, FileNode],
    source_file: str,
) -> List[SegmentPlan]:
    """
    Group tiers into build segments.

    Rules:
      - Adjacent tiers can be merged if combined size < MAX_SEGMENT_ESTIMATED_LINES
        and combined files < MAX_SEGMENT_FILES
      - Facade files always get their own segment (last)
      - Each segment declares dependencies on earlier segments
    """
    segments: List[SegmentPlan] = []
    seg_index = 0

    # Separate facade tiers
    facade_tiers: List[int] = []
    logic_tiers: List[int] = []
    for i, tier_files in enumerate(tiers):
        if all(nodes[fp].is_facade for fp in tier_files):
            facade_tiers.append(i)
        else:
            logic_tiers.append(i)

    # Group logic tiers
    current_files: List[str] = []
    current_tiers: List[int] = []
    current_lines = 0

    for tier_idx in logic_tiers:
        tier_files = tiers[tier_idx]
        tier_lines = sum(nodes[fp].estimated_lines for fp in tier_files)

        # Check if adding this tier would exceed limits
        if current_files and (
            current_lines + tier_lines > MAX_SEGMENT_ESTIMATED_LINES
            or len(current_files) + len(tier_files) > MAX_SEGMENT_FILES
        ):
            # Flush current segment
            segments.append(_make_segment(
                seg_index, current_files, current_tiers,
                nodes, segments, source_file,
            ))
            seg_index += 1
            current_files = []
            current_tiers = []
            current_lines = 0

        current_files.extend(tier_files)
        current_tiers.append(tier_idx)
        current_lines += tier_lines

    # Flush remaining logic files
    if current_files:
        segments.append(_make_segment(
            seg_index, current_files, current_tiers,
            nodes, segments, source_file,
        ))
        seg_index += 1

    # Add facade segment(s)
    for tier_idx in facade_tiers:
        tier_files = tiers[tier_idx]
        segments.append(_make_segment(
            seg_index, tier_files, [tier_idx],
            nodes, segments, source_file,
            is_facade=True,
        ))
        seg_index += 1

    logger.info(
        "[refactor_segmenter] Grouped %d tiers into %d segments",
        len(tiers), len(segments),
    )
    return segments


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
    if is_facade:
        slug = "facade"
    else:
        stems = [nodes[fp].file_stem.lstrip("_") for fp in file_paths if not nodes[fp].is_facade]
        slug = "-and-".join(stems[:3])
        if len(stems) > 3:
            slug += f"-etc-{len(stems)}files"

    seg_id = f"seg-{index + 1:02d}-{slug}"

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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def build_refactor_plan(
    enrichment: Dict[str, Any],
    architecture_text: str,
    source_file: str,
    target_package: str,
    facade_file: Optional[str] = None,
) -> RefactorBuildPlan:
    """
    Build a complete deterministic refactor plan.

    This is the main entry point. Takes enrichment data and architecture
    text, returns a RefactorBuildPlan that can be converted to a standard
    manifest.

    Args:
        enrichment: Enrichment data with functions, classes, constants, source_extract
        architecture_text: The architecture document with File Inventory
        source_file: Path to the source monolith being refactored
        target_package: Package directory for the refactored files
        facade_file: Path to __init__.py (auto-detected if not provided)
    """
    plan = RefactorBuildPlan(
        source_file=source_file,
        target_package=target_package,
    )

    # Step 1: Extract symbols
    symbols = extract_symbols(enrichment)
    if not symbols:
        plan.warnings.append("No symbols extracted from enrichment")
        return plan

    # Step 2: Build file nodes from architecture
    nodes = build_file_nodes(architecture_text, target_package)
    if not nodes:
        plan.warnings.append("No files found in architecture File Inventory")
        return plan

    # Detect facade
    if facade_file:
        plan.facade_file = facade_file
    else:
        for fp, node in nodes.items():
            if node.is_facade:
                plan.facade_file = fp
                break

    # Collect public symbols (non-private, non-dunder)
    plan.public_symbols = [
        s.name for s in symbols
        if not s.is_private and not s.is_dunder
    ]

    # Step 3: Assign symbols to files
    unassigned = assign_symbols_to_files(symbols, nodes, architecture_text)
    plan.graph.unassigned_symbols = unassigned

    # Mark data-only files
    for node in nodes.values():
        if node.symbols and all(
            s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE)
            for s in node.symbols
        ):
            node.is_data_only = True

    # Step 4: Build dependency edges
    build_dependency_edges(nodes)

    # Step 5: Topological sort
    tiers, cycle = sort_into_tiers(nodes)
    plan.graph.nodes = nodes
    plan.graph.tiers = tiers
    plan.graph.cycle_detected = cycle

    # Step 6: Group into segments
    plan.segments = group_tiers_into_segments(tiers, nodes, source_file)

    # Summary warnings
    if unassigned:
        plan.warnings.append(
            f"{len(unassigned)} symbol(s) could not be assigned to any file: "
            + ", ".join(s.name for s in unassigned)
        )
    if cycle:
        plan.warnings.append("Cycle detected in dependency graph — build order may be suboptimal")

    empty_files = [fp for fp, n in nodes.items() if not n.symbols and not n.is_facade]
    if empty_files:
        plan.warnings.append(
            f"{len(empty_files)} non-facade file(s) have no symbols assigned: "
            + ", ".join(os.path.basename(fp) for fp in empty_files)
        )

    logger.info(
        "[refactor_segmenter] Build plan complete: %d symbols → %d files → %d tiers → %d segments",
        len(symbols), len(nodes), len(tiers), len(plan.segments),
    )

    return plan


# =============================================================================
# PERSISTENCE
# =============================================================================

def save_build_plan(plan: RefactorBuildPlan, job_dir: str) -> str:
    """Persist build plan to disk for observability."""
    out_dir = os.path.join(job_dir, "segments")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "refactor_build_plan.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_dict(), f, indent=2)

    logger.info("[refactor_segmenter] Build plan saved: %s", path)
    return path
