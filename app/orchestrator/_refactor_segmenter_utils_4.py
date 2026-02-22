from __future__ import annotations
import logging
import os
import re
from app.orchestrator._refactor_segmenter_utils_2 import _find_common_prefix
from app.orchestrator.refactor_segmenter_models import FileNode, SymbolKind
from typing import Dict, List, Set, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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

def _auto_generate_file_layout(
    symbols: List["Symbol"],
    target_package: str,
    source_file: str,
) -> Dict[str, "FileNode"]:
    """
    Auto-generate a file layout when no architecture File Inventory exists.

    Groups symbols by kind into sensible files:
    - constants + dataclasses/namedtuples → models.py
    - Remaining functions split into groups by estimated size,
      keeping each file under ~500 lines. Groups named by
      the first function's prefix or 'core'.
    - __init__.py facade that re-exports everything.

    Returns dict of {file_path: FileNode}.
    """
    if not symbols:
        return {}

    nodes: Dict[str, FileNode] = {}
    pkg = target_package.rstrip("/")

    # Separate symbols by kind
    constants = [s for s in symbols if s.kind in (SymbolKind.CONSTANT, SymbolKind.DATA_STRUCTURE)]
    classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
    functions = [s for s in symbols if s.kind in (SymbolKind.FUNCTION, SymbolKind.ASYNC_FUNCTION)]

    # v6.1 FIX 26: Dual cap — lines AND brief char budget.
    # The LLM reliably handles briefs up to ~35KB. Beyond that it dumps
    # the entire source instead of writing only assigned symbols.
    # Both caps must be satisfied — whichever is hit first triggers a split.
    MAX_LINES_PER_FILE = 400
    MAX_BRIEF_CHARS = 35_000  # ~35KB of function body content per file

    # Models file: constants + classes — split if too large
    if constants or classes:
        model_syms = constants + classes
        model_lines = sum(s.estimated_lines or 20 for s in model_syms)
        model_chars = sum(s.char_count or 0 for s in model_syms)
        if model_lines > MAX_LINES_PER_FILE or model_chars > MAX_BRIEF_CHARS:
            # Split: data constants into constants.py, classes into models.py
            if constants and classes:
                const_path = f"{pkg}/constants.py"
                nodes[const_path] = FileNode(
                    file_path=const_path,
                    file_stem="constants",
                    description="Data constants and lookup tables",
                )
                models_path = f"{pkg}/models.py"
                nodes[models_path] = FileNode(
                    file_path=models_path,
                    file_stem="models",
                    description="Class definitions and data structures",
                )
                logger.info(
                    "[refactor_segmenter] FIX 25b: Split models (%d lines) into "
                    "constants.py (%d constants) + models.py (%d classes)",
                    model_lines, len(constants), len(classes),
                )
            # If only one kind but still too large, use size-based grouping
            elif len(model_syms) > 1:
                model_groups: List[List] = []
                _mg: List = []
                _ml = 0
                _mc = 0
                for s in model_syms:
                    sl = s.estimated_lines or 20
                    sc = s.char_count or 0
                    if (_ml + sl > MAX_LINES_PER_FILE or _mc + sc > MAX_BRIEF_CHARS) and _mg:
                        model_groups.append(_mg)
                        _mg = []
                        _ml = 0
                        _mc = 0
                    _mg.append(s)
                    _ml += sl
                    _mc += sc
                if _mg:
                    model_groups.append(_mg)
                for idx, mg in enumerate(model_groups):
                    stem = "models" if idx == 0 else f"models_{idx + 1}"
                    mpath = f"{pkg}/{stem}.py"
                    nodes[mpath] = FileNode(
                        file_path=mpath,
                        file_stem=stem,
                        description=f"Data: {', '.join(s.name for s in mg[:3])}...",
                    )
            else:
                # Single oversized symbol (e.g. large dict) — can't split further
                models_path = f"{pkg}/models.py"
                nodes[models_path] = FileNode(
                    file_path=models_path,
                    file_stem="models",
                    description="Constants, data structures, and class definitions",
                )
        else:
            # Fits in one file
            models_path = f"{pkg}/models.py"
            nodes[models_path] = FileNode(
                file_path=models_path,
                file_stem="models",
                description="Constants, data structures, and class definitions",
            )

    # Split functions into groups (dual cap: lines AND chars)
    # Giant functions (individually over cap) get their own dedicated file.
    if functions:
        groups: List[List] = []
        current_group: List = []
        current_lines = 0
        current_chars = 0

        # Sort: largest functions first so giants get isolated early
        sorted_fns = sorted(functions, key=lambda f: f.char_count or 0, reverse=True)

        giant_groups: List[List] = []  # functions that need their own file
        normal_fns: List = []          # functions that can be grouped

        for fn in sorted_fns:
            fn_chars = fn.char_count or 0
            if fn_chars > MAX_BRIEF_CHARS:
                # This function alone exceeds the cap — solo file
                giant_groups.append([fn])
                logger.info(
                    "[refactor_segmenter] FIX 26: Giant function '%s' "
                    "(%d chars) gets dedicated file",
                    fn.name, fn_chars,
                )
            else:
                normal_fns.append(fn)

        # Group the normal functions by cumulative cap
        for fn in normal_fns:
            fn_lines = fn.estimated_lines or 20
            fn_chars = fn.char_count or 0
            if (current_lines + fn_lines > MAX_LINES_PER_FILE
                    or current_chars + fn_chars > MAX_BRIEF_CHARS) and current_group:
                groups.append(current_group)
                current_group = []
                current_lines = 0
                current_chars = 0
            current_group.append(fn)
            current_lines += fn_lines
            current_chars += fn_chars
        if current_group:
            groups.append(current_group)

        # Combine: giant solo files + normal grouped files
        all_groups = giant_groups + groups
        groups = all_groups

        total_fn_chars = sum(fn.char_count or 0 for fn in functions)
        logger.info(
            "[refactor_segmenter] FIX 26: %d functions (%d lines, %d chars) "
            "split into %d group(s) (cap: %d lines / %d chars per file)",
            len(functions),
            sum(fn.estimated_lines or 20 for fn in functions),
            total_fn_chars,
            len(groups),
            MAX_LINES_PER_FILE,
            MAX_BRIEF_CHARS,
        )

        if len(groups) == 1:
            # Single group → core.py
            core_path = f"{pkg}/core.py"
            nodes[core_path] = FileNode(
                file_path=core_path,
                file_stem="core",
                description="Core functions",
            )
        else:
            # Multiple groups → name by common prefix or index
            for idx, group in enumerate(groups):
                # Try to find a common prefix for naming
                names = [fn.name.lower() for fn in group]
                prefix = _find_common_prefix(names)
                if prefix and len(prefix) > 3:
                    stem = prefix.rstrip("_")
                else:
                    stem = f"group_{idx + 1}"
                file_path = f"{pkg}/{stem}.py"
                # Avoid collision with models.py
                if file_path in nodes:
                    file_path = f"{pkg}/{stem}_funcs.py"
                    stem = f"{stem}_funcs"
                nodes[file_path] = FileNode(
                    file_path=file_path,
                    file_stem=stem,
                    description=f"Functions: {', '.join(fn.name for fn in group[:3])}...",
                )

    # Facade __init__.py
    init_path = f"{pkg}/__init__.py"
    nodes[init_path] = FileNode(
        file_path=init_path,
        file_stem="__init__",
        description="Package facade — re-exports all public symbols",
        is_facade=True,
    )

    logger.info(
        "[refactor_segmenter] v6.1 Auto-generated %d file nodes for %s",
        len(nodes), source_file,
    )
    return nodes
