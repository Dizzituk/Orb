# FILE: app/pot_spec/grounded/size_analyzer.py
"""
Size Analyzer — Pre-Segmentation File Size Estimation & Decomposition.

Stage 4 upgrade: runs AFTER SpecGate extracts the file scope but BEFORE
smart segmentation groups files. Ensures no output file will exceed the
implementer's generation window (400 lines / 15 KB).

For refactor jobs:
    - Reads source files from disk
    - AST-parses Python files (via python_parser) for function/class structure
    - Regex-parses TypeScript/JS files (via typescript_parser) for exports
    - Identifies oversized blocks and suggests decomposition sub-files
    - Expands the file scope before segmentation

For greenfield (CREATE) jobs:
    - Heuristic estimation based on spec description complexity

Integration:
    Called from spec_runner.py between _extract_file_scope_from_spec() and
    generate_concept_segments(). Returns enriched file scope and size metadata.

v1.1 (2026-02-14): Split into sub-modules for cap compliance.
    - size_models.py: constants + dataclasses
    - python_parser.py: AST-based Python structure analysis
    - typescript_parser.py: regex-based TS/JS structure analysis
    - size_analyzer.py: main entry, estimation, decomposition (this file)
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional

from .size_models import (
    AVG_CHARS_PER_LINE,
    DEFAULT_PROJECT_ROOTS,
    MAX_FILE_LINES,
    FileSizeEstimate,
    SizeAnalysisResult,
    SourceBlock,
)
from .python_parser import parse_python_structure
from .typescript_parser import parse_typescript_structure
from .size_decomposition import estimate_from_source_blocks, suggest_decomposition_for_block

logger = logging.getLogger(__name__)

SIZE_ANALYZER_BUILD_ID = "2026-02-14-v1.1-split-submodules"
print(f"[SIZE_ANALYZER_LOADED] BUILD_ID={SIZE_ANALYZER_BUILD_ID}")


# =============================================================================
# SOURCE FILE READING
# =============================================================================

def _resolve_to_disk(
    rel_path: str,
    project_roots: Optional[List[str]] = None,
) -> Optional[str]:
    """Resolve a relative path to an absolute path on disk."""
    roots = project_roots or DEFAULT_PROJECT_ROOTS
    normalised = rel_path.replace("/", os.sep).replace("\\", os.sep)
    for root in roots:
        abs_path = os.path.join(root, normalised)
        if os.path.isfile(abs_path):
            return abs_path
    return None


def _read_source_files(
    file_scope: List[str],
    project_roots: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Read existing source files from disk for analysis.

    Returns dict of {relative_path: file_content} for files that exist.
    """
    sources: Dict[str, str] = {}
    MAX_CHARS = 500_000

    for rel_path in file_scope:
        abs_path = _resolve_to_disk(rel_path, project_roots)
        if abs_path:
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    sources[rel_path] = fh.read(MAX_CHARS)
            except Exception as exc:
                logger.warning("[size_analyzer] Failed to read %s: %s", abs_path, exc)

    return sources


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def analyze_file_sizes(
    file_scope: List[str],
    spec_markdown: Optional[str] = None,
    project_roots: Optional[List[str]] = None,
) -> SizeAnalysisResult:
    """
    Pre-segmentation file size analysis.

    Reads source files, parses them, identifies oversized blocks, and
    suggests decompositions. Returns enriched file scope with additional
    sub-files if decomposition is needed.

    This is the main entry point called from spec_runner.py.
    """
    estimates: Dict[str, FileSizeEstimate] = {}
    files_added: List[str] = []
    files_exceeding: List[str] = []

    source_contents = _read_source_files(file_scope, project_roots)
    is_refactor = len(source_contents) > 0

    if not is_refactor and not spec_markdown:
        return SizeAnalysisResult(
            original_file_scope=list(file_scope),
            enriched_file_scope=list(file_scope),
            estimates={},
            files_added=[],
            files_exceeding_cap=[],
            source_files_analyzed=0,
            is_refactor=False,
        )

    logger.info(
        "[size_analyzer] Analyzing %d files (%d source files on disk)",
        len(file_scope), len(source_contents),
    )

    output_package_dir = _detect_output_package_dir(file_scope)

    # Parse source file structures
    source_structures: Dict[str, List[SourceBlock]] = {}
    for rel_path, content in source_contents.items():
        ext = os.path.splitext(rel_path)[1].lower()
        if ext == ".py":
            blocks = parse_python_structure(content)
        elif ext in (".ts", ".tsx", ".js", ".jsx"):
            blocks = parse_typescript_structure(content)
        else:
            blocks = []

        if blocks:
            source_structures[rel_path] = blocks
            total_lines = content.count("\n") + 1
            logger.info(
                "[size_analyzer] Parsed %s: %d lines, %d blocks",
                rel_path, total_lines, len(blocks),
            )

    # Identify oversized blocks and suggest decomposition
    all_suggestions: List[Dict[str, Any]] = []
    for source_path, blocks in source_structures.items():
        source_lines = source_contents[source_path].count("\n") + 1
        output_count = len([f for f in file_scope if f not in source_contents])

        oversized = estimate_from_source_blocks(blocks, source_lines, output_count)

        for block_info in oversized:
            suggestions = suggest_decomposition_for_block(
                block_info, source_path, output_package_dir,
            )
            if suggestions:
                all_suggestions.extend(suggestions)
                logger.info(
                    "[size_analyzer] Block '%s' in %s (%d lines) → %d sub-file(s)",
                    block_info["block_name"], source_path,
                    block_info["total_lines"], len(suggestions),
                )
                print(
                    f"[size_analyzer] ⚠️ OVERSIZED: {source_path}::{block_info['block_name']} "
                    f"= {block_info['total_lines']} lines → suggesting {len(suggestions)} sub-file(s)"
                )

    # Build estimates for all files in scope
    for rel_path in file_scope:
        if rel_path in source_contents:
            content = source_contents[rel_path]
            est = FileSizeEstimate(
                rel_path=rel_path,
                estimated_lines=content.count("\n") + 1,
                estimated_kb=round(len(content) / 1024, 1),
                source_file=rel_path,
                source_blocks=[b.name for b in source_structures.get(rel_path, [])],
            )
        else:
            est = _estimate_output_file_size(rel_path, spec_markdown, source_structures)

        est.exceeds_cap = est.estimated_lines > MAX_FILE_LINES
        if est.exceeds_cap:
            files_exceeding.append(rel_path)
        estimates[rel_path] = est

    # Expand scope with decomposition suggestions
    enriched_scope = list(file_scope)
    for suggestion in all_suggestions:
        new_path = suggestion["rel_path"]
        normalised_scope = {f.replace("\\", "/").lower() for f in enriched_scope}
        if new_path.replace("\\", "/").lower() not in normalised_scope:
            enriched_scope.append(new_path)
            files_added.append(new_path)
            estimates[new_path] = FileSizeEstimate(
                rel_path=new_path,
                estimated_lines=suggestion["estimated_lines"],
                estimated_kb=suggestion["estimated_kb"],
                source_blocks=suggestion.get("source_closures", []),
                decomposition_suggested=True,
            )

    if files_added:
        print(
            f"[size_analyzer] 📐 Size analysis: {len(files_added)} decomposition file(s) "
            f"added to scope: {', '.join(files_added)}"
        )
    else:
        print(f"[size_analyzer] ✅ Size analysis: all {len(file_scope)} files within caps")

    return SizeAnalysisResult(
        original_file_scope=list(file_scope),
        enriched_file_scope=enriched_scope,
        estimates=estimates,
        files_added=files_added,
        files_exceeding_cap=files_exceeding,
        source_files_analyzed=len(source_structures),
        is_refactor=is_refactor,
    )


# =============================================================================
# HELPERS
# =============================================================================

def _detect_output_package_dir(file_scope: List[str]) -> Optional[str]:
    """Detect common package directory for output files."""
    if len(file_scope) < 2:
        return None

    dirs = []
    for fp in file_scope:
        normalised = fp.replace("\\", "/")
        parent = "/".join(normalised.split("/")[:-1])
        if parent:
            dirs.append(parent)

    if not dirs:
        return None

    counts = Counter(dirs)
    most_common_dir, count = counts.most_common(1)[0]

    if count >= len(file_scope) * 0.5:
        return most_common_dir
    return None


def _estimate_output_file_size(
    rel_path: str,
    spec_markdown: Optional[str],
    source_structures: Dict[str, List[SourceBlock]],
) -> FileSizeEstimate:
    """
    Estimate size of an output file that doesn't exist yet.

    Matches output filename to source block names, then applies heuristics.
    """
    estimated_lines = 100
    file_stem = os.path.splitext(os.path.basename(rel_path))[0]

    for _source_path, blocks in source_structures.items():
        for block in blocks:
            block_stem = block.name.lstrip("_").lower()
            if file_stem.lower() in block_stem or block_stem in file_stem.lower():
                estimated_lines = max(estimated_lines, block.line_count)
            for nb in block.nested_blocks:
                nb_stem = nb.name.lstrip("_").lower()
                if file_stem.lower() in nb_stem or nb_stem in file_stem.lower():
                    estimated_lines = max(estimated_lines, nb.line_count + 30)

    if file_stem == "__init__":
        estimated_lines = min(estimated_lines, 80)
    elif file_stem in ("constants", "config"):
        estimated_lines = min(estimated_lines, 60)
    elif file_stem == "prompts":
        estimated_lines = min(estimated_lines, 350)

    return FileSizeEstimate(
        rel_path=rel_path,
        estimated_lines=estimated_lines,
        estimated_kb=round(estimated_lines * AVG_CHARS_PER_LINE / 1024, 1),
    )
