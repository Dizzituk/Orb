from __future__ import annotations
import json
import logging
import os
import re
from app.orchestrator.compiler_models import CompilationResult, FileFunction
from app.orchestrator.implementation_compiler import logger
from typing import Dict, List, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


BUILD_ID = "2026-02-20-v1.0-implementation-compiler"

def _extract_file_design_notes(
    architecture_text: str,
    file_path: str,
) -> str:
    """
    Extract design notes for a specific file from the architecture document.

    Looks for subsections referencing this file path or stem.
    """
    file_stem = os.path.basename(file_path).replace(".py", "")
    lines = architecture_text.split("\n")
    capture = False
    captured: List[str] = []

    # v6.1 FIX 23c: Track the heading level that started capture.
    # Only stop when hitting a heading at the same or higher level.
    # Previously, ### subheadings (Purpose, Re-exports) inside a
    # ## File section would terminate capture immediately.
    capture_level = 0

    for line in lines:
        stripped = line.strip()

        _h_match = re.match(r'^(#{2,4})\s', stripped)
        if _h_match:
            _level = len(_h_match.group(1))
            if file_stem in stripped or file_path in stripped:
                capture = True
                capture_level = _level
                continue
            elif capture and _level <= capture_level:
                # Same or higher level heading = new section
                break
            # Deeper subheading within section — include it

        if capture:
            captured.append(line)

    raw = "\n".join(captured).strip()

    # v6.1 FIX 23d: Strip content that the compiler provides separately
    # to avoid duplicating function bodies in the brief.
    import re as _re
    # 1. Strip "### Contents" sections entirely
    raw = _re.sub(
        r'### Contents.*?(?=### [A-Z]|## [A-Z]|$)',
        '',
        raw,
        flags=_re.DOTALL,
    )
    # 2. Strip individual #### `symbol` blocks with their code fences.
    #    These appear under various subsection headers and duplicate
    #    the function bodies the compiler embeds via enrichment.
    raw = _re.sub(
        r'####\s*`\w+`\s*\([^)]*\).*?(?=####\s*`|### [A-Z]|## [A-Z]|$)',
        '',
        raw,
        flags=_re.DOTALL,
    ).strip()

    return raw

def _extract_import_names(import_line: str) -> Set[str]:
    """Extract imported symbol names from an import statement."""
    names: Set[str] = set()
    line = import_line.strip()

    if line.startswith("from "):
        match = re.search(r'import\s+(.+)', line)
        if match:
            imports_part = match.group(1).strip("()")
            for part in imports_part.split(","):
                part = part.strip()
                if " as " in part:
                    names.add(part.split(" as ")[-1].strip())
                elif part and part != "*":
                    names.add(part.strip())
    elif line.startswith("import "):
        imports_part = line[7:]
        for part in imports_part.split(","):
            part = part.strip()
            if " as " in part:
                names.add(part.split(" as ")[-1].strip())
            elif part:
                names.add(part.split(".")[0].strip())

    return names

def _build_consumed_by(
    file_path: str,
    file_exports: List[str],
    file_function_map: Dict[str, List[FileFunction]],
    file_inventory: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Build map of which sibling files consume symbols from this file."""
    consumed_by: Dict[str, List[str]] = {}

    for entry in file_inventory:
        other_path = entry["path"]
        if other_path == file_path:
            continue
        other_funcs = file_function_map.get(other_path, [])
        for other_func in other_funcs:
            if not other_func.body:
                continue
            for export_name in file_exports:
                if export_name in other_func.body:
                    consumed_by.setdefault(other_path, [])
                    if export_name not in consumed_by[other_path]:
                        consumed_by[other_path].append(export_name)

    return consumed_by

def _build_consumes_from(
    functions: List[FileFunction],
    file_path: str,
    file_function_map: Dict[str, List[FileFunction]],
    file_inventory: List[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Build map of which sibling files this file consumes from."""
    consumes_from: Dict[str, List[str]] = {}
    our_names = {f.name for f in functions}

    used_names: Set[str] = set()
    for func in functions:
        if func.body:
            for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', func.body):
                used_names.add(match.group(1))

    for entry in file_inventory:
        other_path = entry["path"]
        if other_path == file_path:
            continue
        other_funcs = file_function_map.get(other_path, [])
        other_names = {f.name for f in other_funcs}
        needed = (used_names & other_names) - our_names
        if needed:
            consumes_from[other_path] = sorted(needed)

    return consumes_from

def _collect_feedback(
    cohesion_feedback: str,
    implementation_feedback: str,
    import_validation_feedback: str,
) -> List[str]:
    """Collect all feedback into a flat list."""
    feedback: List[str] = []

    if cohesion_feedback:
        feedback.append(f"[COHESION] {cohesion_feedback.strip()}")

    if implementation_feedback:
        for line in implementation_feedback.strip().split("\n"):
            if line.strip():
                feedback.append(f"[IMPL] {line.strip()}")

    if import_validation_feedback:
        for line in import_validation_feedback.strip().split("\n"):
            if line.strip():
                feedback.append(f"[IMPORT] {line.strip()}")

    return feedback

def _filter_feedback_for_file(
    all_feedback: List[str],
    file_path: str,
) -> List[str]:
    """Filter feedback entries relevant to a specific file."""
    file_stem = os.path.basename(file_path).replace(".py", "")
    relevant: List[str] = []

    for fb in all_feedback:
        if file_stem in fb or file_path in fb:
            relevant.append(fb)
        elif not any(kw in fb for kw in [".py", "/"]):
            relevant.append(fb)

    return relevant

def save_compilation_result(
    result: CompilationResult,
    job_dir_path: str,
    segment_id: str,
) -> None:
    """Persist compilation result and individual briefs to disk."""
    seg_dir = os.path.join(job_dir_path, "segments", segment_id)
    compiler_dir = os.path.join(seg_dir, "compiler")
    os.makedirs(compiler_dir, exist_ok=True)

    summary_path = os.path.join(compiler_dir, "compilation_summary.json")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.info("[IMPL_COMPILER] Summary saved: %s", summary_path)
    except Exception as e:
        logger.warning("[IMPL_COMPILER] Failed to save summary: %s", e)

    for brief in result.briefs:
        safe_name = os.path.basename(brief.file_path).replace(".py", "")
        brief_path = os.path.join(compiler_dir, f"brief_{safe_name}.md")
        try:
            with open(brief_path, "w", encoding="utf-8") as f:
                f.write(brief.to_markdown())
            logger.info("[IMPL_COMPILER] Brief saved: %s", brief_path)
        except Exception as e:
            logger.warning("[IMPL_COMPILER] Failed to save brief %s: %s", brief_path, e)
