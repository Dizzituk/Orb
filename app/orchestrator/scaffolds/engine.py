# FILE: app/orchestrator/scaffolds/engine.py
"""
Scaffold Engine — generates code skeletons for implementation briefs.

Sits between the architecture document and the implementer. Takes
a FileBrief, classifies the file pattern, generates the appropriate
scaffold, and injects it into the brief.

v1.0 (2026-03-01): Initial implementation.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

from .pattern_classifier import PatternMatch, classify_file_pattern
from .templates import (
    scaffold_css_module,
    scaffold_data_module,
    scaffold_detail_component,
    scaffold_grid_component,
    scaffold_view_component,
)

logger = logging.getLogger(__name__)

# Template dispatch table
_SCAFFOLD_GENERATORS = {
    "view": scaffold_view_component,
    "grid": scaffold_grid_component,
    "detail": scaffold_detail_component,
    "data": scaffold_data_module,
    "css": scaffold_css_module,
}


def generate_scaffold_for_file(
    file_path: str,
    operation: str,
    architecture_text: str = "",
    design_notes: str = "",
    exports: Optional[List[str]] = None,
    frozen_imports: str = "",
    design_tokens: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Generate a code scaffold for a single file.

    Classifies the file pattern and generates the appropriate template.
    Returns None if the file doesn't match any known pattern or is a
    MODIFY operation (scaffolds are for new files only).

    Args:
        file_path: Target file path.
        operation: "CREATE" or "MODIFY".
        architecture_text: Architecture doc section for this file.
        design_notes: Additional design notes.
        exports: Exported symbol names.
        frozen_imports: Pre-built import block (from Job 2).
        design_tokens: Token registry (from Job 5).

    Returns:
        Scaffold code string with [LLM_FILL] markers, or None.
    """
    # Only scaffold new files — MODIFY files have existing structure
    if operation.upper() == "MODIFY":
        logger.debug("[scaffold] Skipping MODIFY file: %s", file_path)
        return None, "unknown"

    # Classify the pattern
    match = classify_file_pattern(
        file_path=file_path,
        architecture_text=architecture_text,
        exports=exports,
        design_notes=design_notes,
    )

    if match.pattern == "unknown":
        logger.debug(
            "[scaffold] No pattern match for %s (score=%.2f)",
            file_path, match.confidence,
        )
        return None, "unknown"

    # Extract component name from file path
    component_name = _extract_component_name(file_path)

    # Extract data type hints from architecture
    data_type = _extract_data_type(architecture_text)
    data_source = _extract_data_source(architecture_text)

    # Generate the scaffold
    generator = _SCAFFOLD_GENERATORS.get(match.pattern)
    if not generator:
        return None, "unknown"

    try:
        if match.pattern == "view":
            scaffold = generator(
                file_path=file_path,
                component_name=component_name,
                frozen_imports=frozen_imports,
                exports=exports,
                design_tokens=design_tokens,
                architecture_hints=architecture_text,
            )
        elif match.pattern == "grid":
            scaffold = generator(
                file_path=file_path,
                component_name=component_name,
                frozen_imports=frozen_imports,
                exports=exports,
                data_type=data_type,
                data_source=data_source,
                design_tokens=design_tokens,
            )
        elif match.pattern == "detail":
            scaffold = generator(
                file_path=file_path,
                component_name=component_name,
                frozen_imports=frozen_imports,
                exports=exports,
                data_type=data_type,
                design_tokens=design_tokens,
            )
        elif match.pattern == "data":
            scaffold = generator(
                file_path=file_path,
                frozen_imports=frozen_imports,
                exports=exports,
                architecture_hints=architecture_text,
            )
        elif match.pattern == "css":
            scaffold = generator(
                file_path=file_path,
                component_name=component_name,
                design_tokens=design_tokens,
            )
        else:
            return None, "unknown"

        logger.info(
            "[scaffold] Generated '%s' scaffold for %s (%d chars, %.2f confidence)",
            match.pattern, os.path.basename(file_path),
            len(scaffold), match.confidence,
        )
        return scaffold, match.pattern

    except Exception as exc:
        logger.warning("[scaffold] Generation failed for %s: %s", file_path, exc)
        return None, "unknown"


def build_scaffold_prompt_section(scaffold: str, pattern: str) -> str:
    """Wrap a scaffold in a prompt section for the implementation brief.

    The section tells the LLM to use the scaffold as a starting point
    and fill the [LLM_FILL] markers with business logic.

    Args:
        scaffold: Generated scaffold code.
        pattern: Pattern name for context.

    Returns:
        Formatted prompt section string.
    """
    return "\n".join([
        "## Code Scaffold [DETERMINISTIC]",
        "",
        f"Pattern: **{pattern}** (auto-detected from file name and architecture)",
        "",
        "The following code skeleton has been generated deterministically.",
        "Use it as your starting structure. Rules:",
        "",
        "- **Replace** every `[LLM_FILL: ...]` marker with the actual implementation",
        "- **Keep** the overall structure (component name, CSS class names, export pattern)",
        "- **Keep** any pre-filled imports, props interface shape, and section comments",
        "- **Add** additional code as needed (helpers, sub-components, extra state)",
        "- **Do NOT** remove structural elements unless the architecture explicitly requires a different approach",
        "",
        "```typescript",
        scaffold,
        "```",
        "",
    ])


# ─── Extraction helpers ─────────────────────────────────────────────

def _extract_component_name(file_path: str) -> str:
    """Extract PascalCase component name from file path.

    src/components/education/EducationCourseGrid.tsx -> EducationCourseGrid
    src/components/education/education-data.ts -> EducationData
    """
    basename = os.path.basename(file_path.replace("\\", "/"))
    stem = re.sub(r"\.(tsx?|jsx?|css)$", "", basename)

    # If already PascalCase, use as-is
    if stem[0].isupper():
        return stem

    # Convert kebab-case to PascalCase
    return "".join(word.capitalize() for word in stem.split("-"))


def _extract_data_type(architecture_text: str) -> str:
    """Extract the primary data type name from architecture text.

    Looks for patterns like "Course type", "imports Course from",
    "item: Course", etc.
    """
    # Match interface/type declarations
    m = re.search(r"interface\s+(\w+Props)", architecture_text)
    if m:
        # Props interface — look for the data type it uses
        pass

    # Match type references in map/iteration contexts
    m = re.search(r"\.map\(\((\w+):\s*(\w+)\)", architecture_text)
    if m:
        return m.group(2)

    # Match "imports X type" or "X interface"
    m = re.search(r"(?:type|interface)\s+(\w+)\s*\{", architecture_text)
    if m and not m.group(1).endswith("Props"):
        return m.group(1)

    return ""


def _extract_data_source(architecture_text: str) -> str:
    """Extract the data source variable name from architecture text.

    Looks for patterns like "courses array", "maps over courses",
    "imports courses from".
    """
    # Match "maps over X" or "X.map("
    m = re.search(r"(\w+)\.map\(", architecture_text)
    if m:
        return m.group(1)

    # Match "import { X } from" where X is lowercase (data, not type)
    for m in re.finditer(r"import\s*\{[^}]*\b(\w+)\b[^}]*\}", architecture_text):
        names = re.findall(r"\b([a-z]\w+)\b", m.group(0))
        for name in names:
            if name not in ("import", "from", "type"):
                return name

    return ""
