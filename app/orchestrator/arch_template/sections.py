# FILE: app/orchestrator/arch_template/sections.py
"""
Architecture Template Section Generators.

Each function generates one section of the architecture document
deterministically from skeleton contracts, segment specs, and
pattern references. Returns markdown strings.

Sections marked [LLM_FILL] are left for the LLM to complete.
Sections marked [DETERMINISTIC] are fully generated and should
not be modified by the LLM.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def generate_header(
    spec_id: str,
    spec_hash: str,
) -> str:
    """Generate the SPEC_ID/SPEC_HASH header (deterministic)."""
    return (
        f"SPEC_ID: {spec_id}\n"
        f"SPEC_HASH: {spec_hash}\n"
    )


def generate_file_inventory(
    segment_spec: Dict[str, Any],
    skeleton: Dict[str, Any],
) -> str:
    """Generate the File Inventory section (deterministic).

    Extracts file paths from segment spec + grounding data to produce
    the exact file inventory table. No LLM guessing needed.
    """
    parts = ["## 2. File Inventory [DETERMINISTIC]\n"]

    file_scope = segment_spec.get("file_scope", [])
    if isinstance(file_scope, str):
        file_scope = [f.strip() for f in file_scope.split(",") if f.strip()]

    grounding = segment_spec.get("grounding_data", {})
    create_targets = grounding.get("create_targets", [])
    verified_files = grounding.get("verified_files", [])

    # Classify files as CREATE or MODIFY
    create_paths = {
        t.get("path", "").replace("\\", "/")
        for t in create_targets
        if isinstance(t, dict)
    }

    # v1.1 (2026-03-01): When grounding_data is absent or create_targets
    # is empty, fall back to filesystem existence check via INDEX.json.
    # Without this, all files default to one category (wrong either way).
    _fs_existing: set = set()
    if not create_paths:
        _fs_existing = _load_existing_file_set()

    new_files = []
    modified_files = []

    for f in file_scope:
        norm = f.replace("\\", "/")
        # Determine purpose from skeleton exports
        purpose = _infer_file_purpose(norm, skeleton)
        if create_paths:
            # Grounding data available — use it as authoritative source
            if norm in create_paths:
                new_files.append((norm, purpose, "CREATE"))
            else:
                modified_files.append((norm, purpose, "MODIFY"))
        elif _fs_existing:
            # Fallback: check filesystem existence
            _norm_lower = norm.lower()
            # Strip orb-desktop/ prefix for lookup
            _lookup = _norm_lower
            if _lookup.startswith("orb-desktop/"):
                _lookup = _lookup[len("orb-desktop/"):]
            if _lookup in _fs_existing:
                modified_files.append((norm, purpose, "MODIFY"))
            else:
                new_files.append((norm, purpose, "CREATE"))
        else:
            # No grounding data AND no filesystem index — cannot classify.
            # Default to CREATE (safer than MODIFY for new feature work).
            new_files.append((norm, purpose, "CREATE"))

    if new_files:
        parts.append("### New Files\n")
        parts.append("| File | Purpose |")
        parts.append("|------|--------|")
        for path, purpose, _op in new_files:
            parts.append(f"| `{path}` | {purpose} |")
        parts.append("")

    if modified_files:
        parts.append("### Modified Files\n")
        parts.append("| File | Purpose |")
        parts.append("|------|--------|")
        for path, purpose, _op in modified_files:
            # Include 'Modify' in description so parser detects the operation
            parts.append(f"| `{path}` | Modify: {purpose} |")
        parts.append("")

    if not new_files and not modified_files:
        parts.append("*(No files in scope — this may indicate a spec issue.)*\n")

    return "\n".join(parts)


def _infer_file_purpose(file_path: str, skeleton: Dict[str, Any]) -> str:
    """Infer a file's purpose from its name, extension, and skeleton data."""
    name = os.path.basename(file_path)
    stem = re.sub(r"\.(tsx?|jsx?|css|py)$", "", name)

    # Check if it's an export (consumed by other segments)
    for exp in skeleton.get("exports", []):
        if exp.get("file_path", "").replace("\\", "/") == file_path:
            consumers = exp.get("consumed_by", [])
            if consumers:
                return f"Exports to {len(consumers)} downstream segment(s)"

    # Infer from naming conventions
    if file_path.endswith(".css"):
        return "Scoped styles for this feature's components"
    if stem.endswith("-data") or stem.endswith("_data"):
        return "Data models, interfaces, and static/mock data"
    if "View" in stem:
        return "Main orchestrator component for this feature"
    if "Grid" in stem or "List" in stem:
        return "Collection display component"
    if "Detail" in stem or "Card" in stem:
        return "Single item detail/display component"

    return "Feature component"


def _load_existing_file_set() -> set:
    """v1.1: Load normalised relative paths of all existing project files.

    Reads INDEX.json and returns a set of lowercase forward-slash paths
    relative to their project root (e.g. "src/types/index.ts").

    Used by generate_file_inventory() as a fallback when grounding_data
    is absent. Returns empty set if INDEX.json unavailable (non-fatal).
    """
    import json as _json

    arch_index_dir = os.getenv(
        "ASTRA_ARCH_INDEX_DIR",
        os.path.join("D:\\", "Orb", ".architecture"),
    )
    index_path = os.path.join(arch_index_dir, "INDEX.json")

    if not os.path.isfile(index_path):
        return set()

    try:
        with open(index_path, "r", encoding="utf-8") as fh:
            index_data = _json.load(fh)

        roots = index_data.get("roots", [])
        existing: set = set()

        for entry in index_data.get("files", []):
            abs_path = entry.get("path", "")
            if not abs_path:
                continue
            for root in roots:
                root_prefix = root.replace("/", "\\")
                if not root_prefix.endswith("\\"):
                    root_prefix += "\\"
                if abs_path.startswith(root_prefix):
                    rel = abs_path[len(root_prefix):]
                    existing.add(rel.replace("\\", "/").lower())
                    break

        logger.debug(
            "[sections] v1.1 Loaded %d existing paths from INDEX.json",
            len(existing),
        )
        return existing
    except Exception as exc:
        logger.debug("[sections] v1.1 INDEX.json load failed: %s", exc)
        return set()


def generate_import_map(
    skeleton: Dict[str, Any],
    all_skeletons: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate the cross-segment import map (deterministic).

    Shows exactly which files import from which upstream segments.
    """
    imports_from = skeleton.get("imports_from", {})
    if not imports_from:
        return ""

    parts = ["## Import Map [DETERMINISTIC]\n"]
    parts.append("The following cross-segment imports are FIXED by the skeleton contracts.\n")

    for dep_seg_id, dep_files in imports_from.items():
        parts.append(f"### From `{dep_seg_id}`:")
        for dep_file in dep_files:
            norm = dep_file.replace("\\", "/")
            # Find exported names from upstream
            exported_names = _get_upstream_exports(norm, dep_seg_id, all_skeletons)
            if exported_names:
                parts.append(f"- `{norm}` exports: `{', '.join(exported_names)}`")
            else:
                parts.append(f"- `{norm}` (exported symbols to be specified by architecture)")
        parts.append("")

    return "\n".join(parts)


def _get_upstream_exports(
    file_path: str,
    segment_id: str,
    all_skeletons: Optional[List[Dict[str, Any]]],
) -> List[str]:
    """Get exported symbol names from an upstream segment's file."""
    if not all_skeletons:
        return []

    for skel in all_skeletons:
        if skel.get("segment_id") == segment_id:
            for exp in skel.get("exports", []):
                if exp.get("file_path", "").replace("\\", "/") == file_path:
                    return exp.get("names", [])
    return []


def generate_dependency_context(
    skeleton: Dict[str, Any],
) -> str:
    """Generate dependency context section (deterministic)."""
    deps = skeleton.get("dependencies", [])
    if not deps:
        return ""

    parts = ["## Dependency Context [DETERMINISTIC]\n"]
    parts.append(f"This segment depends on {len(deps)} upstream segment(s):\n")
    for dep in deps:
        parts.append(f"- `{dep}`")
    parts.append("")
    parts.append(
        "All imports from these segments are specified in the Import Map above. "
        "Do NOT add additional cross-segment imports.\n"
    )
    return "\n".join(parts)


def generate_design_token_section(
    design_tokens: Optional[Dict[str, Any]],
) -> str:
    """Generate design token constraints section (deterministic).

    Lists CSS variables, font families, and theme tokens that MUST be used.
    """
    if not design_tokens:
        return ""

    parts = ["## Design Tokens [DETERMINISTIC]\n"]
    parts.append(
        "The following CSS variables are established in the codebase. "
        "All styling MUST use these variables — no hardcoded colour values.\n"
    )

    categories = design_tokens.get("categories", {})
    for cat_name, tokens in categories.items():
        if tokens:
            parts.append(f"### {cat_name}")
            parts.append("```css")
            for token_name, token_value in tokens.items():
                parts.append(f"{token_name}: {token_value};")
            parts.append("```")
            parts.append("")

    fonts = design_tokens.get("fonts", [])
    if fonts:
        parts.append(f"### Typography")
        parts.append(f"Font family: `{', '.join(fonts)}`\n")

    return "\n".join(parts)


def generate_pattern_reference(
    pattern_file: Optional[str],
    pattern_content: Optional[str],
) -> str:
    """Generate pattern reference section (deterministic).

    Shows an existing file as a structural template to follow.
    """
    if not pattern_file or not pattern_content:
        return ""

    parts = ["## Codebase Pattern Reference [DETERMINISTIC]\n"]
    parts.append(
        f"The file `{pattern_file}` demonstrates the established component pattern. "
        f"Follow this structure for consistency.\n"
    )

    # Extract key structural patterns
    _patterns = _extract_structural_patterns(pattern_content)
    if _patterns:
        parts.append("**Established patterns to follow:**\n")
        for p in _patterns:
            parts.append(f"- {p}")
        parts.append("")

    # Include truncated source
    max_chars = 3000
    if len(pattern_content) > max_chars:
        parts.append(f"```typescript\n{pattern_content[:max_chars]}\n// ... ({len(pattern_content)} chars total)\n```\n")
    else:
        parts.append(f"```typescript\n{pattern_content}\n```\n")

    return "\n".join(parts)


def _extract_structural_patterns(content: str) -> List[str]:
    """Extract key patterns from a reference file."""
    patterns = []

    # Check for hooks
    hooks = re.findall(r"(use\w+)\s*[(<]", content)
    if hooks:
        unique_hooks = list(dict.fromkeys(hooks))
        patterns.append(f"React hooks used: `{', '.join(unique_hooks[:6])}`")

    # Check for props interface
    if re.search(r"interface\s+\w+Props", content):
        patterns.append("Props defined via TypeScript interface (not inline)")

    # Check for named export
    if re.search(r"export\s+function\s+\w+", content):
        patterns.append("Named function export (not default export)")
    elif re.search(r"export\s+default", content):
        patterns.append("Default export pattern")

    # Check for CSS class naming
    classes = re.findall(r'className="([^"]+)"', content)
    if classes:
        patterns.append(f"CSS class naming: BEM-like (e.g. `{classes[0]}`)")

    return patterns


def generate_llm_fill_sections(
    segment_spec: Dict[str, Any],
    requirements: List[str],
) -> str:
    """Generate the sections the LLM needs to fill.

    These are structured prompts with [LLM_FILL] markers.
    """
    parts = []

    # Section 1: Architecture Vision
    parts.append("## 1. Architecture Vision [LLM_FILL]\n")
    parts.append(
        "Describe the high-level purpose of this segment. What does it build? "
        "How do the files work together? Keep this to 1-2 paragraphs.\n"
    )
    if requirements:
        parts.append("**Requirements to address:**\n")
        for req in requirements[:10]:
            parts.append(f"- {req}")
        parts.append("")

    # Section 3: Detailed file-by-file architecture
    file_scope = segment_spec.get("file_scope", [])
    if isinstance(file_scope, str):
        file_scope = [f.strip() for f in file_scope.split(",") if f.strip()]

    parts.append("## 3. Detailed File-by-File Architecture [LLM_FILL]\n")
    parts.append(
        "For each file below, describe the component architecture, "
        "props interface, component logic, and provide a Code Structure "
        "section with the complete TypeScript/TSX code.\n"
    )
    parts.append(
        "**IMPORTANT:** Imports for each file are already specified in the "
        "Import Map and Frozen Imports sections above. Do NOT re-specify "
        "imports in your Code Structure blocks. Start your code after the "
        "import lines.\n"
    )

    for f in file_scope:
        norm = f.replace("\\", "/")
        name = os.path.basename(norm)
        parts.append(f"### File: `{norm}`\n")
        parts.append(f"#### Purpose\n[LLM_FILL: Describe what `{name}` does]\n")
        parts.append(f"#### Architecture\n[LLM_FILL: Component structure, props, logic]\n")
        parts.append(f"#### Code Structure\n[LLM_FILL: Complete code for `{name}`]\n")
        parts.append("---\n")

    # Critical claims section
    parts.append("## CRITICAL_CLAIMS [LLM_FILL]\n")
    parts.append(
        "List all claims that require evidence. Each claim must have:\n"
        "- id: CC-NNN\n"
        "- claim: What is being claimed\n"
        "- resolution: CITED | DECISION | HUMAN_REQUIRED\n"
        "- evidence: Document/file references\n"
    )

    return "\n".join(parts)
