# FILE: app/orchestrator/deterministic_imports.py
# Purpose: Deterministic Import Generator — Job 2.
# Called-by: no static importers found (dynamic/registry use possible)
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Deterministic Import Generator — Job 2.

v1.0 (2026-03-01): Generates exact import blocks for TypeScript/TSX files
from skeleton contracts + architecture, eliminating LLM-generated imports.

Supports both Python and TypeScript import generation.
For frontend segments, produces ready-to-use TypeScript import lines.
For backend segments, produces Python import lines.

The generated imports are injected as FROZEN headers — the LLM is
instructed not to add, remove, or duplicate them.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DETERMINISTIC_IMPORTS_BUILD_ID = "2026-03-01-v1.0-frozen-imports"
print(f"[DETERMINISTIC_IMPORTS_LOADED] BUILD_ID={DETERMINISTIC_IMPORTS_BUILD_ID}")

# Frontend file extensions
_FRONTEND_EXTS = {".tsx", ".ts", ".jsx", ".js", ".css"}


@dataclass
class FrozenImportBlock:
    """A complete, ready-to-use import block for a single file."""
    file_path: str
    language: str               # "typescript" | "python"
    import_lines: List[str]     # Exact import statements, in order
    notes: List[str] = field(default_factory=list)  # Human-readable notes for the LLM

    def to_code(self) -> str:
        """Return the import block as ready-to-paste code."""
        return "\n".join(self.import_lines)

    def to_prompt_section(self) -> str:
        """Render as a prompt section for the implementation brief."""
        parts = [
            "## ⚠️ FROZEN IMPORTS (deterministic — DO NOT MODIFY)",
            "",
            "The following import block has been generated deterministically from",
            "the skeleton contracts and architecture. It is CORRECT and COMPLETE.",
            "",
            "**Rules:**",
            "- Copy this import block EXACTLY at the top of your file",
            "- Do NOT add additional imports for symbols already listed",
            "- Do NOT remove any import listed here",
            "- Do NOT duplicate any import line",
            "- If you need a symbol not listed here (e.g. React hooks), add it",
            "  BELOW this block with a comment: `// Additional import`",
            "",
            f"```{self.language}",
        ]
        parts.extend(self.import_lines)
        parts.append("```")
        if self.notes:
            parts.append("")
            for note in self.notes:
                parts.append(f"*{note}*")
        parts.append("")
        return "\n".join(parts)


def _compute_relative_import_path(
    from_file: str,
    to_file: str,
) -> str:
    """Compute the TypeScript relative import path between two files.

    Both paths should be relative to the project root (e.g. 'src/components/education/X.tsx').
    Strips the extension for TypeScript imports.

    Examples:
        _compute_relative_import_path(
            'src/components/education/EducationView.tsx',
            'src/components/education/education-data.ts'
        ) -> './education-data'

        _compute_relative_import_path(
            'src/components/jobs/JobPage.tsx',
            'src/components/education/EducationView.tsx'
        ) -> '../education/EducationView'
    """
    from_dir = os.path.dirname(from_file.replace("\\", "/"))
    to_dir = os.path.dirname(to_file.replace("\\", "/"))
    to_name = os.path.basename(to_file.replace("\\", "/"))

    # Strip extension
    to_stem = re.sub(r"\.(tsx?|jsx?|css)$", "", to_name)

    if from_dir == to_dir:
        return f"./{to_stem}"

    # Compute relative path
    from_parts = from_dir.split("/") if from_dir else []
    to_parts = to_dir.split("/") if to_dir else []

    # Find common prefix
    common_len = 0
    for a, b in zip(from_parts, to_parts):
        if a == b:
            common_len += 1
        else:
            break

    # Build relative path
    ups = len(from_parts) - common_len
    downs = to_parts[common_len:]

    if ups == 0:
        rel = "./" + "/".join(downs)
    else:
        rel = "/".join([".."] * ups + downs)

    # Always need a slash before the filename
    if rel.endswith("/"):
        return f"{rel}{to_stem}"
    return f"{rel}/{to_stem}"


def _extract_exported_names_from_arch(
    architecture_text: str,
    file_path: str,
) -> List[str]:
    """Extract exported symbol names from the architecture document for a file.

    Looks for patterns like:
    - "exports: Course, Module, courses"
    - "Exports: `Course`, `Module`"
    - "defines and exports: Course interface, Module interface"
    """
    names: List[str] = []
    norm_path = file_path.replace("\\", "/")
    file_name = os.path.basename(norm_path)

    # Find sections that mention this file
    in_file_section = False
    for line in architecture_text.splitlines():
        # Check if we're in a section about this file
        if file_name in line or norm_path in line:
            in_file_section = True
            continue
        # New section header exits
        if in_file_section and line.startswith("#"):
            in_file_section = False

        if in_file_section:
            # Pattern: "exports: X, Y, Z" or "Exports: X, Y"
            m = re.match(
                r".*(?:exports?|defines?|provides?)\s*:\s*(.+)",
                line, re.IGNORECASE,
            )
            if m:
                raw = m.group(1)
                # Extract identifiers (backtick-wrapped or plain)
                for ident in re.findall(r"`(\w+)`|(?:^|,)\s*(\w+)(?:\s+interface|\s+type|\s+function)?", raw):
                    name = ident[0] or ident[1]
                    if not name:
                        continue
                    # Accept: PascalCase (types/components), camelCase (functions),
                    # UPPER_SNAKE (constants), and known data identifiers
                    # Reject: lowercase keywords like 'from', 'and', 'the', 'this'
                    _REJECT_WORDS = {"from", "and", "the", "this", "with", "for",
                                     "interface", "type", "function", "export",
                                     "default", "import", "class", "const", "let"}
                    if name.lower() not in _REJECT_WORDS and len(name) > 1:
                        names.append(name)

    return list(dict.fromkeys(names))  # dedupe preserving order


def generate_frozen_imports_for_segment(
    skeleton: Dict[str, Any],
    architecture_text: str,
    all_skeletons: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, FrozenImportBlock]:
    """Generate deterministic import blocks for all files in a segment.

    Args:
        skeleton: This segment's skeleton contract (dict form).
        architecture_text: The approved architecture document.
        all_skeletons: All skeletons in the job (for resolving upstream exports).

    Returns:
        Dict of {file_path: FrozenImportBlock} for each file in the segment.
    """
    result: Dict[str, FrozenImportBlock] = {}
    file_scope = skeleton.get("file_scope", [])
    imports_from = skeleton.get("imports_from", {})
    dependencies = skeleton.get("dependencies", [])

    # Build upstream export map: {file_path: {exported_names}}
    upstream_exports: Dict[str, List[str]] = {}
    if all_skeletons:
        for skel in all_skeletons:
            if skel.get("segment_id") == skeleton.get("segment_id"):
                continue
            for exp in skel.get("exports", []):
                exp_path = exp.get("file_path", "")
                exp_names = exp.get("names", [])
                if exp_path:
                    upstream_exports[exp_path] = exp_names

    for file_path in file_scope:
        norm_path = file_path.replace("\\", "/")
        ext = os.path.splitext(norm_path)[1].lower()

        if ext in _FRONTEND_EXTS:
            block = _generate_ts_imports(
                file_path=norm_path,
                file_scope=file_scope,
                imports_from=imports_from,
                upstream_exports=upstream_exports,
                architecture_text=architecture_text,
            )
        else:
            # Python imports handled by existing implementation compiler
            block = FrozenImportBlock(
                file_path=norm_path,
                language="python",
                import_lines=[],
                notes=["Python imports resolved by implementation compiler"],
            )

        if block.import_lines:
            result[norm_path] = block

    return result


def _generate_ts_imports(
    file_path: str,
    file_scope: List[str],
    imports_from: Dict[str, List[str]],
    upstream_exports: Dict[str, List[str]],
    architecture_text: str,
) -> FrozenImportBlock:
    """Generate TypeScript import lines for a single file.

    Sources of import information:
    1. Skeleton contracts — which upstream files this segment depends on
    2. Architecture document — which symbols are exported by each file
    3. Intra-segment imports — files within this segment that depend on each other
    """
    import_lines: List[str] = []
    notes: List[str] = []

    # Source 1: Upstream segment imports (from skeleton contracts)
    for dep_seg_id, dep_files in imports_from.items():
        for dep_file in dep_files:
            dep_norm = dep_file.replace("\\", "/")
            rel_path = _compute_relative_import_path(file_path, dep_norm)

            # Get the exported names from upstream
            exported_names = upstream_exports.get(dep_norm, [])
            if not exported_names:
                # Try to extract from architecture
                exported_names = _extract_exported_names_from_arch(
                    architecture_text, dep_norm,
                )

            if exported_names:
                # Separate type imports from value imports based on naming convention
                type_names = [n for n in exported_names if n[0].isupper() and not n.startswith("DUMMY_")]
                value_names = [n for n in exported_names if n not in type_names]

                if value_names and type_names:
                    import_lines.append(
                        f"import {{ {', '.join(value_names + type_names)} }} from '{rel_path}';"
                    )
                elif type_names:
                    import_lines.append(
                        f"import type {{ {', '.join(type_names)} }} from '{rel_path}';"
                    )
                elif value_names:
                    import_lines.append(
                        f"import {{ {', '.join(value_names)} }} from '{rel_path}';"
                    )
                notes.append(f"From {dep_seg_id}: {', '.join(exported_names)}")
            else:
                # We know the file is a dependency but don't know exact exports
                import_lines.append(f"// TODO: import from '{rel_path}' (resolve exports)")
                notes.append(
                    f"Dependency on {dep_norm} detected but exported symbols unknown — "
                    f"architecture should specify exports"
                )

    # Source 2: Intra-segment imports (files within this segment)
    norm_scope = [f.replace("\\", "/") for f in file_scope]
    for other_file in norm_scope:
        if other_file == file_path:
            continue
        # Check if architecture indicates this file imports from the other
        other_name = os.path.basename(other_file)
        other_stem = re.sub(r"\.(tsx?|jsx?|css)$", "", other_name)

        # For CSS files, check if a .tsx file should import it
        if other_file.endswith(".css") and file_path.endswith((".tsx", ".jsx")):
            rel_path = _compute_relative_import_path(file_path, other_file)
            # CSS imports don't use braces
            import_lines.append(f"import '{rel_path}.css';")
            notes.append(f"CSS stylesheet for this component")
            continue

        # For other intra-segment TS files, check architecture for cross-refs
        if not other_file.endswith((".ts", ".tsx")):
            continue
        # Check if architecture mentions importing from this file
        other_exports = _extract_exported_names_from_arch(architecture_text, other_file)
        if other_exports:
            rel_path = _compute_relative_import_path(file_path, other_file)
            import_lines.append(
                f"import {{ {', '.join(other_exports)} }} from '{rel_path}';"
            )

    # Source 3: Common React imports (if .tsx file)
    if file_path.endswith(".tsx"):
        # Check architecture for React hooks usage
        react_hooks = []
        for hook in ["useState", "useEffect", "useCallback", "useMemo", "useRef"]:
            if hook.lower() in architecture_text.lower():
                # Only include if this file's section mentions it
                # (broad check — could be refined)
                react_hooks.append(hook)

        if react_hooks:
            import_lines.insert(0, f"import React, {{ {', '.join(react_hooks)} }} from 'react';")
        else:
            import_lines.insert(0, "import React from 'react';")

    return FrozenImportBlock(
        file_path=file_path,
        language="typescript",
        import_lines=import_lines,
        notes=notes,
    )


def strip_duplicate_imports_from_output(content: str, frozen_block: FrozenImportBlock) -> str:
    """Post-generation cleanup: remove any LLM-generated imports that duplicate frozen ones.

    If the LLM added its own import lines that import from the same module
    as a frozen import, remove the duplicate. This is the safety net.

    Args:
        content: The LLM-generated file content.
        frozen_block: The frozen import block for this file.

    Returns:
        Cleaned content with duplicate imports removed.
    """
    if not frozen_block.import_lines:
        return content

    # Extract module paths from frozen imports
    frozen_modules: Set[str] = set()
    for line in frozen_block.import_lines:
        m = re.search(r"""from\s+['"]([^'"]+)['"]""", line)
        if m:
            frozen_modules.add(m.group(1))

    lines = content.splitlines()
    clean_lines: List[str] = []
    removed = 0

    # Track which frozen lines we've already seen (allow each exactly once)
    frozen_line_seen: Dict[str, bool] = {fl: False for fl in frozen_block.import_lines}

    for line in lines:
        stripped = line.strip()
        # Check if this is an import from a module we already have frozen
        m = re.search(r"""from\s+['"]([^'"]+)['"]""", stripped)
        if m and m.group(1) in frozen_modules:
            # Check if this is a frozen line we haven't seen yet
            if stripped in frozen_line_seen and not frozen_line_seen[stripped]:
                # First occurrence of this frozen line — keep it
                frozen_line_seen[stripped] = True
            else:
                # Either: already-seen frozen line (duplicate copy) or
                # non-frozen import from a frozen module (LLM-added)
                removed += 1
                logger.info(
                    "[deterministic_imports] Stripped duplicate import: %s",
                    stripped,
                )
                continue
        clean_lines.append(line)

    if removed:
        logger.info(
            "[deterministic_imports] Removed %d duplicate import(s) from %s",
            removed, frozen_block.file_path,
        )

    return "\n".join(clean_lines)
