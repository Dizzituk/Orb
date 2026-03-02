# FILE: app/overwatcher/import_dedup.py
"""
Import deduplication for TypeScript/React files.

v3.4 (2026-03-02): Merges duplicate imports from the same module.
When scaffold templates and LLM output both contain imports from the
same source (e.g. two `import { useState } from 'react'` lines),
the duplicates are merged into a single import with all specifiers.

Only processes .ts, .tsx, .js, .jsx files. Other file types pass through
unchanged.
"""

from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Match: import { X, Y } from 'module';
# Also:  import { X, Y } from "module";
# Also:  import type { X } from 'module';
_NAMED_IMPORT_RE = re.compile(
    r"^(\s*import\s+(?:type\s+)?)\{([^}]+)\}\s+from\s+(['\"])([^'\"]+)\3\s*;?\s*$"
)

# Match: import X from 'module';
_DEFAULT_IMPORT_RE = re.compile(
    r"^(\s*import\s+)(\w+)\s+from\s+(['\"])([^'\"]+)\3\s*;?\s*$"
)

# Match: import 'module'; (side-effect only)
_SIDE_EFFECT_IMPORT_RE = re.compile(
    r"^\s*import\s+(['\"])([^'\"]+)\1\s*;?\s*$"
)

# File extensions that should be deduplicated
_DEDUP_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}


def deduplicate_imports(content: str, path: str) -> str:
    """Merge duplicate imports from the same module.

    For named imports (`import { X } from 'mod'`), all specifiers from
    the same module are merged into a single import line. The last
    occurrence's position is kept; earlier duplicates are removed.

    Default imports and side-effect imports are left untouched (only
    exact-duplicate lines are removed for those).

    Args:
        content: File content string.
        path: File path (used for extension check and logging).

    Returns:
        Content with deduplicated imports. Unchanged if not a JS/TS file.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _DEDUP_EXTENSIONS:
        return content

    lines = content.split("\n")
    # Track named imports per module: module -> {specifiers}
    # We'll collect all, then on second pass replace/remove
    module_specifiers: Dict[str, List[str]] = OrderedDict()
    module_type_specifiers: Dict[str, List[str]] = OrderedDict()
    import_line_indices: Dict[str, List[int]] = {}
    type_import_line_indices: Dict[str, List[int]] = {}

    for i, line in enumerate(lines):
        m = _NAMED_IMPORT_RE.match(line)
        if m:
            prefix = m.group(1).strip()
            specs = [s.strip() for s in m.group(2).split(",") if s.strip()]
            module = m.group(4)
            is_type = "type" in prefix

            if is_type:
                if module not in module_type_specifiers:
                    module_type_specifiers[module] = []
                    type_import_line_indices[module] = []
                module_type_specifiers[module].extend(specs)
                type_import_line_indices[module].append(i)
            else:
                if module not in module_specifiers:
                    module_specifiers[module] = []
                    import_line_indices[module] = []
                module_specifiers[module].extend(specs)
                import_line_indices[module].append(i)

    # Check if any dedup needed
    needs_dedup = any(
        len(indices) > 1
        for indices in list(import_line_indices.values()) + list(type_import_line_indices.values())
    )
    if not needs_dedup:
        return content

    # Build merged lines and mark removals
    remove_lines = set()
    total_merged = 0

    for module, indices in import_line_indices.items():
        if len(indices) <= 1:
            continue
        # Deduplicate specifiers preserving order
        seen = set()
        unique_specs = []
        for s in module_specifiers[module]:
            if s not in seen:
                seen.add(s)
                unique_specs.append(s)
        # Keep last occurrence, remove earlier ones
        keep_idx = indices[-1]
        for idx in indices[:-1]:
            remove_lines.add(idx)
        # Replace the kept line with merged import
        spec_str = ", ".join(unique_specs)
        lines[keep_idx] = f"import {{ {spec_str} }} from '{module}';"
        total_merged += len(indices) - 1

    for module, indices in type_import_line_indices.items():
        if len(indices) <= 1:
            continue
        seen = set()
        unique_specs = []
        for s in module_type_specifiers[module]:
            if s not in seen:
                seen.add(s)
                unique_specs.append(s)
        keep_idx = indices[-1]
        for idx in indices[:-1]:
            remove_lines.add(idx)
        spec_str = ", ".join(unique_specs)
        lines[keep_idx] = f"import type {{ {spec_str} }} from '{module}';"
        total_merged += len(indices) - 1

    if total_merged > 0:
        logger.warning(
            "[import_dedup] v3.4 Merged %d duplicate import(s) in %s",
            total_merged, path,
        )

    # Remove duplicate lines
    result = [line for i, line in enumerate(lines) if i not in remove_lines]
    return "\n".join(result)
