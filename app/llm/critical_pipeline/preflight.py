# FILE: app/llm/critical_pipeline/preflight.py
# Purpose: Pre-Flight Deterministic Enrichment for Architecture Prompt.
# Called-by: app.llm.critical_pipeline.stream_handler
# Depends-on: app.sandbox_fs
# Last-renovated: 2026-06-11
"""
Pre-Flight Deterministic Enrichment for Architecture Prompt.

Gathers file existence + AST exports for each file in scope from the
sandbox. Injected into the architecture prompt so the LLM knows which
files exist (MODIFY) vs don't exist (CREATE) before generating.

Zero LLM cost. Pure deterministic sandbox checks + AST parsing.

v1.0 (2026-03-05): Initial implementation — Job 5 from pipeline fix list.
"""
from __future__ import annotations

import ast
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def gather_preflight_facts(
    file_scope: List[str],
    project_roots: Optional[List[str]] = None,
) -> str:
    """Gather file existence + exports for each file in scope.

    For each file:
    - Check if it exists in the sandbox
    - If it exists: read content, AST-parse for exports, report size
    - If it doesn't exist: report as CREATE target

    Returns a formatted string ready for injection into the arch prompt.
    Returns empty string if file_scope is empty or sandbox unavailable.
    """
    if not file_scope:
        return ""

    try:
        from app.sandbox_fs import sandbox_isfile, sandbox_read_text
    except ImportError:
        logger.debug("[preflight] sandbox_fs not available")
        return ""

    if project_roots is None:
        project_roots = [r"D:\Orb", r"D:\orb-desktop"]

    lines = [
        "## FILE EXISTENCE (Pre-Flight — Deterministic)",
        "",
        "The following facts were gathered from the sandbox filesystem.",
        "Use these to determine CREATE vs MODIFY intent. **Do NOT guess.**",
        "",
    ]

    create_count = 0
    oversized_files = []  # v1.1: Track files over 30KB

    for rel_path in file_scope:
        # Resolve to absolute path
        abs_path = _resolve_path(rel_path, project_roots)
        found = sandbox_isfile(abs_path) if abs_path else False

        if found and abs_path:
            exists_count += 1
            content = sandbox_read_text(abs_path)
            if content:
                size_kb = round(len(content.encode("utf-8")) / 1024, 1)
                line_count = content.count("\n") + 1
                exports = _extract_exports(content, abs_path)
                exports_str = ", ".join(exports[:15]) if exports else "(no public exports found)"
                if len(exports) > 15:
                    exports_str += f", ... (+{len(exports) - 15} more)"

                lines.append(
                    f"**`{rel_path}`** — EXISTS ({size_kb}KB, {line_count} lines) → **MODIFY only**"
                )
                lines.append(f"  Exports: {exports_str}")

                # v1.1: Flag oversized files
                if size_kb > 30:
                    oversized_files.append((rel_path, size_kb, line_count, exports[:20]))
            else:
                lines.append(
                    f"**`{rel_path}`** — EXISTS (content unreadable) → **MODIFY only**"
                )
        else:
            create_count += 1
            lines.append(f"**`{rel_path}`** — DOES NOT EXIST → **CREATE**")

    lines.append("")
    lines.append(
        f"Summary: {exists_count} existing (MODIFY), {create_count} new (CREATE)"
    )
    lines.append("")

    # v1.1: Oversized file warnings
    if oversized_files:
        lines.append("### ⚠️ OVERSIZED FILES DETECTED")
        lines.append("")
        lines.append(
            "The following files EXCEED the 30 KB hard limit. "
            "Before modifying these files, assess whether they can be "
            "decomposed into smaller modules. If decomposition is feasible, "
            "include the decomposition in your architecture. "
            "Do NOT make an oversized file worse."
        )
        lines.append("")
        for path, kb, lc, exports in oversized_files:
            exports_str = ", ".join(exports) if exports else "(unknown)"
            lines.append(f"- **`{path}`** — {kb}KB ({lc} lines)")
            lines.append(f"  Public symbols: {exports_str}")
            if kb > 40:
                lines.append(f"  🔴 CRITICAL: {kb}KB — decomposition strongly recommended")
            else:
                lines.append(f"  🟡 WARNING: {kb}KB — check if it can be split")
        lines.append("")

    logger.info(
        "[preflight] File existence check: %d MODIFY, %d CREATE out of %d files (%d oversized)",
        exists_count, create_count, len(file_scope), len(oversized_files),
    )

    return "\n".join(lines)


def _resolve_path(
    rel_path: str,
    project_roots: List[str],
) -> Optional[str]:
    """Resolve a relative path to an absolute path."""
    # Already absolute
    if len(rel_path) > 2 and rel_path[1] == ":":
        return rel_path

    norm = rel_path.replace("/", os.sep).replace("\\", os.sep)

    # Handle orb-desktop/ prefix
    if norm.startswith("orb-desktop" + os.sep):
        stripped = norm[len("orb-desktop" + os.sep):]
        return os.path.join(r"D:\orb-desktop", stripped)

    # Try each root
    for root in project_roots:
        candidate = os.path.join(root, norm)
        return candidate  # Return first candidate, sandbox_isfile will verify

    return None


def _extract_exports(content: str, file_path: str) -> List[str]:
    """Extract public exports from file content using AST (Python) or regex (TS)."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".py":
        return _extract_python_exports(content)
    elif ext in (".ts", ".tsx", ".js", ".jsx"):
        return _extract_ts_exports(content)
    return []


def _extract_python_exports(content: str) -> List[str]:
    """Extract public function/class names from Python source."""
    exports = []
    try:
        tree = ast.parse(content)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("_"):
                    exports.append(node.name)
            elif isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    exports.append(node.name)
    except SyntaxError:
        pass
    return exports


def _extract_ts_exports(content: str) -> List[str]:
    """Extract exported names from TypeScript/JS source using regex."""
    import re
    exports = []
    for m in re.finditer(
        r"export\s+(?:default\s+)?(?:function|const|class|interface|type|enum)\s+(\w+)",
        content,
    ):
        exports.append(m.group(1))
    return exports
