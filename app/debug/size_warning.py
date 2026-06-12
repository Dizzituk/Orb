# FILE: app/debug/size_warning.py
# Purpose: Universal file size warning for tool results.
# Called-by: app.debug.executors.filesystem, app.debug.executors.user_files
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
"""
Universal file size warning for tool results.

v0.15.0 (2026-03-25): Prepends size warnings to file content returned
by any read tool (read_file, read_user_file, etc.) when the file
exceeds the 30KB hard limit.

Only applies to code files (by extension). Data files, logs, markdown,
JSON configs are excluded — the size cap is about logic, not data.
"""
from __future__ import annotations

import os

_CODE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".kt", ".java", ".css",
    ".html", ".vue", ".svelte", ".go", ".rs", ".rb", ".php",
}

SIZE_TARGET_KB = 20
SIZE_HARD_MAX_KB = 30
SIZE_CRITICAL_KB = 40


def add_size_warning(content: str, path: str) -> str:
    """Prepend a size warning to file content if the file exceeds limits.

    Only applies to code files (by extension). Data files, logs, configs
    are excluded. Returns content unchanged if under limits or non-code.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in _CODE_EXTENSIONS:
        return content

    size_kb = round(len(content.encode("utf-8", errors="replace")) / 1024, 1)
    if size_kb <= SIZE_HARD_MAX_KB:
        return content

    line_count = content.count("\n") + 1
    sep = "=" * 60

    if size_kb > SIZE_CRITICAL_KB:
        severity = "CRITICAL"
        advice = (
            "This file is FAR over the 30KB limit. "
            "It should be decomposed into smaller modules before any new "
            "code is added. Propose a refactoring plan if asked to modify it."
        )
    else:
        severity = "WARNING"
        advice = (
            "This file exceeds the 30KB hard limit. "
            "If you are modifying it, do not make it larger. "
            "Consider whether it can be split into smaller modules."
        )

    warning = (
        f"SIZE ALERT — {severity}: {path}\n"
        f"   Size: {size_kb}KB | Lines: {line_count} | Limit: {SIZE_HARD_MAX_KB}KB\n"
        f"   {advice}\n"
        f"{sep}\n\n"
    )
    return warning + content
