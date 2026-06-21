# FILE: app/pipeline_v2/scaffolds/kotlin_template_helpers.py
# Purpose: Shared formatting helpers for the Kotlin scaffold templates.
# Called-by: app.pipeline_v2.scaffolds.kotlin_templates_base, app.pipeline_v2.scaffolds.kotlin_templates_android, kotlin_scaffolds (shim)
# Depends-on: (stdlib only)
# Last-renovated: 2026-06-21
"""
Shared helpers for the Kotlin scaffold templates.

Split out of kotlin_scaffolds.py (BATCH 4) verbatim. Both the v1 base and v2
Android template cohorts call these, so they live in their own leaf (not the
shim) to keep the template modules acyclic.
"""
from __future__ import annotations

from typing import List


def _req_block(requirements: List[str]) -> str:
    """Format requirements as a Kotlin doc comment block."""
    if not requirements:
        return ""
    lines = [" * Requirements:"]
    for r in requirements:
        lines.append(f" *   - {r}")
    return "\n".join(lines) + "\n"


def _to_snake(name: str) -> str:
    """PascalCase → snake_case."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append("_")
        result.append(c.lower())
    return "".join(result)


def _to_display_name(name: str) -> str:
    """PascalCase → 'Display Name'."""
    result = []
    for i, c in enumerate(name):
        if c.isupper() and i > 0:
            result.append(" ")
        result.append(c)
    return "".join(result)
