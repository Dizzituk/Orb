# Purpose: spec runner utils 10
# Called-by: app.pot_spec.grounded._spec_runner_utils_11, app.pot_spec.grounded._spec_runner_utils_12, app.pot_spec.grounded._spec_runner_utils_13, app.pot_spec.grounded.spec_runner
# Depends-on: stdlib/third-party only
# Last-renovated: 2026-06-11
from __future__ import annotations
import os
from typing import Any, Optional


SPEC_RUNNER_BUILD_ID = "2026-02-14-v5.6-size-analyzer-integration"

_ARCH_INDEX_DIR = os.getenv("ASTRA_ARCH_INDEX_DIR", os.path.join("D:\\", "Orb", ".architecture"))

_ARCH_REPORT_DIR = os.getenv("ASTRA_ARCH_REPORT_DIR", os.path.join("D:\\", "Orb.architecture"))

_PRODUCT_SYNONYMS_RAW = os.getenv("ASTRA_PRODUCT_SYNONYMS", "orb=astra")

_FALLBACK_FRONTEND_PATHS = ['D:\\orb-desktop']

_FALLBACK_BACKEND_PATHS = ['D:\\Orb']

_FALLBACK_ALL_PATHS = ['D:\\orb-desktop', 'D:\\Orb']

def _build_simple_spec(
    goal: str,
    what_to_do: str,
    evidence: Optional[Any] = None,
) -> str:
    """
    Build a simple POT spec for CREATE/MODIFY jobs.
    
    No scan results - just what Weaver said to do.
    """
    lines = []
    
    lines.append("# SPoT Spec")
    lines.append("")
    
    lines.append("## Goal")
    lines.append("")
    if goal:
        lines.append(goal.split('\n')[0].strip())
    else:
        lines.append(what_to_do.split('\n')[0].strip() if what_to_do else "Complete the requested task")
    lines.append("")
    
    lines.append("## What to do")
    lines.append("")
    if what_to_do:
        for line in what_to_do.split('\n')[:10]:
            if line.strip():
                lines.append(line.strip())
    lines.append("")
    
    lines.append("## Acceptance")
    lines.append("")
    lines.append("- [ ] Task completed as specified")
    lines.append("- [ ] No errors")
    lines.append("")
    
    return "\n".join(lines)
