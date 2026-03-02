# FILE: app/llm/pipeline/critique_parts/prohibition_checks.py
"""
Deterministic Critique — Prohibited Pattern Detection.

Check 6: Prohibited pattern detection
    Flag architecture that defines functions/classes already owned
    by other segments. Enforces the "do not define" prohibition
    from skeleton contracts.

Zero LLM calls. Cross-references architecture definitions against
other segments' owned symbols.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

PROHIBITION_CHECKS_BUILD_ID = "2026-02-27-v1.0-prohibited-definitions"


# =========================================================================
# EXTRACT DEFINITIONS FROM ARCHITECTURE
# =========================================================================

def _extract_definitions(arch_content: str) -> Set[str]:
    """
    Extract all function, class, and constant definitions from architecture.

    Returns set of defined names.
    """
    defined: Set[str] = set()

    # Function definitions
    for m in re.finditer(r'(?:async\s+)?def\s+(\w+)\s*\(', arch_content):
        name = m.group(1)
        if not (name.startswith("__") and name.endswith("__")):
            defined.add(name)

    # Class definitions
    for m in re.finditer(r'class\s+(\w+)\s*[(:[\])]', arch_content):
        defined.add(m.group(1))

    # Constants (ALL_CAPS = value)
    for m in re.finditer(r'^([A-Z][A-Z0-9_]+)\s*=', arch_content, re.MULTILINE):
        defined.add(m.group(1))

    return defined


# =========================================================================
# CHECK 6: Prohibited Pattern Detection
# =========================================================================

def check_prohibited_definitions(
    arch_content: str,
    segment_id: str,
    skeleton_contract: Optional[Dict[str, Any]] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
    manifest_dict: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Flag architecture that defines symbols owned by other segments.

    The skeleton contract's "do not define" prohibition means each
    segment should only define symbols within its own scope. If
    this architecture defines a function/class that another segment
    owns, it's a contract violation.

    Args:
        arch_content: Architecture markdown document
        segment_id: This segment's ID
        skeleton_contract: Full skeleton_contract.json dict
        enrichment_data: Enrichment data keyed by segment_id
        manifest_dict: Full manifest dict

    Returns:
        List of issue dicts
    """
    issues: List[Dict[str, Any]] = []

    if not enrichment_data and not manifest_dict:
        return issues

    # Build map of symbol -> owning segment (excluding this segment)
    symbol_owners: Dict[str, str] = {}  # symbol_name -> segment_id

    if enrichment_data:
        for sid, edata in enrichment_data.items():
            if sid == segment_id:
                continue
            if not isinstance(edata, dict):
                continue
            for func in edata.get("functions", []):
                name = func.get("name", "") if isinstance(func, dict) else str(func)
                if name:
                    symbol_owners[name] = sid
            for exp in edata.get("exports", []):
                if isinstance(exp, str) and exp:
                    symbol_owners[exp] = sid

    if not symbol_owners:
        return issues

    # Extract definitions from this architecture
    arch_definitions = _extract_definitions(arch_content)

    # Check for violations
    for name in sorted(arch_definitions):
        if name in symbol_owners:
            owner = symbol_owners[name]
            issues.append({
                "rule_id": "DET-PROHIBIT-REDEFINE",
                "severity": "blocking",
                "file": segment_id,
                "spec_ref": f"skeleton_contract.{owner}.exports",
                "arch_ref": f"def {name}",
                "description": (
                    f"Architecture for {segment_id} defines '{name}' which is "
                    f"owned by {owner}. This segment should import '{name}' "
                    f"from {owner}'s module, not redefine it."
                ),
                "suggested_fix": (
                    f"Remove the definition of '{name}' and add "
                    f"'from .<module> import {name}' instead."
                ),
            })

    if issues:
        logger.info(
            "[det_critique] Prohibition check: %d violations for %s",
            len(issues), segment_id,
        )

    return issues


__all__ = [
    "check_prohibited_definitions",
    "PROHIBITION_CHECKS_BUILD_ID",
]
