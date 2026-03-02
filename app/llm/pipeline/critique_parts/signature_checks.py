# FILE: app/llm/pipeline/critique_parts/signature_checks.py
"""
Deterministic Critique — Function Signature Matching.

Check 4: Function signature matching
    Exported function signatures in the architecture match what
    consuming segments expect. Parses function declarations from
    architecture markdown and compares against skeleton contract
    export bindings with enrichment-augmented signatures.

Zero LLM calls. Pure structural comparison.

v1.0 (2026-02-27): Initial implementation — Stage 1 of deterministic
verification migration.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SIGNATURE_CHECKS_BUILD_ID = "2026-02-27-v1.0-signature-matching"


# =========================================================================
# SIGNATURE PARSING
# =========================================================================

def _parse_function_signatures(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Extract function signatures from architecture text.

    Parses patterns like:
        def func_name(param1: Type, param2: Type = default) -> ReturnType:
        async def func_name(param1, param2):

    Returns {func_name: {"params": [...], "return_type": str, "is_async": bool}}
    """
    signatures: Dict[str, Dict[str, Any]] = {}

    for m in re.finditer(
        r'(async\s+)?def\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\S+))?\s*:',
        text,
    ):
        is_async = bool(m.group(1))
        func_name = m.group(2)
        params_str = m.group(3).strip()
        return_type = m.group(4).strip() if m.group(4) else None

        # Parse parameters
        params: List[Dict[str, str]] = []
        if params_str and params_str != "self":
            for param in params_str.split(","):
                param = param.strip()
                if not param or param == "self" or param == "cls":
                    continue
                if param.startswith("*") or param.startswith("**"):
                    params.append({"name": param, "type": None, "has_default": False})
                    continue

                # Split name: Type = default
                has_default = "=" in param
                if has_default:
                    param = param.split("=")[0].strip()
                if ":" in param:
                    pname, ptype = param.split(":", 1)
                    params.append({
                        "name": pname.strip(),
                        "type": ptype.strip(),
                        "has_default": has_default,
                    })
                else:
                    params.append({
                        "name": param.strip(),
                        "type": None,
                        "has_default": has_default,
                    })

        signatures[func_name] = {
            "params": params,
            "return_type": return_type,
            "is_async": is_async,
        }

    return signatures


def _parse_enrichment_signatures(enrichment: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Extract function signatures from enrichment data.

    Enrichment functions have a 'signature' field with the raw def line.
    """
    signatures: Dict[str, Dict[str, Any]] = {}

    for func in enrichment.get("functions", []):
        if not isinstance(func, dict):
            continue
        name = func.get("name", "")
        sig_str = func.get("signature", "")
        if name and sig_str:
            parsed = _parse_function_signatures(sig_str)
            if name in parsed:
                signatures[name] = parsed[name]

    return signatures


# =========================================================================
# SIGNATURE COMPARISON
# =========================================================================

def _compare_signatures(
    arch_sig: Dict[str, Any],
    expected_sig: Dict[str, Any],
    func_name: str,
) -> List[str]:
    """
    Compare two function signatures and return list of mismatch descriptions.

    Compares:
    - Parameter count (excluding *args/**kwargs)
    - Required parameter names
    - Return type (if both specify one)
    - async/sync mismatch
    """
    mismatches: List[str] = []

    # Async mismatch
    if arch_sig.get("is_async") != expected_sig.get("is_async"):
        arch_kind = "async" if arch_sig.get("is_async") else "sync"
        exp_kind = "async" if expected_sig.get("is_async") else "sync"
        mismatches.append(f"{func_name}: architecture is {arch_kind} but expected {exp_kind}")

    # Parameter comparison
    arch_params = [p for p in arch_sig.get("params", []) if not p["name"].startswith("*")]
    exp_params = [p for p in expected_sig.get("params", []) if not p["name"].startswith("*")]

    # Required params (no default)
    arch_required = [p["name"] for p in arch_params if not p.get("has_default")]
    exp_required = [p["name"] for p in exp_params if not p.get("has_default")]

    if len(arch_required) != len(exp_required):
        mismatches.append(
            f"{func_name}: architecture has {len(arch_required)} required params "
            f"({', '.join(arch_required)}) but expected {len(exp_required)} "
            f"({', '.join(exp_required)})"
        )
    else:
        # Check param names match
        for a, e in zip(arch_required, exp_required):
            if a != e:
                mismatches.append(
                    f"{func_name}: parameter name mismatch — architecture has "
                    f"'{a}' where '{e}' is expected"
                )

    # Return type comparison (only if both specify)
    arch_ret = arch_sig.get("return_type")
    exp_ret = expected_sig.get("return_type")
    if arch_ret and exp_ret and arch_ret != exp_ret:
        mismatches.append(
            f"{func_name}: return type mismatch — architecture says "
            f"'{arch_ret}' but expected '{exp_ret}'"
        )

    return mismatches


# =========================================================================
# CHECK 4: Function Signature Matching
# =========================================================================

def check_function_signatures(
    arch_content: str,
    segment_id: str,
    skeleton_contract: Optional[Dict[str, Any]] = None,
    enrichment_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Check that exported function signatures in the architecture match
    what consuming segments expect.

    Uses enrichment data for expected signatures. If enrichment provides
    full signature strings, parses and compares parameter lists and
    return types.

    Args:
        arch_content: Architecture markdown document
        segment_id: This segment's ID
        skeleton_contract: Full skeleton_contract.json dict
        enrichment_data: Enrichment data keyed by segment_id

    Returns:
        List of issue dicts
    """
    issues: List[Dict[str, Any]] = []

    if not enrichment_data or not skeleton_contract:
        return issues

    # Find which symbols this segment must expose
    must_expose: Set[str] = set()
    if skeleton_contract:
        for skel in skeleton_contract.get("skeletons", []):
            if skel.get("segment_id") == segment_id:
                for export in skel.get("exports", []):
                    names = export.get("names", [])
                    if isinstance(names, list):
                        must_expose.update(names)
                break

    if not must_expose:
        return issues

    # Get expected signatures from this segment's enrichment
    seg_enrichment = enrichment_data.get(segment_id, {})
    if not isinstance(seg_enrichment, dict):
        return issues

    expected_sigs = _parse_enrichment_signatures(seg_enrichment)

    # Parse architecture signatures
    arch_sigs = _parse_function_signatures(arch_content)

    # Compare each exposed function
    for func_name in must_expose:
        if func_name not in expected_sigs:
            continue  # No expected signature to compare against

        if func_name not in arch_sigs:
            # Function expected to be exported but not defined in architecture
            issues.append({
                "rule_id": "DET-SIG-MISSING",
                "severity": "warning",
                "file": segment_id,
                "spec_ref": f"skeleton_contract.exports.{func_name}",
                "arch_ref": "Function definitions",
                "description": (
                    f"Function '{func_name}' is required as an export but no "
                    f"matching function definition found in the architecture."
                ),
                "suggested_fix": (
                    f"Add 'def {func_name}(...)' to the architecture with the "
                    f"expected signature."
                ),
            })
            continue

        # Compare signatures
        mismatches = _compare_signatures(
            arch_sigs[func_name],
            expected_sigs[func_name],
            func_name,
        )
        for mismatch in mismatches:
            issues.append({
                "rule_id": "DET-SIG-MISMATCH",
                "severity": "warning",
                "file": segment_id,
                "spec_ref": f"enrichment.{segment_id}.{func_name}",
                "arch_ref": f"def {func_name}(...)",
                "description": mismatch,
                "suggested_fix": (
                    f"Align the function signature with the expected interface."
                ),
            })

    if issues:
        logger.info(
            "[det_critique] Signature check: %d issues for %s",
            len(issues), segment_id,
        )

    return issues


__all__ = [
    "check_function_signatures",
    "SIGNATURE_CHECKS_BUILD_ID",
]
