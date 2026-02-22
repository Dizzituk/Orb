# FILE: app/overwatcher/signature_checker.py
"""
Signature Checker — Deterministic Post-Implementation Signature Verification.

Layer 3 of Signature Contract Enforcement.

After the Implementer writes a file, this module compares every exported
function's signature against what the skeleton contract requires.  Zero LLM
calls — pure AST parsing.

Checks (in order of severity):
  1. Function exists at all
  2. async/sync matches
  3. Parameter count matches (excluding self/cls)
  4. Parameter types match (when both annotated)
  5. Return type matches (when both annotated)

Parameter NAMES are a soft heuristic only — the Implementer may use synonyms
(e.g. `content` vs `file_content`) and that's acceptable as long as count
and types match.

Mismatches are hard blocks that feed into the three-strike retry loop with
exact error messages showing the required signature.

v1.0 (2026-02-16): Initial implementation — Layer 3 safety net.
"""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from app.overwatcher._signature_checker_utils_4 import SIGNATURE_CHECKER_BUILD_ID, _CONTAINER_BUILTINS, _REVERSE_ALIASES, _TYPE_ALIASES, _extract_base_type, _normalise_type, _types_match, extract_contract_signatures_for_file
from app.overwatcher._signature_checker_utils_5 import SignatureCheckResult, _build_signature_string, _parse_params_from_node, check_file_signatures, compare_signatures

logger = logging.getLogger(__name__)
print(f"[SIGNATURE_CHECKER_LOADED] BUILD_ID={SIGNATURE_CHECKER_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FunctionSignature:
    """Parsed representation of a function signature."""
    name: str
    params: List[str]               # ["rel_path: str", "sandbox_base: str"]
    param_names: List[str]          # ["rel_path", "sandbox_base"]
    param_types: List[Optional[str]]  # ["str", "str"]
    return_type: Optional[str]      # "str", "Dict[str, Any]", None
    is_async: bool
    raw: str = ""                   # The original "def name(params) -> ret" string

    @property
    def param_count(self) -> int:
        return len(self.params)


@dataclass
class SignatureMismatch:
    """A single signature mismatch between contract and implementation."""
    function_name: str
    expected_signature: str     # Full "def name(params) -> ret" from contract
    actual_signature: str       # Full "def name(params) -> ret" from implementation
    differences: List[str]      # Human-readable: ["Missing parameter: sandbox_base: str"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "expected_signature": self.expected_signature,
            "actual_signature": self.actual_signature,
            "differences": self.differences,
        }


# =============================================================================
# TYPE NORMALISATION
# =============================================================================

# Map of Python 3.9+ builtins to their typing module equivalents

# Reverse aliases (typing -> builtin) — we normalise TO lowercase builtins


# v1.1: Builtin container types whose parameterised forms are equivalent
# to their bare form (e.g. ``dict`` matches ``dict[str, Any]``).


# =============================================================================
# AST EXTRACTION — Parse signatures from file content
# =============================================================================


def extract_signatures(file_content: str) -> Dict[str, FunctionSignature]:
    """
    Parse file content and extract all top-level function signatures.

    Also extracts methods from top-level classes (qualified as ClassName.method).

    Returns dict keyed by function name (or ClassName.method_name).
    """
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        logger.warning("[sig_checker] AST parse failed: %s", e)
        return {}

    results: Dict[str, FunctionSignature] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raw = _build_signature_string(node, file_content)
            params, param_names, param_types = _parse_params_from_node(node, file_content)
            ret_type = None
            if node.returns:
                ret_type = ast.get_source_segment(file_content, node.returns)
                if not ret_type:
                    try:
                        ret_type = ast.unparse(node.returns)
                    except (AttributeError, Exception):
                        pass

            results[node.name] = FunctionSignature(
                name=node.name,
                params=params,
                param_names=param_names,
                param_types=param_types,
                return_type=ret_type,
                is_async=isinstance(node, ast.AsyncFunctionDef),
                raw=raw,
            )

        elif isinstance(node, ast.ClassDef):
            for item in ast.iter_child_nodes(node):
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    raw = _build_signature_string(item, file_content)
                    params, param_names, param_types = _parse_params_from_node(item, file_content)
                    ret_type = None
                    if item.returns:
                        ret_type = ast.get_source_segment(file_content, item.returns)
                        if not ret_type:
                            try:
                                ret_type = ast.unparse(item.returns)
                            except (AttributeError, Exception):
                                pass

                    qualified_name = f"{node.name}.{item.name}"
                    results[qualified_name] = FunctionSignature(
                        name=qualified_name,
                        params=params,
                        param_names=param_names,
                        param_types=param_types,
                        return_type=ret_type,
                        is_async=isinstance(item, ast.AsyncFunctionDef),
                        raw=raw,
                    )
                    # Also store unqualified for matching flexibility
                    if item.name not in results:
                        results[item.name] = results[qualified_name]

    return results


# =============================================================================
# CONTRACT SIGNATURE PARSING
# =============================================================================


def parse_contract_signature(sig_string: str) -> Optional[FunctionSignature]:
    """
    Parse a contract signature string like:
        "def _resolve_multi_root_path(rel_path: str, sandbox_base: str) -> str"
        "async def run_execution(spec: ResolvedSpec) -> Dict[str, Any]"

    Returns FunctionSignature or None if unparseable.
    """
    sig = sig_string.strip()
    if not sig:
        return None

    # Must start with 'def ' or 'async def '
    if not (sig.startswith("def ") or sig.startswith("async def ")):
        return None

    # Try AST parsing by adding a stub body
    stub = sig + ":\n    pass"
    try:
        tree = ast.parse(stub)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                raw = _build_signature_string(node, stub)
                params, param_names, param_types = _parse_params_from_node(node, stub)
                ret_type = None
                if node.returns:
                    ret_type = ast.get_source_segment(stub, node.returns)
                    if not ret_type:
                        try:
                            ret_type = ast.unparse(node.returns)
                        except (AttributeError, Exception):
                            pass

                return FunctionSignature(
                    name=node.name,
                    params=params,
                    param_names=param_names,
                    param_types=param_types,
                    return_type=ret_type,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    raw=sig,
                )
    except SyntaxError:
        pass

    # Fallback: regex extraction for cases AST can't handle
    try:
        is_async = sig.startswith("async def ")
        prefix = "async def " if is_async else "def "
        rest = sig[len(prefix):]

        # Extract name
        paren_idx = rest.index("(")
        name = rest[:paren_idx].strip()

        # Extract params and return type
        # Find matching closing paren
        depth = 0
        params_end = paren_idx
        for i, ch in enumerate(rest[paren_idx:], paren_idx):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    params_end = i
                    break

        params_str = rest[paren_idx + 1:params_end].strip()
        after_params = rest[params_end + 1:].strip()

        ret_type = None
        if after_params.startswith("->"):
            ret_type = after_params[2:].strip().rstrip(":")

        # Parse individual params
        params = []
        param_names = []
        param_types = []
        if params_str:
            # Split on commas, but respect brackets
            current = ""
            bracket_depth = 0
            for ch in params_str:
                if ch in "([{":
                    bracket_depth += 1
                elif ch in ")]}":
                    bracket_depth -= 1
                if ch == "," and bracket_depth == 0:
                    params.append(current.strip())
                    current = ""
                else:
                    current += ch
            if current.strip():
                params.append(current.strip())

            for p in params:
                # Strip default values for comparison
                p_no_default = p.split("=")[0].strip() if "=" in p else p
                if ":" in p_no_default:
                    pname, ptype = p_no_default.split(":", 1)
                    param_names.append(pname.strip())
                    param_types.append(ptype.strip())
                else:
                    param_names.append(p_no_default.strip())
                    param_types.append(None)

        return FunctionSignature(
            name=name,
            params=params,
            param_names=param_names,
            param_types=param_types,
            return_type=ret_type,
            is_async=is_async,
            raw=sig,
        )
    except (ValueError, IndexError) as e:
        logger.warning("[sig_checker] Regex fallback failed for '%s': %s", sig_string[:80], e)
        return None


# =============================================================================
# SIGNATURE COMPARISON
# =============================================================================


# =============================================================================
# CONTRACT MARKDOWN PARSING
# =============================================================================


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FunctionSignature",
    "SignatureMismatch",
    "SignatureCheckResult",
    "extract_signatures",
    "parse_contract_signature",
    "compare_signatures",
    "extract_contract_signatures_for_file",
    "check_file_signatures",
    "SIGNATURE_CHECKER_BUILD_ID",
]
