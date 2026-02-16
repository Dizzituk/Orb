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

logger = logging.getLogger(__name__)

SIGNATURE_CHECKER_BUILD_ID = "2026-02-16-v1.0-layer3-safety-net"
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


@dataclass
class SignatureCheckResult:
    """Result of signature verification for one file."""
    passed: bool
    file_path: str
    mismatches: List[SignatureMismatch] = field(default_factory=list)
    missing_functions: List[str] = field(default_factory=list)
    extra_info: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "file_path": self.file_path,
            "mismatches": [m.to_dict() for m in self.mismatches],
            "missing_functions": self.missing_functions,
            "extra_info": self.extra_info,
        }


# =============================================================================
# TYPE NORMALISATION
# =============================================================================

# Map of Python 3.9+ builtins to their typing module equivalents
_TYPE_ALIASES = {
    "dict": "Dict",
    "list": "List",
    "tuple": "Tuple",
    "set": "Set",
    "frozenset": "FrozenSet",
    "type": "Type",
}

# Reverse aliases (typing -> builtin) — we normalise TO lowercase builtins
_REVERSE_ALIASES = {v: k for k, v in _TYPE_ALIASES.items()}


def _normalise_type(type_str: Optional[str]) -> Optional[str]:
    """
    Normalise a type annotation string for comparison.

    Handles:
      - Dict vs dict, List vs list, etc.
      - Optional[X] vs X | None vs Union[X, None]
      - Whitespace differences
      - Quoting differences

    Returns normalised string, or None if input is None/empty.
    """
    if not type_str or not type_str.strip():
        return None

    t = type_str.strip()

    # Strip quotes (forward references)
    t = t.strip("'\"")

    # Normalise whitespace
    t = re.sub(r'\s+', ' ', t).strip()

    # Normalise Optional[X] -> X | None
    optional_match = re.match(r'^Optional\[(.+)\]$', t)
    if optional_match:
        inner = optional_match.group(1).strip()
        t = f"{inner} | None"

    # Normalise Union[X, None] -> X | None
    union_match = re.match(r'^Union\[(.+),\s*None\]$', t)
    if union_match:
        inner = union_match.group(1).strip()
        t = f"{inner} | None"

    # Normalise Dict -> dict, List -> list, etc. (case-sensitive mapping)
    for typing_name, builtin_name in _REVERSE_ALIASES.items():
        # Replace at word boundary: Dict[str, Any] -> dict[str, Any]
        t = re.sub(rf'\b{typing_name}\b', builtin_name, t)

    # Also handle lowercase -> lowercase (already normalised, but ensure)
    # No-op since we already mapped to lowercase

    return t


# =============================================================================
# AST EXTRACTION — Parse signatures from file content
# =============================================================================


def _build_signature_string(node: ast.FunctionDef, source: str) -> str:
    """
    Build a 'def name(params) -> ret' string from an AST node.

    Uses ast.get_source_segment for type annotations when available,
    falls back to ast.dump for complex cases.
    """
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    name = node.name

    # Build parameter list
    params = []
    args = node.args

    # Positional args (skip self/cls)
    all_args = args.args or []
    start_idx = 0
    if all_args and all_args[0].arg in ("self", "cls"):
        start_idx = 1

    # Calculate defaults alignment: defaults align to the END of the args list
    num_defaults = len(args.defaults or [])
    num_positional = len(all_args) - start_idx
    default_offset = num_positional - num_defaults

    for i, arg in enumerate(all_args[start_idx:]):
        param = arg.arg
        if arg.annotation:
            ann = ast.get_source_segment(source, arg.annotation)
            if ann:
                param = f"{param}: {ann}"
            else:
                # Fallback: try to unparse (Python 3.9+)
                try:
                    param = f"{param}: {ast.unparse(arg.annotation)}"
                except (AttributeError, Exception):
                    pass

        # Check for default value
        if i >= default_offset and num_defaults > 0:
            default_idx = i - default_offset
            if 0 <= default_idx < num_defaults:
                default_node = args.defaults[default_idx]
                default_str = ast.get_source_segment(source, default_node)
                if not default_str:
                    try:
                        default_str = ast.unparse(default_node)
                    except (AttributeError, Exception):
                        default_str = "..."
                param = f"{param} = {default_str}"

        params.append(param)

    # *args
    if args.vararg:
        vp = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            ann = ast.get_source_segment(source, args.vararg.annotation)
            if ann:
                vp = f"*{args.vararg.arg}: {ann}"
        params.append(vp)
    elif args.kwonlyargs:
        # Bare * separator when there are keyword-only args but no *args
        params.append("*")

    # Keyword-only args
    for j, kwarg in enumerate(args.kwonlyargs or []):
        kp = kwarg.arg
        if kwarg.annotation:
            ann = ast.get_source_segment(source, kwarg.annotation)
            if ann:
                kp = f"{kp}: {ann}"
        if j < len(args.kw_defaults or []) and args.kw_defaults[j] is not None:
            default_str = ast.get_source_segment(source, args.kw_defaults[j])
            if not default_str:
                try:
                    default_str = ast.unparse(args.kw_defaults[j])
                except (AttributeError, Exception):
                    default_str = "..."
            kp = f"{kp} = {default_str}"
        params.append(kp)

    # **kwargs
    if args.kwarg:
        kp = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            ann = ast.get_source_segment(source, args.kwarg.annotation)
            if ann:
                kp = f"**{args.kwarg.arg}: {ann}"
        params.append(kp)

    params_str = ", ".join(params)

    # Return type
    ret = ""
    if node.returns:
        ret_ann = ast.get_source_segment(source, node.returns)
        if not ret_ann:
            try:
                ret_ann = ast.unparse(node.returns)
            except (AttributeError, Exception):
                ret_ann = None
        if ret_ann:
            ret = f" -> {ret_ann}"

    return f"{prefix} {name}({params_str}){ret}"


def _parse_params_from_node(node: ast.FunctionDef, source: str) -> Tuple[List[str], List[str], List[Optional[str]]]:
    """
    Extract parameter details from an AST FunctionDef node.

    Returns: (params, param_names, param_types)
      - params: ["rel_path: str", "sandbox_base: str"]
      - param_names: ["rel_path", "sandbox_base"]
      - param_types: ["str", "str"]
    """
    params = []
    param_names = []
    param_types = []

    args = node.args
    all_args = args.args or []
    start_idx = 0
    if all_args and all_args[0].arg in ("self", "cls"):
        start_idx = 1

    for arg in all_args[start_idx:]:
        name = arg.arg
        param_names.append(name)
        if arg.annotation:
            ann = ast.get_source_segment(source, arg.annotation)
            if not ann:
                try:
                    ann = ast.unparse(arg.annotation)
                except (AttributeError, Exception):
                    ann = None
            if ann:
                params.append(f"{name}: {ann}")
                param_types.append(ann)
            else:
                params.append(name)
                param_types.append(None)
        else:
            params.append(name)
            param_types.append(None)

    # *args
    if args.vararg:
        param_names.append(f"*{args.vararg.arg}")
        ann = None
        if args.vararg.annotation:
            ann = ast.get_source_segment(source, args.vararg.annotation)
        params.append(f"*{args.vararg.arg}" + (f": {ann}" if ann else ""))
        param_types.append(ann)

    # keyword-only args
    for kwarg in (args.kwonlyargs or []):
        param_names.append(kwarg.arg)
        ann = None
        if kwarg.annotation:
            ann = ast.get_source_segment(source, kwarg.annotation)
            if not ann:
                try:
                    ann = ast.unparse(kwarg.annotation)
                except (AttributeError, Exception):
                    pass
        if ann:
            params.append(f"{kwarg.arg}: {ann}")
        else:
            params.append(kwarg.arg)
        param_types.append(ann)

    # **kwargs
    if args.kwarg:
        param_names.append(f"**{args.kwarg.arg}")
        ann = None
        if args.kwarg.annotation:
            ann = ast.get_source_segment(source, args.kwarg.annotation)
        params.append(f"**{args.kwarg.arg}" + (f": {ann}" if ann else ""))
        param_types.append(ann)

    return params, param_names, param_types


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


def compare_signatures(
    required: FunctionSignature,
    actual: FunctionSignature,
) -> Optional[SignatureMismatch]:
    """
    Compare a required (contract) signature against an actual (implementation).

    Returns SignatureMismatch if they differ, None if they match.

    Comparison rules:
      1. async/sync must match — hard check
      2. Parameter count must match — hard check
      3. Parameter types must match when both annotated — hard check
      4. Return type must match when both annotated — hard check
      5. Parameter names — soft heuristic, only reported when count matches
         and types aren't available for disambiguation
    """
    differences = []

    # 1. async/sync
    if required.is_async != actual.is_async:
        expected = "async def" if required.is_async else "def"
        got = "async def" if actual.is_async else "def"
        differences.append(f"Expected {expected}, got {got}")

    # 2. Parameter count (filter out *args/**kwargs for positional count)
    req_positional = [n for n in required.param_names if not n.startswith("*")]
    act_positional = [n for n in actual.param_names if not n.startswith("*")]

    if len(req_positional) != len(act_positional):
        differences.append(
            f"Parameter count: expected {len(req_positional)}, got {len(act_positional)}"
        )
        # List missing/extra params
        if len(req_positional) > len(act_positional):
            for i in range(len(act_positional), len(req_positional)):
                if i < len(required.params):
                    differences.append(f"Missing parameter: {required.params[i]}")
        elif len(act_positional) > len(req_positional):
            for i in range(len(req_positional), len(act_positional)):
                if i < len(actual.params):
                    differences.append(f"Extra parameter: {actual.params[i]}")

    # 3. Parameter types (when counts match)
    if len(req_positional) == len(act_positional):
        for i in range(len(req_positional)):
            req_type = required.param_types[i] if i < len(required.param_types) else None
            act_type = actual.param_types[i] if i < len(actual.param_types) else None

            if req_type and act_type:
                req_norm = _normalise_type(req_type)
                act_norm = _normalise_type(act_type)
                if req_norm and act_norm and req_norm != act_norm:
                    req_name = req_positional[i] if i < len(req_positional) else f"param_{i}"
                    act_name = act_positional[i] if i < len(act_positional) else f"param_{i}"
                    differences.append(
                        f"Parameter {i} type: expected `{req_type}` (for `{req_name}`), "
                        f"got `{act_type}` (for `{act_name}`)"
                    )

    # 4. Return type
    req_ret = _normalise_type(required.return_type)
    act_ret = _normalise_type(actual.return_type)
    if req_ret and act_ret and req_ret != act_ret:
        differences.append(
            f"Return type: expected `{required.return_type}`, got `{actual.return_type}`"
        )

    # 5. Parameter names — soft heuristic only
    # Only flag when: counts match, no type mismatches found, and names differ
    # This catches swapped-argument-order bugs like (file_path, content) vs (content, file_path)
    if (len(req_positional) == len(act_positional)
            and not any("Parameter" in d and "type:" in d for d in differences)):
        name_diffs = []
        for i in range(len(req_positional)):
            if req_positional[i] != act_positional[i]:
                # Check if both have types and they're the same — if so, name diff is cosmetic
                req_type = required.param_types[i] if i < len(required.param_types) else None
                act_type = actual.param_types[i] if i < len(actual.param_types) else None
                if req_type and act_type:
                    req_norm = _normalise_type(req_type)
                    act_norm = _normalise_type(act_type)
                    if req_norm == act_norm:
                        continue  # Same type, different name — cosmetic, skip
                name_diffs.append(
                    f"Parameter {i} name: expected `{req_positional[i]}`, "
                    f"got `{act_positional[i]}`"
                )

        # Check for argument reordering regardless of whether individual
        # name diffs were suppressed.  Swapping (file_path: str, content: str)
        # to (content: str, file_path: str) is invisible to per-position type
        # checks when types are identical, but it's still a real bug.
        req_set = set(req_positional)
        act_set = set(act_positional)
        if req_set == act_set and req_positional != act_positional:
            differences.append(
                f"Parameter order may be wrong: expected ({', '.join(req_positional)}), "
                f"got ({', '.join(act_positional)})"
            )
        elif name_diffs:
            # Names differ AND it's not a simple reorder — flag them
            differences.extend(name_diffs)

    if differences:
        return SignatureMismatch(
            function_name=required.name,
            expected_signature=required.raw,
            actual_signature=actual.raw,
            differences=differences,
        )

    return None


# =============================================================================
# CONTRACT MARKDOWN PARSING
# =============================================================================


def extract_contract_signatures_for_file(
    interface_contract: str,
    file_path: str,
) -> List[str]:
    """
    Parse the interface_contract markdown (from format_contract_for_segment())
    to find signature strings for a specific file.

    The contract format has:
        - `path/to/file.py` → consumed by `seg-05`, `seg-06`
          **MUST EXPORT these symbols** (downstream segments depend on them):
            - `def func_name(params) -> ret`
            - `bare_name`

    Returns list of signature strings (only the "def ..." ones, not bare names).
    """
    if not interface_contract or not file_path:
        return []

    signatures = []
    file_path_norm = file_path.replace("\\", "/").strip()

    # State machine: find the file path line, then collect signatures
    lines = interface_contract.split("\n")
    in_file_section = False
    in_exports = False

    for line in lines:
        stripped = line.strip()

        # Detect file path line: "  - `path/to/file.py` → consumed by ..."
        if f"`{file_path_norm}`" in stripped:
            in_file_section = True
            in_exports = False
            continue

        if in_file_section:
            # Detect MUST EXPORT header
            if "MUST EXPORT" in stripped:
                in_exports = True
                continue

            # Detect end of this file's section
            # New section header always ends the file section
            if stripped.startswith("###") or stripped.startswith("## "):
                in_file_section = False
                in_exports = False
                continue

            # New file entry: starts with "- `", contains a file path (has /)
            # but is NOT a function signature (doesn't contain 'def ')
            if stripped.startswith("- `") and "`" in stripped[3:]:
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    candidate = match.group(1).strip()
                    # File paths contain '/' or end with '.py'; signatures start with 'def '/'async def '
                    is_file_path = ("/" in candidate or candidate.endswith(".py"))
                    is_signature = candidate.startswith("def ") or candidate.startswith("async def ")
                    if is_file_path and not is_signature:
                        if candidate.replace("\\", "/") != file_path_norm:
                            in_file_section = False
                            in_exports = False
                            continue

            # Collect signature lines
            if in_exports and stripped.startswith("- `"):
                # Extract content between backticks
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    sig = match.group(1).strip()
                    # Only include actual function signatures (def ... or async def ...)
                    if sig.startswith("def ") or sig.startswith("async def "):
                        signatures.append(sig)

    if signatures:
        logger.debug(
            "[sig_checker] Found %d contract signature(s) for %s",
            len(signatures), file_path,
        )

    return signatures


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def check_file_signatures(
    file_content: str,
    file_path: str,
    contract_signatures: List[str],
) -> SignatureCheckResult:
    """
    Main entry point: verify a written file's signatures against contract.

    Args:
        file_content: The actual file content that was written
        file_path: Relative path (for logging/display)
        contract_signatures: List of "def name(params) -> ret" strings
                             from the skeleton contract's ExportBinding.signatures

    Returns:
        SignatureCheckResult with pass/fail and detailed mismatches
    """
    if not contract_signatures:
        return SignatureCheckResult(
            passed=True,
            file_path=file_path,
            extra_info="No contract signatures to verify against",
        )

    # Parse the implementation
    actual_sigs = extract_signatures(file_content)
    if not actual_sigs:
        # Could be a syntax error or non-Python file
        return SignatureCheckResult(
            passed=True,
            file_path=file_path,
            extra_info="Could not extract signatures from file (parse failure or non-Python)",
        )

    mismatches = []
    missing = []

    for contract_sig_str in contract_signatures:
        required = parse_contract_signature(contract_sig_str)
        if required is None:
            logger.debug(
                "[sig_checker] Could not parse contract signature: %s",
                contract_sig_str[:80],
            )
            continue

        # Find the matching function in the implementation
        actual = actual_sigs.get(required.name)
        if actual is None:
            missing.append(required.name)
            continue

        # Compare
        mismatch = compare_signatures(required, actual)
        if mismatch is not None:
            mismatches.append(mismatch)

    passed = len(mismatches) == 0 and len(missing) == 0

    if not passed:
        logger.info(
            "[sig_checker] %s: FAILED — %d mismatch(es), %d missing",
            file_path, len(mismatches), len(missing),
        )
    else:
        logger.debug("[sig_checker] %s: PASSED (%d signatures verified)", file_path, len(contract_signatures))

    return SignatureCheckResult(
        passed=passed,
        file_path=file_path,
        mismatches=mismatches,
        missing_functions=missing,
    )


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
