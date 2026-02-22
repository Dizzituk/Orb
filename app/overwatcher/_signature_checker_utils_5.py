from __future__ import annotations
import ast
import logging
from app.overwatcher._signature_checker_utils_4 import _normalise_type, _types_match
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


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

    # 2. Parameter count (filter out *args/**kwargs and bare * for positional count)
    # v1.3: Also filter param_types in sync to avoid index misalignment.
    # Contract parser includes bare `*` as a param (keyword-only separator),
    # but AST extraction strips it. Without synced filtering, type comparison
    # uses wrong indices and produces false mismatches.
    req_positional = [n for n in required.param_names if not n.startswith("*") and n != "*"]
    act_positional = [n for n in actual.param_names if not n.startswith("*") and n != "*"]
    req_types_filtered = [t for n, t in zip(required.param_names, required.param_types) if not n.startswith("*") and n != "*"]
    act_types_filtered = [t for n, t in zip(actual.param_names, actual.param_types) if not n.startswith("*") and n != "*"]

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
    # v1.3: Use filtered type lists (synced with positional names) to avoid
    # index misalignment from bare `*` keyword-only separator.
    if len(req_positional) == len(act_positional):
        for i in range(len(req_positional)):
            req_type = req_types_filtered[i] if i < len(req_types_filtered) else None
            act_type = act_types_filtered[i] if i < len(act_types_filtered) else None

            if req_type and act_type:
                req_norm = _normalise_type(req_type)
                act_norm = _normalise_type(act_type)
                if req_norm and act_norm and not _types_match(req_norm, act_norm):
                    req_name = req_positional[i] if i < len(req_positional) else f"param_{i}"
                    act_name = act_positional[i] if i < len(act_positional) else f"param_{i}"
                    differences.append(
                        f"Parameter {i} type: expected `{req_type}` (for `{req_name}`), "
                        f"got `{act_type}` (for `{act_name}`)"
                    )

    # 4. Return type
    req_ret = _normalise_type(required.return_type)
    act_ret = _normalise_type(actual.return_type)
    if req_ret and act_ret and not _types_match(req_ret, act_ret):
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
                # v1.3: Use filtered type lists for correct alignment
                req_type = req_types_filtered[i] if i < len(req_types_filtered) else None
                act_type = act_types_filtered[i] if i < len(act_types_filtered) else None
                if req_type and act_type:
                    req_norm = _normalise_type(req_type)
                    act_norm = _normalise_type(act_type)
                    if _types_match(req_norm, act_norm):
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
