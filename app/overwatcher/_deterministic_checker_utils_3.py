from __future__ import annotations
import ast
import logging
import os
from app.overwatcher._deterministic_checker_utils import DetCheckIssue, DetCheckResult, extract_required_exports
from app.overwatcher.deterministic_checker import logger
from typing import Dict, List, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def deterministic_check(
    file_path: str,
    file_content: str,
    interface_contract: str = "",
    sandbox_base: str = "",
    existing_sandbox_files: Optional[Set[str]] = None,
    manifest_file_scope: Optional[Set[str]] = None,
    expected_exports: Optional[List[str]] = None,
) -> DetCheckResult:
    """
    Deterministic post-write verification — NO LLM calls.

    Checks:
    1. SYNTAX: File parses as valid Python (ast.parse)
    2. EXPORT VERIFICATION: Contract-required symbols exist as definitions or re-exports
    3. IMPORT RESOLUTION: Relative imports reference files that exist on disk or in manifest
    4. COMPLETENESS: No bare 'pass' stubs, NotImplementedError in function bodies

    Returns DetCheckResult. If all checks pass, the LLM-based checker can be skipped.
    """
    issues: List[DetCheckIssue] = []
    file_path_norm = file_path.replace("\\", "/").strip()
    basename = os.path.basename(file_path)

    # v3.0 FIX 21: Validate __init__.py facade files instead of skipping.
    # These are the most critical files — if re-exports are wrong, the
    # entire package fails to import. Run focused facade checks.
    is_facade = (basename == "__init__.py")

    # Skip non-Python
    if not basename.endswith(".py"):
        return DetCheckResult(passed=True, reasoning="v2.5-det: non-python skipped", skipped=True, skip_reason="non-python")

    # ── CHECK 1: SYNTAX ──────────────────────────────────────────────────
    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        issues.append(DetCheckIssue(
            severity="blocking",
            category="syntax_error",
            description=f"SyntaxError: {e.msg} at line {e.lineno}",
            line_hint=f"line {e.lineno}" if e.lineno else None,
        ))
        logger.warning("[det_checker] SYNTAX ERROR in %s: %s", file_path, e.msg)
        return DetCheckResult(passed=False, issues=issues, reasoning="v2.5-det: syntax error")

    # ── CHECK 1b: ANNOTATION NAME RESOLUTION (v6.1 FIX 24c) ───────────
    # ast.parse succeeds even when type annotations reference undefined
    # names (e.g. Optional, List, Dict). These cause NameError at runtime.
    # Collect all imported/defined names, then check annotations resolve.
    _imported_names: Set[str] = set()
    _builtins = {'int', 'str', 'float', 'bool', 'bytes', 'list', 'dict',
                 'set', 'tuple', 'type', 'None', 'True', 'False',
                 'object', 'Exception', 'print', 'len', 'range',
                 'enumerate', 'zip', 'map', 'filter', 'isinstance',
                 'issubclass', 'super', 'property', 'staticmethod',
                 'classmethod', 'dataclass'}
    for _node in ast.iter_child_nodes(tree):
        if isinstance(_node, ast.Import):
            for _alias in _node.names:
                _imported_names.add(_alias.asname or _alias.name.split('.')[0])
        elif isinstance(_node, ast.ImportFrom):
            if _node.names:
                for _alias in _node.names:
                    _imported_names.add(_alias.asname or _alias.name)
        elif isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _imported_names.add(_node.name)
        elif isinstance(_node, ast.ClassDef):
            _imported_names.add(_node.name)
        elif isinstance(_node, ast.Assign):
            for _t in _node.targets:
                if isinstance(_t, ast.Name):
                    _imported_names.add(_t.id)

    def _collect_annotation_names(node) -> Set[str]:
        """Recursively collect all Name references in an annotation."""
        names: Set[str] = set()
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Subscript):
            names |= _collect_annotation_names(node.value)
            names |= _collect_annotation_names(node.slice)
        elif isinstance(node, ast.Attribute):
            pass  # e.g. module.Type - the module is imported
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                names |= _collect_annotation_names(elt)
        elif isinstance(node, ast.BinOp):  # X | Y union syntax
            names |= _collect_annotation_names(node.left)
            names |= _collect_annotation_names(node.right)
        return names

    _all_defined = _imported_names | _builtins
    _seen_undef: Set[str] = set()
    for _node in ast.iter_child_nodes(tree):
        if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _annot_nodes = []
            for _arg in _node.args.args + _node.args.kwonlyargs:
                if _arg.annotation:
                    _annot_nodes.append((_arg.annotation, "parameter"))
            if _node.returns:
                _annot_nodes.append((_node.returns, "return"))
            for _ann_node, _ann_kind in _annot_nodes:
                for _name in _collect_annotation_names(_ann_node):
                    if _name not in _all_defined and _name not in _seen_undef:
                        _seen_undef.add(_name)
                        issues.append(DetCheckIssue(
                            severity="blocking",
                            category="undefined_annotation",
                            description=(
                                f"'{_name}' used in {_ann_kind} annotation of "
                                f"'{_node.name}' (line {_ann_node.lineno}) but "
                                f"is not imported. Add: 'from typing import {_name}'"
                            ),
                            line_hint=f"line {_ann_node.lineno}",
                        ))

    if _seen_undef:
        logger.warning(
            "[det_checker] FIX 24c: %s has %d undefined annotation name(s): %s",
            basename, len(_seen_undef), ", ".join(sorted(_seen_undef)),
        )

    # ── Extract top-level definitions from AST ───────────────────────────
    top_level_names: Set[str] = set()
    top_level_funcs: Dict[str, ast.FunctionDef] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            top_level_names.add(node.name)
            top_level_funcs[node.name] = node
        elif isinstance(node, ast.ClassDef):
            top_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    top_level_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            top_level_names.add(node.target.id)

    # Also extract re-exports: `from .module import name` at top level
    re_exported_names: Set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            if node.names:
                for alias in node.names:
                    actual_name = alias.asname if alias.asname else alias.name
                    re_exported_names.add(actual_name)

    all_available = top_level_names | re_exported_names

    # Extract __all__ if present
    dunder_all: Optional[Set[str]] = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        dunder_all = set()
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                dunder_all.add(elt.value)

    # ── CHECK 2: CONTRACT EXPORT VERIFICATION ────────────────────────────
    if interface_contract and interface_contract.strip():
        required_exports = extract_required_exports(interface_contract, file_path_norm)
        if required_exports:
            logger.info(
                "[det_checker] v2.5 Export check for %s: %d required, %d available",
                basename, len(required_exports), len(all_available),
            )
        for symbol_name in required_exports:
            if symbol_name not in all_available:
                issues.append(DetCheckIssue(
                    severity="blocking",
                    category="missing_export",
                    description=(
                        f"Contract requires '{symbol_name}' to be exported from "
                        f"{basename}, but it is not defined or re-exported. "
                        f"Available: {sorted(all_available)[:10]}"
                    ),
                ))
            elif dunder_all is not None and symbol_name not in dunder_all:
                if not symbol_name.startswith("_"):
                    issues.append(DetCheckIssue(
                        severity="warning",
                        category="export_visibility",
                        description=(
                            f"'{symbol_name}' exists but is not listed in __all__."
                        ),
                    ))

    # ── CHECK 3: IMPORT RESOLUTION ───────────────────────────────────────
    _effective_files = existing_sandbox_files or set()
    if manifest_file_scope:
        _effective_files = _effective_files | manifest_file_scope

    if _effective_files:
        file_dir = os.path.dirname(file_path_norm)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                if not node.module:
                    continue  # `from . import X` — package init, skip

                # Resolve the module path
                parts = file_dir.split("/") if file_dir else []
                # Go up 'level-1' directories (level 1 = same package)
                up = node.level - 1
                base_parts = parts[:max(0, len(parts) - up)] if up > 0 else parts
                mod_parts = node.module.split(".")
                resolved_file = "/".join(base_parts + mod_parts) + ".py"
                resolved_pkg = "/".join(base_parts + mod_parts) + "/__init__.py"

                # Check against known files (suffix match for flexibility)
                found = False
                for known in _effective_files:
                    known_norm = known.replace("\\", "/")
                    if (known_norm.endswith(resolved_file) or
                            known_norm.endswith(resolved_pkg) or
                            known_norm == resolved_file or
                            known_norm == resolved_pkg):
                        found = True
                        break

                if not found:
                    # Check host filesystem as fallback
                    _base = sandbox_base or os.getenv("SANDBOX_BASE", "")
                    if _base:
                        if (os.path.isfile(os.path.join(_base, resolved_file)) or
                                os.path.isfile(os.path.join(_base, resolved_pkg))):
                            found = True

                if not found:
                    issues.append(DetCheckIssue(
                        severity="blocking",
                        category="import_error",
                        description=(
                            f"Relative import 'from {'.' * node.level}{node.module} import ...' "
                            f"resolves to '{resolved_file}' which is not found in sandbox or manifest."
                        ),
                        line_hint=f"line {node.lineno}" if node.lineno else None,
                    ))

    # ── CHECK 4: COMPLETENESS ────────────────────────────────────────────
    for func_name, func_node in top_level_funcs.items():
        body = func_node.body
        if not body:
            continue

        # Check for bare 'pass' as only statement (excluding docstrings)
        real_stmts = [s for s in body if not (
            isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
            and isinstance(s.value.value, str)
        )]
        if len(real_stmts) == 1 and isinstance(real_stmts[0], ast.Pass):
            issues.append(DetCheckIssue(
                severity="blocking",
                category="stub_function",
                description=f"Function '{func_name}' is a bare 'pass' stub — no implementation.",
                line_hint=f"line {func_node.lineno}",
            ))

        # Check for NotImplementedError raises
        for child in ast.walk(func_node):
            if isinstance(child, ast.Raise) and child.exc:
                if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                    if child.exc.func.id == "NotImplementedError":
                        issues.append(DetCheckIssue(
                            severity="blocking",
                            category="stub_function",
                            description=f"Function '{func_name}' raises NotImplementedError.",
                            line_hint=f"line {child.lineno}",
                        ))

    # TODO/FIXME markers — warning only
    for i, line in enumerate(file_content.split("\n"), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            upper = stripped.upper()
            if "TODO" in upper or "FIXME" in upper:
                issues.append(DetCheckIssue(
                    severity="warning",
                    category="todo_marker",
                    description=f"TODO/FIXME marker: {stripped[:80]}",
                    line_hint=f"line {i}",
                ))

    # ── CHECK 5: FACADE VALIDATION (v3.0 FIX 21) ────────────────────────
    # For __init__.py files: verify that expected re-exports are present.
    if is_facade and expected_exports:
        for symbol_name in expected_exports:
            if symbol_name not in all_available:
                issues.append(DetCheckIssue(
                    severity="blocking",
                    category="facade_missing_reexport",
                    description=(
                        f"Facade __init__.py must re-export '{symbol_name}' but it "
                        f"is not imported or defined. The parent package's __init__.py "
                        f"expects this symbol. Available: {sorted(all_available)[:10]}"
                    ),
                ))
        # Check that re-exports reference sibling modules that exist
        # (delegate to CHECK 3 import resolution which already handles this)

    # ── CHECK 6: UNEXPECTED SYMBOL DETECTION (v6.1 FIX 25b) ───────────
    # For non-facade files with expected_exports: BLOCK if the LLM added
    # public FUNCTIONS or CLASSES not assigned to this file. This prevents
    # the LLM from duplicating function bodies across sibling files.
    #
    # We only check functions and classes, NOT variable assignments.
    # Type aliases (e.g. ProgressCallback = Optional[...]) and constants
    # are small and sometimes legitimately needed in multiple files for
    # annotations or configuration. The duplication problem is function
    # bodies being copied wholesale across files.
    _ALLOWED_INFRASTRUCTURE = {'logger', 'log', 'LOG', 'app'}
    if not is_facade and expected_exports:
        expected_set = set(expected_exports)
        # Build set of only function and class names (not assignments)
        _func_and_class_names: Set[str] = set()
        for _node in ast.iter_child_nodes(tree):
            if isinstance(_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                _func_and_class_names.add(_node.name)
            elif isinstance(_node, ast.ClassDef):
                _func_and_class_names.add(_node.name)
        unexpected_public = []
        for name in _func_and_class_names:
            if name.startswith("_"):  # skip private helpers
                continue
            if name in _ALLOWED_INFRASTRUCTURE:
                continue
            if name not in expected_set:
                unexpected_public.append(name)
        if unexpected_public:
            unexpected_sorted = sorted(unexpected_public)
            issues.append(DetCheckIssue(
                severity="blocking",
                category="unexpected_symbol",
                description=(
                    f"UNEXPECTED {len(unexpected_sorted)} PUBLIC FUNCTION/CLASS(ES) "
                    f"not assigned to this file: {', '.join(unexpected_sorted)}. "
                    f"These belong to sibling files in the package. "
                    f"ONLY implement the functions listed in your brief. Remove: "
                    f"{', '.join(unexpected_sorted)}"
                ),
            ))
            logger.warning(
                "[det_checker] FIX 25b: %s has %d unexpected public funcs/classes: %s",
                basename, len(unexpected_sorted),
                ", ".join(unexpected_sorted[:8]),
            )

    # ── CHECK 7: MISSING SYMBOL VERIFICATION (v6.1 FIX 24) ───────────────
    # For non-facade files: verify that ALL expected symbols from the
    # architecture are actually present in the output. This catches the
    # most common LLM failure: silently dropping symbols from the tail
    # end of a long brief. Blocking — triggers re-strike with specific
    # feedback listing exactly which symbols were missed.
    if not is_facade and expected_exports:
        expected_set = set(expected_exports)
        missing_symbols = expected_set - all_available
        if missing_symbols:
            # Sort for deterministic ordering in feedback
            missing_sorted = sorted(missing_symbols)
            issues.append(DetCheckIssue(
                severity="blocking",
                category="missing_expected_symbol",
                description=(
                    f"MISSING {len(missing_sorted)} SYMBOL(S) from architecture: "
                    f"{', '.join(missing_sorted)}. "
                    f"These were specified in the brief with TRANSPLANT VERBATIM "
                    f"but are not defined or imported in the output. "
                    f"You MUST include ALL symbols from the brief."
                ),
            ))
            logger.warning(
                "[det_checker] FIX 24: %s missing %d/%d expected symbols: %s",
                basename, len(missing_sorted), len(expected_set),
                ", ".join(missing_sorted[:8]),
            )

    # ── RESULT ───────────────────────────────────────────────────────────
    blocking = [i for i in issues if i.severity == "blocking"]
    passed = len(blocking) == 0

    if blocking:
        logger.warning(
            "[det_checker] v2.5 FAIL %s: %d blocking issues",
            basename, len(blocking),
        )
    else:
        logger.info(
            "[det_checker] v2.5 PASS %s (%d warnings)",
            basename, len(issues),
        )

    return DetCheckResult(
        passed=passed,
        issues=issues,
        reasoning=f"v2.5-det: {len(blocking)} blocking, {len(issues) - len(blocking)} warnings",
    )
