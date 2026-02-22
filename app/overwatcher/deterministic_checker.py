# FILE: app/overwatcher/deterministic_checker.py
"""
Deterministic Job Checker — Zero LLM calls.

v2.5 (2026-02-20): Initial implementation.

Performs post-write verification of implemented files using AST parsing
and string matching only. No LLM calls, no hallucination risk.

Checks:
1. SYNTAX: File parses as valid Python
2. EXPORT VERIFICATION: Contract-required symbols exist in file
3. IMPORT RESOLUTION: Relative imports reference files that exist
4. COMPLETENESS: No bare 'pass' stubs, NotImplementedError in function bodies

If all checks pass, the LLM-based job checker can be skipped entirely.
"""

from __future__ import annotations

import ast
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

DETERMINISTIC_CHECKER_BUILD_ID = "2026-02-21-v3.0-facade-gate-and-divergence-check"
print(f"[DETERMINISTIC_CHECKER_LOADED] BUILD_ID={DETERMINISTIC_CHECKER_BUILD_ID}")


# =============================================================================
# RESULT TYPES (mirrors job_checker.CheckResult for compatibility)
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class DetCheckIssue:
    severity: str           # "blocking" or "warning"
    category: str           # e.g. "missing_export", "syntax_error", "import_error"
    description: str
    line_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "line_hint": self.line_hint,
        }


@dataclass
class DetCheckResult:
    passed: bool = True
    issues: List[DetCheckIssue] = field(default_factory=list)
    reasoning: str = ""
    skipped: bool = False
    skip_reason: str = ""

    @property
    def blocking_issues(self) -> List[DetCheckIssue]:
        return [i for i in self.issues if i.severity == "blocking"]

    @property
    def warning_issues(self) -> List[DetCheckIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": [i.to_dict() for i in self.issues],
            "blocking_count": len(self.blocking_issues),
            "warning_count": len(self.warning_issues),
            "reasoning": self.reasoning,
            "skipped": self.skipped,
        }


# =============================================================================
# MAIN DETERMINISTIC CHECK
# =============================================================================

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


# =============================================================================
# CONTRACT EXPORT EXTRACTION
# =============================================================================

def extract_required_exports(
    interface_contract: str,
    file_path: str,
) -> List[str]:
    """
    Extract symbol names the contract says this file MUST export.

    Parses the 'MUST DEFINE AND EXPORT' section of skeleton contract markdown.
    Uses the same multi-occurrence scanning approach as signature_checker v1.3.

    Returns list of bare names (not full signatures).
    """
    file_path_norm = file_path.replace("\\", "/").strip()
    required: List[str] = []

    lines = interface_contract.split("\n")
    in_file_section = False
    in_exports = False

    for line in lines:
        stripped = line.strip()
        stripped_norm = stripped.replace("\\", "/")

        # Detect file path reference (backtick-wrapped)
        if f"`{file_path_norm}`" in stripped_norm:
            in_file_section = True
            in_exports = False
            continue

        if in_file_section:
            # Detect MUST EXPORT header (handles MUST DEFINE AND EXPORT too)
            if "MUST" in stripped and "EXPORT" in stripped:
                in_exports = True
                continue

            # Section boundary — stop
            if stripped.startswith("###") or stripped.startswith("## "):
                # v1.3: Don't give up — keep scanning for more occurrences
                in_file_section = False
                in_exports = False
                continue

            # New file entry — check if it's a DIFFERENT file
            if stripped.startswith("- `") and "`" in stripped[3:]:
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    candidate = match.group(1).strip().replace("\\", "/")
                    # Is this a file path (not a signature)?
                    is_file = ("/" in candidate or candidate.endswith(".py"))
                    is_sig = candidate.startswith("def ") or candidate.startswith("async def ")
                    if is_file and not is_sig and candidate != file_path_norm:
                        in_file_section = False
                        in_exports = False
                        continue

            # Collect export names from indented bullet items
            if in_exports and stripped.startswith("- `"):
                match = re.match(r'^-\s*`([^`]+)`', stripped)
                if match:
                    symbol = match.group(1).strip()
                    # Extract just the function name from full signature
                    if symbol.startswith("def ") or symbol.startswith("async def "):
                        name_match = re.match(r'(?:async\s+)?def\s+(\w+)\s*\(', symbol)
                        if name_match:
                            name = name_match.group(1)
                            if name not in required:
                                required.append(name)
                    else:
                        # Bare name (e.g. run_segmented_job)
                        if re.match(r'^\w+$', symbol) and symbol not in required:
                            required.append(symbol)

    return required


# =============================================================================
# ARCHITECTURE EXPORT EXTRACTION (v3.0 FIX 21)
# =============================================================================


def extract_expected_exports_from_arch(
    architecture_section: str,
) -> List[str]:
    """
    Extract the expected exports from an architecture file section.

    Looks for:
    1. An `__all__` block in the architecture's code blocks
    2. A `### Re-exports` section with `from .module import (...)` blocks
    3. A `### Exports` section listing symbol names

    Returns a list of symbol names that this file is expected to define or re-export.
    """
    if not architecture_section:
        return []

    exports: List[str] = []

    # Strategy 1: Parse __all__ from code blocks in the architecture
    all_match = re.search(
        r'__all__\s*=\s*\[([^\]]+)\]',
        architecture_section,
        re.DOTALL,
    )
    if all_match:
        content = all_match.group(1)
        for name_match in re.finditer(r'["\']([a-zA-Z_][a-zA-Z0-9_]*)["\']', content):
            exports.append(name_match.group(1))
        if exports:
            return exports

    # Strategy 2: Parse function/class/constant headers from architecture
    # Matches: #### `symbol_name` (function, ...) or ### `symbol_name` (...)
    for match in re.finditer(
        r'^#{3,4}\s+`([a-zA-Z_][a-zA-Z0-9_]*)`\s+\(',
        architecture_section,
        re.MULTILINE,
    ):
        exports.append(match.group(1))

    return exports


# =============================================================================
# SEGMENT INTERFACE EXTRACTION (Job 3)
# =============================================================================

def extract_segment_interface(
    file_path: str,
    file_content: str,
) -> Dict[str, Any]:
    """
    Deterministic extraction of a file's public interface using AST.

    Returns a structured dict with:
    - exports: list of exported symbol names
    - functions: dict of func_name -> {async, params, return_type, line}
    - classes: dict of class_name -> {bases, methods, line}
    - type_aliases: dict of name -> annotation_str
    - imports_from: list of {module, names}

    This is injected as hard evidence into seg-06's prompt so the LLM
    has zero room to hallucinate sibling interfaces.
    """
    result: Dict[str, Any] = {
        "file_path": file_path,
        "exports": [],
        "functions": {},
        "classes": {},
        "type_aliases": {},
        "imports_from": [],
    }

    try:
        tree = ast.parse(file_content)
    except SyntaxError:
        result["error"] = "SyntaxError — cannot parse"
        return result

    # __all__
    dunder_all = None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        dunder_all = []
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                dunder_all.append(elt.value)

    # Top-level definitions
    all_names = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_names.append(node.name)
            func_info = {
                "async": isinstance(node, ast.AsyncFunctionDef),
                "line": node.lineno,
                "params": [],
                "return_type": _unparse_safe(node.returns),
            }
            for arg in node.args.args:
                param = {"name": arg.arg, "type": _unparse_safe(arg.annotation)}
                func_info["params"].append(param)
            # Keyword-only args
            for arg in node.args.kwonlyargs:
                param = {"name": arg.arg, "type": _unparse_safe(arg.annotation), "keyword_only": True}
                func_info["params"].append(param)
            result["functions"][node.name] = func_info

        elif isinstance(node, ast.ClassDef):
            all_names.append(node.name)
            cls_info = {
                "line": node.lineno,
                "bases": [_unparse_safe(b) for b in node.bases],
                "methods": [],
            }
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cls_info["methods"].append(item.name)
            result["classes"][node.name] = cls_info

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    all_names.append(target.id)
                    # Check if it's a type alias (assigned from typing construct)
                    val_str = _unparse_safe(node.value)
                    if val_str and ("Optional" in val_str or "Callable" in val_str
                                    or "List" in val_str or "Dict" in val_str
                                    or "Union" in val_str or "Tuple" in val_str):
                        result["type_aliases"][target.id] = val_str

        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            all_names.append(node.target.id)

    # Imports from siblings
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.level and node.level > 0:
                names = [alias.name for alias in (node.names or [])]
                result["imports_from"].append({
                    "module": "." * node.level + node.module,
                    "names": names,
                })

    result["exports"] = dunder_all if dunder_all is not None else all_names
    return result


def format_segment_interfaces(
    interfaces: List[Dict[str, Any]],
) -> str:
    """
    Format extracted interfaces into evidence text for injection into prompts.

    Produces a structured, unambiguous representation that leaves no room
    for LLM guessing about sibling module contents.
    """
    lines = []
    lines.append("## Deterministic Sibling Interface Evidence (GROUND TRUTH)")
    lines.append("Extracted by AST from actual implemented files on disk.")
    lines.append("DO NOT invent, guess, or assume any interface not listed here.")
    lines.append("")

    for iface in interfaces:
        fp = iface.get("file_path", "?")
        lines.append(f"### {fp}")

        if iface.get("error"):
            lines.append(f"  ERROR: {iface['error']}")
            lines.append("")
            continue

        exports = iface.get("exports", [])
        if exports:
            lines.append(f"  EXPORTS: {', '.join(exports)}")

        for fname, finfo in iface.get("functions", {}).items():
            prefix = "async " if finfo.get("async") else ""
            params = []
            for p in finfo.get("params", []):
                pstr = p["name"]
                if p.get("type"):
                    pstr += f": {p['type']}"
                params.append(pstr)
            ret = f" -> {finfo['return_type']}" if finfo.get("return_type") else ""
            lines.append(f"  {prefix}def {fname}({', '.join(params)}){ret}")

        for cname, cinfo in iface.get("classes", {}).items():
            bases = f"({', '.join(cinfo['bases'])})" if cinfo.get("bases") else ""
            lines.append(f"  class {cname}{bases}: methods={cinfo.get('methods', [])}")

        for tname, tval in iface.get("type_aliases", {}).items():
            lines.append(f"  {tname} = {tval}")

        if iface.get("imports_from"):
            lines.append("  IMPORTS:")
            for imp in iface["imports_from"]:
                lines.append(f"    from {imp['module']} import {', '.join(imp['names'])}")

        lines.append("")

    return "\n".join(lines)


def _unparse_safe(node) -> Optional[str]:
    """Safely unparse an AST node to string. Returns None if not possible."""
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "DetCheckResult",
    "DetCheckIssue",
    "deterministic_check",
    "extract_required_exports",
    "extract_segment_interface",
    "extract_expected_exports_from_arch",
    "format_segment_interfaces",
    "DETERMINISTIC_CHECKER_BUILD_ID",
]
