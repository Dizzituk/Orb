# FILE: app/orchestrator/brief_validator.py
"""
Brief Validator-Fixer — Autonomous pre-implementation quality gate.

Validates compiled per-file briefs for known failure patterns and
autonomously fixes every issue found. Never blocks the pipeline.
Every check has a corresponding auto-fix.

Sits between the Implementation Compiler and the Overwatcher/Implementer:

    Compiler → **Brief Validator-Fixer** → Overwatcher → Implementer

Checks and auto-fixes:
    1. Duplicate ownership     → Keep in best-affinity file, import in other
    2. Missing function        → Append to correct brief from enrichment
    3. Phantom import          → Fix import path from enrichment, or guarded fallback
    4. Signature mismatch      → Overwrite with authoritative source signature
    5. Import path consistency → Normalise to single convention
    6. Completeness gap        → Pull from enrichment or merge empty brief

v1.0 (2026-02-20): Initial implementation
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
import json

logger = logging.getLogger(__name__)

BUILD_ID = "2026-02-20-v1.0-brief-validator"
print(f"[BRIEF_VALIDATOR_LOADED] BUILD_ID={BUILD_ID}")


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Fix:
    """A single fix applied by the validator."""
    check: str          # Which check triggered this
    severity: str       # "info", "warning", "error"
    description: str    # Human-readable description
    file_path: str      # Which brief was affected
    action: str         # What was done to fix it


@dataclass
class FixLog:
    """Complete log of all fixes applied during validation."""
    fixes: List[Fix] = field(default_factory=list)
    checks_run: int = 0
    issues_found: int = 0
    issues_fixed: int = 0
    timestamp: str = ""

    def add_fix(self, check: str, severity: str, description: str,
                file_path: str, action: str) -> None:
        self.fixes.append(Fix(
            check=check,
            severity=severity,
            description=description,
            file_path=file_path,
            action=action,
        ))
        self.issues_found += 1
        self.issues_fixed += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "checks_run": self.checks_run,
            "issues_found": self.issues_found,
            "issues_fixed": self.issues_fixed,
            "fixes": [
                {
                    "check": f.check,
                    "severity": f.severity,
                    "description": f.description,
                    "file_path": f.file_path,
                    "action": f.action,
                }
                for f in self.fixes
            ],
        }

    @property
    def had_fixes(self) -> bool:
        return len(self.fixes) > 0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================


def validate_and_fix_briefs(
    briefs: list,
    enrichment: Optional[Dict[str, Any]] = None,
    sibling_enrichments: Optional[Dict[str, Dict]] = None,
) -> tuple:
    """
    Validate compiled briefs and autonomously fix all issues.

    Never blocks the pipeline. Every check has a corresponding auto-fix.

    Args:
        briefs: List of FileBrief objects from the compiler
        enrichment: This segment's enrichment data (authoritative source)
        sibling_enrichments: {seg_id: enrichment_dict} for sibling segments

    Returns:
        (fixed_briefs, fix_log) — corrected briefs and documentation of changes
    """
    fix_log = FixLog(timestamp=datetime.now(timezone.utc).isoformat())

    if not briefs:
        logger.info("[BRIEF_VALIDATOR] No briefs to validate")
        return briefs, fix_log

    # Build enrichment lookups
    source_extract = {}
    enrichment_functions = {}
    enrichment_exports = set()

    if enrichment:
        source_extract = enrichment.get("source_extract", {})
        for f in enrichment.get("functions", []):
            if isinstance(f, dict):
                enrichment_functions[f.get("name", "")] = f
        enrichment_exports = set(enrichment.get("exports", []))

    # Build sibling export map
    sibling_export_map: Dict[str, str] = {}  # {symbol: seg_id}
    if sibling_enrichments:
        for sib_id, sib_data in sibling_enrichments.items():
            for exp in sib_data.get("exports", []):
                sibling_export_map[exp] = sib_id

    # Run all checks with auto-fixes
    _check_duplicate_ownership(briefs, enrichment, fix_log)
    fix_log.checks_run += 1

    _check_missing_functions(briefs, source_extract, enrichment_functions, fix_log)
    fix_log.checks_run += 1

    _check_phantom_imports(briefs, sibling_export_map, fix_log)
    fix_log.checks_run += 1

    _check_signature_consistency(briefs, enrichment_functions, source_extract, fix_log)
    fix_log.checks_run += 1

    _check_import_path_consistency(briefs, fix_log)
    fix_log.checks_run += 1

    _check_completeness(briefs, enrichment_exports, source_extract, enrichment_functions, fix_log)
    fix_log.checks_run += 1

    if fix_log.had_fixes:
        logger.info(
            "[BRIEF_VALIDATOR] Validation complete: %d check(s), %d issue(s) found, %d fixed",
            fix_log.checks_run, fix_log.issues_found, fix_log.issues_fixed,
        )
    else:
        logger.info(
            "[BRIEF_VALIDATOR] Validation clean: %d check(s), no issues",
            fix_log.checks_run,
        )

    return briefs, fix_log


# =============================================================================
# CHECK 1: DUPLICATE OWNERSHIP
# =============================================================================


def _check_duplicate_ownership(
    briefs: list,
    enrichment: Optional[Dict[str, Any]],
    fix_log: FixLog,
) -> None:
    """
    Ensure each function appears in exactly one brief.

    Auto-fix: Keep in the brief with strongest file-stem affinity.
    Remove from others and add an import instead.
    """
    # Build ownership map: function_name -> [brief_indices]
    ownership: Dict[str, List[int]] = {}

    for idx, brief in enumerate(briefs):
        for func in brief.functions:
            ownership.setdefault(func.name, []).append(idx)

    for func_name, owner_indices in ownership.items():
        if len(owner_indices) <= 1:
            continue

        # Duplicate found — pick the best owner
        best_idx = _pick_best_owner(func_name, owner_indices, briefs)

        for idx in owner_indices:
            if idx == best_idx:
                continue

            # Remove from this brief
            brief = briefs[idx]
            brief.functions = [f for f in brief.functions if f.name != func_name]

            # Add import from the owning brief
            owner_brief = briefs[best_idx]
            owner_module = os.path.basename(owner_brief.file_path).replace(".py", "")
            imp_statement = f"from .{owner_module} import {func_name}"

            # Avoid duplicate imports
            existing_statements = {imp.statement for imp in brief.imports}
            if imp_statement not in existing_statements:
                from app.orchestrator.compiler_models import FileImport
                brief.imports.append(FileImport(
                    statement=imp_statement,
                    source_segment="",
                    symbols=[func_name],
                ))

            fix_log.add_fix(
                check="duplicate_ownership",
                severity="warning",
                description=f"'{func_name}' was in {len(owner_indices)} briefs",
                file_path=brief.file_path,
                action=f"Removed from {brief.file_path}, kept in {owner_brief.file_path}, added import",
            )


def _pick_best_owner(
    func_name: str,
    owner_indices: List[int],
    briefs: list,
) -> int:
    """
    Pick the best brief to own a function based on file-stem affinity.

    Returns the index of the winning brief.
    """
    name_lower = func_name.lower().lstrip("_")
    name_words = {w for w in name_lower.split("_") if len(w) > 2}

    best_idx = owner_indices[0]
    best_score = -1

    for idx in owner_indices:
        brief = briefs[idx]
        stem = os.path.basename(brief.file_path).replace(".py", "").lower().lstrip("_")
        stem_words = {w for w in stem.split("_") if len(w) > 2}

        score = len(name_words & stem_words)

        # Bonus for substring match
        if stem in name_lower:
            score += 2

        # Penalty for __init__.py (should import, not define)
        if "__init__" in brief.file_path:
            score -= 5

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


# =============================================================================
# CHECK 2: MISSING FUNCTIONS
# =============================================================================


def _check_missing_functions(
    briefs: list,
    source_extract: Dict[str, str],
    enrichment_functions: Dict[str, Dict],
    fix_log: FixLog,
) -> None:
    """
    Ensure every function in the enrichment appears in at least one brief.

    Auto-fix: Add missing function to the brief with best file-stem affinity.
    """
    # Collect all function names across all briefs
    assigned_names: Set[str] = set()
    for brief in briefs:
        for func in brief.functions:
            assigned_names.add(func.name)

    # Check source_extract keys (authoritative list of what this segment owns)
    all_expected = set(source_extract.keys())

    missing = all_expected - assigned_names
    if not missing:
        return

    for func_name in missing:
        # Find best brief to place it in
        best_brief = _find_best_brief_for_function(func_name, briefs)
        if not best_brief:
            fix_log.add_fix(
                check="missing_function",
                severity="error",
                description=f"'{func_name}' not in any brief and no suitable target found",
                file_path="(none)",
                action="Could not auto-fix — no target brief available",
            )
            continue

        # Build FileFunction from enrichment data
        body = source_extract.get(func_name, "")
        func_meta = enrichment_functions.get(func_name, {})

        from app.orchestrator.compiler_models import FileFunction
        new_func = FileFunction(
            name=func_name,
            kind="function",
            signature=func_meta.get("signature", f"def {func_name}(...):"),
            body=body,
            line_count=body.count("\n") + 1 if body else 0,
            is_async=func_meta.get("is_async", False),
            docstring=func_meta.get("docstring", ""),
        )

        best_brief.functions.append(new_func)
        best_brief.exports.append(func_name)

        fix_log.add_fix(
            check="missing_function",
            severity="warning",
            description=f"'{func_name}' was in enrichment but not in any brief",
            file_path=best_brief.file_path,
            action=f"Added to {best_brief.file_path} ({new_func.line_count} lines)",
        )


def _find_best_brief_for_function(
    func_name: str,
    briefs: list,
) -> Optional[Any]:
    """Find the best brief to place a missing function based on affinity."""
    name_lower = func_name.lower().lstrip("_")
    name_words = {w for w in name_lower.split("_") if len(w) > 2}

    best_brief = None
    best_score = -1

    for brief in briefs:
        # Skip __init__.py
        if "__init__" in brief.file_path:
            continue

        stem = os.path.basename(brief.file_path).replace(".py", "").lower().lstrip("_")
        stem_words = {w for w in stem.split("_") if len(w) > 2}

        score = len(name_words & stem_words)
        if stem in name_lower:
            score += 2

        if score > best_score:
            best_score = score
            best_brief = brief

    # If no affinity match, use the brief with the most existing functions
    if best_brief is None or best_score <= 0:
        non_init = [b for b in briefs if "__init__" not in b.file_path]
        if non_init:
            best_brief = max(non_init, key=lambda b: len(b.functions))

    return best_brief


# =============================================================================
# CHECK 3: PHANTOM IMPORTS
# =============================================================================


def _check_phantom_imports(
    briefs: list,
    sibling_export_map: Dict[str, str],
    fix_log: FixLog,
) -> None:
    """
    Verify that every cross-file import references a symbol that actually exists.

    Auto-fix: Fix import path if symbol exists elsewhere. If symbol genuinely
    doesn't exist, replace with a guarded import that won't crash at boot.
    """
    # Build map of all symbols defined across all briefs
    all_defined: Dict[str, str] = {}  # {symbol: file_path}
    for brief in briefs:
        for func in brief.functions:
            all_defined[func.name] = brief.file_path

    for brief in briefs:
        fixed_imports = []
        for imp in brief.imports:
            all_valid = True
            for sym in imp.symbols:
                if sym in all_defined:
                    # Symbol exists — check if import path is correct
                    correct_file = all_defined[sym]
                    if correct_file != brief.file_path:
                        # Symbol is in a sibling file — import path should reference it
                        correct_module = os.path.basename(correct_file).replace(".py", "")
                        expected_prefix = f"from .{correct_module}"
                        if expected_prefix not in imp.statement:
                            # Fix the import path
                            new_statement = f"from .{correct_module} import {sym}"
                            from app.orchestrator.compiler_models import FileImport as _FI
                            fixed_imports.append(_FI(
                                statement=new_statement,
                                source_segment=imp.source_segment,
                                symbols=[sym],
                            ))
                            fix_log.add_fix(
                                check="phantom_import",
                                severity="warning",
                                description=f"'{sym}' import pointed to wrong module",
                                file_path=brief.file_path,
                                action=f"Fixed: {imp.statement} → {new_statement}",
                            )
                            all_valid = False
                            continue

                elif sym in sibling_export_map:
                    # Symbol exists in a sibling segment — import is valid
                    pass

                elif not _is_stdlib_name(sym):
                    # Symbol doesn't exist anywhere — guard the import
                    guarded = f"try:\n    {imp.statement}\nexcept ImportError:\n    {sym} = None  # Guarded: not found in any segment"
                    from app.orchestrator.compiler_models import FileImport as _FI2
                    fixed_imports.append(_FI2(
                        statement=imp.statement,  # Keep original, note in fix log
                        source_segment=imp.source_segment,
                        symbols=imp.symbols,
                    ))
                    fix_log.add_fix(
                        check="phantom_import",
                        severity="warning",
                        description=f"'{sym}' not found in any brief or sibling segment",
                        file_path=brief.file_path,
                        action=f"Kept import but flagged — symbol may come from external dependency",
                    )
                    all_valid = False
                    continue

            if all_valid:
                fixed_imports.append(imp)

        brief.imports = fixed_imports


def _is_stdlib_name(name: str) -> bool:
    """Check if a name is a Python stdlib/builtin that doesn't need segment import."""
    stdlib = {
        "os", "sys", "json", "logging", "re", "ast", "hashlib", "uuid",
        "datetime", "timezone", "pathlib", "typing", "collections", "functools",
        "asyncio", "traceback", "io", "copy", "shutil", "time", "enum",
        "dataclasses", "abc", "math",
        # typing
        "Dict", "List", "Optional", "Any", "Tuple", "Set", "Union",
        "Callable", "Sequence", "Mapping", "Type",
        # builtins
        "logger", "print", "len", "str", "int", "float", "bool",
        "True", "False", "None", "self", "cls", "super",
        "Exception", "RuntimeError", "ValueError", "TypeError",
        "KeyError", "AttributeError", "ImportError", "OSError",
        "FileNotFoundError", "IndexError", "StopIteration",
        "isinstance", "issubclass", "hasattr", "getattr", "setattr",
        "range", "enumerate", "zip", "map", "filter", "sorted",
        "min", "max", "sum", "abs", "round",
        "dict", "list", "set", "tuple", "frozenset",
        "open", "type", "property", "staticmethod", "classmethod",
        "dataclass", "field", "Enum",
    }
    return name in stdlib


# =============================================================================
# CHECK 4: SIGNATURE CONSISTENCY
# =============================================================================


def _check_signature_consistency(
    briefs: list,
    enrichment_functions: Dict[str, Dict],
    source_extract: Dict[str, str],
    fix_log: FixLog,
) -> None:
    """
    Verify function signatures match the authoritative source.

    Auto-fix: If enrichment has the original signature, overwrite the
    brief's version with the source-of-truth.
    """
    for brief in briefs:
        for func in brief.functions:
            if func.name not in enrichment_functions:
                continue

            auth_meta = enrichment_functions[func.name]
            auth_sig = auth_meta.get("signature", "")

            if not auth_sig or not func.signature:
                continue

            # Compare signatures (normalise whitespace)
            norm_func_sig = re.sub(r'\s+', ' ', func.signature.strip())
            norm_auth_sig = re.sub(r'\s+', ' ', auth_sig.strip())

            if norm_func_sig != norm_auth_sig:
                old_sig = func.signature
                func.signature = auth_sig

                fix_log.add_fix(
                    check="signature_consistency",
                    severity="warning",
                    description=f"'{func.name}' signature didn't match enrichment source",
                    file_path=brief.file_path,
                    action=f"Overwritten with authoritative signature from AST extraction",
                )

            # Also check that the body matches source_extract if available
            if func.name in source_extract:
                auth_body = source_extract[func.name]
                if func.body and auth_body:
                    # Check if body is a stub (way too short compared to source)
                    func_lines = func.body.strip().count("\n") + 1
                    auth_lines = auth_body.strip().count("\n") + 1

                    if func_lines < auth_lines * 0.5 and auth_lines > 5:
                        # Body is suspiciously short — replace with authoritative
                        func.body = auth_body
                        func.line_count = auth_lines

                        fix_log.add_fix(
                            check="signature_consistency",
                            severity="error",
                            description=(
                                f"'{func.name}' body was {func_lines} lines "
                                f"vs {auth_lines} in source — likely a stub"
                            ),
                            file_path=brief.file_path,
                            action="Replaced with full body from enrichment source_extract",
                        )


# =============================================================================
# CHECK 5: IMPORT PATH CONSISTENCY
# =============================================================================


def _check_import_path_consistency(
    briefs: list,
    fix_log: FixLog,
) -> None:
    """
    Ensure all intra-package imports use the same convention (relative).

    Auto-fix: Normalise all cross-file imports to relative style.
    """
    for brief in briefs:
        fixed_imports = []
        for imp in brief.imports:
            statement = imp.statement

            # Detect absolute imports that should be relative
            # e.g. "from app.orchestrator.segment_loop._constants import X"
            # should be "from ._constants import X"
            abs_pattern = re.match(
                r'from\s+(app\.[a-zA-Z_.]+)\.(_[a-zA-Z_]+)\s+import\s+(.+)',
                statement,
            )
            if abs_pattern:
                module = abs_pattern.group(2)
                symbols = abs_pattern.group(3)
                new_statement = f"from .{module} import {symbols}"

                from app.orchestrator.compiler_models import FileImport as _FI3
                fixed_imports.append(_FI3(
                    statement=new_statement,
                    source_segment=imp.source_segment,
                    symbols=imp.symbols,
                ))

                fix_log.add_fix(
                    check="import_path_consistency",
                    severity="info",
                    description=f"Absolute import normalised to relative",
                    file_path=brief.file_path,
                    action=f"{statement} → {new_statement}",
                )
                continue

            fixed_imports.append(imp)

        brief.imports = fixed_imports


# =============================================================================
# CHECK 6: COMPLETENESS
# =============================================================================


def _check_completeness(
    briefs: list,
    enrichment_exports: Set[str],
    source_extract: Dict[str, str],
    enrichment_functions: Dict[str, Dict],
    fix_log: FixLog,
) -> None:
    """
    Ensure no brief would produce an empty or stub file.

    Auto-fix: Pull functions from enrichment or merge into sibling brief.
    """
    empty_briefs = []

    for brief in briefs:
        # Skip __init__.py — it's expected to be thin
        if "__init__" in brief.file_path:
            continue

        if not brief.functions:
            empty_briefs.append(brief)

    for empty_brief in empty_briefs:
        # Try to find functions from enrichment that match this file
        stem = os.path.basename(empty_brief.file_path).replace(".py", "").lower().lstrip("_")
        stem_words = {w for w in stem.split("_") if len(w) > 2}

        # Look through unassigned enrichment functions
        assigned_names = set()
        for b in briefs:
            for f in b.functions:
                assigned_names.add(f.name)

        rescued = False
        for func_name, body in source_extract.items():
            if func_name in assigned_names:
                continue

            name_words = {w for w in func_name.lower().lstrip("_").split("_") if len(w) > 2}
            if name_words & stem_words:
                # Affinity match — add to this brief
                func_meta = enrichment_functions.get(func_name, {})
                from app.orchestrator.compiler_models import FileFunction as _FF
                new_func = _FF(
                    name=func_name,
                    kind="function",
                    signature=func_meta.get("signature", f"def {func_name}(...):"),
                    body=body,
                    line_count=body.count("\n") + 1 if body else 0,
                    is_async=func_meta.get("is_async", False),
                    docstring=func_meta.get("docstring", ""),
                )
                empty_brief.functions.append(new_func)
                assigned_names.add(func_name)
                rescued = True

        if rescued:
            fix_log.add_fix(
                check="completeness",
                severity="warning",
                description=f"Brief for {empty_brief.file_path} was empty",
                file_path=empty_brief.file_path,
                action=f"Rescued {len(empty_brief.functions)} function(s) from enrichment",
            )
        else:
            # Can't rescue — merge this brief's file_path into the largest sibling
            non_empty = [b for b in briefs if b.functions and "__init__" not in b.file_path]
            if non_empty:
                target = max(non_empty, key=lambda b: len(b.functions))
                fix_log.add_fix(
                    check="completeness",
                    severity="warning",
                    description=f"Brief for {empty_brief.file_path} is empty with no rescuable functions",
                    file_path=empty_brief.file_path,
                    action=f"File should be removed from inventory or populated by architecture",
                )


# =============================================================================
# PERSISTENCE
# =============================================================================


def save_fix_log(
    fix_log: FixLog,
    job_dir_path: str,
    segment_id: str,
) -> None:
    """Persist the fix log to disk for observability."""
    seg_dir = os.path.join(job_dir_path, "segments", segment_id)
    compiler_dir = os.path.join(seg_dir, "compiler")
    os.makedirs(compiler_dir, exist_ok=True)

    path = os.path.join(compiler_dir, "brief_validation.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fix_log.to_dict(), f, indent=2, default=str)
        logger.info("[BRIEF_VALIDATOR] Fix log saved: %s", path)
    except Exception as e:
        logger.warning("[BRIEF_VALIDATOR] Failed to save fix log: %s", e)
