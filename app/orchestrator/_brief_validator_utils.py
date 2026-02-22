from __future__ import annotations
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


BUILD_ID = "2026-02-20-v1.0-brief-validator"

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
    from .brief_validator import FixLog
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

def _check_import_path_consistency(
    briefs: list,
    fix_log: FixLog,
) -> None:
    """
    Ensure all intra-package imports use the same convention (relative).

    Auto-fix: Normalise all cross-file imports to relative style.
    """
    from .brief_validator import FixLog
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
    from .brief_validator import FixLog
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

def save_fix_log(
    fix_log: FixLog,
    job_dir_path: str,
    segment_id: str,
) -> None:
    """Persist the fix log to disk for observability."""
    from .brief_validator import FixLog
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
