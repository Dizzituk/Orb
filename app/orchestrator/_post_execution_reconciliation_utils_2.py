from __future__ import annotations
import ast
import logging
import os
import re
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Set, Tuple
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


POST_RECON_BUILD_ID = "2026-02-18-v2.0-deferred-consumer-recon"

def _build_export_registry(
    file_contents: Dict[str, str],
) -> Dict[str, Set[str]]:
    """
    Build a registry of what each module exports.

    Parses each Python file's AST to extract top-level names:
    - Functions (def / async def)
    - Classes
    - Constants (ALL_CAPS assignments)
    - __all__ entries (if defined, these take precedence)

    Args:
        file_contents: {relative_path: file_content}

    Returns:
        {module_dotted_path: set_of_exported_names}
        e.g. {"app.overwatcher.architecture_executor.source_context": {"_detect_source_files", ...}}
    """
    registry: Dict[str, Set[str]] = {}

    for rel_path, content in file_contents.items():
        if not rel_path.endswith(".py"):
            continue

        # Convert file path to module path
        # app/overwatcher/architecture_executor/source_context.py
        # -> app.overwatcher.architecture_executor.source_context
        module_path = rel_path.replace("\\", "/").replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        names: Set[str] = set()
        all_names: Set[str] = set()

        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning("[post_recon] SyntaxError parsing %s — skipping", rel_path)
            continue

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id == "__all__":
                            # Parse __all__ list
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        all_names.add(elt.value)
                        elif target.id.isupper() or target.id.startswith("_"):
                            names.add(target.id)

        # __all__ takes precedence if defined
        registry[module_path] = all_names if all_names else names

        # Also register by just the filename stem for simpler lookups
        stem = module_path.rsplit(".", 1)[-1]
        if stem not in registry:
            registry[stem] = registry[module_path]

        logger.debug(
            "[post_recon] Registry: %s → %d names (%s)",
            module_path, len(registry[module_path]),
            "from __all__" if all_names else "from AST",
        )

    return registry

def _extract_imports_with_lines(
    content: str,
) -> List[Tuple[int, str, str, List[str]]]:
    """
    Extract import statements with line numbers from source code.

    Returns list of (line_number, full_line, module_path, [imported_names])

    Only extracts `from X import Y, Z` style imports, not `import X`.
    """
    imports = []
    lines = content.split("\n")

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Match: from app.foo.bar import X, Y, Z
        # Also: from .foo import X
        # Also: from ..foo import X
        m = re.match(
            r'^from\s+([\w.]+)\s+import\s+(.+?)(?:\s*#.*)?$',
            stripped,
        )
        if m:
            module = m.group(1)
            names_str = m.group(2).strip()

            # Handle parenthesised imports across multiple lines
            if names_str.startswith("(") and ")" not in names_str:
                # Multi-line import — collect until closing paren
                j = i + 1
                while j < len(lines) and ")" not in lines[j]:
                    names_str += " " + lines[j].strip()
                    j += 1
                if j < len(lines):
                    names_str += " " + lines[j].strip()
                names_str = names_str.replace("(", "").replace(")", "")

            # Parse individual names
            imported_names = []
            for part in names_str.split(","):
                name = part.strip()
                # Handle 'X as Y' aliases
                if " as " in name:
                    name = name.split(" as ")[0].strip()
                if name and name != "\\" and not name.startswith("#"):
                    imported_names.append(name)

            if imported_names:
                imports.append((i + 1, line, module, imported_names))

    return imports

def _find_best_match(
    wrong_name: str,
    available_names: Set[str],
    cutoff: float = 0.5,
) -> Optional[Tuple[str, str, float]]:
    """
    Find the best matching name from available exports.

    Returns (correct_name, method, confidence) or None.

    Matching strategies (in order):
    1. Case-insensitive exact match
    2. Prefix/suffix containment (e.g. "extract_source_files" matches
       "_detect_source_files_from_architecture" because "source_files" overlaps)
    3. Fuzzy match via difflib
    """
    if not available_names:
        return None

    wrong_lower = wrong_name.lower()

    # Strategy 1: Case-insensitive exact match
    for name in available_names:
        if name.lower() == wrong_lower:
            return (name, "case_match", 0.95)

    # Strategy 2: Significant substring overlap
    # Break both names into word parts and check overlap
    def _word_parts(name: str) -> Set[str]:
        # Split on underscores and camelCase boundaries
        parts = set()
        for segment in name.split("_"):
            if segment:
                parts.add(segment.lower())
                # Split camelCase
                camel_parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', segment)
                for cp in camel_parts:
                    parts.add(cp.lower())
        return parts

    wrong_parts = _word_parts(wrong_name)
    if len(wrong_parts) >= 2:  # Only use this for multi-word names
        best_overlap = 0
        best_candidate = None
        for name in available_names:
            name_parts = _word_parts(name)
            overlap = len(wrong_parts & name_parts)
            # Require at least 2 matching word parts
            if overlap >= 2 and overlap > best_overlap:
                best_overlap = overlap
                best_candidate = name
        if best_candidate:
            # Confidence based on overlap ratio
            max_parts = max(len(wrong_parts), len(_word_parts(best_candidate)))
            confidence = best_overlap / max_parts if max_parts > 0 else 0
            if confidence >= 0.4:
                return (best_candidate, "word_overlap", min(confidence, 0.85))

    # Strategy 3: Fuzzy match
    # v1.1: Constants (ALL_CAPS_NAMES) are NOT safe to fuzzy-match. They represent
    # distinct values (timeouts, IDs, limits), not renamed functions. Fuzzy-matching
    # ARCHITECTURE_EXECUTOR_BUILD_ID -> ARCHITECTURE_LOG_PREFIX is semantically wrong.
    # For constants, only exact or case-insensitive matches are safe (Strategy 1).
    _is_constant = bool(re.match(r'^[A-Z][A-Z0-9_]+$', wrong_name))
    if _is_constant:
        # Constants: skip fuzzy matching entirely — return None so the caller
        # knows the symbol is genuinely missing, not just renamed
        return None

    matches = get_close_matches(wrong_name, list(available_names), n=1, cutoff=cutoff)
    if matches:
        return (matches[0], "fuzzy_match", 0.7)

    return None

def apply_import_fixes(
    file_content: str,
    fixes: List[ImportFix],
) -> str:
    """
    Apply import fixes to file content by line replacement.

    Processes fixes in reverse line order to avoid offset issues.

    Args:
        file_content: Original file content
        fixes: List of ImportFix objects for this file

    Returns:
        Patched file content
    """
    if not fixes:
        return file_content

    lines = file_content.split("\n")

    # Sort fixes by line number descending (apply from bottom up)
    sorted_fixes = sorted(fixes, key=lambda f: f.line_number, reverse=True)

    for fix in sorted_fixes:
        idx = fix.line_number - 1  # Convert to 0-indexed
        if 0 <= idx < len(lines):
            old_line = lines[idx]
            # Replace the wrong name with the correct one
            new_line = old_line.replace(fix.wrong_name, fix.correct_name)
            if new_line != old_line:
                lines[idx] = new_line
                logger.debug(
                    "[post_recon] Line %d: '%s' → '%s'",
                    fix.line_number,
                    old_line.strip(),
                    new_line.strip(),
                )

    return "\n".join(lines)

def reconcile_deferred_consumers(
    manifest: Any,
    sandbox_base: str = "D:\\Orb",
    dry_run: bool = False,
    on_progress: Any = None,
) -> ReconciliationResult:
    """
    Fix import paths in external consumer files after a file->package refactor.

    When a monolith (e.g. segment_loop.py) is refactored into a package
    (segment_loop/), external files that imported from the monolith need
    their imports updated. These files were excluded from segment scope
    (v5.18 consumer exclusion) and are listed in manifest.deferred_consumer_files.

    The fix is mechanical: the __init__.py facade should re-export everything,
    so most imports like `from .segment_loop import X` still work. But if
    the monolith was `from app.orchestrator.segment_loop import X` and the
    monolith has been quarantined, we need to ensure the package __init__.py
    actually re-exports X.

    This function:
    1. Reads deferred consumer file list from manifest
    2. Builds export registry from new package __init__.py
    3. Scans each consumer for imports from the old monolith path
    4. Verifies each imported name exists in the package exports
    5. Reports (or fixes) any missing re-exports

    Returns ReconciliationResult with fix details.
    """
    _emit = on_progress or (lambda msg: None)
    result = ReconciliationResult()

    deferred = getattr(manifest, 'deferred_consumer_files', []) or []
    if not deferred:
        _emit("🔧 [CONSUMER-RECON] No deferred consumer files — skipping")
        return result

    _emit(f"🔧 [CONSUMER-RECON] Processing {len(deferred)} deferred consumer file(s)...")

    # Detect which packages were created (monolith -> package pattern)
    # Look for quarantined monoliths or package __init__.py files
    _package_exports: Dict[str, Set[str]] = {}  # dotted_module -> {exported_names}

    for seg in manifest.segments:
        for rel_path in seg.file_scope:
            norm = rel_path.replace("\\", "/")
            if norm.endswith("/__init__.py"):
                # This is a package init — read its exports
                abs_path = os.path.join(sandbox_base, norm.replace("/", os.sep))
                if os.path.isfile(abs_path):
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                            init_content = f.read()
                        # Parse exports from __init__.py
                        _pkg_dir = norm.rsplit("/", 1)[0]  # e.g. "app/orchestrator/segment_loop"
                        _dotted = _pkg_dir.replace("/", ".")
                        # Get names from __all__ or from imports
                        try:
                            tree = ast.parse(init_content)
                            names: Set[str] = set()
                            for node in ast.walk(tree):
                                if isinstance(node, ast.ImportFrom):
                                    for alias in (node.names or []):
                                        names.add(alias.asname or alias.name)
                                elif isinstance(node, ast.Assign):
                                    for target in node.targets:
                                        if isinstance(target, ast.Name) and target.id == "__all__":
                                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                                for elt in node.value.elts:
                                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                                        names.add(elt.value)
                            _package_exports[_dotted] = names
                            logger.info(
                                "[consumer_recon] Package %s exports %d name(s)",
                                _dotted, len(names),
                            )
                        except SyntaxError:
                            logger.warning("[consumer_recon] Cannot parse %s", abs_path)
                    except Exception as e:
                        logger.warning("[consumer_recon] Cannot read %s: %s", abs_path, e)

    if not _package_exports:
        _emit("🔧 [CONSUMER-RECON] No package __init__.py exports found — skipping")
        return result

    # Process each deferred consumer file
    _missing_reexports: Dict[str, List[str]] = {}  # package -> [missing_names]

    for consumer_path in deferred:
        abs_path = os.path.join(sandbox_base, consumer_path.replace("/", os.sep))
        if not os.path.isfile(abs_path):
            logger.warning("[consumer_recon] Deferred consumer not found: %s", abs_path)
            continue

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                consumer_content = f.read()
        except Exception as e:
            logger.warning("[consumer_recon] Cannot read %s: %s", abs_path, e)
            continue

        result.files_scanned += 1

        # Find imports that reference any of our refactored packages
        for line_no, line in enumerate(consumer_content.splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith(("from ", "import ")):
                continue

            for pkg_dotted, pkg_names in _package_exports.items():
                # Match: from app.orchestrator.segment_loop import X, Y
                # Or:    from .segment_loop import X, Y  (relative)
                _rel_dotted = "." + pkg_dotted.rsplit(".", 1)[-1] if "." in pkg_dotted else pkg_dotted
                if pkg_dotted in stripped or _rel_dotted in stripped:
                    # Extract imported names
                    _import_match = re.match(
                        r'from\s+[\w.]+\s+import\s+(.+)',
                        stripped.rstrip("\\").rstrip(),
                    )
                    if _import_match:
                        _imported = [
                            n.strip().split(" as ")[0].strip()
                            for n in _import_match.group(1).split(",")
                            if n.strip() and n.strip() != "\\"
                        ]
                        for _name in _imported:
                            if _name and _name not in pkg_names and _name != "*":
                                _missing_reexports.setdefault(pkg_dotted, []).append(_name)
                                logger.warning(
                                    "[consumer_recon] %s:%d imports '%s' from %s "
                                    "but __init__.py does not re-export it",
                                    consumer_path, line_no, _name, pkg_dotted,
                                )

    # Report findings
    if _missing_reexports:
        for pkg, names in _missing_reexports.items():
            unique_names = sorted(set(names))
            _emit(
                f"⚠️ [CONSUMER-RECON] Package {pkg} missing re-exports: "
                f"{', '.join(unique_names)}"
            )
            result.errors.append(
                f"Package {pkg} __init__.py must re-export: {', '.join(unique_names)}"
            )
    else:
        _emit("✅ [CONSUMER-RECON] All deferred consumer imports are satisfied")

    return result
