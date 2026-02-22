# FILE: app/orchestrator/post_execution_reconciliation.py
"""
Post-Execution Import Reconciliation — Option B: Fix naming drift after execution.

Fallback layer that runs AFTER segment execution completes (with partial or full
failures). Reads actual implemented files from the sandbox, detects import
mismatches against what sibling modules actually export, and surgically edits
the import lines.

This catches what Option A (pre-execution interface injection) couldn't prevent:
  - Implementer ignored the DEPENDENCY REALITY block
  - A retry strike changed a file's exports after reconciliation was generated
  - The __all__ list was present but the Implementer used a wrong alias

Flow:
  1. Collect all files written by completed/failed segments from sandbox
  2. Build an export registry: {module_path: {function_names, class_names, constants}}
  3. For each file, parse imports and check against the registry
  4. For mismatched imports, find the closest match in the target module's exports
  5. Rewrite the import line with the correct name
  6. Write the fixed file back to the sandbox

v1.0 (2026-02-15): Initial implementation
"""

from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

POST_RECON_BUILD_ID = "2026-02-18-v2.0-deferred-consumer-recon"
print(f"[POST_EXECUTION_RECON_LOADED] BUILD_ID={POST_RECON_BUILD_ID}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ImportFix:
    """A single import fix that was applied."""
    file_path: str             # File that was fixed
    line_number: int           # Line number of the import
    original_line: str         # Original import line
    fixed_line: str            # Corrected import line
    wrong_name: str            # The incorrect name
    correct_name: str          # The correct name
    target_module: str         # Module being imported from
    fix_method: str            # "exact_match", "close_match", "ast_match"
    confidence: float = 1.0    # 0.0-1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "original_line": self.original_line,
            "fixed_line": self.fixed_line,
            "wrong_name": self.wrong_name,
            "correct_name": self.correct_name,
            "target_module": self.target_module,
            "fix_method": self.fix_method,
            "confidence": self.confidence,
        }


@dataclass
class ReconciliationResult:
    """Result of post-execution reconciliation."""
    files_scanned: int = 0
    files_fixed: int = 0
    fixes_applied: List[ImportFix] = field(default_factory=list)
    files_with_errors: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "files_fixed": self.files_fixed,
            "fixes_applied": [f.to_dict() for f in self.fixes_applied],
            "files_with_errors": self.files_with_errors,
            "errors": self.errors,
        }


# =============================================================================
# EXPORT REGISTRY BUILDER
# =============================================================================

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


# =============================================================================
# IMPORT MISMATCH DETECTION
# =============================================================================

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


def detect_import_mismatches(
    file_path: str,
    file_content: str,
    export_registry: Dict[str, Set[str]],
    package_prefix: str = "app.overwatcher.architecture_executor",
) -> List[ImportFix]:
    """
    Detect import mismatches in a single file against the export registry.

    For each `from X import Y` in the file:
    1. Resolve X to a module in the registry
    2. Check if Y exists in that module's exports
    3. If not, find the best match
    4. Create an ImportFix if a match is found

    Args:
        file_path: Relative path of the file being checked
        file_content: Content of the file
        export_registry: {module_path: set_of_names} from _build_export_registry
        package_prefix: The package prefix for resolving relative imports

    Returns:
        List of ImportFix objects (may be empty if no mismatches)
    """
    fixes: List[ImportFix] = []

    imports = _extract_imports_with_lines(file_content)

    for line_num, full_line, module_path, imported_names in imports:
        # Resolve relative imports
        resolved_module = module_path
        if module_path.startswith("."):
            # Count dots for relative depth
            dots = len(module_path) - len(module_path.lstrip("."))
            relative_part = module_path.lstrip(".")

            # Go up from current file's package
            current_parts = file_path.replace("\\", "/").replace("/", ".").split(".")
            if current_parts[-1] == "py":
                current_parts = current_parts[:-1]
            # Remove filename to get package
            if len(current_parts) > 1:
                current_parts = current_parts[:-1]

            # Go up `dots` levels
            if dots <= len(current_parts):
                base = ".".join(current_parts[:len(current_parts) - dots + 1])
                resolved_module = f"{base}.{relative_part}" if relative_part else base
            else:
                resolved_module = relative_part

        # Find this module in the registry
        available_names = None

        # Try full path first
        if resolved_module in export_registry:
            available_names = export_registry[resolved_module]
        else:
            # Try just the module stem
            stem = resolved_module.rsplit(".", 1)[-1] if "." in resolved_module else resolved_module
            if stem in export_registry:
                available_names = export_registry[stem]
            else:
                # Try with package prefix
                prefixed = f"{package_prefix}.{stem}"
                if prefixed in export_registry:
                    available_names = export_registry[prefixed]

        if available_names is None:
            # Module not in registry — can't check, skip
            continue

        # Check each imported name
        for name in imported_names:
            if name in available_names:
                continue  # Name exists, all good

            # Name not found — try to find a match
            match = _find_best_match(name, available_names)
            if match:
                correct_name, method, confidence = match

                # Build the fixed line
                fixed_line = full_line.replace(name, correct_name)

                fixes.append(ImportFix(
                    file_path=file_path,
                    line_number=line_num,
                    original_line=full_line.strip(),
                    fixed_line=fixed_line.strip(),
                    wrong_name=name,
                    correct_name=correct_name,
                    target_module=module_path,
                    fix_method=method,
                    confidence=confidence,
                ))
                logger.info(
                    "[post_recon] MISMATCH in %s:%d — '%s' → '%s' (%s, %.0f%% confidence)",
                    file_path, line_num, name, correct_name, method, confidence * 100,
                )
            else:
                logger.warning(
                    "[post_recon] UNRESOLVABLE in %s:%d — '%s' not found in %s exports: %s",
                    file_path, line_num, name, module_path,
                    sorted(available_names)[:10],
                )

    return fixes


# =============================================================================
# SURGICAL FILE EDITOR
# =============================================================================

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


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_post_execution_reconciliation(
    manifest: Any,
    state: Any,
    sandbox_base: str = "D:\\Orb",
    min_confidence: float = 0.6,
    dry_run: bool = False,
    on_progress: Any = None,
) -> ReconciliationResult:
    """
    Run post-execution import reconciliation across all segment outputs.

    This is the Option B fallback. Call after segment execution completes
    (partial or full) to fix import mismatches between segments.

    Args:
        manifest: SegmentManifest with all segment specs
        state: JobState with segment statuses and output files
        sandbox_base: Root of the sandbox filesystem (e.g. D:\\Orb)
        min_confidence: Minimum confidence threshold for applying fixes (0.0-1.0)
        dry_run: If True, detect but don't apply fixes
        on_progress: Optional callback for progress messages

    Returns:
        ReconciliationResult with details of all fixes
    """
    _emit = on_progress or (lambda msg: None)
    result = ReconciliationResult()

    _emit("🔧 [POST-RECON] Starting post-execution import reconciliation...")

    # =========================================================================
    # Step 1: Collect all files from segments
    # =========================================================================
    all_file_contents: Dict[str, str] = {}  # rel_path -> content
    segment_file_map: Dict[str, str] = {}   # rel_path -> segment_id

    for seg_spec in manifest.segments:
        seg_id = seg_spec.segment_id
        for rel_path in seg_spec.file_scope:
            if not rel_path.endswith(".py"):
                continue

            abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
            try:
                if os.path.isfile(abs_path):
                    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    all_file_contents[rel_path] = content
                    segment_file_map[rel_path] = seg_id
                    result.files_scanned += 1
            except Exception as e:
                logger.warning("[post_recon] Cannot read %s: %s", abs_path, e)

    _emit(f"🔧 [POST-RECON] Scanned {result.files_scanned} files from {len(set(segment_file_map.values()))} segments")

    if result.files_scanned < 2:
        _emit("🔧 [POST-RECON] Not enough files to check — skipping")
        return result

    # =========================================================================
    # Step 2: Build export registry
    # =========================================================================
    export_registry = _build_export_registry(all_file_contents)
    total_exports = sum(len(v) for v in export_registry.values())
    _emit(f"🔧 [POST-RECON] Built export registry: {len(export_registry)} modules, {total_exports} total names")

    # =========================================================================
    # Step 3: Detect mismatches in each file
    # =========================================================================
    all_fixes: Dict[str, List[ImportFix]] = {}  # rel_path -> fixes

    # Determine the common package prefix from file paths
    if all_file_contents:
        first_path = next(iter(all_file_contents.keys()))
        parts = first_path.replace("\\", "/").split("/")
        # Find the package directory (where __init__.py would be)
        # e.g. "app/overwatcher/architecture_executor/foo.py" -> "app.overwatcher.architecture_executor"
        if len(parts) > 1:
            package_prefix = ".".join(parts[:-1])
        else:
            package_prefix = ""
    else:
        package_prefix = ""

    for rel_path, content in all_file_contents.items():
        fixes = detect_import_mismatches(
            file_path=rel_path,
            file_content=content,
            export_registry=export_registry,
            package_prefix=package_prefix,
        )

        # Filter by confidence
        confident_fixes = [f for f in fixes if f.confidence >= min_confidence]
        if confident_fixes:
            all_fixes[rel_path] = confident_fixes

        # Log low-confidence fixes that were filtered out
        low_conf = [f for f in fixes if f.confidence < min_confidence]
        for lf in low_conf:
            logger.info(
                "[post_recon] LOW CONFIDENCE (%.0f%%) — %s:%d '%s' → '%s' — skipped",
                lf.confidence * 100, lf.file_path, lf.line_number,
                lf.wrong_name, lf.correct_name,
            )

    total_fixes = sum(len(f) for f in all_fixes.values())
    _emit(f"🔧 [POST-RECON] Found {total_fixes} import mismatch(es) in {len(all_fixes)} file(s)")

    if total_fixes == 0:
        _emit("✅ [POST-RECON] No import mismatches detected")
        return result

    # =========================================================================
    # Step 4: Apply fixes
    # =========================================================================
    for rel_path, fixes in all_fixes.items():
        seg_id = segment_file_map.get(rel_path, "?")

        for fix in fixes:
            _emit(
                f"  🔧 {rel_path}:{fix.line_number} — "
                f"`{fix.wrong_name}` → `{fix.correct_name}` "
                f"({fix.fix_method}, {fix.confidence:.0%})"
            )
            result.fixes_applied.append(fix)

        if dry_run:
            _emit(f"  📋 [DRY RUN] Would fix {len(fixes)} import(s) in {rel_path}")
            continue

        # Apply fixes to content
        original_content = all_file_contents[rel_path]
        patched_content = apply_import_fixes(original_content, fixes)

        if patched_content == original_content:
            _emit(f"  ⚠️ No changes after applying fixes to {rel_path} — skipping write")
            continue

        # Write back to sandbox
        abs_path = os.path.join(sandbox_base, rel_path.replace("/", os.sep))
        try:
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(patched_content)
            result.files_fixed += 1
            _emit(f"  ✅ Fixed and saved: {rel_path} ({len(fixes)} fix(es))")
            logger.info(
                "[post_recon] Wrote fixed file: %s (%d fixes, %d→%d chars)",
                rel_path, len(fixes), len(original_content), len(patched_content),
            )
        except Exception as e:
            error_msg = f"Failed to write {rel_path}: {e}"
            result.errors.append(error_msg)
            result.files_with_errors.append(rel_path)
            _emit(f"  ❌ {error_msg}")
            logger.error("[post_recon] %s", error_msg)

    # =========================================================================
    # Step 5: Summary
    # =========================================================================
    _emit(
        f"🔧 [POST-RECON] Complete: {result.files_fixed} file(s) fixed, "
        f"{len(result.fixes_applied)} import(s) corrected"
    )
    if result.errors:
        _emit(f"⚠️ [POST-RECON] {len(result.errors)} error(s) occurred")

    return result


# =============================================================================
# v5.18: DEFERRED CONSUMER IMPORT RECONCILIATION
# =============================================================================

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


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "ImportFix",
    "ReconciliationResult",
    "run_post_execution_reconciliation",
    "detect_import_mismatches",
    "apply_import_fixes",
    "POST_RECON_BUILD_ID",
    "reconcile_deferred_consumers",
]
