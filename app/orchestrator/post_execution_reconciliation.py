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
from app.orchestrator._post_execution_reconciliation_utils_2 import POST_RECON_BUILD_ID, _build_export_registry, _extract_imports_with_lines, _find_best_match, apply_import_fixes, reconcile_deferred_consumers

logger = logging.getLogger(__name__)
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


# =============================================================================
# IMPORT MISMATCH DETECTION
# =============================================================================


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
