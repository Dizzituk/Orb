# FILE: app/orchestrator/deterministic_cohesion.py
"""
Deterministic cohesion pre-check — zero LLM cost.

Runs structural cross-segment validation BEFORE the LLM cohesion
check. Catches import mismatches, interface inconsistencies, and
missing exports that the LLM would otherwise need to discover.

When structural issues are found, they're injected into the LLM
cohesion prompt so it can focus on semantic correctness rather
than rediscovering structural problems.

Reuses import_validator.py logic for cross-segment import checks.

Called from: seg_pipeline_step3.py or cohesion_check.py
Returns: DeterministicCohesionResult with structural issues found

Usage:
    from app.orchestrator.deterministic_cohesion import run_deterministic_cohesion

    result = run_deterministic_cohesion(segment_files, enrichment_data)
    if result.has_blockers:
        # Inject into LLM prompt or return directly
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class CohesionIssue:
    """A structural cross-segment issue."""
    check: str               # "import", "interface", "export", "collision"
    source_segment: str      # Segment with the issue
    target_segment: str      # Related segment (if applicable)
    severity: str            # "blocker" or "warning"
    message: str


@dataclass
class DeterministicCohesionResult:
    """Result of deterministic cohesion checks."""
    passed: bool
    issues: List[CohesionIssue] = field(default_factory=list)
    segments_checked: int = 0

    @property
    def has_blockers(self) -> bool:
        return any(i.severity == "blocker" for i in self.issues)

    def format_for_prompt(self) -> str:
        """Format issues for injection into LLM cohesion prompt."""
        if not self.issues:
            return ""

        lines = ["[DETERMINISTIC COHESION PRE-CHECK FINDINGS]"]
        for issue in self.issues:
            tag = "BLOCKER" if issue.severity == "blocker" else "WARNING"
            lines.append(
                f"  [{tag}] {issue.check}: {issue.message} "
                f"(segments: {issue.source_segment} → {issue.target_segment})"
            )
        lines.append("[/DETERMINISTIC COHESION PRE-CHECK FINDINGS]")
        return "\n".join(lines)


# =========================================================================
# Cross-segment import check
# =========================================================================

def check_cross_segment_imports(
    segment_files: Dict[str, Dict[str, str]],
) -> List[CohesionIssue]:
    """
    Check that imports between segments resolve correctly.

    Args:
        segment_files: Dict of {segment_id: {file_path: content}}

    Returns list of CohesionIssue for unresolved cross-segment imports.
    """
    # Build a global file index
    all_files: Set[str] = set()
    file_to_segment: Dict[str, str] = {}
    for seg_id, files in segment_files.items():
        for file_path in files:
            all_files.add(file_path)
            file_to_segment[file_path] = seg_id

    # Build export map: what each file exports
    exports_by_file: Dict[str, Set[str]] = {}
    for seg_id, files in segment_files.items():
        for file_path, content in files.items():
            if not file_path.endswith(".py"):
                continue
            exports_by_file[file_path] = _extract_exports(content, file_path)

    issues: List[CohesionIssue] = []

    for seg_id, files in segment_files.items():
        for file_path, content in files.items():
            if not file_path.endswith(".py"):
                continue

            imports = _extract_imports(content, file_path)
            for imp_module, imp_names, line_no in imports:
                # Only check app.* imports (internal)
                if not imp_module.startswith("app."):
                    continue

                # Find the target file
                target_path = imp_module.replace(".", os.sep) + ".py"
                target_init = imp_module.replace(".", os.sep) + os.sep + "__init__.py"

                target = None
                if target_path in all_files:
                    target = target_path
                elif target_init in all_files:
                    target = target_init

                if target is None:
                    # Module not in any segment — might be pre-existing
                    continue

                target_seg = file_to_segment.get(target, "")
                if target_seg == seg_id:
                    # Same segment — not a cross-segment issue
                    continue

                # Cross-segment import — check names resolve
                target_exports = exports_by_file.get(target, set())
                for name in imp_names:
                    if name not in target_exports and target_exports:
                        issues.append(CohesionIssue(
                            check="import",
                            source_segment=seg_id,
                            target_segment=target_seg,
                            severity="blocker",
                            message=(
                                f"'{file_path}' imports '{name}' from "
                                f"'{imp_module}' but it's not exported. "
                                f"Available: {sorted(target_exports)[:5]}"
                            ),
                        ))

    return issues


# =========================================================================
# Function name collision check
# =========================================================================

def check_function_collisions(
    segment_files: Dict[str, Dict[str, str]],
) -> List[CohesionIssue]:
    """Check for public function name collisions across segments."""
    func_locations: Dict[str, List[tuple]] = {}  # name → [(segment, file)]

    for seg_id, files in segment_files.items():
        for file_path, content in files.items():
            if not file_path.endswith(".py"):
                continue
            try:
                tree = ast.parse(content, filename=file_path)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        func_locations.setdefault(node.name, []).append(
                            (seg_id, file_path)
                        )

    issues = []
    for func_name, locations in func_locations.items():
        segments = set(loc[0] for loc in locations)
        if len(segments) > 1:
            loc_str = ", ".join(f"{s}:{f}" for s, f in locations[:3])
            issues.append(CohesionIssue(
                check="collision",
                source_segment=locations[0][0],
                target_segment=locations[1][0],
                severity="warning",
                message=f"Public function '{func_name}' defined in multiple segments: {loc_str}",
            ))

    return issues


# =========================================================================
# AST helpers
# =========================================================================

def _extract_exports(content: str, file_path: str) -> Set[str]:
    """Extract all public names defined in a file."""
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return set()

    names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)

    return names


def _extract_imports(
    content: str,
    file_path: str,
) -> List[tuple]:
    """
    Extract import-from statements.

    Returns: [(module, [names], line_number), ...]
    """
    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [alias.name for alias in (node.names or [])]
            imports.append((node.module, names, node.lineno))

    return imports


# =========================================================================
# Main entry point
# =========================================================================

def run_deterministic_cohesion(
    segment_files: Dict[str, Dict[str, str]],
) -> DeterministicCohesionResult:
    """
    Run all deterministic cross-segment cohesion checks.

    Args:
        segment_files: Dict of {segment_id: {file_path: content}}

    Returns:
        DeterministicCohesionResult with all issues found.
    """
    all_issues: List[CohesionIssue] = []

    # Cross-segment imports
    all_issues.extend(check_cross_segment_imports(segment_files))

    # Function collisions
    all_issues.extend(check_function_collisions(segment_files))

    has_blockers = any(i.severity == "blocker" for i in all_issues)

    result = DeterministicCohesionResult(
        passed=not has_blockers,
        issues=all_issues,
        segments_checked=len(segment_files),
    )

    if all_issues:
        logger.info(
            "[det_cohesion] %d issues (%d blockers) across %d segments",
            len(all_issues),
            sum(1 for i in all_issues if i.severity == "blocker"),
            len(segment_files),
        )

    return result


__all__ = [
    "CohesionIssue",
    "DeterministicCohesionResult",
    "run_deterministic_cohesion",
    "check_cross_segment_imports",
    "check_function_collisions",
]
