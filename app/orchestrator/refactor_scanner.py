# FILE: app/orchestrator/refactor_scanner.py
"""
Refactor Scanner — Deterministic file selection for the refactor loop.

Scans the codebase and returns the next file to refactor, ordered by
extractability (easiest first). No LLM calls. Pure filesystem + AST analysis.

The scanner:
1. Walks the codebase finding all Python files
2. Measures size, function count, class count
3. Builds an import dependency map (who imports whom)
4. Scores each file by extractability (lower = easier to extract from)
5. Returns the top candidate with full metadata

Extractability score (lower = easier):
- Base: file size in KB
- Penalty: +5 per file that imports from this file (dependents are costly)
- Bonus: -2 per function that's standalone (no cross-references)
- Bonus: -10 if leaf node (nothing imports from it)

Usage:
    from app.orchestrator.refactor_scanner import scan_for_refactor
    
    result = scan_for_refactor()
    if result.next_file is None:
        print("Nothing left to refactor!")
    else:
        print(f"Next: {result.next_file.path} ({result.next_file.size_kb:.1f}KB)")
"""

import ast
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Thresholds
MIN_SIZE_KB = 20  # Don't consider files under this size
SCAN_ROOTS = [r"D:\Orb\app"]
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", "dist", "build",
}


@dataclass
class FileCandidate:
    """A file that could be refactored."""
    path: str
    size_bytes: int
    size_kb: float
    line_count: int
    function_count: int
    class_count: int
    constant_count: int
    total_symbol_count: int
    imported_by: List[str] = field(default_factory=list)
    imports_from: List[str] = field(default_factory=list)
    extractability_score: float = 0.0
    is_leaf: bool = False
    role: str = "unknown"  # leaf, branch, root


@dataclass
class RefactorScanResult:
    """Result of a refactor scan."""
    timestamp: str
    scan_duration_ms: float
    total_files: int
    oversized_files: int
    next_file: Optional[FileCandidate]
    all_candidates: List[FileCandidate]
    progress: dict = field(default_factory=dict)


def _walk_python_files() -> Dict[str, int]:
    """Walk codebase, return {path: size_bytes} for all .py files."""
    files = {}
    for root_path in SCAN_ROOTS:
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                full = os.path.join(dirpath, fname)
                try:
                    files[full] = os.path.getsize(full)
                except OSError:
                    continue
    return files


def _count_symbols(path: str) -> dict:
    """Count top-level symbols in a Python file using AST."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, OSError, UnicodeDecodeError):
        return {"functions": 0, "classes": 0, "constants": 0, "lines": 0}

    functions = 0
    classes = 0
    constants = 0

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions += 1
        elif isinstance(node, ast.ClassDef):
            classes += 1
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            constants += 1

    return {
        "functions": functions,
        "classes": classes,
        "constants": constants,
        "lines": source.count("\n") + 1,
    }


def _build_import_map(files: Dict[str, int]) -> Dict[str, Set[str]]:
    """
    Build a map of {file_path: set of file_paths it imports from}.

    Only tracks internal imports (between files in our codebase).
    """
    # Map module paths to file paths for resolution
    module_to_file = {}
    for path in files:
        # D:\Orb\app\rag\lifecycle.py -> app.rag.lifecycle
        rel = path.replace(r"D:\Orb\\", "").replace("\\", ".").replace("/", ".")
        if rel.endswith(".py"):
            rel = rel[:-3]
        if rel.endswith(".__init__"):
            rel = rel[:-9]
        module_to_file[rel] = path

    imports_from = {p: set() for p in files}

    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source)
        except (SyntaxError, OSError, UnicodeDecodeError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
                # Check if this module maps to one of our files
                if mod in module_to_file:
                    target = module_to_file[mod]
                    if target != path:
                        imports_from[path].add(target)
                # Also check parent modules (for package imports)
                parts = mod.split(".")
                for i in range(len(parts), 0, -1):
                    prefix = ".".join(parts[:i])
                    if prefix in module_to_file:
                        target = module_to_file[prefix]
                        if target != path:
                            imports_from[path].add(target)
                        break

    return imports_from


def _compute_imported_by(
    imports_from: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """Invert the import map: {file: list of files that import it}."""
    imported_by: Dict[str, List[str]] = {p: [] for p in imports_from}
    for importer, targets in imports_from.items():
        for target in targets:
            if target in imported_by:
                imported_by[target].append(importer)
    return imported_by


def _score_candidate(
    candidate: FileCandidate,
    imported_by_count: int,
) -> float:
    """
    Compute extractability score. Lower = easier to refactor.

    Factors:
    - Size (bigger = harder, but also more reward)
    - Dependents (files that import this = harder to change)
    - Leaf bonus (nothing imports it = much easier)
    - Symbol density (many small functions = easier to extract)
    """
    score = candidate.size_kb

    # Penalty per dependent file
    score += imported_by_count * 5.0

    # Leaf bonus
    if imported_by_count == 0:
        score -= 10.0
        candidate.is_leaf = True
        candidate.role = "leaf"
    elif imported_by_count <= 3:
        candidate.role = "branch"
    else:
        candidate.role = "root"

    # Symbol density bonus (many functions = easy extractions)
    if candidate.function_count > 5:
        score -= (candidate.function_count - 5) * 0.5

    return max(score, 0.1)


def scan_for_refactor(
    min_size_kb: float = MIN_SIZE_KB,
) -> RefactorScanResult:
    """
    Scan codebase and return the next file to refactor.

    Returns RefactorScanResult with:
    - next_file: the easiest file to refactor (or None if done)
    - all_candidates: all oversized files ranked by extractability
    - progress: {total_oversized, leaf_count, branch_count, root_count}
    """
    start = time.time()

    # Walk filesystem
    all_files = _walk_python_files()
    total_files = len(all_files)

    # Filter to oversized
    oversized = {
        p: s for p, s in all_files.items()
        if s > min_size_kb * 1024
    }

    if not oversized:
        elapsed = (time.time() - start) * 1000
        return RefactorScanResult(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            scan_duration_ms=elapsed,
            total_files=total_files,
            oversized_files=0,
            next_file=None,
            all_candidates=[],
            progress={"total_oversized": 0, "status": "complete"},
        )

    # Build import map (only for oversized files + their deps)
    imports_from = _build_import_map(all_files)
    imported_by = _compute_imported_by(imports_from)

    # Build candidates
    candidates = []
    for path, size_bytes in oversized.items():
        symbols = _count_symbols(path)
        candidate = FileCandidate(
            path=path,
            size_bytes=size_bytes,
            size_kb=size_bytes / 1024,
            line_count=symbols["lines"],
            function_count=symbols["functions"],
            class_count=symbols["classes"],
            constant_count=symbols["constants"],
            total_symbol_count=(
                symbols["functions"] + symbols["classes"] + symbols["constants"]
            ),
            imported_by=imported_by.get(path, []),
            imports_from=list(imports_from.get(path, set())),
        )
        candidate.extractability_score = _score_candidate(
            candidate, len(candidate.imported_by)
        )
        candidates.append(candidate)

    # Sort by extractability (lowest = easiest = first)
    candidates.sort(key=lambda c: c.extractability_score)

    # Progress stats
    leaves = sum(1 for c in candidates if c.role == "leaf")
    branches = sum(1 for c in candidates if c.role == "branch")
    roots = sum(1 for c in candidates if c.role == "root")

    elapsed = (time.time() - start) * 1000

    logger.info(
        f"[refactor_scanner] Scanned {total_files} files in {elapsed:.0f}ms. "
        f"Oversized: {len(candidates)} (leaf={leaves}, branch={branches}, root={roots})"
    )

    return RefactorScanResult(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        scan_duration_ms=elapsed,
        total_files=total_files,
        oversized_files=len(candidates),
        next_file=candidates[0] if candidates else None,
        all_candidates=candidates,
        progress={
            "total_oversized": len(candidates),
            "leaves": leaves,
            "branches": branches,
            "roots": roots,
            "status": "refactoring",
        },
    )
