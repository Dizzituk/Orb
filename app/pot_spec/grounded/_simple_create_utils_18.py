# Purpose: Grounding helpers for build_grounded_create_spec (v2.3–v2.6).
# Called-by: app.pot_spec.grounded._simple_create_utils_17
# Depends-on: app.pot_spec.grounded._dependency_presence
# Last-renovated: 2026-06-11
"""
Grounding helpers for build_grounded_create_spec (v2.3–v2.6).
=============================================================
Extracted from _simple_create_utils_17.py to keep that file under the
20 KB target. All four helpers below are pure functions with explicit
inputs — they mutate nothing outside the values they return.

v2.3 inject_brief_mentioned_roots   — scan Weaver brief for real project roots
v2.4 priority_sort_project_paths    — mobile/native roots walked first
v2.5 run_dependency_gate            — append Dropped Dependencies block
v2.6 filter_phantom_files           — drop React/Python entry-point phantoms
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# v2.3: BRIEF-MENTIONED ROOT INJECTION
# ============================================================================

_ROOT_MARKERS = (
    "app/build.gradle.kts", "build.gradle.kts", "settings.gradle.kts",
    "package.json", "pyproject.toml", "Cargo.toml", "go.mod", ".git",
)

_PATH_PAT = re.compile(r'([A-Za-z]:[\\/](?:[^\r\n`\'<>|*?]+[\\/])+)')


def inject_brief_mentioned_roots(project_paths: List[str], combined_text: str) -> Tuple[List[str], List[str]]:
    """Scan brief text for absolute paths, walk up to root markers, prepend new roots.

    Returns (new_project_paths, added_roots).
    """
    added: List[str] = []
    try:
        candidates = set()
        for m in _PATH_PAT.finditer(combined_text or ""):
            p = m.group(1).rstrip("\\/").replace("/", os.sep)
            cur = p
            for _ in range(8):
                if not cur or len(cur) < 4:
                    break
                if any(os.path.exists(os.path.join(cur, mk.replace("/", os.sep))) for mk in _ROOT_MARKERS):
                    candidates.add(cur)
                    break
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent
        existing = {os.path.normcase(os.path.normpath(p)) for p in project_paths}
        for c in sorted(candidates):
            if os.path.normcase(os.path.normpath(c)) not in existing:
                project_paths = list(project_paths) + [c]
                added.append(c)
    except Exception as e:
        logger.warning("[v2.3] Brief-root injection failed: %s", e)
    return project_paths, added


# ============================================================================
# v2.4: PRIORITY SORT — mobile/native walked first
# ============================================================================

_PRIORITY_MARKERS = (
    "app/build.gradle.kts", "build.gradle.kts", "settings.gradle.kts",
    "AndroidManifest.xml", "Cargo.toml", "go.mod",
)


def _priority_key(p: str) -> int:
    try:
        for marker in _PRIORITY_MARKERS:
            if os.path.exists(os.path.join(p, marker.replace("/", os.sep))):
                return 0
    except Exception:
        pass
    return 1


def priority_sort_project_paths(project_paths: List[str]) -> List[str]:
    """Sort so mobile/native project roots are walked first."""
    return sorted(project_paths, key=_priority_key)


# ============================================================================
# v2.5: DEPENDENCY-PRESENCE GATE
# ============================================================================

def run_dependency_gate(llm_analysis: str, project_paths: List[str]) -> str:
    """Scan llm_analysis for library markers not declared in any project root.
    Append a 'Dropped Dependencies' block if any are missing. Returns
    the possibly-modified llm_analysis string.
    """
    if not llm_analysis:
        return llm_analysis
    try:
        from app.pot_spec.grounded._dependency_presence import scan_proposed_content, is_dependency_available
        all_missing: List[Tuple[str, str, str]] = []
        seen = set()
        for root in project_paths:
            if not root:
                continue
            for friendly, dep in scan_proposed_content(llm_analysis, root):
                if (friendly, dep) not in seen:
                    seen.add((friendly, dep))
                    all_missing.append((friendly, dep, root))
        if not all_missing:
            print("[v2.5] DEPENDENCY GATE: no library markers in proposal")
            return llm_analysis
        truly_missing: List[Tuple[str, str]] = []
        for friendly, dep, _src in all_missing:
            if not any(is_dependency_available(r, dep) for r in project_paths if r):
                truly_missing.append((friendly, dep))
        if not truly_missing:
            print("[v2.5] DEPENDENCY GATE: all proposed libs are declared")
            return llm_analysis
        block = ["", "## Dropped Dependencies", "",
                 "The following libraries were referenced in the proposed implementation",
                 "but are NOT declared in the target project's build files. They have",
                 "been flagged so they don't reach the implementer as silent compile errors.",
                 "Add them to your build.gradle.kts (or equivalent) first if you actually",
                 "want to use them:", ""]
        for f, d in truly_missing:
            block.append(f"- **{f}** — declared dependency `{d}` not found")
        block.append("")
        print(f"[v2.5] DEPENDENCY GATE: flagged {len(truly_missing)} missing lib(s): {[d for _, d in truly_missing]}")
        logger.info("[v2.5] Dependency gate flagged: %s", truly_missing)
        return llm_analysis + "\n" + "\n".join(block)
    except Exception as e:
        logger.warning("[v2.5] Dependency gate failed: %s", e)
        return llm_analysis


# ============================================================================
# v2.6: PHANTOM-FILE FILTER
# ============================================================================

_PHANTOM_NAMES = frozenset({
    "app.tsx", "app.jsx", "main.tsx", "main.jsx",
    "main.py", "index.tsx", "index.jsx",
})


def filter_phantom_files(all_points: List[Any], tech_stack: Any) -> List[Any]:
    """Drop React/Python entry-point phantom integration points when the
    target is a mobile/native project. Returns the filtered list.
    """
    try:
        stack_str = " ".join(str(getattr(tech_stack, attr, "") or "")
                              for attr in ("frontend_framework", "frontend_language", "backend_language")).lower()
        if not any(k in stack_str for k in ("android", "kotlin", "ios", "swift")):
            return all_points
        filtered: List[Any] = []
        dropped: List[str] = []
        for p in all_points:
            fname = (getattr(p, "file_name", "") or "").lower()
            if fname in _PHANTOM_NAMES:
                dropped.append(fname)
                continue
            filtered.append(p)
        if dropped:
            print(f"[v2.6] PHANTOM-FILE FILTER: dropped {len(dropped)} non-mobile entry point(s): {dropped}")
            logger.info("[v2.6] Phantom filter dropped: %s", dropped)
        return filtered
    except Exception as e:
        logger.warning("[v2.6] Phantom filter failed: %s", e)
        return all_points