from __future__ import annotations
import os
import re
from typing import List

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.
_SBX_FS_OK = False


SIMPLE_CREATE_BUILD_ID = "2026-02-09-v5.1-pre-resolve-mentioned-files"

_EVIDENCE_MAX_LOOPS = int(os.getenv("ASTRA_EVIDENCE_MAX_LOOPS", "8"))  # v4.3: Safety ceiling only. Normal exit is LLM emitting no more ERs.

_CREATE_ANALYSIS_MODEL = os.getenv("ASTRA_CREATE_ANALYSIS_MODEL", "")

def _find_file_in_projects(
    filename: str,
    project_paths: List[str],
    max_results: int = 3,
) -> List[str]:
    """
    v5.1: Walk project directories to find files matching the given filename.

    Returns up to max_results absolute paths. Skips common junk directories.
    """
    skip_dirs = {
        '__pycache__', '.git', 'node_modules', '.venv', 'venv',
        '.tox', '.mypy_cache', '.pytest_cache', 'dist', 'build',
        '.next', '.nuxt', 'eggs', '*.egg-info',
    }
    found = []
    for root_path in project_paths:
        if not (_sbx_isdir(root_path) if _SBX_FS_OK else os.path.isdir(root_path)):
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            # Prune junk dirs
            dirnames[:] = [
                d for d in dirnames
                if d not in skip_dirs and not d.endswith('.egg-info')
            ]
            if filename in filenames:
                full = os.path.join(dirpath, filename)
                found.append(full)
                if len(found) >= max_results:
                    return found
    return found

_CONTENT_SIGNALS = [
    (re.compile(r'FastAPI\s*\('), +10),
    (re.compile(r'from\s+fastapi\s+import\s+FastAPI'), +10),
    (re.compile(r'include_router\s*\('), +5),
    (re.compile(r'app\.mount\s*\('), +3),
    (re.compile(r'@app\.on_event'), +3),
    (re.compile(r'uvicorn\.run'), +3),
]

_NEGATIVE_PATH_SEGMENTS = {'static', 'dist', 'build', 'public', 'assets', 'out', '.next'}

_FALLBACK_MODELS = [
    ("openai", "gpt-5-mini"),
    ("anthropic", "claude-sonnet-4-5-20250929"),
]

def _extract_acceptance_from_constraints(constraints: List[str]) -> List[str]:
    """
    v2.0: Generate task-specific acceptance criteria from constraints.
    """
    criteria = []
    for constraint in constraints:
        c_lower = constraint.lower()
        if 'no cloud' in c_lower or 'local' in c_lower:
            criteria.append("No network traffic during transcription (verify with network monitor)")
        if 'no audio' in c_lower and 'leave' in c_lower:
            criteria.append("Audio data never leaves the machine — no outbound connections")
        if 'desktop' in c_lower:
            criteria.append("Works on desktop platform as specified")
        if 'phase 1' in c_lower:
            criteria.append("Only Phase 1 features implemented — no scope creep")
    return criteria
