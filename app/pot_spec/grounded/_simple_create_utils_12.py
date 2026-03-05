from __future__ import annotations
import os
import re
from typing import List
from app.pot_spec.grounded._sbx_fs import _sbx_isfile, _sbx_isdir

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.


SIMPLE_CREATE_BUILD_ID = "2026-02-09-v5.1-pre-resolve-mentioned-files"

_EVIDENCE_MAX_LOOPS = int(os.getenv("ASTRA_EVIDENCE_MAX_LOOPS", "8"))  # v4.3: Safety ceiling only. Normal exit is LLM emitting no more ERs.

_CREATE_ANALYSIS_MODEL = os.getenv("ASTRA_CREATE_ANALYSIS_MODEL", "")

def _find_file_in_projects(
    filename: str,
    project_paths: List[str],
    max_results: int = 3,
) -> List[str]:
    """
    v10.0: Find files matching filename using the architecture INDEX.json
    and sandbox existence checks. No host os.walk.

    Strategy:
    1. Check architecture INDEX.json for known paths matching the filename
    2. Try common subdirectory patterns (src/, app/) via sandbox
    3. Fall back to direct root+filename check via sandbox

    Returns up to max_results absolute paths.
    """
    found = []
    basename = os.path.basename(filename)

    # Strategy 1: Search architecture INDEX.json for matching filenames
    try:
        import json as _json
        _idx_path = os.path.join("D:\\Orb", ".architecture", "INDEX.json")
        if os.path.isfile(_idx_path):  # INDEX.json is host operational data
            with open(_idx_path, "r", encoding="utf-8") as _f:
                _idx = _json.load(_f)
            for _file_entry in _idx.get("files", []):
                _entry_name = _file_entry.get("name", "")
                _entry_path = _file_entry.get("path", "")
                if _entry_name == basename and _entry_path:
                    # Verify it exists in the sandbox before returning
                    if _sbx_isfile(_entry_path):
                        found.append(_entry_path)
                        if len(found) >= max_results:
                            return found
    except Exception:
        pass

    if found:
        return found

    # Strategy 2: Try common subdirectory patterns via sandbox
    _subdirs = ["src", "app", "src/components", "src/services", "src/types"]
    for root_path in project_paths:
        for subdir in _subdirs:
            candidate = os.path.join(root_path, subdir, filename)
            if _sbx_isfile(candidate):
                found.append(candidate)
                if len(found) >= max_results:
                    return found
        # Also try direct root + filename
        candidate = os.path.join(root_path, filename)
        if _sbx_isfile(candidate):
            found.append(candidate)
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
