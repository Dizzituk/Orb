from __future__ import annotations
import os
import re
from typing import List


SIMPLE_CREATE_BUILD_ID = "2026-02-09-v5.1-pre-resolve-mentioned-files"

_EVIDENCE_MAX_LOOPS = int(os.getenv("ASTRA_EVIDENCE_MAX_LOOPS", "3"))  # v4.2: Increased from 2 to 3

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
        if not os.path.isdir(root_path):
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


# Auto-generated re-exports for symbols in numbered _utils files
_REEXPORT_MAP = {
    "CreateEvidence": "_simple_create_utils_10",
    "IntegrationPoint": "_simple_create_utils_10",
    "_detect_tech_stack": "_simple_create_utils_10",
    "TechStack": "_simple_create_utils_11",
    "_run_llm_analysis": "_simple_create_utils_11",
    "build_grounded_create_spec": "_simple_create_utils_11",
    "ARCHITECTURAL_FILE_PATTERNS": "_simple_create_utils_7",
    "CONCEPT_KEYWORDS": "_simple_create_utils_7",
    "KEYWORD_STOPWORDS": "_simple_create_utils_7",
    "MIN_KEYWORD_LENGTH": "_simple_create_utils_7",
    "NEGATION_PATTERNS": "_simple_create_utils_7",
    "PLACEHOLDER_GOALS": "_simple_create_utils_7",
    "_resolve_mentioned_files": "_simple_create_utils_7",
    "_score_integration_point": "_simple_create_utils_7",
    "CONCEPT_DIRECTORY_PATTERNS": "_simple_create_utils_8",
    "_CREATE_ANALYSIS_TIMEOUT": "_simple_create_utils_8",
    "_EVIDENCE_MAX_FILE_CHARS": "_simple_create_utils_8",
    "_extract_constraints": "_simple_create_utils_8",
    "_extract_task_keywords": "_simple_create_utils_8",
    "_host_list_directory": "_simple_create_utils_8",
    "_sanitize_goal": "_simple_create_utils_8",
    "_suggest_new_files": "_simple_create_utils_8",
    "_extract_patterns": "_simple_create_utils_9",
    "_find_integration_points": "_simple_create_utils_9",
    "_host_read_file": "_simple_create_utils_9",
    "_read_text_any_encoding": "_simple_create_utils_9",
    "build_create_spec": "_simple_create_utils_9",
}

def __getattr__(name):
    if name in _REEXPORT_MAP:
        import importlib
        mod = importlib.import_module(f"app.pot_spec.grounded.{_REEXPORT_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
