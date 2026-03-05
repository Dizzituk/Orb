from __future__ import annotations
import os
import re
from app.pot_spec.grounded._simple_create_utils_12 import _CONTENT_SIGNALS, _NEGATIVE_PATH_SEGMENTS, _find_file_in_projects
from app.pot_spec.grounded._sbx_fs import _sbx_read
from typing import Any, Dict, List

# v3.2-fix: Sandbox-aware filesystem checks for codebase paths.
# v4.3: Spec gate uses HOST filesystem, not sandbox.


KEYWORD_STOPWORDS = {
    # Articles, prepositions, conjunctions
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'of', 'for',
    'by', 'or', 'and', 'but', 'not', 'no', 'nor', 'so', 'as', 'if',
    'do', 'be', 'am', 'are', 'was', 'were', 'has', 'had', 'have',
    'will', 'can', 'may', 'with', 'from', 'that', 'this', 'than',
    'its', 'my', 'any', 'all', 'each', 'every',
    # Common verbs that match too many files
    'get', 'set', 'run', 'use', 'add', 'new', 'put', 'let',
    # Task description noise
    'should', 'must', 'need', 'want', 'like', 'also', 'just',
    'only', 'ever', 'never', 'always', 'still', 'yet',
    # Generic computing terms too broad for filename matching
    'file', 'data', 'type', 'name', 'path', 'list', 'item',
    'mode', 'test', 'main', 'base', 'core', 'util', 'help',
}

MIN_KEYWORD_LENGTH = 3

CONCEPT_KEYWORDS = {
    "voice": ["voice", "audio", "speech", "microphone", "mic", "record"],
    "text_input": ["input field", "text input", "textarea", "text entry"],
    "button": ["button", "toggle", "click handler"],
    "api_endpoint": ["endpoint", "api route", "http endpoint", "fastapi route"],
    "file_upload": ["upload", "file upload", "blob upload"],
    "transcription": ["transcribe", "transcription", "whisper", "speech-to-text", "stt"],
    "ui_component": ["component", "widget", "ui element"],
    "state_management": ["useState", "context", "store", "redux", "state management"],
    "streaming": ["stream", "sse", "server-sent", "websocket", "real-time"],
    "model_management": ["model download", "model loading", "model manager", "gpu detection"],
    "wake_word": ["wake word", "hotword", "always listening", "voice activation"],
    "noise_suppression": ["noise", "noise suppression", "vad", "voice activity"],
}

NEGATION_PATTERNS = [
    (r'no\s+(cloud\s+api|cloud\s+service|external\s+api|remote\s+api)', 'no_cloud_api'),
    (r'no\s+audio.*leave', 'no_audio_upload'),
    (r'local[\s-]+only', 'local_only'),
    (r'no\s+mobile', 'no_mobile'),
    (r'no\s+multi[\s-]*language', 'no_multilingual'),
    (r'never\s+leav(?:e|ing)\s+the\s+machine', 'local_only'),
    (r'not\s+(?:a|an)?\s*cloud', 'no_cloud_api'),
    (r'privacy\s+constraint', 'privacy_critical'),
    (r'no\s+network\s+traffic', 'local_only'),
    (r'phase\s+1\s+only', 'phase_1_scope'),
    (r'desktop[\s-]+only', 'desktop_only'),
    (r'no\s+(?:tts|text[\s-]*to[\s-]*speech)', 'no_tts'),
]

def _resolve_mentioned_files(
    text: str,
    project_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    v5.1: Extract filenames mentioned in the job description and resolve
    them to real filesystem paths under project_paths.

    Looks for patterns like:
    - `filename.py` (backtick-quoted)
    - filename.py (bare, with .py/.js/.ts/.jsx/.tsx/.json/.md/.yaml/.yml extension)

    Returns list of dicts: {mentioned, resolved_path, size_bytes}
    """
    results = []
    seen_filenames = set()

    # Extract filenames from backtick-quoted references
    backtick_pattern = re.compile(r'`([\w./_-]+\.(?:py|js|ts|jsx|tsx|json|md|yaml|yml|toml|cfg|txt))`')
    # Also match bare filenames with extensions (word boundaries)
    bare_pattern = re.compile(r'\b([\w_-]+\.(?:py|js|ts|jsx|tsx|json|md|yaml|yml|toml|cfg|txt))\b')

    for pattern in [backtick_pattern, bare_pattern]:
        for match in pattern.finditer(text):
            filename = match.group(1)
            # Normalise — take just the basename for searching
            basename = os.path.basename(filename)
            if basename in seen_filenames or not basename:
                continue
            seen_filenames.add(basename)

            # Search for this file across all project paths
            resolved = _find_file_in_projects(basename, project_paths)
            if resolved:
                for rpath in resolved:
                    try:
                        size = os.path.getsize(rpath)
                    except OSError:
                        size = 0
                    results.append({
                        "mentioned": filename,
                        "resolved_path": rpath,
                        "size_bytes": size,
                    })

    return results

ARCHITECTURAL_FILE_PATTERNS = [
    # v3.3: Tightened - NO files auto-flagged as "modify".
    # The LLM decides what to modify based on the actual task.
    # These are structural reference points only.
    (r'^App\.(tsx|jsx)$', "Root component", "reference"),
    (r'^main\.(tsx|jsx)$', "Frontend entry point", "reference"),
    (r'^index\.(tsx|jsx)$', "Frontend entry point", "reference"),
    (r'^main\.py$', "Backend entry point", "reference"),
    (r'^types\.(ts|d\.ts)$', "Type definitions", "reference"),
]
def _score_integration_point(file_path: str, project_path: str) -> int:
    """
    v3.6: Score an integration point based on content signals and path heuristics.
    Only reads the first 3KB of the file for efficiency.
    Positive scores = more likely architecturally relevant.
    Negative scores = likely a false positive.
    """
    score = 0
    rel_path = os.path.relpath(file_path, project_path).lower().replace('\\', '/')
    path_segments = set(rel_path.split('/'))

    if path_segments & _NEGATIVE_PATH_SEGMENTS:
        score -= 10

    filename = os.path.basename(file_path).lower()
    if filename in ('main.py', 'app.py', 'index.py', 'server.py'):
        head = _sbx_read(file_path)
        if head:
            head = head[:3072]
            for pattern, delta in _CONTENT_SIGNALS:
                if pattern.search(head):
                    score += delta

    return score

PLACEHOLDER_GOALS = {
    "job description",
    "job description from weaver", 
    "weaver output",
    "task description",
    "implement requested feature",
    "complete the requested task",
}
