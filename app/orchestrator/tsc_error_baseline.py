# FILE: app/orchestrator/tsc_error_baseline.py
"""
Known pre-existing TypeScript errors that the sandbox will always produce.

v1.0 (2026-03-01): Initial baseline.
    The Windows Sandbox does not have numpy, whisper models, or other
    ML dependencies installed. Test files that import these will always
    fail tsc. These errors are pre-existing and must not trigger
    stage_8_overwatcher routing or any corrective action.

    Root cause: job sg-a798331a regression — phase checkout failed on
    pre-existing VoiceInput.test.tsx errors, triggering overwatcher to
    rewrite JobPage.tsx from scratch and destroy all existing tab routing.
"""

from __future__ import annotations

import re
from typing import List, Set

# ─── Known baseline error patterns ───────────────────────────────────────────
# Each entry is a tuple of (file_pattern, error_code, description).
# file_pattern: regex matched against the error's file path.
# error_code: exact TS error code, or None to match any code for that file.
#
# If a tsc error matches ANY baseline entry, it is treated as pre-existing
# and excluded from failure routing decisions.

BASELINE_PATTERNS: List[tuple] = [
    # VoiceInput.test.tsx — depends on numpy/whisper not available in sandbox
    (re.compile(r"__tests__/VoiceInput\.test\.tsx$"), None,
     "VoiceInput test depends on ML models not available in sandbox"),

    # Any .test.tsx / .test.ts file — test files may have dependencies
    # not installed in the sandbox. Only flag if they're in segment output.
    # (This is a soft baseline — filter_errors_by_segment handles scoping.)
]


def is_baseline_error(file_path: str, error_code: str) -> bool:
    """Check if a tsc error matches a known pre-existing baseline pattern.

    Args:
        file_path: Normalised file path from tsc output.
        error_code: TS error code (e.g. "TS1002").

    Returns:
        True if this error is known pre-existing and should be ignored.
    """
    norm = file_path.replace("\\", "/")
    for pattern, code, _desc in BASELINE_PATTERNS:
        if pattern.search(norm):
            if code is None or code == error_code:
                return True
    return False


def filter_baseline_errors(errors: list) -> tuple:
    """Split tsc errors into baseline (pre-existing) and new (actionable).

    Args:
        errors: List of TscError objects.

    Returns:
        Tuple of (new_errors, baseline_errors).
    """
    new_errors = []
    baseline = []
    for err in errors:
        if is_baseline_error(err.file, err.code):
            baseline.append(err)
        else:
            new_errors.append(err)
    return new_errors, baseline