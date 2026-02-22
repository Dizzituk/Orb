# FILE: app/overwatcher/implementer.py
"""Implementer: Executes approved work and verifies results.

Handles:
- Writing files to sandbox based on spec
- Enforcing must_exist constraint for modify actions
- Verifying output matches spec requirements
- APPEND_IN_PLACE mode for appending to existing files (v1.2)
- REWRITE_IN_PLACE mode for multi-question file edits (v1.3)
- CHAT_ONLY mode safety: no file writes at all (v1.4)
- OVERWRITE_FULL mode for complete file replacement (v1.6)
- Multi-file batch operations (search and refactor) (v1.11)

v1.15 (2026-02-14): PERMANENT FIX - remove all hardcoded WDAGUtilityAccount paths
    - Root cause: Windows Sandbox (WDAG) clones the host desktop session, it does
      NOT create a WDAGUtilityAccount profile with initialised temp directories.
      The WDAG AppData Temp path has never existed in sandbox.
    - Fix 1: Temp file now written to target file's PARENT DIRECTORY (dynamic per job)
    - Fix 2: Chunk size reduced from 20000 to 8000 chars (safe under 32767 cmd limit)
    - Fix 3: Chunks written via Set-Content/Add-Content with CHUNK_OK verification
      tokens, NOT embedded in .NET method arguments
    - Fix 4: DIR_OK pre-flight ensures parent directory exists before any writes
    - Fix 5: DESKTOP target now dynamically queries $env:USERPROFILE from
      the sandbox instead of hardcoding a WDAGUtilityAccount Desktop path
    - Fix 6: All error paths prefixed with INFRASTRUCTURE_ERROR for strike system
    - Rule: NO path in this module should EVER reference WDAGUtilityAccount temp dirs.
      The sandbox environment mirrors the host - use dynamic resolution always.
v1.14 (2026-02-11): ATTEMPTED FIX (INCOMPLETE - changelog described fixes not applied to code)
    - v1.14 BUILD_ID was set but the actual code still contained v1.13 bugs
    - v1.15 delivers what v1.14 claimed to do
v1.13 (2026-02-10): WinError 206 fix (SUPERSEDED by v1.14) — temp-file write for large files
    - Added _write_content_to_sandbox(): auto-selects inline or temp-file method
    - Added INLINE_BASE64_CHAR_LIMIT constant (24000 chars)
    - All 6 write call sites now use shared helper
    - Files >~18KB safely written via temp file chunking
    - Fixes WinError 206 (command line too long) for large file writes
v1.11 (2026-01-28): Multi-file batch operations (Level 3 - Phase 5)
    - Added run_multi_file_search(): read-only search across multiple files
    - Added run_multi_file_refactor(): batch search/replace with verification
    - Added _multi_file_read_content(): helper for reading files via PowerShell
    - Added _multi_file_write_content(): helper for Base64-safe writes
    - Added MULTI_FILE_MAX_ERRORS constant (10) for consecutive error limit
    - Added MULTI_FILE_VERIFY_TIMEOUT constant (30s) per file
    - Progress callbacks supported for streaming updates
v1.10 (2026-01-28): Intelligent Q&A correction for REWRITE_IN_PLACE
    - Added _find_question_answer_pairs(): flexible detection of any Q&A format
    - Added _parse_corrections(): parse SpecGate's Q#: [STATUS] format
    - Added _apply_qa_corrections(): apply corrections to file in-place
    - REWRITE_IN_PLACE now tries intelligent correction first
    - Works with unnumbered, mixed format Q&A files
v1.9 (2026-01-27): Fix Answer marker detection (with or without colon)
    - Detects both "Answer" and "Answer:" patterns in _block_has_answer()
    - Detects both patterns in _insert_answers_under_questions()
    - Fixes duplicate "Answer:" sections being added to files
v1.8 (2026-01-27): Base64 encoding for PowerShell writes
    - Fixes escaping issues with embedded quotes (e.g., "works on my machine")
    - Uses Base64 encoding to safely transmit complex content
    - Completely avoids shell escaping problems
v1.7 (2026-01-27): Pattern 3 for standalone numbered lines
    - Added detection of "1)" or "2." format question headers
    - Fixes parsing for files with format: "1)\nQuestion\n..."
v1.6 (2026-01-27): OVERWRITE_FULL mode for complete file replacement
v1.5 (2026-01-25): REWRITE_IN_PLACE improvements for Q&A file tasks
v1.4.1 (2026-01-25): CHAT_ONLY safety fix - BULLETPROOF EDITION
v1.4 (2026-01-24): CHAT_ONLY safety fix - CRITICAL BUG FIX
v1.3 (2026-01-24): Added REWRITE_IN_PLACE support for multi-question file edits
v1.2 (2026-01-24): Added APPEND_IN_PLACE support with insertion_format
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.overwatcher.overwatcher import OverwatcherOutput, Decision
from app.overwatcher.sandbox_client import (
    SandboxClient,
    SandboxError,
    get_sandbox_client,
)

from .spec_resolution import ResolvedSpec, SpecMissingDeliverableError
from app.overwatcher._implementer_utils_2 import IMPLEMENTER_BUILD_ID, INLINE_BASE64_CHAR_LIMIT, _block_has_answer, _build_powershell_write_command_base64, _encode_for_powershell_base64, _escape_powershell_string, _find_question_block_starts, _generate_sandbox_path_candidates
from app.overwatcher._implementer_utils_3 import MULTI_FILE_MAX_ERRORS, _apply_qa_corrections, _find_question_answer_pairs, _insert_answers_under_questions, _is_absolute_windows_path, _is_specgate_correction_format, _multi_file_read_content, _multi_file_write_content
from app.overwatcher._implementer_utils_4 import AtomicTaskResult, EditTaskResult, MULTI_FILE_VERIFY_TIMEOUT, VerificationResult, _parse_answers_from_reply, _parse_corrections, run_implementer_task
from app.overwatcher._implementer_utils_5 import run_implementer_edit_task, run_verification
from app.overwatcher._implementer_utils_6 import ImplementerResult, _write_content_to_sandbox, run_multi_file_operation
from app.overwatcher._implementer_utils_7 import run_multi_file_search
from app.overwatcher._implementer_utils_8 import run_implementer
from app.overwatcher._implementer_utils_9 import MultiFileResult, run_multi_file_refactor

logger = logging.getLogger(__name__)

# =============================================================================
# v1.11 BUILD VERIFICATION - Proves correct code is running
# v1.11: Multi-file batch operations (Level 3 - Phase 5)
# =============================================================================
print(f"[IMPLEMENTER_LOADED] BUILD_ID={IMPLEMENTER_BUILD_ID}")
logger.info(f"[implementer] Module loaded: BUILD_ID={IMPLEMENTER_BUILD_ID}")

# =============================================================================
# v1.11: MULTI-FILE OPERATION CONSTANTS
# =============================================================================

# v1.13: WinError 206 fix — command line length limit
# Windows has a 32,767 character command line limit. Base64 encoding inflates
# content by ~33%, so a 20KB file becomes ~27KB Base64 which, with the
# PowerShell wrapper, exceeds the limit. When the Base64 string exceeds this
# threshold, we write it to a temp file first and have PowerShell read from it.


# =============================================================================
# v1.10: INTELLIGENT Q&A CORRECTION
# =============================================================================


# =============================================================================
# REWRITE_IN_PLACE HELPERS (v1.3, updated v1.7, v1.9)
# =============================================================================


# =============================================================================
# DATA CLASSES
# =============================================================================


# =============================================================================
# v1.11: MULTI-FILE OPERATIONS (Level 3 - Phase 5)
# =============================================================================


# =============================================================================
# v1.12: ATOMIC TASK INTERFACE (Architecture Execution Support)
# =============================================================================


# =============================================================================
# v1.13: TARGETED EDIT INTERFACE (Phase 0B — MODIFY without full-file rewrite)
# =============================================================================

__all__ = [
    "ImplementerResult",
    "VerificationResult",
    "MultiFileResult",
    "AtomicTaskResult",
    "EditTaskResult",
    "run_implementer",
    "run_implementer_task",
    "run_implementer_edit_task",
    "run_verification",
    "run_multi_file_search",
    "run_multi_file_refactor",
    "run_multi_file_operation",
    "MULTI_FILE_MAX_ERRORS",
    "MULTI_FILE_VERIFY_TIMEOUT",
]
