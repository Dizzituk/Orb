# FILE: app/pot_spec/grounded/rename_policy.py
# Purpose: Rename Policy Engine (v1.0)
# Called-by: app.pot_spec.grounded._multi_file_detection_utils_2, app.pot_spec.grounded._rename_policy_utils_2, app.pot_spec.grounded.multi_file_detection
# Depends-on: app.pot_spec.grounded._rename_policy_utils_2
# Last-renovated: 2026-06-11
"""
Rename Policy Engine (v1.0)

Implements the "features > data" policy for multi-file refactors.
Classifies matches into SAFE / UNSAFE / MIGRATION-REQUIRED based on
whether they break core invariants.

Core Invariants (must not break):
1. Backend boots cleanly
2. Desktop boots (electron)
3. Auth still works (session/token creation + validation)
4. Encryption still works (master key load + DB decrypt/encrypt)
5. Job pipeline runs end-to-end (Weaver→SpecGate→Critical→Implementer)
6. Sandbox safety rules still block forbidden paths
7. Tool routing still works (local tools / stream endpoints)

Not Core (allowed to break/wipe):
- stored memory DB contents
- old job artifacts
- backups, logs
- historical reports
- cached embeddings/indexes

Token Categories:
A) Branding/UI text → SAFE to rename
B) Code identifiers → SAFE if fully refactored (must compile)
C) Protocol/compatibility tokens → UNSAFE unless migration added

Version: v1.0 (2026-02-01)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from app.pot_spec.grounded._rename_policy_utils_2 import Invariant, MatchDecision, RenameDecision, RenamePlan, build_rename_plan, classify_match

logger = logging.getLogger(__name__)


# =============================================================================
# Core Invariants Definition
# =============================================================================


# =============================================================================
# Protocol Token Patterns (HIGH RISK - breaks invariants if renamed)
# =============================================================================

# These are exact patterns that are protocol/compatibility tokens
# Renaming them breaks auth, encryption, or persistence
PROTOCOL_TOKENS: Dict[str, Tuple[Invariant, str]] = {
    # Auth tokens
    r'orb_session_': (Invariant.AUTH_WORKS, "Session token prefix used in validation"),
    r'orb_api_': (Invariant.AUTH_WORKS, "API key prefix"),
    r'orb_': (Invariant.AUTH_WORKS, "Generic token prefix (if at start of token)"),
    
    # Encryption
    r'ORB_MASTER_KEY': (Invariant.ENCRYPTION_WORKS, "Master encryption key env var"),
    r'ORB_ENCRYPTION': (Invariant.ENCRYPTION_WORKS, "Encryption config env var"),
    
    # Database paths (persisted)
    r'orb_memory': (Invariant.BACKEND_BOOT, "Database filename - persisted on disk"),
    r'\.db': (Invariant.BACKEND_BOOT, "Database file extension context"),
}

# Env var patterns that are protocol tokens
PROTOCOL_ENV_VARS: Set[str] = {
    'ORB_MASTER_KEY',
    'ORB_SESSION_SECRET',
    'ORB_API_KEY',
    'ORB_ENCRYPTION_KEY',
    'ORB_DB_PATH',
    'ORB_JOB_ARTIFACT_ROOT',
}

# Files that contain protocol-critical code
INVARIANT_FILES: Dict[str, Invariant] = {
    'auth/middleware.py': Invariant.AUTH_WORKS,
    'auth/config.py': Invariant.AUTH_WORKS,
    'auth/session.py': Invariant.AUTH_WORKS,
    'encryption/': Invariant.ENCRYPTION_WORKS,
    'crypto': Invariant.ENCRYPTION_WORKS,
    'master_key': Invariant.ENCRYPTION_WORKS,
    'sandbox/manager.py': Invariant.SANDBOX_SAFETY,
    'sandbox/tools.py': Invariant.SANDBOX_SAFETY,
    'overwatcher/': Invariant.JOB_PIPELINE,
    'translation/': Invariant.TOOL_ROUTING,
    'stream_router': Invariant.TOOL_ROUTING,
}


# =============================================================================
# Safe Categories (can rename without breaking invariants)
# =============================================================================

# File patterns that are safe to rename (won't break invariants)
SAFE_FILE_PATTERNS: List[str] = [
    # Documentation
    r'\.md$',
    r'README',
    r'docs/',
    r'\.txt$',
    
    # Tests (won't break prod)
    r'tests/',
    r'test_',
    r'_test\.py$',
    
    # Static/UI
    r'static/',
    r'\.html$',
    r'\.css$',
    
    # Old artifacts (allowed to break)
    r'jobs/jobs/',  # Old job outputs
    r'backup',
    r'/logs/',
    r'\.log$',
    r'leak_scan',
    r'_backup',
]

# Content patterns that are safe to rename
SAFE_CONTENT_PATTERNS: List[str] = [
    # Branding/UI text
    r'You are Orb',
    r"Orb's engineering brain",
    r'Starting Orb',
    r'ZOMBIE ORB',
    r'Orb Desktop',
    
    # Comments/docs
    r'^#.*Orb',
    r'^//.*Orb',
    r'""".*Orb',
    r"'''.*Orb",
    
    # CSS classes (UI only)
    r'\.orb\s*\{',
    r'class.*orb',
    r'className.*orb',
]


# =============================================================================
# Rename Decision Engine
# =============================================================================


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "Invariant",
    "RenameDecision",
    "MatchDecision",
    "RenamePlan",
    "classify_match",
    "build_rename_plan",
    "PROTOCOL_TOKENS",
    "PROTOCOL_ENV_VARS",
    "SAFE_FILE_PATTERNS",
]
