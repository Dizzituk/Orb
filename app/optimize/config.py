# FILE: app/optimize/config.py
"""
Optimize Tab configuration.

Category definitions, risk thresholds, file size limits,
and paths to existing architecture data.

v1.0 (2026-03-10): Initial implementation per ASTRA-SPEC-OPT-001.
"""
from __future__ import annotations

import os

# ── Architecture data paths ──────────────────────────────────────
ARCH_DIR = os.path.join("D:", os.sep, "Orb", ".architecture")
IMPORT_GRAPH_PATH = os.path.join(ARCH_DIR, "IMPORT_GRAPH.json")
INDEX_PATH = os.path.join(ARCH_DIR, "INDEX.json")
SIGNATURES_DIR = ARCH_DIR
TECH_DEBT_PATH = os.path.join("D:", os.sep, "Orb", "TECH_DEBT.md")

# ── File size thresholds ─────────────────────────────────────────
FILE_SIZE_TARGET_KB = 20
FILE_SIZE_MAX_KB = 30
FILE_SIZE_OVERSIZED_BYTES = FILE_SIZE_MAX_KB * 1024  # 30KB

# ── Complexity thresholds ────────────────────────────────────────
HIGH_COMPLEXITY_LINES = 500
HIGH_COUPLING_THRESHOLD = 15  # dependents + dependencies

# ── Proposal scoring ────────────────────────────────────────────
MIN_PROPOSAL_CONFIDENCE = 0.5
MIN_IMPROVEMENT_PCT = 10  # Proposals predicting <10% improvement are low priority

# ── Risk defaults per category ───────────────────────────────────
CATEGORY_RISK = {
    "dead_code": "low",
    "seq_to_parallel": "medium",
    "redundant_call": "medium",
    "logic_simplification": "medium",
    "cache_introduction": "medium",
    "model_call_reduction": "medium",
    "file_split_merge": "low",
    "dependency_cleanup": "low",
}

# ── Dead code detection ─────────────────────────────────────────
# Files matching these patterns are excluded from dead code analysis
DEAD_CODE_IGNORE_PATTERNS = [
    "__pycache__",
    ".pyc",
    "__init__.py",
    "test_",
    "conftest.py",
    "migrations.py",
    "seed.py",
    "alembic",
]

# ── Profiling ────────────────────────────────────────────────────
PROFILE_TIMEOUT_SEC = 30  # Max time for a single chunk profile
MAX_CHUNKS_TO_PROFILE = 200  # Don't profile more than this

# ── Execution ────────────────────────────────────────────────────
MAX_CONCURRENT_PROPOSALS = 1  # Execute one at a time for safety
POST_EXECUTION_COOLDOWN_SEC = 5
