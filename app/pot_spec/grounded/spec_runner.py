# FILE: app/pot_spec/grounded/spec_runner.py
"""
SpecGate v4.0 - Direct Spec Builder

NO GATES. NO CLASSIFICATION. NO RISK ASSESSMENT.

Flow:
1. Get Weaver spec (what to do)
2. Run scan (evidence of where)
3. Build POT spec (output for Implementer)

Only ask questions if something CRITICAL is missing.

v4.0 (2026-02-01): Stripped all gates - simple but powerful
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from app.pot_spec.grounded._spec_runner_utils import SPEC_RUNNER_BUILD_ID, _ARCH_INDEX_DIR, _ARCH_REPORT_DIR, _FALLBACK_ALL_PATHS, _FALLBACK_BACKEND_PATHS, _FALLBACK_FRONTEND_PATHS, _PRODUCT_SYNONYMS_RAW, _build_simple_spec
from app.pot_spec.grounded._spec_runner_utils import SCOPE_BACKEND, SCOPE_FRONTEND, _PRODUCT_SYNONYMS, _detect_search_replace_terms, _extract_requirements_from_spec, _generate_aliases_for_root, _get_job_dir_for_segmentation, _parse_product_synonyms
from app.pot_spec.grounded._spec_runner_utils import _build_single_segment_manifest, _dedup_evidence_requests, _extract_project_paths, _write_segmentation_output
from app.pot_spec.grounded._spec_runner_utils import _discover_project_roots, _extract_acceptance_from_spec, _extract_file_scope_from_spec
from app.pot_spec.grounded._spec_runner_utils import _reconcile_ac_names_against_source
from app.pot_spec.grounded._spec_runner_utils import run_spec_gate_grounded

logger = logging.getLogger(__name__)
print(f"[SPEC_RUNNER_LOADED] BUILD_ID={SPEC_RUNNER_BUILD_ID}")


# =============================================================================
# IMPORTS
# =============================================================================

from .spec_models import GroundedFact, FileTarget, GroundedPOTSpec
from .domain_detection import detect_domains
from .sandbox_discovery import extract_sandbox_hints
from .evidence_gathering import gather_filesystem_evidence, sandbox_read_file
from .multi_file_detection import _detect_multi_file_intent, _build_multi_file_operation
from .weaver_parser import parse_weaver_intent, _is_placeholder_goal

# Direct spec builder (no LLM, no classification)
try:
    from .simple_refactor import build_direct_spec, SIMPLE_REFACTOR_BUILD_ID
    _DIRECT_BUILDER_AVAILABLE = True
except ImportError:
    _DIRECT_BUILDER_AVAILABLE = False
    build_direct_spec = None

# CREATE spec builder (grounded feature specs)
try:
    from .simple_create import build_grounded_create_spec, SIMPLE_CREATE_BUILD_ID
    _CREATE_BUILDER_AVAILABLE = True
except ImportError:
    _CREATE_BUILDER_AVAILABLE = False
    build_grounded_create_spec = None

# Evidence collector
try:
    from ..evidence_collector import EvidenceBundle, load_evidence
    _EVIDENCE_AVAILABLE = True
except ImportError:
    _EVIDENCE_AVAILABLE = False
    EvidenceBundle = None
    load_evidence = None

# SpecGateResult type
try:
    from ..spec_gate_types import SpecGateResult
except ImportError:
    from dataclasses import dataclass, field
    @dataclass
    class SpecGateResult:
        ready_for_pipeline: bool = False
        open_questions: List[str] = field(default_factory=list)
        spot_markdown: Optional[str] = None
        db_persisted: bool = False
        spec_id: Optional[str] = None
        spec_hash: Optional[str] = None
        spec_version: Optional[int] = None
        hard_stopped: bool = False
        hard_stop_reason: Optional[str] = None
        notes: Optional[str] = None
        blocking_issues: List[str] = field(default_factory=list)
        validation_status: str = "pending"
        grounding_data: Optional[Dict] = None


__all__ = ["run_spec_gate_grounded"]


# =============================================================================
# PATH EXTRACTION - v4.5 DYNAMIC PROJECT DISCOVERY
# =============================================================================
#
# v4.5 (2026-02-04): DYNAMIC PROJECT DISCOVERY
# - Replaced hardcoded EXPLICIT_PROJECT_PATTERNS with architecture-driven discovery
# - Reads INDEX.json from .architecture/ to discover project roots
# - Classifies roots as frontend/backend from file zone metadata
# - Generates product name aliases from folder names + configurable synonyms
# - Falls back to codebase report JSON if INDEX.json unavailable
# - Hardcoded paths kept ONLY as last-resort fallback
#
# Key insight: "Astra" and "Orb" are the same product. Future jobs may be
# for completely different projects. System must discover, not assume.
#

# --- Architecture document locations (configurable via env) ---

# --- Product synonyms: names that refer to the same product ---
# Format: comma-separated pairs like "orb=astra,foo=bar"
# These are BIDIRECTIONAL: orb=astra means both 'orb' and 'astra' map to the same roots


# --- Scope indicators: UI/frontend vs backend ---
# Key insight: If user explicitly says "UI" or "frontend", DON'T include backend
#
# v4.6: TIGHTENED FRONTEND DETECTION
# Only set frontend=True when the user requests CHANGES to the frontend.
# Merely MENTIONING the frontend (e.g., "the desktop app will call it",
# "the frontend will handle sending") does NOT mean frontend scope.
# Removed: 'the app', 'desktop app', "app's" — too broad, triggers on
# consumer/client mentions without requesting frontend code changes.
#

# LEGACY FALLBACK: Only used if dynamic discovery fails completely


# =============================================================================
# SIMPLE SPEC BUILDER (for non-scan jobs)
# =============================================================================


# =============================================================================
# v4.7: ER DEDUPLICATION — collapse duplicate EVIDENCE_REQUEST blocks by id
# =============================================================================
#
# LLM outputs sometimes emit the same ER block twice (e.g., ER-001 appears
# in both scaffold and LLM analysis sections). Duplicate ERs confuse the
# Critical Pipeline and inflate the CRITICAL ER count.
#
# Strategy: Parse all EVIDENCE_REQUEST blocks from the spec markdown,
# keep the first occurrence of each id, drop duplicates, and reconstruct.
#


# =============================================================================
# SEGMENTATION HELPERS (v4.8 — Pipeline Segmentation Phase 1)
# =============================================================================


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
