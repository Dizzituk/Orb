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

async def run_spec_gate_grounded(
    db: Session,
    job_id: str,
    user_intent: str,
    provider_id: str,
    model_id: str,
    project_id: int,
    constraints_hint: Optional[Dict] = None,
    spec_version: int = 1,
    user_answers: Optional[Dict[str, str]] = None,
) -> SpecGateResult:
    """
    v4.0: Direct spec builder - NO GATES.
    
    Flow:
    1. Parse Weaver spec
    2. Run scan if needed
    3. Build POT spec
    4. Return result
    
    Only asks questions if something CRITICAL is missing (e.g., no target path).
    """
    try:
        round_n = max(1, min(3, int(spec_version or 1)))
        
        logger.info("[spec_runner] v4.0 Starting: job=%s, round=%d", job_id, round_n)
        print(f"[spec_runner] v4.0 DIRECT PATH: No gates, no classification")
        
        # =================================================================
        # STEP 1: Get Weaver spec
        # =================================================================
        
        weaver_job_text = (constraints_hint or {}).get('weaver_job_description_text', '')
        combined_text = f"{user_intent or ''} {weaver_job_text}"
        
        intent = parse_weaver_intent(constraints_hint or {})
        
        # v4.4: Extract goal with placeholder filtering
        # Priority: 1) Intent goal, 2) weaver text, 3) user intent
        # But NEVER use placeholder text like "Job Description from Weaver"
        goal = ""
        
        # Try intent goal first
        intent_goal = intent.get("goal", "")
        if intent_goal and not _is_placeholder_goal(intent_goal):
            goal = intent_goal
            logger.info("[spec_runner] v4.4 Using goal from intent: %s", goal[:80])
        
        # Fallback to weaver text (but filter out placeholder headers)
        if not goal and weaver_job_text:
            # Try to extract real content from weaver text
            # Skip lines that are just headers/placeholders
            for line in weaver_job_text.split('\n'):
                line = line.strip()
                if line and not _is_placeholder_goal(line):
                    # Found a real line of content
                    goal = line[:200]
                    logger.info("[spec_runner] v4.4 Using goal from weaver text: %s", goal[:80])
                    break
        
        # Final fallback to user intent
        if not goal and user_intent:
            goal = user_intent[:200]
            logger.info("[spec_runner] v4.4 Using goal from user intent: %s", goal[:80])
        
        logger.info("[spec_runner] v4.0 Weaver goal: %s", goal[:100])
        
        # =================================================================
        # STEP 2: Detect if this needs a scan
        # =================================================================
        
        project_paths = _extract_project_paths(combined_text)
        
        multi_file_meta = _detect_multi_file_intent(
            combined_text=combined_text,
            constraints_hint=constraints_hint,
            project_paths=project_paths,
            vision_results=constraints_hint.get('vision_results') if constraints_hint else None,
        )
        
        # =================================================================
        # STEP 3: Run scan if multi-file operation
        # =================================================================
        
        multi_file_op = None
        spot_markdown = None
        
        if multi_file_meta and multi_file_meta.get("is_multi_file"):
            logger.info(
                "[spec_runner] v4.0 Multi-file detected: %s '%s' → '%s'",
                multi_file_meta.get("operation_type"),
                multi_file_meta.get("search_pattern"),
                multi_file_meta.get("replacement_pattern"),
            )
            print(f"[spec_runner] v4.0 SCANNING: {multi_file_meta.get('search_pattern')}")
            
            # Run scan
            multi_file_op = await _build_multi_file_operation(
                operation_type=multi_file_meta.get("operation_type", "search"),
                search_pattern=multi_file_meta.get("search_pattern", ""),
                replacement_pattern=multi_file_meta.get("replacement_pattern", ""),
                file_filter=multi_file_meta.get("file_filter"),
                sandbox_client=None,
                job_description=weaver_job_text or combined_text,
                provider_id=provider_id,
                model_id=model_id,
                explicit_roots=project_paths if project_paths else None,
                vision_context=constraints_hint.get("vision_context", "") if constraints_hint else "",
            )
            
            logger.info(
                "[spec_runner] v4.0 Scan complete: %d files, %d matches",
                multi_file_op.total_files,
                multi_file_op.total_occurrences,
            )
            print(f"[spec_runner] v4.0 FOUND: {multi_file_op.total_occurrences} matches in {multi_file_op.total_files} files")
            
            # =================================================================
            # CRITICAL CHECK: Do we have what we need?
            # =================================================================
            
            if multi_file_op.total_occurrences == 0 and not multi_file_op.error_message:
                # No matches found - this might be a problem
                logger.warning("[spec_runner] v4.0 NO MATCHES found for '%s'", multi_file_meta.get("search_pattern"))
                
                # Ask the user if no matches were found
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=[
                        f"No matches found for '{multi_file_meta.get('search_pattern')}' in {project_paths}. "
                        f"Is the search term correct? Is the path correct?"
                    ],
                    spec_version=round_n,
                    validation_status="needs_clarification",
                    notes="v4.0: No scan matches found",
                )
            
            if multi_file_op.error_message:
                # Scan error - this is a real problem
                return SpecGateResult(
                    ready_for_pipeline=False,
                    blocking_issues=[f"Scan error: {multi_file_op.error_message}"],
                    spec_version=round_n,
                    validation_status="blocked",
                    notes="v4.0: Scan failed",
                )
            
            # =================================================================
            # STEP 4: Build POT spec from evidence
            # =================================================================
            
            if _DIRECT_BUILDER_AVAILABLE and multi_file_op.raw_matches:
                # Use direct builder - no LLM, no classification
                spot_markdown = build_direct_spec(
                    search_term=multi_file_op.search_pattern,
                    replace_term=multi_file_op.replacement_pattern,
                    raw_matches=multi_file_op.raw_matches,
                    goal=goal,
                    total_files=multi_file_op.total_files,
                )
                logger.info("[spec_runner] v4.0 Direct spec built: %d chars", len(spot_markdown))
                print(f"[spec_runner] v4.0 POT SPEC READY: {len(spot_markdown)} chars")
            else:
                # Fallback: use classification markdown if available
                spot_markdown = multi_file_op.classification_markdown
                if not spot_markdown:
                    # Build minimal spec
                    spot_markdown = f"""# SPoT Spec — {multi_file_op.search_pattern} → {multi_file_op.replacement_pattern}

## Goal
{goal}

## Evidence
Found **{multi_file_op.total_occurrences} occurrences** in **{multi_file_op.total_files} files**

## Replace
- `{multi_file_op.search_pattern}` → `{multi_file_op.replacement_pattern}`

## Acceptance
- [ ] App boots
- [ ] Changes applied
- [ ] No errors
"""
        else:
            # =================================================================
            # Non-scan job: CREATE, MODIFY, etc.
            # v4.1: Use grounded CREATE spec if we have project paths
            # =================================================================
            
            logger.info("[spec_runner] v4.3 Non-scan job, checking for CREATE grounding")
            print(f"[spec_runner] v4.3 NON-SCAN JOB: project_paths={project_paths}")
            
            # Check if we have enough info
            if not goal and not weaver_job_text and not user_intent:
                return SpecGateResult(
                    ready_for_pipeline=False,
                    open_questions=["What would you like me to do?"],
                    spec_version=round_n,
                    validation_status="needs_clarification",
                    notes="v4.1: No goal specified",
                )
            
            # v4.3: Try grounded CREATE spec if we have project paths
            valid_paths = [p for p in project_paths if os.path.isdir(p)]
            print(f"[spec_runner] v4.3 VALID PATHS: {valid_paths}")
            print(f"[spec_runner] v4.3 CREATE_BUILDER_AVAILABLE: {_CREATE_BUILDER_AVAILABLE}")
            
            if _CREATE_BUILDER_AVAILABLE and valid_paths:
                logger.info("[spec_runner] v4.3 Using grounded CREATE builder for paths: %s", valid_paths)
                print(f"[spec_runner] v4.3 GROUNDED CREATE: scanning {len(valid_paths)} project(s)")
                
                try:
                    spot_markdown, create_evidence = await build_grounded_create_spec(
                        goal=goal,
                        what_to_do=weaver_job_text or user_intent,
                        project_paths=valid_paths,
                        sandbox_client=None,
                        provider_id=provider_id,
                        model_id=model_id,
                    )
                    print(f"[spec_runner] v4.3 CREATE SPEC READY: {len(spot_markdown)} chars")
                except Exception as create_err:
                    logger.warning("[spec_runner] v4.3 Grounded CREATE failed, falling back: %s", create_err)
                    print(f"[spec_runner] v4.3 CREATE FAILED: {create_err}")
                    spot_markdown = _build_simple_spec(
                        goal=goal,
                        what_to_do=weaver_job_text or user_intent,
                    )
            else:
                # No project paths or CREATE builder not available - use simple spec
                logger.info("[spec_runner] v4.3 No project paths, using simple spec")
                print(f"[spec_runner] v4.3 FALLBACK: No valid paths or builder unavailable")
                spot_markdown = _build_simple_spec(
                    goal=goal,
                    what_to_do=weaver_job_text or user_intent,
                )
        
        # =================================================================
        # STEP 4b: Segmentation check (Phase 3A — Needle-Count Classification)
        # =================================================================
        # v5.5 PHASE 3A: Replace crude file-count trigger with needle-count
        # classification. A lightweight LLM call estimates the cognitive load
        # of the spec, then segmentation is triggered if needles >= 5.
        # Falls back to the legacy needs_segmentation() if needle classifier
        # is unavailable.
        
        segmentation_manifest = None
        _needle_estimate = None
        try:
            from .segmentation import needs_segmentation, generate_segments
            
            # Extract file scope from the spec (all files mentioned as targets)
            _file_scope = _extract_file_scope_from_spec(
                spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
            )

            # v6.1 FIX 25: Also extract file paths from the raw user intent
            # and weaver text. The LLM-generated spec may rewrite/drop
            # explicit file paths the user specified. Merging from the raw
            # inputs ensures user-specified targets are never lost.
            _user_scope = _extract_file_scope_from_spec(
                combined_text, grounding_data=None, multi_file_op=None,
            )
            if _user_scope:
                _existing = {fp.replace("\\", "/").lower() for fp in _file_scope}
                _added = []
                for _ufp in _user_scope:
                    if _ufp.replace("\\", "/").lower() not in _existing:
                        _file_scope.append(_ufp)
                        _added.append(_ufp)
                if _added:
                    logger.info(
                        "[spec_runner] FIX 25: Merged %d file(s) from user intent "
                        "into file_scope: %s",
                        len(_added), _added[:5],
                    )

            logger.info("[spec_runner] v5.16 File scope: %d file(s) extracted", len(_file_scope))
            print(f"[spec_runner] v5.16 FILE_SCOPE: {len(_file_scope)} file(s): {_file_scope[:5]}")

            if _file_scope:
                # v5.6: Pre-segmentation size analysis
                # Reads source files, AST-parses them, identifies oversized
                # blocks, and expands scope with decomposition sub-files.
                _size_analysis = None
                _size_metadata = {}  # flows to smart segmenter
                try:
                    from .size_analyzer import analyze_file_sizes
                    _size_analysis = analyze_file_sizes(
                        file_scope=_file_scope,
                        spec_markdown=spot_markdown,
                    )
                    if _size_analysis.files_added:
                        logger.info(
                            "[spec_runner] v5.6 Size analysis expanded scope: "
                            "%d → %d files (+%d decomposition)",
                            len(_file_scope),
                            len(_size_analysis.enriched_file_scope),
                            len(_size_analysis.files_added),
                        )
                        print(
                            f"[spec_runner] v5.6 SIZE ANALYSIS: "
                            f"{len(_size_analysis.files_added)} file(s) added "
                            f"from decomposition: {', '.join(_size_analysis.files_added)}"
                        )
                        _file_scope = _size_analysis.enriched_file_scope
                    else:
                        logger.info(
                            "[spec_runner] v5.6 Size analysis: all %d files within caps",
                            len(_file_scope),
                        )
                    _size_metadata = {
                        est.rel_path: est.to_dict()
                        for est in _size_analysis.estimates.values()
                    } if _size_analysis.estimates else {}
                except (ImportError, Exception) as _sa_err:
                    logger.debug(
                        "[spec_runner] Size analyzer unavailable: %s — continuing without",
                        _sa_err,
                    )

                # v5.5: Try needle classifier first
                _should_segment = False
                _seg_reason = ""
                try:
                    from .needle_classifier import classify_needles
                    _needle_estimate = await classify_needles(
                        spec_markdown=spot_markdown,
                        file_scope=_file_scope,
                    )
                    logger.info(
                        "[spec_runner] v5.5 Needle estimate: %d (%s) — blast=%d concept=%d interface=%d",
                        _needle_estimate.needle_estimate,
                        _needle_estimate.difficulty_tier,
                        _needle_estimate.blast_radius_count,
                        _needle_estimate.concept_count,
                        _needle_estimate.interface_count,
                    )
                    print(
                        f"[spec_runner] v5.5 NEEDLES: {_needle_estimate.needle_estimate} "
                        f"({_needle_estimate.difficulty_tier}) — "
                        f"blast={_needle_estimate.blast_radius_count} "
                        f"concept={_needle_estimate.concept_count} "
                        f"interface={_needle_estimate.interface_count}"
                    )
                    _should_segment = _needle_estimate.needs_segmentation
                    _seg_reason = (
                        f"Needle count {_needle_estimate.needle_estimate} "
                        f"({_needle_estimate.difficulty_tier}): "
                        f"blast={_needle_estimate.blast_radius_count}, "
                        f"concept={_needle_estimate.concept_count}, "
                        f"interface={_needle_estimate.interface_count}"
                    )
                except (ImportError, Exception) as _nc_err:
                    logger.debug("[spec_runner] Needle classifier unavailable: %s — using legacy", _nc_err)
                    _should_segment, _seg_reason = needs_segmentation(_file_scope)

                # =============================================================
                # v6.1: DETERMINISTIC REFACTOR CHECK
                # Before LLM segmentation, check if this is a refactor job.
                # If the spec mentions refactoring an existing file that exists
                # on disk, use the deterministic pipeline instead of the LLM.
                # =============================================================
                _deterministic_manifest = None
                try:
                    from app.orchestrator.refactor_pipeline import (
                        is_refactor_job,
                        run_deterministic_refactor,
                    )
                    from app.pot_spec.grounded.segment_schemas import (
                        SegmentManifest,
                        SegmentSpec as _DetSegSpec,
                    )

                    _is_refactor, _refactor_reason = is_refactor_job(
                        spot_markdown, _file_scope,
                    )
                    # v6.1 FIX 6: Log detection result regardless of outcome
                    logger.info(
                        "[spec_runner] v6.1 Refactor detection: is_refactor=%s, reason=%s, file_scope=%d files",
                        _is_refactor, _refactor_reason, len(_file_scope),
                    )

                    if _is_refactor:
                        logger.info(
                            "[spec_runner] v6.1 DETERMINISTIC REFACTOR detected: %s",
                            _refactor_reason,
                        )
                        print(f"[spec_runner] v6.1 DETERMINISTIC REFACTOR: {_refactor_reason}")

                        # Find the source monolith and target package
                        # v6.1 FIX 5: Improved source/target identification.
                        # Strategy 1: Exact stem match (file.py → file/ package)
                        # Strategy 2: Source .py exists on disk + any __init__.py
                        #             in a subdir of the same parent (LLM may have
                        #             chosen a different package name).
                        # v6.1 FIX 13: Multi-file refactor support.
                        # Collect ALL source/target pairs, not just the first.
                        _refactor_pairs = []  # List of (source_file, target_pkg)
                        _matched_sources = set()  # Track already-matched to avoid dupes

                        # Normalise all paths once
                        _scope_norm = [fp.replace("\\", "/") for fp in _file_scope]

                        # v6.1 FIX 14: Strategy 0 — Direct source detection.
                        # Don't rely on the LLM's file_scope having package patterns.
                        # Any existing .py file on disk that's in file_scope is a
                        # refactor source. Auto-generate its target package path.
                        for _fp_norm in _scope_norm:
                            if not _fp_norm.endswith(".py") or "/" not in _fp_norm:
                                continue
                            if _fp_norm.endswith("__init__.py"):
                                continue
                            _abs = os.path.join("D:\\Orb", _fp_norm.replace("/", os.sep))
                            if not os.path.isfile(_abs):
                                continue
                            # Check file size — only refactor files that are actually large
                            try:
                                _fsize = os.path.getsize(_abs)
                                if _fsize < 10_000:  # < 10KB, not worth refactoring
                                    continue
                            except OSError:
                                continue
                            _stem = _fp_norm.rsplit("/", 1)[-1].replace(".py", "")
                            _parent = _fp_norm.rsplit("/", 1)[0]
                            _auto_pkg = f"{_parent}/{_stem}/"
                            _refactor_pairs.append((_fp_norm, _auto_pkg))
                            _matched_sources.add(_fp_norm)
                            logger.info(
                                "[spec_runner] v6.1 Strategy 0 (direct): %s -> %s (%d bytes)",
                                _fp_norm, _auto_pkg, _fsize,
                            )

                        if _refactor_pairs:
                            logger.info(
                                "[spec_runner] v6.1 Strategy 0 found %d pair(s) — skipping Strategy 1/2",
                                len(_refactor_pairs),
                            )

                        # Strategy 1 & 2: Legacy fallbacks — only if Strategy 0 found nothing.
                        # These check the LLM's file_scope for package patterns,
                        # which only works if the LLM happened to propose a package layout.
                        if not _refactor_pairs:
                            for _fp_norm in _scope_norm:
                                if _fp_norm.endswith(".py") and "/" in _fp_norm:
                                    _stem = _fp_norm.rsplit("/", 1)[-1].replace(".py", "")
                                    _parent = _fp_norm.rsplit("/", 1)[0]
                                    _pkg_prefix = f"{_parent}/{_stem}/"
                                    _has_pkg = any(
                                        f2.startswith(_pkg_prefix)
                                        for f2 in _scope_norm if f2 != _fp_norm
                                    )
                                    _on_disk = os.path.isfile(
                                        os.path.join("D:\\Orb", _fp_norm.replace("/", os.sep))
                                    )
                                    if _has_pkg and _on_disk:
                                        _refactor_pairs.append((_fp_norm, _pkg_prefix))
                                        _matched_sources.add(_fp_norm)
                                        logger.info(
                                            "[spec_runner] v6.1 Strategy 1 match: %s -> %s",
                                            _fp_norm, _pkg_prefix,
                                        )
                                    elif _on_disk and not _has_pkg:
                                        logger.info(
                                            "[spec_runner] v6.1 Strategy 1 miss: %s exists on disk but no package at %s",
                                            _fp_norm, _pkg_prefix,
                                        )

                        if not _refactor_pairs:
                            # Strategy 2: Source .py on disk + __init__.py in child dir
                            _unmatched = [
                                fp for fp in _scope_norm
                                if fp.endswith(".py") and "/" in fp and fp not in _matched_sources
                                and os.path.isfile(os.path.join("D:\\Orb", fp.replace("/", os.sep)))
                            ]
                            if _unmatched:
                                logger.info(
                                    "[spec_runner] v6.1 Strategy 2 checking %d unmatched file(s)",
                                    len(_unmatched),
                                )
                                for _fp_norm in _unmatched:
                                    _parent = _fp_norm.rsplit("/", 1)[0]
                                    for _f2 in _scope_norm:
                                        if _f2 == _fp_norm:
                                            continue
                                        if _f2.startswith(_parent + "/") and "__init__.py" in _f2:
                                            _pkg_dir = _f2.rsplit("__init__.py", 1)[0]
                                            _refactor_pairs.append((_fp_norm, _pkg_dir))
                                            _matched_sources.add(_fp_norm)
                                            logger.info(
                                                "[spec_runner] v6.1 Strategy 2 match: %s -> %s",
                                                _fp_norm, _pkg_dir,
                                            )
                                            break

                        if not _refactor_pairs:
                            logger.info(
                                "[spec_runner] v6.1 Refactor detected but couldn't identify any source/target pairs — falling back to LLM",
                            )

                        logger.info(
                            "[spec_runner] v6.1 Found %d refactor pair(s): %s",
                            len(_refactor_pairs),
                            [(s, t) for s, t in _refactor_pairs],
                        )

                        # Run deterministic pipeline for EACH pair and merge segments
                        if _refactor_pairs:
                            _det_job_dir = _get_job_dir_for_segmentation(job_id)
                            _all_segments = []
                            _all_sources = []

                            for _src, _tgt in _refactor_pairs:
                                logger.info(
                                    "[spec_runner] v6.1 Running deterministic refactor: %s -> %s",
                                    _src, _tgt,
                                )

                                # v6.1 FIX 20: Always use auto-layout for Strategy 0.
                                # The LLM's file_scope proposes file names (e.g.
                                # general_intents.py, helpers.py) that don't correspond
                                # to actual symbol groupings. The deterministic auto-layout
                                # in build_refactor_plan analyses symbols and creates
                                # properly-assigned files. Passing the LLM inventory causes
                                # empty files and misassigned symbols.
                                _inv_text = ""

                                _det_plan, _det_archs, _det_manifest_data = run_deterministic_refactor(
                                    source_file_path=_src,
                                    architecture_file_inventory=_inv_text,
                                    target_package=_tgt,
                                    job_dir=_det_job_dir,
                                    spec_id=f"sg-{uuid.uuid4().hex[:12]}",
                                )

                                # Convert segments and stamp each with its source
                                for _seg_data in _det_manifest_data["segments"]:
                                    _seg = _DetSegSpec.from_dict(_seg_data)
                                    _seg.deterministic_source = _src
                                    _all_segments.append(_seg)
                                _all_sources.append(_src)

                            _deterministic_manifest = SegmentManifest(
                                segments=_all_segments,
                                total_segments=len(_all_segments),
                                total_files=sum(len(s.file_scope) for s in _all_segments),
                                deterministic_sources=_all_sources,
                            )

                            logger.info(
                                "[spec_runner] v6.1 Deterministic manifest: %d segments, %d files, %d source(s)",
                                len(_all_segments),
                                sum(len(s.file_scope) for s in _all_segments),
                                len(_all_sources),
                            )
                            print(
                                f"[spec_runner] v6.1 DETERMINISTIC: "
                                f"{len(_all_segments)} segments, "
                                f"{sum(len(s.file_scope) for s in _all_segments)} files, "
                                f"{len(_all_sources)} source(s)"
                            )
                except ImportError:
                    logger.debug("[spec_runner] v6.1 refactor_pipeline not available")
                except Exception as _det_err:
                    logger.warning(
                        "[spec_runner] v6.1 Deterministic refactor error (falling back to LLM): %s",
                        _det_err,
                    )

                if _deterministic_manifest:
                    # Use deterministic manifest — skip LLM segmentation entirely
                    segmentation_manifest = _deterministic_manifest
                    _should_segment = True
                    _seg_reason = f"Deterministic refactor: {_refactor_reason}"
                    logger.info("[spec_runner] v6.1 Using deterministic manifest")

                if _should_segment:
                    logger.info("[spec_runner] v5.5 Segmentation triggered: %s", _seg_reason)
                    print(f"[spec_runner] v5.5 SEGMENTATION: {_seg_reason}")
                    
                    # Extract requirements and acceptance criteria from spec
                    _requirements = _extract_requirements_from_spec(spot_markdown)
                    _acceptance = _extract_acceptance_from_spec(spot_markdown)
                    
                    # v5.18 PHASE 3D: External consumer exclusion for file→package refactors
                    # When a monolith (e.g. segment_loop.py) is being refactored into a
                    # package (segment_loop/), external files that only need import-path
                    # updates (e.g. cohesion_check.py, phase_loop.py) should NOT be in
                    # segment scope. They waste segments on MODIFY operations that fail
                    # on large files. Post-recon handles their imports mechanically.
                    _deferred_consumers: List[str] = []
                    _refactor_packages: Dict[str, str] = {}  # package_dir -> monolith_path
                    _norm_scope = [f.replace("\\", "/") for f in _file_scope]

                    # Detect file→package pattern: both "foo.py" and "foo/_bar.py" in scope
                    for _fp in _norm_scope:
                        if _fp.endswith(".py") and "/" in _fp:
                            _stem = _fp.rsplit("/", 1)[-1].replace(".py", "")
                            _parent = _fp.rsplit("/", 1)[0]
                            _pkg_prefix = f"{_parent}/{_stem}/"
                            # Check if any other file lives inside this package
                            _has_pkg_files = any(
                                _other.startswith(_pkg_prefix) for _other in _norm_scope if _other != _fp
                            )
                            if _has_pkg_files:
                                _refactor_packages[_pkg_prefix] = _fp

                    if _refactor_packages:
                        _products = set()
                        for _pkg_prefix, _mono_path in _refactor_packages.items():
                            # The monolith itself is a product (gets quarantined)
                            _products.add(_mono_path)
                            # All files inside the package dir are products
                            for _fp2 in _norm_scope:
                                if _fp2.startswith(_pkg_prefix):
                                    _products.add(_fp2)

                        # Files NOT in any product set are external consumers
                        for _fp3 in _norm_scope:
                            if _fp3 not in _products:
                                # Check if file exists on disk (it's being modified, not created)
                                _abs_check = None
                                for _root in (_disc.get('roots', []) if '_disc' in dir() else ["D:\\Orb"]):
                                    _candidate = os.path.join(_root, _fp3.replace("/", os.sep))
                                    if os.path.isfile(_candidate):
                                        _abs_check = _candidate
                                        break
                                if _abs_check:
                                    _deferred_consumers.append(_fp3)

                        if _deferred_consumers:
                            # Remove consumers from file scope
                            _file_scope = [
                                f for f in _file_scope
                                if f.replace("\\", "/") not in set(_deferred_consumers)
                            ]
                            logger.info(
                                "[spec_runner] v5.18 External consumer exclusion: "
                                "deferred %d file(s) to post-recon: %s",
                                len(_deferred_consumers),
                                _deferred_consumers,
                            )
                            print(
                                f"[spec_runner] v5.18 CONSUMER EXCLUSION: "
                                f"{len(_deferred_consumers)} external file(s) deferred to post-recon: "
                                f"{', '.join(_deferred_consumers)}"
                            )

                    # v5.5 PHASE 3B: Try concept-aware grouping first
                    _concept_groups = None
                    _target_segs = _needle_estimate.recommended_segment_count if _needle_estimate else 0
                    if _target_segs >= 2 and not _deterministic_manifest:  # v6.1: skip for deterministic:
                        try:
                            from .smart_segmentation import generate_concept_segments
                            _concept_groups = await generate_concept_segments(
                                spec_markdown=spot_markdown,
                                file_scope=_file_scope,
                                target_segments=_target_segs,
                                requirements=_requirements,
                                size_metadata=_size_metadata,
                            )
                            if _concept_groups:
                                logger.info("[spec_runner] v5.5 Concept grouping: %d groups",
                                            len(_concept_groups))
                                print(f"[spec_runner] v5.5 CONCEPT GROUPS: {len(_concept_groups)}")
                            else:
                                logger.info("[spec_runner] v5.5 Concept grouping returned None — using legacy")
                        except (ImportError, Exception) as _cg_err:
                            logger.debug("[spec_runner] Smart segmentation unavailable: %s", _cg_err)
                    
                    if not _deterministic_manifest:  # v6.1: skip LLM segmenter for deterministic
                        segmentation_manifest = generate_segments(
                            file_scope=_file_scope,
                            requirements=_requirements,
                            acceptance_criteria=_acceptance,
                            parent_spec_id=f"sg-{uuid.uuid4().hex[:12]}",
                            parent_spec_hash=hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else None,
                            concept_groups=_concept_groups,
                        )
                    
                    if segmentation_manifest:
                        # v5.18: Attach deferred consumer files to manifest
                        if _deferred_consumers:
                            segmentation_manifest.deferred_consumer_files = _deferred_consumers
                            logger.info(
                                "[spec_runner] v5.18 Attached %d deferred consumer(s) to manifest",
                                len(_deferred_consumers),
                            )

                        # Write manifest and segment specs to job directory
                        _write_segmentation_output(job_id, segmentation_manifest)
                        logger.info(
                            "[spec_runner] v4.8 Segmentation complete: %s",
                            segmentation_manifest.summary(),
                        )
                        print(f"[spec_runner] v4.8 SEGMENTED: {segmentation_manifest.summary()}")

                        # v4.9 PHASE 2: Return early with "segmented" status.
                        # This prevents the spec from falling through to single-pass.
                        # The caller (spec_gate_stream.py) routes to the segment loop
                        # instead of the critical pipeline.
                        _seg_spec_id = f"sg-{uuid.uuid4().hex[:12]}"
                        _seg_spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else ""
                        _seg_grounding = {
                            "job_kind": "architecture",
                            "job_kind_confidence": 0.9,
                            "job_kind_reason": "Segmented job — Phase 2 segment loop",
                            "goal": goal,
                            "segmentation": {
                                "segmented": True,
                                "total_segments": segmentation_manifest.total_segments,
                                "segment_ids": [s.segment_id for s in segmentation_manifest.segments],
                                "manifest_path": os.path.join(
                                    _get_job_dir_for_segmentation(job_id),
                                    'segments', 'manifest.json',
                                ),
                            },
                        }
                        logger.info(
                            "[spec_runner] v4.9 PHASE 2: Returning segmented result for segment loop"
                        )
                        print("[spec_runner] v4.9 PHASE 2: Segmented — routing to segment loop")
                        return SpecGateResult(
                            ready_for_pipeline=True,
                            open_questions=[],
                            spot_markdown=spot_markdown,
                            db_persisted=False,
                            spec_id=_seg_spec_id,
                            spec_hash=_seg_spec_hash,
                            spec_version=round_n,
                            notes="v4.9: Job segmented — use segment loop for execution",
                            blocking_issues=[],
                            validation_status="segmented",
                            grounding_data=_seg_grounding,
                        )
                    else:
                        logger.info("[spec_runner] v4.8 Segmentation returned None — single pass")
                        print("[spec_runner] v4.8 Segmentation validation failed or not needed — single pass")
                else:
                    logger.info("[spec_runner] v4.8 No segmentation needed: %s", _seg_reason)
        except ImportError:
            logger.debug("[spec_runner] v4.8 Segmentation module not available")
        except Exception as seg_err:
            # Segmentation failure is NEVER fatal — fall back to single pass
            logger.warning("[spec_runner] v4.8 Segmentation failed (non-fatal): %s", seg_err)
            print(f"[spec_runner] v4.8 SEGMENTATION FAILED (non-fatal): {seg_err}")
            segmentation_manifest = None
        
        # =================================================================
        # STEP 5: Return result
        # =================================================================
        
        spec_id = f"sg-{uuid.uuid4().hex[:12]}"
        spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest()
        
        # v4.8: Proper job_kind classification for grounding_data
        # CREATE jobs that went through simple_create should be classified as
        # "architecture" so Critical Pipeline routes them correctly.
        # Previously hardcoded to "other" which caused 0.0 confidence and
        # downstream parsing failures.
        if multi_file_op:
            _job_kind = "refactor"
            _job_kind_confidence = 0.85
            _job_kind_reason = "Multi-file operation detected"
        elif spot_markdown and _CREATE_BUILDER_AVAILABLE and valid_paths:
            _job_kind = "architecture"
            _job_kind_confidence = 0.9
            _job_kind_reason = "Grounded CREATE spec with project paths"
        else:
            _job_kind = "other"
            _job_kind_confidence = 0.5
            _job_kind_reason = "Simple spec without grounded evidence"
        
        # =================================================================
        # v5.4 PHASE 1: Always-Manifest — wrap non-segmented specs too
        # =================================================================
        # If segmentation didn't trigger, build a single-segment manifest.
        # This ensures SpecGate ALWAYS outputs a manifest, regardless of
        # whether segmentation was needed. Downstream only handles one format.
        
        if segmentation_manifest is None:
            # Extract file scope, requirements, acceptance for wrapping
            _wrap_file_scope = _extract_file_scope_from_spec(
                spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
            )
            _wrap_requirements = _extract_requirements_from_spec(spot_markdown)
            _wrap_acceptance = _extract_acceptance_from_spec(spot_markdown)
            
            segmentation_manifest = _build_single_segment_manifest(
                spec_markdown=spot_markdown,
                spec_id=spec_id,
                spec_hash=spec_hash,
                goal=goal or "",
                file_scope=_wrap_file_scope,
                requirements=_wrap_requirements,
                acceptance_criteria=_wrap_acceptance,
                job_kind=_job_kind,
            )
            _write_segmentation_output(job_id, segmentation_manifest)
            print(f"[spec_runner] v5.4 ALWAYS-MANIFEST: single-segment manifest written for job {job_id}")
            logger.info("[spec_runner] v5.4 Single-segment manifest written for job %s", job_id)
        
        # Build manifest path (always available now)
        _manifest_path = os.path.join(
            _get_job_dir_for_segmentation(job_id),
            'segments', 'manifest.json',
        )
        
        grounding_data = {
            "job_kind": _job_kind,
            "job_kind_confidence": _job_kind_confidence,
            "job_kind_reason": _job_kind_reason,
            "multi_file": {
                "is_multi_file": multi_file_op.is_multi_file if multi_file_op else False,
                "operation_type": multi_file_op.operation_type if multi_file_op else None,
                "search_pattern": multi_file_op.search_pattern if multi_file_op else None,
                "replacement_pattern": multi_file_op.replacement_pattern if multi_file_op else None,
                "total_files": multi_file_op.total_files if multi_file_op else 0,
                "total_occurrences": multi_file_op.total_occurrences if multi_file_op else 0,
            } if multi_file_op else None,
            "goal": goal,
            "segmentation": {
                "segmented": segmentation_manifest.total_segments > 1,
                "total_segments": segmentation_manifest.total_segments,
                "segment_ids": [s.segment_id for s in segmentation_manifest.segments],
                "manifest_path": _manifest_path,
            },
            # v5.5 PHASE 3A: Needle estimate for downstream model selection
            "needle_estimate": _needle_estimate.to_dict() if _needle_estimate else None,
        }
        
        # =================================================================
        # v5.5: AC NAME RECONCILIATION — catch spec/source name mismatches
        # =================================================================
        if spot_markdown:
            _recon_file_scope = _extract_file_scope_from_spec(
                spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
            )
            _recon_warnings = _reconcile_ac_names_against_source(spot_markdown, _recon_file_scope)
            if _recon_warnings:
                recon_note = "\n\n## ⚠️ AC Name Reconciliation Warnings\n\n"
                recon_note += (
                    "The following identifiers in Acceptance Criteria may not match "
                    "the actual source code definitions. The Critical Pipeline should "
                    "use the SOURCE CODE names, not the AC names, if they differ.\n\n"
                )
                for w in _recon_warnings:
                    recon_note += f"- {w}\n"
                spot_markdown += recon_note

        # =================================================================
        # v4.7: DEDUP EVIDENCE_REQUESTs before counting
        # =================================================================
        if spot_markdown:
            spot_markdown = _dedup_evidence_requests(spot_markdown)

        # =================================================================
        # v5.0: STATUS SEMANTICS — check for unfulfilled EVIDENCE_REQUESTs
        # =================================================================
        # v4.0 of simple_create.py now fulfils ERs during spec generation,
        # so CRITICAL ERs should no longer appear in the final spec. Any
        # surviving ERs are either:
        #   a) Force-resolved (FORCED_RESOLUTION markers) — already handled
        #   b) Edge cases where fulfilment wasn't available (import failure)
        #
        # v5.0 CHANGE: NEVER return "pending_evidence" — it caused a deadlock
        # where SpecGate and Critical Pipeline each told the user to go to the
        # other. Instead:
        #   - No EVIDENCE_REQUESTs → validated
        #   - Surviving CRITICAL ERs → force-resolve them HERE as a safety net,
        #     then set validated_with_gaps (proceeds with honest acknowledgment)
        #   - Only non-CRITICAL ERs → validated (nice-to-have, not blocking)
        
        has_critical_er = False
        critical_er_count = 0
        if spot_markdown:
            # v4.6.1: Robust CRITICAL detection — multiple strategies to avoid
            # false negatives from YAML formatting variations.
            # Catches: severity: "CRITICAL", severity: CRITICAL,
            #          severity:CRITICAL, severity : 'critical', etc.
            
            _spot_lower = spot_markdown.lower()
            
            # Strategy 1: Line-level scan (handles any indentation/quoting)
            # Look for lines containing both "severity" and "critical"
            for line in _spot_lower.split('\n'):
                stripped = line.strip()
                if stripped.startswith('severity') and 'critical' in stripped:
                    critical_er_count += 1
            
            # Strategy 2: Regex fallback (catches inline/compact YAML)
            # Only used if Strategy 1 found nothing — avoids double-counting
            if critical_er_count == 0:
                er_blocks = re.findall(
                    r'severity\s*:\s*["\']?critical["\']?',
                    _spot_lower,
                )
                critical_er_count = len(er_blocks)
            
            if critical_er_count > 0:
                has_critical_er = True
                print(f"[spec_runner] v5.0 CRITICAL EVIDENCE_REQUEST survived fulfilment: {critical_er_count} block(s)")
                logger.warning(
                    "[spec_runner] v5.0 Spec has %d CRITICAL EVIDENCE_REQUEST(s) after fulfilment — force-resolving",
                    critical_er_count
                )
        
        if has_critical_er:
            # v5.0: Force-resolve surviving CRITICAL ERs instead of deadlocking
            # Import the stripping utility to convert ERs to FORCED_RESOLUTION markers
            try:
                from app.llm.pipeline.evidence_loop import (
                    parse_evidence_requests,
                    strip_forced_stop_requests,
                )
                remaining_ers = parse_evidence_requests(spot_markdown)
                if remaining_ers:
                    remaining_ids = {r.get("id", "UNKNOWN") for r in remaining_ers}
                    spot_markdown = strip_forced_stop_requests(spot_markdown, remaining_ids)
                    # Add a visible note to the spec about unfulfilled evidence
                    gap_note = (
                        "\n\n## ⚠️ Evidence Gaps\n\n"
                        "The following evidence requests could not be fulfilled during spec generation "
                        "and have been force-resolved. The Critical Pipeline's architecture stage "
                        "should gather this evidence directly.\n\n"
                    )
                    for r in remaining_ers:
                        gap_note += f"- **{r.get('id', '?')}**: {r.get('need', 'No description')}\n"
                    spot_markdown += gap_note
                    logger.info("[spec_runner] v5.0 Force-resolved %d surviving ER(s): %s",
                                len(remaining_ids), remaining_ids)
                    print(f"[spec_runner] v5.0 Force-resolved {len(remaining_ids)} surviving ER(s)")
            except ImportError as _imp_err:
                logger.warning("[spec_runner] v5.0 Cannot import evidence_loop for force-resolve: %s", _imp_err)
                # Can't strip, but still don't deadlock — proceed with gaps acknowledged
            
            final_status = "validated_with_gaps"
            final_notes = "v5.0: Spec has force-resolved CRITICAL EVIDENCE_REQUESTs. Proceeding with acknowledged gaps."
            print("[spec_runner] v5.0 STATUS: validated_with_gaps (CRITICAL ERs force-resolved)")
        else:
            final_status = "validated"
            final_notes = "v5.0: Direct path, evidence fulfilled"
            print("[spec_runner] v5.0 SUCCESS: POT spec ready for pipeline")
        
        logger.info("[spec_runner] v4.6 DONE: ready_for_pipeline=True, status=%s", final_status)
        
        return SpecGateResult(
            ready_for_pipeline=True,
            open_questions=[],
            spot_markdown=spot_markdown,
            db_persisted=False,
            spec_id=spec_id,
            spec_hash=spec_hash,
            spec_version=round_n,
            notes=final_notes,
            blocking_issues=[],
            validation_status=final_status,
            grounding_data=grounding_data,
        )
        
    except Exception as e:
        logger.exception("[spec_runner] v4.0 HARD STOP: %s", e)
        return SpecGateResult(
            ready_for_pipeline=False,
            hard_stopped=True,
            hard_stop_reason=str(e),
            spec_version=int(spec_version) if isinstance(spec_version, int) else None,
            validation_status="error",
        )
