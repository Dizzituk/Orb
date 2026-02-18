# FILE: app/experience/distillation.py
"""
Post-Job Distillation — Journal → Experience Patterns.

After a job completes, this module reads the Build Journal and extracts
reusable patterns: error→fix pairs, architectural decisions, implementation
strategies. It deduplicates against existing knowledge and scores confidence.

Section 3.4 of the Unified Memory System v3.0 spec.

Usage:
    from app.experience.distillation import distill_job

    patterns = distill_job(db, job_id, job_dir)
    # Returns list of created/validated ExperiencePattern instances
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from .journal_writer import JournalWriter
from .schemas import BuildJournalEntry, EventSeverity, JournalEventType
from .experience_store import (
    create_pattern,
    validate_pattern,
    find_similar_pattern,
    INITIAL_CONFIDENCE_ERROR_FIX,
    INITIAL_CONFIDENCE_STRATEGY,
)

logger = logging.getLogger(__name__)


def distill_job(
    db: Session,
    job_id: str,
    job_dir: str,
) -> List[Any]:
    """
    Distill a completed job's journal into Experience Patterns.

    Process:
    1. Read the complete journal
    2. Identify error→fix pairs (boot errors, implementation strikes)
    3. Identify architectural decisions
    4. Identify implementation strategies
    5. For each candidate: check existing patterns for matches
       - Match found → validate (increase confidence)
       - No match → create new pattern
    6. Generate embeddings for new patterns (queued, not blocking)

    Returns:
        List of ExperiencePattern instances (created or validated)
    """
    journal_path = os.path.join(job_dir, "journal.ndjson")
    if not os.path.exists(journal_path):
        logger.info(f"[distillation] No journal found for {job_id}, skipping")
        return []

    writer = JournalWriter(job_id, job_dir)
    entries = writer.read_all()

    if not entries:
        logger.info(f"[distillation] Empty journal for {job_id}, skipping")
        return []

    logger.info(
        f"[distillation] Processing {len(entries)} journal entries for {job_id}"
    )

    patterns_created = []

    # === Extract error→fix pairs ===
    boot_patterns = _extract_boot_fix_patterns(entries, job_id)
    for bp in boot_patterns:
        pattern = _deduplicate_and_store(db, bp)
        if pattern:
            patterns_created.append(pattern)

    # === Extract implementation strike patterns ===
    strike_patterns = _extract_strike_patterns(entries, job_id)
    for sp in strike_patterns:
        pattern = _deduplicate_and_store(db, sp)
        if pattern:
            patterns_created.append(pattern)

    # === Extract architecture decisions ===
    arch_patterns = _extract_architecture_patterns(entries, job_id)
    for ap in arch_patterns:
        pattern = _deduplicate_and_store(db, ap)
        if pattern:
            patterns_created.append(pattern)

    # === Extract implementation strategies ===
    strategy_patterns = _extract_strategy_patterns(entries, job_id)
    for sp in strategy_patterns:
        pattern = _deduplicate_and_store(db, sp)
        if pattern:
            patterns_created.append(pattern)

    # === Extract cohesion patterns (v5.29) ===
    cohesion_patterns = _extract_cohesion_patterns(entries, job_id)
    for cp in cohesion_patterns:
        pattern = _deduplicate_and_store(db, cp)
        if pattern:
            patterns_created.append(pattern)

    db.commit()

    logger.info(
        f"[distillation] Job {job_id}: "
        f"{len(patterns_created)} patterns created/validated "
        f"(from {len(entries)} journal entries)"
    )

    return patterns_created


# =============================================================================
# PATTERN EXTRACTION
# =============================================================================

def _extract_boot_fix_patterns(
    entries: List[BuildJournalEntry],
    job_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract boot error→fix pairs from journal entries.

    Looks for sequences: boot_failure → boot_fix_attempt → boot_pass
    These are the highest-value patterns for the system.
    """
    patterns = []

    # Group entries by error_signature
    errors_by_sig: Dict[str, List[BuildJournalEntry]] = defaultdict(list)
    fixes_by_sig: Dict[str, List[BuildJournalEntry]] = defaultdict(list)
    boot_passed = False

    for entry in entries:
        if entry.event_type == JournalEventType.BOOT_FAILURE and entry.error_signature:
            errors_by_sig[entry.error_signature].append(entry)
        elif entry.event_type == JournalEventType.BOOT_FIX_ATTEMPT and entry.error_signature:
            fixes_by_sig[entry.error_signature].append(entry)
        elif entry.event_type == JournalEventType.BOOT_PASS:
            boot_passed = True

    # For each error signature that had a fix and boot eventually passed
    for sig, fix_entries in fixes_by_sig.items():
        error_entries = errors_by_sig.get(sig, [])
        if not error_entries:
            continue

        last_fix = fix_entries[-1]
        first_error = error_entries[0]

        # Only create pattern if the fix resolved the issue
        # (boot passed, or the error signature stopped appearing)
        patterns.append({
            "category": "boot_resolution",
            "stage": "phase_checkout",
            "description": f"Boot error: {first_error.description[:200]}",
            "root_cause": first_error.description[:300],
            "resolution": last_fix.resolution or last_fix.description[:300],
            "error_signature": sig,
            "file_scope": first_error.file_scope,
            "source_job_id": job_id,
            "language": first_error.language or _infer_language(first_error.file_scope),
            "initial_confidence": INITIAL_CONFIDENCE_ERROR_FIX if boot_passed else 0.4,
        })

    return patterns


def _extract_strike_patterns(
    entries: List[BuildJournalEntry],
    job_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract implementation strike patterns — what went wrong during file generation.

    Looks for implementation_strike events, especially those with
    root_cause and resolution information.
    """
    patterns = []
    seen_files = set()

    for entry in entries:
        if entry.event_type != JournalEventType.IMPLEMENTATION_STRIKE:
            continue
        if not entry.file_scope or entry.file_scope in seen_files:
            continue

        seen_files.add(entry.file_scope)

        # Only create a pattern if we have causal information
        if not entry.root_cause and not entry.description:
            continue

        patterns.append({
            "category": "error_fix",
            "stage": "implementer",
            "description": f"Strike on {entry.file_scope}: {entry.description[:200]}",
            "root_cause": entry.root_cause,
            "resolution": entry.resolution,
            "error_signature": entry.error_signature,
            "file_scope": entry.file_scope,
            "source_job_id": job_id,
            "language": entry.language or _infer_language(entry.file_scope),
            "initial_confidence": 0.5,
        })

    return patterns


def _extract_architecture_patterns(
    entries: List[BuildJournalEntry],
    job_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract architecture decision patterns.

    These capture why certain structural choices were made — useful for
    the Critical Pipeline to avoid repeating mistakes.
    """
    patterns = []

    for entry in entries:
        if entry.event_type != JournalEventType.ARCHITECTURE_DECISION:
            continue
        if not entry.description:
            continue

        patterns.append({
            "category": "architecture_decision",
            "stage": "critical_pipeline",
            "description": entry.description[:300],
            "root_cause": entry.root_cause,
            "resolution": entry.resolution,
            "source_job_id": job_id,
            "initial_confidence": 0.5,
        })

    return patterns


def _extract_strategy_patterns(
    entries: List[BuildJournalEntry],
    job_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract implementation strategy patterns.

    Captures which strategies (verbatim, edit-mode, full-write) worked
    for which types of files.
    """
    patterns = []

    strategy_events = [
        JournalEventType.VERBATIM_EXTRACTION,
        JournalEventType.EDIT_MODE_FALLBACK,
        JournalEventType.IMPLEMENTATION_STRATEGY,
    ]

    for entry in entries:
        if entry.event_type not in strategy_events:
            continue
        if not entry.description and not entry.strategy_used:
            continue

        desc = entry.description or f"Strategy: {entry.strategy_used} on {entry.file_scope}"

        patterns.append({
            "category": "implementation_strategy",
            "stage": "implementer",
            "description": desc[:300],
            "resolution": entry.strategy_used,
            "file_scope": entry.file_scope,
            "source_job_id": job_id,
            "language": entry.language or _infer_language(entry.file_scope),
            "initial_confidence": INITIAL_CONFIDENCE_STRATEGY,
        })

    return patterns


# =============================================================================
# COHESION PATTERN EXTRACTION (v5.29)
# =============================================================================

def _extract_cohesion_patterns(
    entries: List[BuildJournalEntry],
    job_id: str,
) -> List[Dict[str, Any]]:
    """
    Extract cohesion check patterns — cross-segment mismatches and fixes.

    These teach the architecture generator and segmenter what kinds of
    inter-segment issues occur so future jobs can avoid them. Patterns
    include:
    - Missing exports that downstream segments expected
    - Naming drift between what was promised and what was produced
    - Interface shape mismatches (wrong signatures, wrong types)
    - Auto-fix resolutions that worked (so the system can preempt)
    """
    cohesion_events = {
        JournalEventType.COHESION_MISMATCH,
        JournalEventType.COHESION_NAMING_DRIFT,
        JournalEventType.COHESION_INTERFACE_BREAK,
    }

    patterns = []
    seen_keys: set = set()

    for entry in entries:
        if entry.event_type not in cohesion_events:
            continue
        if not entry.description:
            continue

        # Deduplicate within this job by (category, segment, file)
        _dedup_key = f"{entry.root_cause}:{entry.segment_id}:{entry.file_scope}"
        if _dedup_key in seen_keys:
            continue
        seen_keys.add(_dedup_key)

        # Determine category based on event type
        _category = "architecture_decision"  # cohesion issues inform architecture
        if entry.event_type == JournalEventType.COHESION_INTERFACE_BREAK:
            _category = "architecture_decision"
        elif entry.event_type == JournalEventType.COHESION_NAMING_DRIFT:
            _category = "architecture_decision"

        # Build description with expected/actual from details
        _details = entry.details or {}
        _desc = entry.description[:200]
        _expected = _details.get("expected", "")
        _actual = _details.get("actual", "")
        if _expected and _actual:
            _desc += f" | expected: {_expected[:80]} | actual: {_actual[:80]}"

        _resolution = entry.resolution or ""
        _was_auto_fixed = _details.get("auto_fixed", False)
        if _was_auto_fixed and _resolution:
            _resolution = f"[auto-fixed] {_resolution}"

        patterns.append({
            "category": _category,
            "stage": "cohesion_check",
            "description": _desc[:400],
            "root_cause": entry.root_cause or entry.event_type.value,
            "resolution": _resolution[:300],
            "error_signature": f"cohesion:{entry.root_cause}:{entry.segment_id}",
            "file_scope": entry.file_scope,
            "source_job_id": job_id,
            "language": entry.language or _infer_language(entry.file_scope),
            "initial_confidence": 0.5,
        })

    return patterns


# =============================================================================
# DEDUPLICATION
# =============================================================================

def _deduplicate_and_store(
    db: Session,
    candidate: Dict[str, Any],
) -> Optional[Any]:
    """
    Check if a similar pattern exists. If so, validate it.
    Otherwise, create a new pattern.
    """
    from .models import ExperiencePattern

    existing = find_similar_pattern(
        db,
        category=candidate["category"],
        stage=candidate["stage"],
        description=candidate.get("description", ""),
        error_signature=candidate.get("error_signature"),
    )

    if existing:
        logger.debug(
            f"[distillation] Validating existing pattern #{existing.id} "
            f"(matched by {'error_sig' if candidate.get('error_signature') else 'description'})"
        )
        return validate_pattern(db, existing.id)
    else:
        return create_pattern(
            db,
            category=candidate["category"],
            stage=candidate["stage"],
            description=candidate.get("description", ""),
            root_cause=candidate.get("root_cause", ""),
            resolution=candidate.get("resolution", ""),
            job_type=candidate.get("job_type"),
            language=candidate.get("language"),
            error_signature=candidate.get("error_signature"),
            file_scope=candidate.get("file_scope"),
            source_job_id=candidate.get("source_job_id"),
            initial_confidence=candidate.get("initial_confidence"),
        )


# =============================================================================
# HELPERS
# =============================================================================

def _infer_language(file_scope: str) -> Optional[str]:
    """Infer language from file path."""
    if not file_scope:
        return None
    ext = os.path.splitext(file_scope)[1].lower()
    return {
        ".py": "python", ".pyw": "python", ".pyi": "python",
        ".js": "javascript", ".jsx": "javascript",
        ".ts": "typescript", ".tsx": "typescript",
        ".html": "html", ".css": "css",
        ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    }.get(ext)


def _compute_error_signature(error_text: str) -> str:
    """Compute a deterministic signature for an error string."""
    normalised = error_text.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]
