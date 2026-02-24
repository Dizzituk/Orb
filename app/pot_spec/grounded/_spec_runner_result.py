# FILE: app/pot_spec/grounded/_spec_runner_result.py
"""
Spec runner Step 5: Result assembly.

Handles always-manifest wrapping, AC name reconciliation, ER dedup,
CRITICAL ER force-resolution, and final SpecGateResult construction.
Extracted from spec_runner.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from app.pot_spec.grounded._spec_runner_utils_11 import (
    _extract_requirements_from_spec,
    _get_job_dir_for_segmentation,
)
from app.pot_spec.grounded._spec_runner_utils_12 import (
    _build_single_segment_manifest,
    _dedup_evidence_requests,
    _write_segmentation_output,
)
from app.pot_spec.grounded._spec_runner_utils_13 import (
    _extract_acceptance_from_spec,
    _extract_file_scope_from_spec,
)
from app.pot_spec.grounded._spec_runner_utils_14 import _reconcile_ac_names_against_source

logger = logging.getLogger(__name__)


def build_spec_result(
    spot_markdown: Optional[str],
    multi_file_op: Any,
    valid_paths: List[str],
    goal: str,
    job_id: str,
    round_n: int,
    segmentation_manifest: Any,
    needle_estimate: Any,
    create_builder_available: bool,
    SpecGateResult: type,
) -> Any:
    """
    Assemble the final SpecGateResult.

    Handles:
    - Job kind classification
    - Always-manifest wrapping for non-segmented specs
    - AC name reconciliation
    - ER dedup and CRITICAL ER force-resolution
    - Spec caching
    """
    spec_id = f"sg-{uuid.uuid4().hex[:12]}"
    spec_hash = hashlib.sha256(spot_markdown.encode()).hexdigest() if spot_markdown else ""

    # Job kind classification
    _job_kind, _job_kind_confidence, _job_kind_reason = _classify_job_kind(
        multi_file_op, spot_markdown, create_builder_available, valid_paths,
    )

    # v5.4: Always-manifest — wrap non-segmented specs
    if segmentation_manifest is None:
        segmentation_manifest = _wrap_single_segment(
            spot_markdown, multi_file_op, spec_id, spec_hash, goal, _job_kind, job_id,
        )

    # Manifest path
    _manifest_path = os.path.join(
        _get_job_dir_for_segmentation(job_id), 'segments', 'manifest.json',
    )

    # v5.5: AC name reconciliation
    spot_markdown = _reconcile_ac_names(spot_markdown, multi_file_op)

    # v4.7: Dedup ERs
    if spot_markdown:
        spot_markdown = _dedup_evidence_requests(spot_markdown)

    # v5.0: Status semantics — check and force-resolve CRITICAL ERs
    final_status, final_notes, spot_markdown = _resolve_critical_ers(spot_markdown)

    # Build grounding data
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
            "segmented": segmentation_manifest.total_segments > 1 if segmentation_manifest else False,
            "total_segments": segmentation_manifest.total_segments if segmentation_manifest else 0,
            "segment_ids": [s.segment_id for s in segmentation_manifest.segments] if segmentation_manifest else [],
            "manifest_path": _manifest_path,
        },
        "needle_estimate": needle_estimate.to_dict() if needle_estimate else None,
    }

    # v2.2: Cache spec
    _cache_spec(goal, _job_kind, spec_id, grounding_data, spot_markdown, multi_file_op)

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


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _classify_job_kind(
    multi_file_op, spot_markdown, create_builder_available, valid_paths,
):
    """Classify job kind for grounding data."""
    if multi_file_op:
        return "refactor", 0.85, "Multi-file operation detected"
    elif spot_markdown and create_builder_available and valid_paths:
        return "architecture", 0.9, "Grounded CREATE spec with project paths"
    else:
        return "other", 0.5, "Simple spec without grounded evidence"


def _wrap_single_segment(
    spot_markdown, multi_file_op, spec_id, spec_hash, goal, job_kind, job_id,
):
    """Build single-segment manifest for non-segmented specs."""
    _file_scope = _extract_file_scope_from_spec(
        spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
    )
    _requirements = _extract_requirements_from_spec(spot_markdown)
    _acceptance = _extract_acceptance_from_spec(spot_markdown)

    manifest = _build_single_segment_manifest(
        spec_markdown=spot_markdown,
        spec_id=spec_id,
        spec_hash=spec_hash,
        goal=goal or "",
        file_scope=_file_scope,
        requirements=_requirements,
        acceptance_criteria=_acceptance,
        job_kind=job_kind,
    )
    _write_segmentation_output(job_id, manifest)
    print(f"[spec_runner] v5.4 ALWAYS-MANIFEST: single-segment manifest written for job {job_id}")
    logger.info("[spec_runner] v5.4 Single-segment manifest written for job %s", job_id)
    return manifest


def _reconcile_ac_names(spot_markdown, multi_file_op):
    """v5.5: AC name reconciliation."""
    if not spot_markdown:
        return spot_markdown
    _recon_file_scope = _extract_file_scope_from_spec(
        spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
    )
    _warnings = _reconcile_ac_names_against_source(spot_markdown, _recon_file_scope)
    if _warnings:
        note = "\n\n## ⚠️ AC Name Reconciliation Warnings\n\n"
        note += (
            "The following identifiers in Acceptance Criteria may not match "
            "the actual source code definitions. The Critical Pipeline should "
            "use the SOURCE CODE names, not the AC names, if they differ.\n\n"
        )
        for w in _warnings:
            note += f"- {w}\n"
        spot_markdown += note
    return spot_markdown


def _resolve_critical_ers(spot_markdown):
    """v5.0: Check for CRITICAL ERs and force-resolve them."""
    if not spot_markdown:
        return "validated", "v5.0: Direct path, no spec", spot_markdown

    critical_er_count = 0
    _spot_lower = spot_markdown.lower()

    # Strategy 1: Line-level scan
    for line in _spot_lower.split('\n'):
        stripped = line.strip()
        if stripped.startswith('severity') and 'critical' in stripped:
            critical_er_count += 1

    # Strategy 2: Regex fallback
    if critical_er_count == 0:
        er_blocks = re.findall(
            r'severity\s*:\s*["\']?critical["\']?', _spot_lower,
        )
        critical_er_count = len(er_blocks)

    if critical_er_count == 0:
        print("[spec_runner] v5.0 SUCCESS: POT spec ready for pipeline")
        return "validated", "v5.0: Direct path, evidence fulfilled", spot_markdown

    # Force-resolve surviving CRITICAL ERs
    logger.warning(
        "[spec_runner] v5.0 Spec has %d CRITICAL ER(s) — force-resolving",
        critical_er_count,
    )
    try:
        from app.llm.pipeline.evidence_loop import (
            parse_evidence_requests,
            strip_forced_stop_requests,
        )
        remaining_ers = parse_evidence_requests(spot_markdown)
        if remaining_ers:
            remaining_ids = {r.get("id", "UNKNOWN") for r in remaining_ers}
            spot_markdown = strip_forced_stop_requests(spot_markdown, remaining_ids)
            gap_note = (
                "\n\n## ⚠️ Evidence Gaps\n\n"
                "The following evidence requests could not be fulfilled during spec generation "
                "and have been force-resolved.\n\n"
            )
            for r in remaining_ers:
                gap_note += f"- **{r.get('id', '?')}**: {r.get('need', 'No description')}\n"
            spot_markdown += gap_note
            logger.info("[spec_runner] v5.0 Force-resolved %d surviving ER(s)", len(remaining_ids))
    except ImportError as err:
        logger.warning("[spec_runner] v5.0 Cannot import evidence_loop: %s", err)

    print("[spec_runner] v5.0 STATUS: validated_with_gaps")
    return (
        "validated_with_gaps",
        "v5.0: Spec has force-resolved CRITICAL ERs. Proceeding with acknowledged gaps.",
        spot_markdown,
    )


def _cache_spec(goal, job_kind, spec_id, grounding_data, spot_markdown, multi_file_op):
    """v2.2: Cache spec for reuse."""
    try:
        from app.pot_spec.spec_cache import store_spec as _store
        _cache_files = _extract_file_scope_from_spec(
            spot_markdown, grounding_data=None, multi_file_op=multi_file_op,
        ) if spot_markdown else []
        _store(
            goal=goal or "",
            in_scope_files=_cache_files,
            operation_type=job_kind,
            spec_id=spec_id,
            spec_json=json.dumps(grounding_data or {}),
        )
    except Exception as err:
        logger.debug("[spec_runner] v2.2 Spec cache store failed: %s", err)
