from __future__ import annotations
import json
import logging
import os
from app.orchestrator._segment_loop_utils import _build_sibling_interfaces
from app.orchestrator.segment_state import JobState
from app.pot_spec.grounded.segment_schemas import SegmentManifest, SegmentSpec, SegmentStatus
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def _save_execution_trace(seg_id: str, job_dir: str, arch_result: dict) -> None:
    """
    Persist the architecture execution trace to disk on failure.

    The architecture_executor returns an in-memory trace list with per-file
    success/failure details, but this was previously discarded — only the
    summary error string was saved to state.json.  This function writes the
    full trace to the segment's ledger directory so we can diagnose which
    specific file failed and why.

    v5.8: Closes the observability gap where partial failures (e.g. "4/5
    succeeded, 1 failed") gave no indication of which file broke.
    """
    trace = arch_result.get("trace", [])
    summary = arch_result.get("summary", {})
    if not trace and not summary:
        return

    try:
        trace_dir = os.path.join(job_dir, "segments", seg_id, "execution_trace")
        os.makedirs(trace_dir, exist_ok=True)

        trace_path = os.path.join(trace_dir, "trace.json")
        trace_data = {
            "segment_id": seg_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": arch_result.get("error"),
            "success": arch_result.get("success", False),
            "summary": summary,
            "artifacts_written": arch_result.get("artifacts_written", []),
            "trace_events": trace,
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2, default=str)

        logger.info(
            "[SEGMENT_LOOP] v5.8 Execution trace saved: %s (%d events)",
            trace_path, len(trace),
        )
    except Exception as e:
        logger.warning("[SEGMENT_LOOP] v5.8 Failed to save execution trace: %s", e)

def can_execute_segment(
    segment: SegmentSpec,
    state: JobState,
    require_complete: bool = False,
) -> bool:
    """
    Check if all dependencies of a segment are met.

    By default, APPROVED or COMPLETE both count as met (for architecture
    generation where we just need the spec, not the files on disk).

    When require_complete=True, only COMPLETE counts (for facade segments
    that need actual files from their dependencies).

    v5.26: Added require_complete parameter for facade deferral.
    """
    if not segment.dependencies:
        return True

    _allowed = {SegmentStatus.COMPLETE.value}
    if not require_complete:
        _allowed.add(SegmentStatus.APPROVED.value)

    for dep_id in segment.dependencies:
        dep_state = state.segments.get(dep_id)
        if dep_state is None:
            logger.warning(
                "[SEGMENT_LOOP] Segment %s depends on unknown segment %s",
                segment.segment_id, dep_id,
            )
            return False
        if dep_state.status not in _allowed:
            return False

    return True

def _is_facade_segment(segment: SegmentSpec, manifest: SegmentManifest) -> bool:
    """
    v5.26: Detect facade segments — segments that depend on ALL other segments.

    A facade is the last segment to build. It ties everything together by
    importing from all other segments. Its architecture should only be
    generated AFTER all dependencies are COMPLETE (files on disk), not just
    APPROVED, because it needs actual export data to produce correct imports.
    """
    if not segment.dependencies:
        return False
    other_ids = {
        s.segment_id for s in manifest.segments
        if s.segment_id != segment.segment_id
    }
    return other_ids.issubset(set(segment.dependencies))

def mark_dependents_blocked(
    state: JobState,
    failed_segment_id: str,
    manifest: SegmentManifest,
    job_dir_path: str,
) -> List[str]:
    """
    Mark all segments that depend (directly or transitively) on a
    failed segment as BLOCKED.

    Returns list of segment IDs that were blocked.
    """
    blocked_ids: List[str] = []
    # Build reverse dependency map: blocked set grows transitively
    blocked_set = {failed_segment_id}

    for seg in manifest.segments:
        if seg.segment_id in blocked_set:
            continue
        # Check if any dependency is in the blocked set
        for dep_id in seg.dependencies:
            if dep_id in blocked_set:
                seg_state = state.segments.get(seg.segment_id)
                if seg_state and seg_state.status in (SegmentStatus.PENDING.value, SegmentStatus.APPROVED.value):
                    update_segment_status(
                        state, seg.segment_id, SegmentStatus.BLOCKED, job_dir_path,
                        error=f"Blocked by failed segment {failed_segment_id}",
                    )
                    blocked_set.add(seg.segment_id)
                    blocked_ids.append(seg.segment_id)
                break

    if blocked_ids:
        logger.info(
            "[SEGMENT_LOOP] Blocked %d segment(s) due to %s failure: %s",
            len(blocked_ids), failed_segment_id, blocked_ids,
        )
    return blocked_ids

def unblock_recovered_segments(
    state: JobState,
    manifest: SegmentManifest,
    job_dir_path: str,
) -> List[str]:
    """
    Re-evaluate BLOCKED segments: if all their dependencies are now
    COMPLETE (or at least no longer FAILED/BLOCKED), unblock them.

    This handles the case where a failed segment is re-tried (e.g. via
    cohesion regen or retry loop) and eventually succeeds — its
    dependents should become runnable again.

    Returns list of segment IDs that were unblocked.
    """
    unblocked_ids: List[str] = []

    for seg in manifest.segments:
        seg_state = state.segments.get(seg.segment_id)
        if seg_state is None or seg_state.status != SegmentStatus.BLOCKED.value:
            continue

        # Check if ALL dependencies are now in a non-blocking state
        still_blocked = False
        for dep_id in seg.dependencies:
            dep_state = state.segments.get(dep_id)
            if dep_state and dep_state.status in (
                SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value,
            ):
                still_blocked = True
                break

        if not still_blocked:
            # Restore to APPROVED if architecture exists, else PENDING
            _seg_dir = os.path.join(job_dir_path, "segments", seg.segment_id, "arch")
            if os.path.isdir(_seg_dir) and any(f.endswith(".md") for f in os.listdir(_seg_dir)):
                restore_status = SegmentStatus.APPROVED
            else:
                restore_status = SegmentStatus.PENDING

            update_segment_status(
                state, seg.segment_id, restore_status, job_dir_path,
                error=None,
            )
            unblocked_ids.append(seg.segment_id)
            logger.info(
                "[SEGMENT_LOOP] v5.15 UNBLOCKED %s -> %s (blocker recovered)",
                seg.segment_id, restore_status.value,
            )

    return unblocked_ids

def build_evidence_bundle(
    segment: SegmentSpec,
    state: JobState,
    job_dir_path: str,
) -> Dict[str, Any]:
    """
    Assemble evidence from completed upstream segments.

    Returns a dict containing:
        - upstream_files: dict of {segment_id: [file_paths]} from completed deps
        - interface_contracts: what this segment consumes
        - parent_evidence_files: files from the segment's own evidence_files list
    """
    upstream_files: Dict[str, List[str]] = {}

    for dep_id in segment.dependencies:
        dep_state = state.segments.get(dep_id)
        if dep_state and dep_state.status == SegmentStatus.COMPLETE.value:
            upstream_files[dep_id] = dep_state.output_files

            # Record that this dep's evidence was provided to this segment
            if segment.segment_id not in dep_state.evidence_provided_to:
                dep_state.evidence_provided_to.append(segment.segment_id)

    return {
        "upstream_files": upstream_files,
        "consumes": segment.consumes.to_dict() if segment.consumes else None,
        "evidence_files": segment.evidence_files,
    }

def verify_contracts_fulfilled(
    segment_id: str,
    state: JobState,
    manifest: SegmentManifest,
) -> List[str]:
    """
    Lightweight check: did the completed segment actually create the files
    it promised in its 'exposes' contracts?

    Returns a list of warning messages (empty if all contracts fulfilled).
    This is advisory — warnings are logged but don't block execution.
    """
    warnings: List[str] = []

    seg_spec = manifest.get_segment(segment_id)
    if seg_spec is None or seg_spec.exposes is None or seg_spec.exposes.is_empty():
        return warnings

    seg_state = state.segments.get(segment_id)
    if seg_state is None:
        return warnings

    output_files_lower = {f.lower().replace("\\", "/") for f in seg_state.output_files}

    # Check if files in the segment's file_scope were actually created
    for scope_file in seg_spec.file_scope:
        normalised = scope_file.lower().replace("\\", "/")
        # Check if any output file ends with this relative path
        found = any(
            out.endswith(normalised) or normalised in out
            for out in output_files_lower
        )
        if not found:
            warnings.append(
                f"Segment {segment_id}: file_scope entry '{scope_file}' "
                f"not found in output files"
            )

    if warnings:
        for w in warnings:
            logger.warning("[SEGMENT_LOOP] CONTRACT WARNING: %s", w)
            print(f"[SEGMENT_LOOP] ⚠️ {w}")

    return warnings

def build_segment_context(
    segment: SegmentSpec,
    state: JobState,
    parent_spec: dict,
    job_dir_path: str,
    contract_set: Any = None,
    source_file_evidence: Optional[Dict[str, str]] = None,
    enrichment: Optional[Dict[str, Any]] = None,  # v5.17: Stage 4B enrichment data
) -> Dict[str, Any]:
    """
    Build the execution context for a segment.

    This context is passed to each pipeline stage so they know they're
    processing a segment, not a full job.

    Contains:
        - segment_spec: the segment's own spec (primary input)
        - parent_spec: full SPoT spec (for reference)
        - evidence: upstream files and interface contracts
        - file_scope: what files this segment owns
        - exposes: what this segment must create for downstream
        - consumes: what this segment needs from upstream
        - interface_contract: (v5.4 Phase 2A) formatted contract markdown for this segment

    Args:
        contract_set: Optional SupervisorContractSet from the Critical Supervisor.
                      If provided, the relevant contract is formatted and injected.
    """
    evidence = build_evidence_bundle(segment, state, job_dir_path)

    # v5.4 PHASE 2A: Format interface contract for this segment
    _contract_markdown = ""
    if contract_set is not None:
        try:
            _contract_markdown = contract_set.format_contract_for_segment(segment.segment_id)
        except Exception as _ce:
            logger.warning("[build_segment_context] Failed to format contract for %s: %s",
                           segment.segment_id, _ce)

    # v5.5 PHASE 3C: Extract grounding_data for needle-based model selection.
    # The selector reads _grounding_data.needle_estimate to choose the model tier.
    _grounding_data = None
    if isinstance(parent_spec, dict):
        _grounding_data = parent_spec.get("grounding_data")
        if _grounding_data is None:
            # Try nested — spec_json might wrap it
            _grounding_data = parent_spec.get("grounding")

    return {
        "segment_id": segment.segment_id,
        "segment_spec": segment.to_dict(),
        "parent_spec": parent_spec,
        "file_scope": segment.file_scope,
        "evidence": evidence,
        "exposes": segment.exposes.to_dict() if segment.exposes else None,
        "consumes": segment.consumes.to_dict() if segment.consumes else None,
        "requirements": segment.requirements,
        "acceptance_criteria": segment.acceptance_criteria,
        "dependencies": segment.dependencies,
        "interface_contract": _contract_markdown,
        "_grounding_data": _grounding_data,
        "source_file_evidence": source_file_evidence or {},
        "enrichment": enrichment,  # v5.17: Stage 4B enrichment bundle
        "sibling_interfaces": _build_sibling_interfaces(segment, state, job_dir_path),  # v2.5: Deterministic evidence
    }
