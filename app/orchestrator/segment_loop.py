# FILE: app/orchestrator/segment_loop.py
"""
Core orchestrator segment loop.

Reads a segment manifest, processes segments in dependency order through
the existing pipeline (Critical Pipeline → Critique → Overwatcher →
Implementer), threads evidence forward between segments, and tracks
state for crash recovery.

Phase 2 of Pipeline Segmentation.

Evidence collection is inlined here rather than in a separate module —
the functions are small, tightly coupled to loop state, and have no
external reuse case.

v1.0 (2026-02-08): Initial implementation
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

SEGMENT_LOOP_BUILD_ID = "2026-02-19-v5.34-cohesion-halt-gate"
print(f"[SEGMENT_LOOP_LOADED] BUILD_ID={SEGMENT_LOOP_BUILD_ID}")


def _find_latest_arch(seg_dir: str) -> Optional[str]:
    """
    Find the latest architecture version file in a segment's arch directory.

    Scans for arch_v{N}.md files and returns the path to the highest version.
    Used by both execution and cohesion checking to ensure consistent version
    resolution across the entire pipeline.

    v5.8: Replaces hardcoded v1/v2 checks and static v3/v2/v1 fallback lists.
    """
    arch_dir = os.path.join(seg_dir, "arch")
    if not os.path.isdir(arch_dir):
        return None

    max_version = 0
    max_path = None
    for fname in os.listdir(arch_dir):
        if fname.startswith("arch_v") and fname.endswith(".md"):
            try:
                v = int(fname.replace("arch_v", "").replace(".md", ""))
                if v > max_version:
                    max_version = v
                    max_path = os.path.join(arch_dir, fname)
            except ValueError:
                pass
    return max_path


def _clear_stale_arch_versions(seg_dir: str) -> int:
    """
    Remove stale autofix arch versions when a fresh regen produces arch_v1.md.

    When the Critical Pipeline regenerates an architecture (e.g. after cohesion
    regen feedback), it writes to arch_v1.md. Any existing v2, v3, etc. from
    previous cohesion autofixes are now stale and must be removed so that:
      1. The cohesion checker reads the fresh regen (not old autofix patches)
      2. The executor loads the correct version
      3. Version numbers don't drift upward across runs

    v5.8: Fixes the recurring import-logging cohesion loop where regen wrote
    a correct v1 but stale v2/v3 (without the fix) kept being loaded instead.

    Returns:
        Number of stale files removed.
    """
    arch_dir = os.path.join(seg_dir, "arch")
    if not os.path.isdir(arch_dir):
        return 0

    removed = 0
    for fname in os.listdir(arch_dir):
        if fname.startswith("arch_v") and fname.endswith(".md") and fname != "arch_v1.md":
            try:
                stale_path = os.path.join(arch_dir, fname)
                os.remove(stale_path)
                removed += 1
                logger.info("[SEGMENT_LOOP] v5.8 Removed stale arch: %s", stale_path)
            except OSError as e:
                logger.warning("[SEGMENT_LOOP] v5.8 Could not remove stale arch %s: %s", fname, e)
    return removed


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


# --- Internal imports ---
from app.pot_spec.grounded.segment_schemas import (
    SegmentManifest,
    SegmentSpec,
    SegmentStatus,
    InterfaceContract,
)
from app.orchestrator.segment_state import (
    JobState,
    SegmentState,
    load_or_init_state,
    save_state,
    get_job_dir,
)

# --- Pipeline stage imports (optional — graceful degradation) ---
try:
    from app.llm.critical_pipeline_stream import generate_critical_pipeline_stream
    _CRITICAL_PIPELINE_AVAILABLE = True
except ImportError:
    _CRITICAL_PIPELINE_AVAILABLE = False

try:
    from app.overwatcher.overwatcher import run_overwatcher, run_pot_spec_execution
    _OVERWATCHER_AVAILABLE = True
except ImportError:
    _OVERWATCHER_AVAILABLE = False

try:
    from app.overwatcher.implementer import run_implementer
    _IMPLEMENTER_AVAILABLE = True
except ImportError:
    _IMPLEMENTER_AVAILABLE = False

try:
    from app.overwatcher.architecture_executor import run_architecture_execution
    from app.overwatcher.spec_resolution import resolve_latest_spec, ResolvedSpec
    from app.llm.overwatcher_stream import create_overwatcher_llm_fn
    _ARCH_EXECUTOR_AVAILABLE = True
except ImportError as _ae:
    _ARCH_EXECUTOR_AVAILABLE = False
    logger.warning("[SEGMENT_LOOP] Architecture executor not available: %s", _ae)
    print(f"[SEGMENT_LOOP] [WARNING] Architecture executor import failed: {_ae}")

# v5.12: Interface Reconciliation (Option A — prevent naming drift)
try:
    from app.orchestrator.interface_reconciliation import (
        read_dependency_interfaces_from_sandbox,
        inject_reconciliation_into_architecture,
    )
    _RECONCILIATION_AVAILABLE = True
except ImportError:
    _RECONCILIATION_AVAILABLE = False
    logger.debug("[SEGMENT_LOOP] Interface reconciliation not available")


# Type alias for progress callback
ProgressCallback = Optional[Callable[[str], None]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================


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


def is_segment_blocked(segment: SegmentSpec, state: JobState) -> bool:
    """
    Check if a segment should be BLOCKED (dependency FAILED or BLOCKED).

    Distinct from "can't execute yet" (dependency PENDING/IN_PROGRESS).
    """
    for dep_id in segment.dependencies:
        dep_state = state.segments.get(dep_id)
        if dep_state is None:
            continue
        if dep_state.status in (SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value):
            return True
    return False


# =============================================================================
# STATE UPDATES
# =============================================================================


def update_segment_status(
    state: JobState,
    segment_id: str,
    new_status: SegmentStatus,
    job_dir_path: str,
    *,
    error: Optional[str] = None,
    output_files: Optional[List[str]] = None,
) -> None:
    """
    Update a segment's status and persist state.json immediately.

    Every status change is written to disk before continuing — this is
    the foundation of crash recovery.
    """
    seg = state.segments.get(segment_id)
    if seg is None:
        logger.error("[SEGMENT_LOOP] Cannot update unknown segment: %s", segment_id)
        return

    seg.status = new_status.value

    if new_status == SegmentStatus.IN_PROGRESS:
        seg.started_at = _now_iso()
    elif new_status == SegmentStatus.COMPLETE:
        seg.completed_at = _now_iso()
        if output_files is not None:
            seg.output_files = output_files
    elif new_status == SegmentStatus.FAILED:
        seg.completed_at = _now_iso()
        seg.error = error
    elif new_status == SegmentStatus.BLOCKED:
        seg.error = error or "Blocked by failed dependency"

    save_state(state, job_dir_path)

    logger.info(
        "[SEGMENT_LOOP] %s → %s%s",
        segment_id, new_status.value,
        f" (error: {error})" if error else "",
    )


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


# =============================================================================
# EVIDENCE COLLECTION & THREADING
# =============================================================================


def collect_segment_outputs(segment_id: str, job_dir_path: str) -> List[str]:
    """
    After implementation, collect what files were actually created/modified
    by this segment.

    Checks the segment's output directory for any files. Also checks the
    state for output_files recorded by the implementer.
    """
    output_dir = os.path.join(job_dir_path, "segments", segment_id, "output")
    output_files: List[str] = []

    if os.path.isdir(output_dir):
        for root, _dirs, files in os.walk(output_dir):
            for f in files:
                output_files.append(os.path.join(root, f))

    logger.info(
        "[SEGMENT_LOOP] Collected %d output file(s) for %s",
        len(output_files), segment_id,
    )
    return output_files


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


def _load_source_file_evidence(
    manifest: "SegmentManifest",
    project_roots: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    v2.2: Pre-load existing source files for refactor jobs.

    Scans ALL segments' file_scope entries across the manifest, finds files
    that already exist on disk (i.e. source files being refactored, not
    CREATE targets), reads their content, and returns it.

    This ensures every segment has access to the original source code it's
    extracting from — preventing the LLM from fabricating function signatures,
    constant values, and API shapes.

    Args:
        manifest: The full segment manifest
        project_roots: Project root directories to resolve relative paths.
                       Defaults to ["D:\\Orb", "D:\\orb-desktop"].

    Returns:
        Dict of {relative_path: file_content} for files that exist on disk.
        Content is capped at 250K chars per file.
    """
    if project_roots is None:
        project_roots = ["D:\\Orb", "D:\\orb-desktop"]

    MAX_SOURCE_CHARS = 250_000
    source_files: Dict[str, str] = {}
    seen_paths: set = set()

    for seg in manifest.segments:
        for rel_path in seg.file_scope:
            normalised = rel_path.replace("/", os.sep).replace("\\", os.sep).lower()
            if normalised in seen_paths:
                continue
            seen_paths.add(normalised)

            # Try to find on disk under each project root
            for root in project_roots:
                abs_path = os.path.join(root, rel_path.replace("/", os.sep).replace("\\", os.sep))
                if os.path.isfile(abs_path):
                    try:
                        with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                            content = fh.read(MAX_SOURCE_CHARS)
                        source_files[rel_path] = content
                        logger.info(
                            "[segment_loop] v2.2 Source file pre-loaded: %s (%d chars)",
                            rel_path, len(content),
                        )
                    except Exception as exc:
                        logger.warning(
                            "[segment_loop] v2.2 Failed to read source file %s: %s",
                            abs_path, exc,
                        )
                    break

    if source_files:
        print(
            f"[segment_loop] 📖 Pre-loaded {len(source_files)} source file(s) "
            f"for refactor evidence: {', '.join(source_files.keys())}"
        )

    return source_files


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


# =============================================================================
# SEGMENT CONTEXT BUILDER
# =============================================================================


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
    }


# =============================================================================
# CORE ORCHESTRATOR LOOP
# =============================================================================


async def run_segment_through_pipeline(
    segment: SegmentSpec,
    segment_context: Dict[str, Any],
    job_id: str,
    db: Any,
    project_id: int,
    on_progress: ProgressCallback = None,
    contract_set: Any = None,      # v2.0: Skeleton contract for pre-flight
    job_dir_path: str = "",         # v2.0: Job dir for rejection persistence
    manifest: Any = None,           # v2.0: Manifest for pre-flight context
    parent_spec: Any = None,        # v2.0: SPoT spec for rejection context
    quarantine_result: Any = None,  # v5.9: Job-level quarantine result for MODIFY->CREATE promotion
) -> Dict[str, Any]:
    """
    Run a single segment through: Critical Pipeline → Critique → Overwatcher → Implementer.

    v1.1: Overwatcher + Implementer wired via run_architecture_execution.

    Returns a dict with:
        - success: bool
        - output_files: list[str]
        - error: str | None
        - critique_warnings: list[str]

    This function calls the existing pipeline stages with segment context
    injected as optional parameters. Each stage checks for the presence
    of segment_context and scopes its work accordingly.
    """
    result = {
        "success": False,
        "output_files": [],
        "error": None,
        "critique_warnings": [],
    }

    seg_id = segment.segment_id
    _emit = on_progress or (lambda msg: None)

    # Use a segment-specific sub-job-id so architecture files don't
    # overwrite each other across segments sharing the same parent job.
    seg_job_id = f"{job_id}__{seg_id}"

    # =====================================================================
    # Step 0.5: v5.25 Load previous implementation failure feedback (if any)
    # When a previous attempt failed at the Implementer stage, the execution
    # trace contains the exact strike errors. Inject these into segment_context
    # so the Critical Pipeline can avoid producing architectures that cause
    # the same implementation failures.
    # =====================================================================
    try:
        _prev_trace_path = os.path.join(
            get_job_dir(job_id), "segments", seg_id, "execution_trace", "trace.json",
        )
        if os.path.isfile(_prev_trace_path):
            with open(_prev_trace_path, "r", encoding="utf-8") as _tf:
                _prev_trace = json.load(_tf)
            if not _prev_trace.get("success", True):
                _feedback_parts = []
                _feedback_parts.append(f"Overall error: {_prev_trace.get('error', 'Unknown')}")
                for _evt in _prev_trace.get("trace_events", []):
                    if _evt.get("stage", "") in ("FILE_TASK_STRIKE", "FILE_TASK_FAILED", "JOB_CHECK_FAIL", "SIGNATURE_CHECK_FAIL"):
                        _det = _evt.get("details", {})
                        _path = _det.get("path", "")
                        _err = _det.get("error", _det.get("last_error", ""))
                        if _err:
                            _feedback_parts.append(f"- [{_evt['stage']}] {_path}: {_err[:300]}")
                if len(_feedback_parts) > 1:  # More than just the overall error
                    _impl_feedback = "\n".join(_feedback_parts)
                    segment_context["implementation_feedback"] = _impl_feedback
                    _emit(f"  📊 Loaded previous implementation failure feedback ({len(_feedback_parts)-1} issue(s))")
                    logger.info(
                        "[SEGMENT_LOOP] v5.25 Implementation feedback loaded for %s: %d issue(s)",
                        seg_id, len(_feedback_parts) - 1,
                    )
    except Exception as _fb_err:
        logger.warning("[SEGMENT_LOOP] v5.25 Failed to load implementation feedback (non-fatal): %s", _fb_err)

    # =====================================================================
    # Step 1: Critical Pipeline (architecture generation + critique)
    # =====================================================================
    _emit(f"  📝 Running Critical Pipeline for {seg_id}...")

    if not _CRITICAL_PIPELINE_AVAILABLE:
        result["error"] = "Critical Pipeline not available"
        return result

    arch_content_parts: List[str] = []
    done_metadata: Dict[str, Any] = {}

    try:
        async for event in generate_critical_pipeline_stream(
            project_id=project_id,
            message=json.dumps(segment_context.get("segment_spec", {})),
            db=db,
            job_id=seg_job_id,
            segment_context=segment_context,
        ):
            if not isinstance(event, str):
                continue
            # Parse SSE events: each is "data: {json}\n\n"
            for line in event.split("\n"):
                if not line.startswith("data: "):
                    continue
                try:
                    payload = json.loads(line[6:])
                except (json.JSONDecodeError, ValueError):
                    continue
                evt_type = payload.get("type")
                if evt_type == "token":
                    arch_content_parts.append(payload.get("content", ""))
                elif evt_type == "done":
                    done_metadata = payload

        if not arch_content_parts:
            result["error"] = f"Critical Pipeline produced no output for {seg_id}"
            return result

        arch_text = "".join(arch_content_parts)
        critique_passed = done_metadata.get("critique_passed", False)
        arch_id = done_metadata.get("arch_id", "unknown")

        _emit(f"  ✅ Architecture generated for {seg_id} ({len(arch_text)} chars, arch_id={arch_id})")
        if not critique_passed:
            _emit(f"  ⚠️ Critique did not fully pass — proceeding with caution")

    except Exception as e:
        result["error"] = f"Critical Pipeline failed for {seg_id}: {e}"
        logger.exception("[SEGMENT_LOOP] Critical Pipeline error for %s", seg_id)
        return result

    # --- v5.18: Architecture Sanitiser (deterministic post-generation cleanup) ---
    # Catches known LLM hallucination patterns BEFORE architecture hits disk:
    #   1. Package self-naming (foo/foo.py alongside foo/__init__.py)
    #   2. Out-of-scope files not in this segment's file_scope
    #   3. Paths previously flagged as hallucinated by segmentation
    try:
        from app.orchestrator.architecture_sanitiser import sanitise_architecture
        _sanitiser_scope = segment_context.get("file_scope", segment.file_scope)
        arch_text, _san_result = sanitise_architecture(
            arch_text=arch_text,
            file_scope=_sanitiser_scope,
            segment_id=seg_id,
        )
        if _san_result.had_fixes:
            _emit(f"  🧹 Architecture sanitiser: {_san_result.fix_count} fix(es) applied")
            for _fix in _san_result.fixes_applied:
                _emit(f"    🔧 [{_fix['type']}] {_fix['description'][:120]}")
            logger.info(
                "[SEGMENT_LOOP] v5.18 Sanitiser applied %d fix(es) for %s",
                _san_result.fix_count, seg_id,
            )
            # Persist sanitiser result alongside architecture
            try:
                import json as _json_san
                _san_path = os.path.join(
                    get_job_dir(job_id), "segments", seg_id, "arch", "sanitiser_result.json",
                )
                os.makedirs(os.path.dirname(_san_path), exist_ok=True)
                with open(_san_path, "w", encoding="utf-8") as _sf:
                    _json_san.dump({
                        "segment_id": seg_id,
                        "original_length": _san_result.original_length,
                        "sanitised_length": _san_result.sanitised_length,
                        "fixes": _san_result.fixes_applied,
                    }, _sf, indent=2)
            except Exception:
                pass  # Non-fatal — logging is sufficient
        else:
            logger.debug("[SEGMENT_LOOP] v5.18 Sanitiser: no issues for %s", seg_id)
    except ImportError:
        logger.debug("[SEGMENT_LOOP] v5.18 Architecture sanitiser not available")
    except Exception as _san_err:
        logger.warning("[SEGMENT_LOOP] v5.18 Sanitiser error (non-fatal): %s", _san_err)
        _emit(f"  ⚠️ Architecture sanitiser error (non-fatal): {_san_err}")

    # --- Save architecture per-segment on disk ---
    seg_arch_dir = os.path.join(
        get_job_dir(job_id), "segments", seg_id, "arch",
    )
    os.makedirs(seg_arch_dir, exist_ok=True)

    # v5.8: Clear stale autofix versions before writing fresh regen.
    # Prevents the cohesion checker from reading old v2/v3 instead of
    # the new v1 that includes the fix.
    _seg_dir_for_clear = os.path.join(get_job_dir(job_id), "segments", seg_id)
    _stale_removed = _clear_stale_arch_versions(_seg_dir_for_clear)
    if _stale_removed:
        _emit(f"  🧹 Cleared {_stale_removed} stale arch version(s)")
        logger.info("[SEGMENT_LOOP] v5.8 Cleared %d stale arch version(s) for %s", _stale_removed, seg_id)

    seg_arch_path = os.path.join(seg_arch_dir, "arch_v1.md")
    try:
        with open(seg_arch_path, "w", encoding="utf-8") as f:
            f.write(arch_text)
        _emit(f"  💾 Architecture saved: segments/{seg_id}/arch/arch_v1.md")
    except Exception as e:
        logger.warning("[SEGMENT_LOOP] Failed to save segment arch: %s", e)

    # --- v3.0 / v3.1: Show File Inventory from architecture for transparency ---
    # v3.1 FIX #3: Only extract from the actual File Inventory section, not from
    # evidence tables or prose that happen to contain backtick-wrapped paths.
    try:
        import re as _re
        _file_lines = []
        # Find the File Inventory section and extract only from it
        _in_inventory = False
        _past_header_row = False
        for _line in arch_text.split("\n"):
            _stripped = _line.strip()
            # Detect section start
            if _re.match(r'#{1,4}\s*.*[Ff]ile\s*[Ii]nventory', _stripped):
                _in_inventory = True
                _past_header_row = False
                continue
            # Detect section end (next heading or horizontal rule after table)
            if _in_inventory and (_stripped.startswith('#') or _stripped == '---'):
                if _past_header_row:  # Only stop if we've seen table rows
                    _in_inventory = False
                    continue
            if not _in_inventory:
                continue
            # Skip non-table lines
            if not _stripped.startswith('|'):
                continue
            # Skip separator rows and header rows
            if _re.match(r'\|[-\s|]+\|', _stripped):
                _past_header_row = True
                continue
            if 'File' in _stripped and 'Purpose' in _stripped:
                continue
            # Skip *(none)* / _(none)_ rows
            _lower = _stripped.lower()
            if '*(none' in _lower or '_(none' in _lower or '*(n/a' in _lower or '_(n/a' in _lower:
                continue
            # Extract file path from backtick-wrapped cell
            _m = _re.search(r'\|\s*`([^`]+)`\s*\|\s*([^|]+)', _stripped)
            if _m:
                _fp = _m.group(1).strip()
                _desc = _m.group(2).strip()
                if _fp and _fp.lower() != 'file':
                    _op = 'CREATE' if 'new' in _desc.lower() or 'create' in _desc.lower() or 'package' in _desc.lower() else 'MODIFY'
                    _file_lines.append(f"    {_op}: `{_fp}` — {_desc[:80]}")
        if _file_lines:
            _emit(f"  📂 File Inventory ({len(_file_lines)} operations):")
            for _fl in _file_lines:
                _emit(_fl)
        else:
            _emit(f"  📂 File Inventory: (could not parse — check arch_v1.md)")
    except Exception:
        pass  # Non-fatal

    # =====================================================================
    # Step 2: Human Approval Gate (v3.0)
    # Architecture is generated and critique-approved. STOP here and
    # wait for explicit human approval before executing any writes.
    #
    # v5.8: Cohesion regen bypass — if this segment was previously approved
    # and is only being re-run because cohesion found a fixable issue, skip
    # the approval gate. The regen is a targeted patch, not new work.
    # =====================================================================
    auto_execute = os.getenv("ASTRA_SEGMENT_AUTO_EXECUTE", "0").strip()
    _is_cohesion_regen = bool(segment_context and segment_context.get("cohesion_feedback"))
    # v5.26: Facade segments bypass approval when triggered from implement_only
    _is_facade_auto = bool(segment_context and segment_context.get("_facade_auto_execute"))

    if auto_execute != "1" and not _is_cohesion_regen and not _is_facade_auto:
        _emit(f"  ⏸️ AWAITING APPROVAL: Architecture ready for {seg_id}")
        _emit(f"  📄 Review: jobs/{os.path.basename(get_job_dir(job_id))}/segments/{seg_id}/arch/arch_v1.md")
        _emit(f"  💡 To implement: say 'Astra, command: implement segments'")
        result["success"] = True
        result["awaiting_approval"] = True
        result["architecture_path"] = seg_arch_path
        return result

    if _is_facade_auto:
        _emit(f"  🏗️ Facade auto-execute — bypassing approval gate (implement_only mode)")
        logger.info("[SEGMENT_LOOP] v5.26 Facade approval bypass for %s", seg_id)

    if _is_cohesion_regen:
        _emit(f"  🧩 Cohesion regen — bypassing approval gate (was previously approved)")
        logger.info("[SEGMENT_LOOP] v5.8 Cohesion regen bypass for %s", seg_id)

    # =====================================================================
    # Step 3: Overwatcher Pre-Flight + Architecture Executor
    # Only reached if ASTRA_SEGMENT_AUTO_EXECUTE=1, explicit approval,
    # or cohesion regen bypass (v5.8)
    # =====================================================================
    _emit(f"  🔧 Running Overwatcher for {seg_id}...")

    if not _ARCH_EXECUTOR_AVAILABLE:
        _emit(f"  ⚠️ Architecture executor not available — architecture generated only")
        result["success"] = True
        return result

    # -----------------------------------------------------------------
    # Step 3a: Overwatcher Coherence Pre-Flight (deterministic)
    # Verifies architecture against skeleton contract BEFORE implementation.
    # If this fails, route back to Critical Pipeline for this segment only.
    # -----------------------------------------------------------------
    try:
        from app.overwatcher.preflight import (
            run_segment_preflight,
            save_rejection,
        )
        _seg_contract = segment_context.get("interface_contract", "")
        _skeleton_json = None
        if contract_set:
            _skeleton_json = contract_set.to_json()

        _manifest_dict = None
        if manifest and hasattr(manifest, 'to_dict'):
            _manifest_dict = manifest.to_dict()

        _spec_md = ""
        if isinstance(parent_spec, str):
            _spec_md = parent_spec
        elif parent_spec:
            try:
                _spec_md = json.dumps(parent_spec)
            except Exception:
                pass

        _preflight_rejection = run_segment_preflight(
            segment_id=seg_id,
            architecture_content=arch_text,
            skeleton_json=_skeleton_json,
            manifest_dict=_manifest_dict,
            job_id=job_id,
            architecture_path=seg_arch_path,
            skeleton_contract_markdown=_seg_contract,
            spec_markdown=_spec_md,
            attempt_number=segment_context.get("_attempt_number", 1),
        )

        if _preflight_rejection:
            _emit(f"  ❌ PRE-FLIGHT FAILED for {seg_id}: {_preflight_rejection.summary}")
            for _iss in _preflight_rejection.issues:
                _emit(f"    🚫 [{_iss.get('category', '?')}] {_iss.get('description', '?')}")
            _emit(f"  🔄 Route: back to Critical Pipeline (segment only)")

            # Save rejection for Experience Database
            try:
                save_rejection(_preflight_rejection, job_dir_path)
                _emit(f"  💾 Rejection saved: {_preflight_rejection.rejection_id}")
            except Exception as _sav_err:
                logger.warning("[execute_segment] Failed to save rejection: %s", _sav_err)

            result["success"] = False
            result["preflight_failed"] = True
            result["rejection"] = _preflight_rejection.to_dict()
            return result
        else:
            _emit(f"  ✅ Pre-flight PASSED for {seg_id}")

    except ImportError:
        logger.debug("[execute_segment] Preflight module not available — skipping")
    except Exception as _pf_err:
        logger.warning("[execute_segment] Pre-flight check error (non-fatal): %s", _pf_err)
        _emit(f"  ⚠️ Pre-flight check error (non-fatal): {_pf_err}")

    # -----------------------------------------------------------------
    # Step 3b: Overwatcher Architecture Execution
    # Pre-flight passed — proceed to implementation.
    # -----------------------------------------------------------------
    try:
        # Resolve the spec (the parent SPoT spec)
        spec = resolve_latest_spec(project_id, db)
        if spec is None:
            _emit(f"  ⚠️ No spec found for project {project_id} — skipping Overwatcher")
            result["success"] = True
            return result

        # Create LLM function for Overwatcher
        llm_call_fn = create_overwatcher_llm_fn()

        # Run architecture execution for this segment
        _seg_contract = segment_context.get("interface_contract", "")

        # v5.7: Promote quarantined MODIFY->CREATE in architecture text
        # When quarantine renames a file, the Implementer can't MODIFY it.
        # Rewrite the File Inventory to list it as New Files instead.
        if quarantine_result and quarantine_result.has_quarantined:
            try:
                from app.orchestrator.package_quarantine import promote_quarantined_in_architecture
                _orig_len = len(arch_text)
                arch_text = promote_quarantined_in_architecture(
                    arch_text, quarantine_result.quarantined_rel_paths,
                )
                if len(arch_text) != _orig_len:
                    _emit(f"  [quarantine] Promoted quarantined file(s) from MODIFY->CREATE")
            except Exception as _promo_err:
                logger.warning("[SEGMENT_LOOP] v5.7 Quarantine promotion failed (non-fatal): %s", _promo_err)

        # v5.12: Interface Reconciliation (Option A)
        # Read actual interfaces from completed dependency segments
        # and inject into architecture so Implementer uses correct names
        _recon_arch_text = arch_text
        if _RECONCILIATION_AVAILABLE and segment.dependencies:
            try:
                # Need access to state — get it from the caller's scope via segment_context
                # The state isn't directly available here, but we can get completed segments
                # from the evidence bundle
                from app.orchestrator.segment_state import load_or_init_state
                _job_dir = get_job_dir(job_id.split('__')[0])  # Strip segment suffix from job_id
                _recon_state = load_or_init_state(job_id.split('__')[0], manifest) if manifest else None
                if _recon_state:
                    _recon_block = read_dependency_interfaces_from_sandbox(
                        segment=segment,
                        completed_segments=_recon_state.segments,
                        manifest=manifest,
                    )
                    if _recon_block:
                        _recon_arch_text = inject_reconciliation_into_architecture(
                            arch_text, _recon_block,
                        )
                        _emit(f"  \U0001f9e9 Interface reconciliation: injected real interfaces from {len(segment.dependencies)} dependency segment(s)")
                        logger.info(
                            "[SEGMENT_LOOP] v5.12 Reconciliation injected for %s (%d chars added)",
                            seg_id, len(_recon_block),
                        )
            except Exception as _recon_err:
                logger.warning("[SEGMENT_LOOP] v5.12 Reconciliation failed (non-fatal): %s", _recon_err)
                _emit(f"  \u26a0\ufe0f Interface reconciliation failed (non-fatal): {_recon_err}")

        # v5.26: Extraction Binding — inject enrichment source extractions
        # so the Implementer gets the exact function bodies to transplant.
        try:
            from app.orchestrator.extraction_binding import (
                load_segment_enrichment,
                build_extraction_block,
                build_facade_export_map,
                inject_extraction_into_architecture,
            )
            _parent_job_id = job_id.split('__')[0]
            _eb_job_dir = get_job_dir(_parent_job_id)
            _eb_enrichment = load_segment_enrichment(_eb_job_dir, seg_id)
            if _eb_enrichment:
                _is_facade = _is_facade_segment(segment, manifest) if manifest else False
                if _is_facade and manifest:
                    _eb_block = build_facade_export_map(
                        _eb_job_dir, manifest.segments, seg_id,
                    )
                else:
                    _eb_block = build_extraction_block(_eb_enrichment, seg_id)
                if _eb_block:
                    _recon_arch_text = inject_extraction_into_architecture(
                        _recon_arch_text, _eb_block,
                    )
                    _emit(f"  🧬 Extraction binding: injected source code for {seg_id} ({len(_eb_block)} chars)")
                    logger.info(
                        "[SEGMENT_LOOP] v5.26 Extraction binding for %s (%d chars)",
                        seg_id, len(_eb_block),
                    )
        except Exception as _eb_err:
            logger.warning("[SEGMENT_LOOP] v5.26 Extraction binding failed (non-fatal): %s", _eb_err)
            _emit(f"  ⚠️ Extraction binding failed (non-fatal): {_eb_err}")

        # v4.0: Skip boot check — segments are intermediate builds.
        # Boot check runs once at Phase Checkout after ALL segments complete.
        # v5.32: Pass manifest files for import validation
        _manifest_all_files2 = set()
        if manifest:
            for _ms2 in manifest.segments:
                for _mf2 in _ms2.file_scope:
                    _manifest_all_files2.add(_mf2.replace("\\", "/"))
        arch_result = await run_architecture_execution(
            spec=spec,
            architecture_content=_recon_arch_text,
            architecture_path=seg_arch_path,
            job_id=seg_job_id,
            llm_call_fn=llm_call_fn,
            artifact_root=os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs"),
            interface_contract=_seg_contract,
            skip_boot_check=True,
            manifest_all_files=_manifest_all_files2 if _manifest_all_files2 else None,
        )

        if arch_result.get("success", False):
            result["success"] = True
            result["output_files"] = arch_result.get("artifacts_written", [])
            result["critique_warnings"] = [
                e.get("status", "")
                for e in arch_result.get("trace", [])
                if e.get("stage", "").startswith("WARN")
            ]
            _emit(
                f"  ✅ Overwatcher + Implementer completed for {seg_id} "
                f"({len(result['output_files'])} artifact(s) written)"
            )
            # v3.0: List individual output files for transparency
            for _of in result['output_files']:
                _emit(f"    ✅ {_of}")
        else:
            error_msg = arch_result.get("error", "Unknown error")
            result["error"] = f"Architecture execution failed for {seg_id}: {error_msg}"
            _emit(f"  ❌ Architecture execution failed for {seg_id}: {error_msg}")

            # v5.8: Persist execution trace for failure diagnosis
            _save_execution_trace(seg_id, get_job_dir(job_id), arch_result)
            _n_trace = len(arch_result.get("trace", []))
            if _n_trace:
                _emit(f"  💾 Execution trace saved ({_n_trace} events) — check segments/{seg_id}/execution_trace/trace.json")

    except Exception as e:
        result["error"] = f"Overwatcher failed for {seg_id}: {e}"
        logger.exception("[SEGMENT_LOOP] Overwatcher error for %s", seg_id)

    return result


async def run_segmented_job(
    job_id: str,
    manifest_path: str,
    parent_spec: dict,
    db: Any = None,
    project_id: int = 0,
    on_progress: ProgressCallback = None,
    implement_only: bool = False,
) -> JobState:
    """
    Main entry point for segmented execution.

    1. Load manifest from disk
    2. Initialise or resume state (crash recovery)
    3. Process segments in dependency order
    4. Thread evidence between segments
    5. Return final job state

    Args:
        job_id: Unique job identifier
        manifest_path: Path to manifest.json on disk
        parent_spec: The parent SPoT spec dict (for reference)
        db: SQLAlchemy session (passed to pipeline stages)
        project_id: Project ID (passed to pipeline stages)
        on_progress: Optional callback for streaming progress messages

    Returns:
        Final JobState with all segments processed
    """
    _emit = on_progress or (lambda msg: None)

    job_dir_path = get_job_dir(job_id)

    # v5.16: Set journal context so all pipeline stages can emit learning entries
    try:
        from app.experience.context import set_job_context
        set_job_context(job_id=job_id, job_dir=job_dir_path, job_type="segmented")
    except Exception:
        pass  # Non-fatal — journal is optional

    # --- Load manifest ---
    logger.info("[SEGMENT_LOOP] Starting segmented execution for job %s", job_id)
    _emit(f"📋 Loading manifest from {manifest_path}...")

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest = SegmentManifest.from_dict(manifest_data)
    except Exception as e:
        logger.error("[SEGMENT_LOOP] Failed to load manifest: %s", e)
        _emit(f"❌ Failed to load manifest: {e}")
        # Return a failed state
        state = JobState(job_id=job_id, overall_status="failed")
        return state

    _emit(f"📋 Manifest loaded: {manifest.total_segments} segment(s)")

    # =================================================================
    # v5.4 PHASE 1C: Single-segment fast path
    # =================================================================
    # When the manifest has exactly 1 segment (non-segmented job wrapped
    # by Phase 1A always-manifest), skip:
    #   - State persistence (nothing to resume)
    #   - Dependency checking (no deps)
    #   - Evidence threading (no upstream)
    #   - Contract verification (no interfaces)
    #   - Integration checks (nothing to integrate)
    #   - Blocker cascading (no dependents)
    # Same pipeline stages, less ceremony.
    
    if manifest.total_segments == 1:
        seg_spec = manifest.segments[0]
        seg_id = seg_spec.segment_id
        _emit(f"⚡ Single-segment fast path: {seg_id}")
        _emit(f"  Files: {', '.join(seg_spec.file_scope[:5])}"
               f"{'...' if len(seg_spec.file_scope) > 5 else ''}")
        
        # Build minimal context — no evidence bundle, no upstream
        segment_context = {
            "segment_id": seg_id,
            "segment_spec": seg_spec.to_dict(),
            "parent_spec": parent_spec,
            "file_scope": seg_spec.file_scope,
            "evidence": [],
            "exposes": None,
            "consumes": None,
            "requirements": seg_spec.requirements,
            "acceptance_criteria": seg_spec.acceptance_criteria,
            "dependencies": [],
        }
        
        try:
            pipeline_result = await run_segment_through_pipeline(
                segment=seg_spec,
                segment_context=segment_context,
                job_id=job_id,
                db=db,
                project_id=project_id,
                on_progress=on_progress,
            )
        except Exception as e:
            pipeline_result = {
                "success": False,
                "output_files": [],
                "error": str(e),
                "critique_warnings": [],
            }
            logger.exception("[SEGMENT_LOOP] Single-segment error: %s", e)
        
        # Build minimal final state (no disk persistence)
        state = JobState(job_id=job_id)
        state.segments[seg_id] = SegmentState(
            segment_id=seg_id,
            status=(
                SegmentStatus.COMPLETE.value if pipeline_result["success"]
                else SegmentStatus.FAILED.value
            ),
            output_files=pipeline_result.get("output_files", []),
            error=pipeline_result.get("error"),
        )
        state.overall_status = "complete" if pipeline_result["success"] else "failed"
        
        output_count = len(pipeline_result.get("output_files", []))
        if pipeline_result["success"]:
            _emit(f"\n✅ Pipeline complete ({output_count} file(s) written)")
        else:
            _emit(f"\n❌ Pipeline failed: {pipeline_result.get('error', 'Unknown')}")
        
        logger.info(
            "[SEGMENT_LOOP] v5.4 Single-segment fast path %s: %s",
            state.overall_status, job_id,
        )
        return state
    
    # =================================================================
    # Multi-segment path (existing logic)
    # =================================================================

    # --- v5.6 SKELETON CONTRACTS — Deterministic Interface Binding ---
    # Before generating any architectures, generate skeleton contracts
    # deterministically from the manifest. Zero LLM calls.
    # These contracts bind segments together by defining:
    #   - File scope constraints (prevent scope creep)
    #   - Export contracts (what downstream needs)
    #   - Import contracts (what upstream provides)
    _contract_set = None
    try:
        from app.orchestrator.skeleton_contracts import (
            generate_skeleton_contract,
            save_skeleton_contract,
            load_skeleton_contract,
        )
        _SKELETON_AVAILABLE = True
    except ImportError:
        _SKELETON_AVAILABLE = False
        logger.debug("[SEGMENT_LOOP] Skeleton contracts not available")

    if _SKELETON_AVAILABLE:
        # Check if skeleton already exists (crash recovery)
        _contract_set = load_skeleton_contract(job_dir_path)
        if _contract_set and _contract_set.skeletons:
            _emit(f"🦴 Loaded existing skeleton contract: {_contract_set.total_segments} segment(s), "
                  f"{len(_contract_set.cross_segment_bindings)} binding(s)")
        else:
            _emit("🦴 Generating skeleton contracts (deterministic)...")
            try:
                _contract_set = generate_skeleton_contract(
                    manifest_dict=manifest.to_dict(),
                    job_id=job_id,
                )
                if _contract_set.skeletons:
                    save_skeleton_contract(_contract_set, job_dir_path)
                    _total_exports = sum(len(s.exports) for s in _contract_set.skeletons)
                    _emit(f"🦴 Skeleton: {_contract_set.total_segments} segments, "
                          f"{_total_exports} exports, "
                          f"{len(_contract_set.cross_segment_bindings)} cross-segment bindings")
                    for _binding in _contract_set.cross_segment_bindings:
                        _emit(f"  🔗 {_binding['from_segment']} → {_binding['to_segment']}: "
                              f"`{_binding['file_path']}` ({_binding['binding_type']})")
                else:
                    _emit("ℹ️ No cross-segment bindings detected (segments may be independent)")
            except Exception as skel_err:
                logger.warning("[SEGMENT_LOOP] Skeleton generation failed (non-fatal): %s", skel_err)
                _emit(f"⚠️ Skeleton generation failed (non-fatal): {skel_err}")
                _contract_set = None

    # --- v2.2: Pre-load source file evidence for refactor jobs ---
    _source_evidence = _load_source_file_evidence(manifest)

    # --- v5.17 Stage 4B: SEGMENT ENRICHMENT ---
    # Enrich segments with AST-extracted source code, cross-segment symbol
    # maps, and LLM implementation intelligence BEFORE architecture generation.
    # Non-fatal — if enrichment fails, pipeline continues as before.
    _enrichment_data = {}
    if _source_evidence and manifest.total_segments > 1:
        try:
            from app.orchestrator.segment_enrichment import enrich_segments
            _emit("🔬 Running segment enrichment (Stage 4B)...")
            _enrichment_data = await enrich_segments(
                manifest=manifest,
                source_evidence=_source_evidence,
                job_dir_path=job_dir_path,
                db=db,
                project_id=project_id,
            )
            if _enrichment_data:
                _n_enriched = len(_enrichment_data)
                _total_symbols = sum(
                    e.get("extraction_stats", {}).get("constants", 0)
                    + e.get("extraction_stats", {}).get("functions", 0)
                    + e.get("extraction_stats", {}).get("classes", 0)
                    for e in _enrichment_data.values()
                )
                _n_unresolved = sum(
                    len(e.get("unresolved", []))
                    for e in _enrichment_data.values()
                )
                _emit(f"🔬 Segment enrichment complete: {_n_enriched} segment(s), "
                      f"{_total_symbols} symbol(s) extracted")
                if _n_unresolved:
                    _emit(f"  ⚠️ {_n_unresolved} unresolved symbol(s) detected")
                # Show per-segment summary
                for _seg_id, _seg_enrich in _enrichment_data.items():
                    _stats = _seg_enrich.get("extraction_stats", {})
                    _risk = _seg_enrich.get("risk_level", "low")
                    _order = _seg_enrich.get("implementation_order", 0)
                    _risk_icon = "🔴" if _risk == "high" else "🟡" if _risk == "medium" else "🟢"
                    _emit(f"  {_risk_icon} {_seg_id}: "
                          f"{_stats.get('constants', 0)}C/{_stats.get('functions', 0)}F/{_stats.get('classes', 0)}Cl "
                          f"risk={_risk} order={_order}")
            else:
                _emit("🔬 Segment enrichment: no data produced (pipeline continues as before)")
        except Exception as enrich_err:
            logger.warning("[SEGMENT_LOOP] Segment enrichment failed (non-fatal): %s", enrich_err)
            _emit(f"⚠️ Segment enrichment failed (non-fatal): {enrich_err}")
            _enrichment_data = {}

    # --- v5.21 POST-ENRICHMENT SKELETON AUGMENTATION ---
    # Enrichment extracted function names/signatures per segment. Wire them
    # into the skeleton contracts so the architecture generator knows EXACTLY
    # which symbols each file must export (not just which files are consumed).
    # This prevents the #1 source of cohesion failures: missing_symbol errors
    # where seg-05 imports build_evidence_bundle from seg-04 but seg-04's
    # architecture never defined it.
    if _enrichment_data and _contract_set and _SKELETON_AVAILABLE:
        try:
            from app.orchestrator.skeleton_contracts import augment_skeleton_with_enrichment
            _augmented = augment_skeleton_with_enrichment(
                contract_set=_contract_set,
                enrichment_data=_enrichment_data,
                job_dir=job_dir_path,
            )
            if _augmented:
                _emit(f"🦴 Skeleton augmented: {_augmented} export binding(s) now have named symbols")
                logger.info(
                    "[SEGMENT_LOOP] v5.21 Skeleton augmented with %d enriched export binding(s)",
                    _augmented,
                )
            else:
                logger.debug("[SEGMENT_LOOP] v5.21 No export bindings to augment")
        except Exception as _aug_err:
            logger.warning("[SEGMENT_LOOP] v5.21 Skeleton augmentation failed (non-fatal): %s", _aug_err)
            _emit(f"⚠️ Skeleton augmentation failed (non-fatal): {_aug_err}")

    # --- v2.2: Evidence Ledger — create/load and seed with source files ---
    _ledger = None
    try:
        from app.orchestrator.evidence_ledger import (
            create_ledger, load_ledger, save_ledger,
            seed_ledger_with_source_files,
        )
        _ledger = load_ledger(job_dir_path)
        if _ledger is None:
            _ledger = create_ledger(job_id, job_dir_path)
            if _source_evidence:
                seed_ledger_with_source_files(_ledger, job_dir_path, _source_evidence)
        else:
            _emit(f"📚 Evidence ledger loaded: {_ledger.entry_count} entries")
    except Exception as _ledger_err:
        logger.warning("[SEGMENT_LOOP] Evidence ledger init failed (non-fatal): %s", _ledger_err)
        _ledger = None

    # --- v5.7 PRE-EXECUTION QUARANTINE — File→Package Refactors ---
    # When a job converts a .py file into a package directory, the original
    # must be quarantined BEFORE any segments execute. The per-segment shadow
    # check (arch_executor v2.9) can't handle this because __init__.py is
    # typically in a different segment than the files that need the directory.
    # v5.15: Only quarantine during implement_only (implement segments),
    # NOT during run segments (architecture design). Quarantining during
    # architecture design breaks evidence gathering because the monolith
    # gets moved before the Critical Pipeline can read it for grounding.
    # --- Initialise or resume state ---
    # v5.19: Moved BEFORE quarantine so we can check segment readiness.
    state = load_or_init_state(job_id, manifest)
    _emit(f"📊 State: {state.summary()}")

    _quarantine_result = None
    if not implement_only:
        logger.debug("[SEGMENT_LOOP] v5.15 Skipping quarantine (run segments mode — architecture design only)")
    else:
        # v5.22: Auto-recover FAILED/BLOCKED segments on retry.
        # When the user says 'implement segments' after a failure, they're
        # explicitly retrying. Segments that have architectures should be
        # restored to APPROVED so quarantine doesn't skip and shadow
        # detection doesn't block every file operation.
        _failed_or_blocked = [
            (sid, s) for sid, s in state.segments.items()
            if s.status in (SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value)
        ]
        if _failed_or_blocked:
            _recovered = []
            for _fb_sid, _fb_state in _failed_or_blocked:
                _fb_arch_dir = os.path.join(job_dir_path, "segments", _fb_sid, "arch")
                _has_arch = (
                    os.path.isdir(_fb_arch_dir)
                    and any(f.endswith(".md") for f in os.listdir(_fb_arch_dir))
                )
                if _has_arch:
                    _fb_state.status = SegmentStatus.APPROVED.value
                    _fb_state.error = None
                    _fb_state.started_at = None
                    _fb_state.completed_at = None
                    _recovered.append(_fb_sid)
                    logger.info(
                        "[SEGMENT_LOOP] v5.22 Auto-recovered %s: FAILED/BLOCKED -> APPROVED (retry)",
                        _fb_sid,
                    )
            if _recovered:
                save_state(state, job_dir_path)
                _emit(
                    f"🔄 Auto-recovered {len(_recovered)} segment(s) for retry: "
                    f"{', '.join(_recovered[:5])}{'...' if len(_recovered) > 5 else ''}"
                )

        # v5.31: Quarantine DEFERRED to just before Phase Checkout (Stage 9).
        # Previously ran here (before segment execution), but this caused
        # the monolith to be moved before strike-loop retries could re-read
        # it as source evidence. The monolith is only needed gone for the
        # boot test, so we defer quarantine until all segments are complete.
        logger.info("[SEGMENT_LOOP] v5.31 Quarantine deferred to Phase Checkout")

    # --- Process segments in dependency order (multi-pass) ---
    # v5.11: The loop repeats until no further progress is made.
    # This handles segments that are skipped on early passes because
    # their dependencies aren't COMPLETE yet (e.g. seg-01 depends on seg-02..seg-09).
    # Also handles PENDING segments that get architectures generated and need
    # a second pass to execute once approved.
    execution_order = manifest.get_execution_order()
    total = len(execution_order)
    _pass_number = 0
    MAX_PASSES = 5  # Safety limit to prevent infinite loops

    _emit(f"🔄 Processing {total} segment(s) in dependency order...\n")

    while _pass_number < MAX_PASSES:
        _pass_number += 1
        _progress_this_pass = 0

        # v5.15: Re-evaluate BLOCKED segments at start of each pass.
        # If a blocker was re-tried and succeeded, its dependents
        # should become runnable again.
        if _pass_number > 1:
            _unblocked = unblock_recovered_segments(state, manifest, job_dir_path)
            if _unblocked:
                _emit(f"\n🔓 Unblocked {len(_unblocked)} segment(s) (blocker recovered): {_unblocked}")
                _progress_this_pass += len(_unblocked)  # Count as progress to keep loop alive

        for idx, seg_id in enumerate(execution_order, 1):
            seg_state = state.segments.get(seg_id)
            seg_spec = manifest.get_segment(seg_id)

            if seg_state is None or seg_spec is None:
                logger.error("[SEGMENT_LOOP] Missing state/spec for segment %s", seg_id)
                continue

            # --- Skip already COMPLETE segments (crash recovery) ---
            if seg_state.status == SegmentStatus.COMPLETE.value:
                _emit(f"⏭️ [{idx}/{total}] {seg_id}: already COMPLETE (skipping)")
                continue

            # --- Skip BLOCKED segments (with inline recovery check) ---
            if seg_state.status == SegmentStatus.BLOCKED.value:
                # v5.15: Check if blocker has recovered since we were marked BLOCKED
                if not is_segment_blocked(seg_spec, state):
                    # Blocker recovered! Determine restore status
                    _seg_arch_dir = os.path.join(job_dir_path, "segments", seg_id, "arch")
                    _has_arch = os.path.isdir(_seg_arch_dir) and any(f.endswith(".md") for f in os.listdir(_seg_arch_dir))
                    _restore = SegmentStatus.APPROVED if _has_arch else SegmentStatus.PENDING
                    update_segment_status(state, seg_id, _restore, job_dir_path, error=None)
                    seg_state = state.segments[seg_id]  # refresh
                    _emit(f"🔓 [{idx}/{total}] {seg_id}: UNBLOCKED (blocker recovered) -> {_restore.value}")
                    logger.info("[SEGMENT_LOOP] v5.15 Inline unblock: %s -> %s", seg_id, _restore.value)
                    # Fall through to be processed in this pass
                else:
                    _emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED — {seg_state.error or 'dependency failed'}")
                    continue

            # --- v3.0: APPROVED segments — skip architecture, go straight to execution ---
            if seg_state.status == SegmentStatus.APPROVED.value:
                # v5.13: If NOT in implement_only mode, skip APPROVED segments.
                # They need a separate "implement segments" command to execute.
                if not implement_only:
                    _emit(f"⏸️ [{idx}/{total}] {seg_id}: APPROVED — awaiting 'implement segments' command")
                    continue
                # v3.1: Check if dependencies failed/blocked BEFORE executing
                if is_segment_blocked(seg_spec, state):
                    update_segment_status(
                        state, seg_id, SegmentStatus.BLOCKED, job_dir_path,
                        error="Dependency failed or blocked",
                    )
                    _emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED by failed dependency (was APPROVED)")
                    continue
                # v5.10: APPROVED execution requires deps COMPLETE (files on disk),
                # not just APPROVED. APPROVED-as-met is only for architecture generation.
                _deps_complete = True
                for _dep_id in (seg_spec.dependencies or []):
                    _dep_st = state.segments.get(_dep_id)
                    if _dep_st and _dep_st.status != SegmentStatus.COMPLETE.value:
                        _deps_complete = False
                        break
                if not _deps_complete:
                    _emit(f"⏳ [{idx}/{total}] {seg_id}: APPROVED but dependencies not yet COMPLETE (skipping)")
                    continue

                _emit(f"\n✅ [{idx}/{total}] {seg_id}: APPROVED — executing...")
                _emit(f"  Files: {', '.join(seg_spec.file_scope[:5])}"
                       f"{'...' if len(seg_spec.file_scope) > 5 else ''}")
                update_segment_status(state, seg_id, SegmentStatus.IN_PROGRESS, job_dir_path)

                # Load the saved architecture and execute directly
                # v5.8: Use consistent version resolution (find highest arch_v{N}.md)
                seg_dir = os.path.join(job_dir_path, "segments", seg_id)
                arch_path = _find_latest_arch(seg_dir)

                if arch_path is None or not os.path.isfile(arch_path):
                    update_segment_status(
                        state, seg_id, SegmentStatus.FAILED, job_dir_path,
                        error=f"Architecture file not found: {arch_path}",
                    )
                    _emit(f"  ❌ Architecture file missing: {arch_path}")
                    blocked = mark_dependents_blocked(state, seg_id, manifest, job_dir_path)
                    if blocked:
                        _emit(f"  🚫 Blocked {len(blocked)} dependent segment(s)")
                    continue

                with open(arch_path, 'r', encoding='utf-8') as f:
                    arch_text = f.read()
                _emit(f"  📄 Loaded architecture: {arch_path} ({len(arch_text)} chars)")

                # v5.18: Sanitise loaded architecture before execution
                try:
                    from app.orchestrator.architecture_sanitiser import sanitise_architecture
                    arch_text, _san_result = sanitise_architecture(
                        arch_text=arch_text,
                        file_scope=seg_spec.file_scope,
                        segment_id=seg_id,
                    )
                    if _san_result.had_fixes:
                        _emit(f"  🧹 Sanitiser: {_san_result.fix_count} fix(es) applied to loaded architecture")
                        # Re-save the sanitised version
                        try:
                            with open(arch_path, "w", encoding="utf-8") as _sf:
                                _sf.write(arch_text)
                        except Exception:
                            pass
                except ImportError:
                    pass
                except Exception as _san_err:
                    logger.warning("[SEGMENT_LOOP] v5.18 Sanitiser error on load (non-fatal): %s", _san_err)

                # v2.2: Build segment context for approved-resume path
                segment_context = build_segment_context(
                    seg_spec, state, parent_spec, job_dir_path,
                    contract_set=_contract_set,
                    source_file_evidence=_source_evidence,
                    enrichment=_enrichment_data.get(seg_spec.segment_id),  # v5.17
                )

                # Execute via Overwatcher + Implementer
                pipeline_result = {"success": False, "error": None, "output_files": []}
                try:
                    if not _ARCH_EXECUTOR_AVAILABLE:
                        pipeline_result["error"] = "Architecture executor not available"
                        _emit(f"  ⚠️ Architecture executor not available")
                    else:
                        spec = resolve_latest_spec(project_id, db)
                        if spec is None:
                            pipeline_result["error"] = f"No spec found for project {project_id}"
                            _emit(f"  ⚠️ No spec found")
                        else:
                            llm_call_fn = create_overwatcher_llm_fn()
                            seg_job_id = f"{job_id}__{seg_id}"
                            # v5.5 PHASE 4A: Pass interface contract for Job Checker
                            _seg_contract_md = segment_context.get("interface_contract", "") if segment_context else ""
                            # v5.12: Interface Reconciliation (Option A)
                            # Read actual interfaces from completed dependency segments
                            # and inject into architecture so Implementer uses correct names
                            _recon_arch_text = arch_text
                            if _RECONCILIATION_AVAILABLE and seg_spec.dependencies:
                                try:
                                    _recon_block = read_dependency_interfaces_from_sandbox(
                                        segment=seg_spec,
                                        completed_segments=state.segments,
                                        manifest=manifest,
                                    )
                                    if _recon_block:
                                        _recon_arch_text = inject_reconciliation_into_architecture(
                                            arch_text, _recon_block,
                                        )
                                        _emit(f"  \U0001f9e9 Interface reconciliation: injected real interfaces from {len(seg_spec.dependencies)} dependency segment(s)")
                                        logger.info(
                                            "[SEGMENT_LOOP] v5.12 Reconciliation injected for %s (%d chars added)",
                                            seg_id, len(_recon_block),
                                        )
                                except Exception as _recon_err:
                                    logger.warning("[SEGMENT_LOOP] v5.12 Reconciliation failed (non-fatal): %s", _recon_err)
                                    _emit(f"  \u26a0\ufe0f Interface reconciliation failed (non-fatal): {_recon_err}")

                            # v5.26: Extraction Binding (call site 2 — implement_only path)
                            try:
                                from app.orchestrator.extraction_binding import (
                                    load_segment_enrichment,
                                    build_extraction_block,
                                    build_facade_export_map,
                                    inject_extraction_into_architecture,
                                )
                                _eb_enrichment = load_segment_enrichment(job_dir_path, seg_id)
                                if _eb_enrichment:
                                    _is_facade = _is_facade_segment(seg_spec, manifest) if manifest else False
                                    if _is_facade and manifest:
                                        _eb_block = build_facade_export_map(
                                            job_dir_path, manifest.segments, seg_id,
                                        )
                                    else:
                                        _eb_block = build_extraction_block(_eb_enrichment, seg_id)
                                    if _eb_block:
                                        _recon_arch_text = inject_extraction_into_architecture(
                                            _recon_arch_text, _eb_block,
                                        )
                                        _emit(f"  🧬 Extraction binding: injected source code for {seg_id} ({len(_eb_block)} chars)")
                                        logger.info(
                                            "[SEGMENT_LOOP] v5.26 Extraction binding for %s (%d chars)",
                                            seg_id, len(_eb_block),
                                        )
                            except Exception as _eb_err:
                                logger.warning("[SEGMENT_LOOP] v5.26 Extraction binding failed (non-fatal): %s", _eb_err)
                                _emit(f"  ⚠️ Extraction binding failed (non-fatal): {_eb_err}")
                            # v4.0: Skip boot check — Phase Checkout handles it
                            # v5.32: Pass all manifest file paths so job checker
                            # treats future segment files as expected imports
                            _manifest_all_files = set()
                            for _ms in manifest.segments:
                                for _mf in _ms.file_scope:
                                    _manifest_all_files.add(_mf.replace("\\", "/"))
                            arch_result = await run_architecture_execution(
                                spec=spec,
                                architecture_content=_recon_arch_text,
                                architecture_path=arch_path,
                                job_id=seg_job_id,
                                llm_call_fn=llm_call_fn,
                                artifact_root=os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs"),
                                interface_contract=_seg_contract_md,
                                skip_boot_check=True,
                                manifest_all_files=_manifest_all_files,
                            )
                            if arch_result.get("success", False):
                                pipeline_result["success"] = True
                                pipeline_result["output_files"] = arch_result.get("artifacts_written", [])
                                _emit(f"  ✅ Overwatcher + Implementer completed ({len(pipeline_result['output_files'])} files)")
                                for _of in pipeline_result['output_files']:
                                    _emit(f"    ✅ {_of}")
                            else:
                                pipeline_result["error"] = arch_result.get("error", "Unknown error")
                                _emit(f"  ❌ Execution failed: {pipeline_result['error']}")

                                # v5.8: Persist execution trace for failure diagnosis
                                _save_execution_trace(seg_id, job_dir_path, arch_result)
                                _n_trace = len(arch_result.get("trace", []))
                                if _n_trace:
                                    _emit(f"  💾 Execution trace saved ({_n_trace} events) — check segments/{seg_id}/execution_trace/trace.json")
                except Exception as e:
                    pipeline_result["error"] = f"Execution error: {e}"
                    logger.exception("[SEGMENT_LOOP] Execution error for approved %s", seg_id)
                    _emit(f"  ❌ Execution error: {e}")

                # Handle result (same as normal flow)
                if pipeline_result["success"]:
                    output_files = pipeline_result.get("output_files", [])
                    update_segment_status(
                        state, seg_id, SegmentStatus.COMPLETE, job_dir_path,
                        output_files=output_files,
                    )
                    _emit(f"  ✅ {seg_id}: COMPLETE ({len(output_files)} output file(s))")
                    _progress_this_pass += 1
                else:
                    error_msg = pipeline_result.get("error", "Unknown")
                    update_segment_status(
                        state, seg_id, SegmentStatus.FAILED, job_dir_path,
                        error=error_msg,
                    )
                    _emit(f"  ❌ {seg_id}: FAILED — {error_msg}")
                    print(f"[SEGMENT_LOOP] v3.1 ❌ SEGMENT FAILED: {seg_id} — {error_msg}")
                    blocked = mark_dependents_blocked(state, seg_id, manifest, job_dir_path)
                    if blocked:
                        _emit(f"  🚫 STOPPING: Blocked {len(blocked)} dependent segment(s): {blocked}")
                        print(f"[SEGMENT_LOOP] v3.1 🚫 BLOCKED dependents: {blocked}")
                continue  # v3.1: CRITICAL — must continue after APPROVED handling to avoid fall-through
            # --- v5.13 / v5.26: In implement_only mode, skip PENDING segments ---
            # They need architecture generation first (via 'run segments').
            #
            # v5.26 EXCEPTION: Facade segments (depend on ALL other segments) should
            # auto-generate their architecture during implement_only if all deps are
            # COMPLETE. This is because facades need real interface data from completed
            # segments — running 'run segments' separately would generate architecture
            # without that data, leading to truncation/MODIFY failures.
            if implement_only and seg_state.status == SegmentStatus.PENDING.value:
                if _is_facade_segment(seg_spec, manifest):
                    # Check if all deps are COMPLETE
                    if can_execute_segment(seg_spec, state, require_complete=True):
                        _emit(f"\n🏗️ [{idx}/{total}] {seg_id}: FACADE — all deps COMPLETE, auto-generating architecture + implementing")
                        _emit(f"  Files: {', '.join(seg_spec.file_scope[:5])}")
                        update_segment_status(state, seg_id, SegmentStatus.IN_PROGRESS, job_dir_path)

                        # Build context with real interface data from completed segments
                        segment_context = build_segment_context(
                            seg_spec, state, parent_spec, job_dir_path,
                            contract_set=_contract_set,
                            source_file_evidence=_source_evidence,
                            enrichment=_enrichment_data.get(seg_spec.segment_id),
                        )

                        # v5.26: Flag for approval gate bypass — facades in implement_only
                        # go straight through since we're already in the implement phase
                        segment_context["_facade_auto_execute"] = True

                        # v5.26: Pre-read dependency output files and inject their
                        # contents into source_file_evidence. The facade needs to
                        # see the ACTUAL code it's importing from, not just paths.
                        _dep_file_contents: Dict[str, str] = {}
                        for _dep_id in seg_spec.dependencies:
                            _dep_state = state.segments.get(_dep_id)
                            if _dep_state and _dep_state.status == SegmentStatus.COMPLETE.value:
                                for _dep_file in (_dep_state.output_files or []):
                                    try:
                                        with open(_dep_file, "r", encoding="utf-8", errors="replace") as _df:
                                            _dep_content = _df.read(60_000)  # Cap at 60K per file
                                        # Convert absolute path to relative for the prompt
                                        _rel_path = _dep_file
                                        for _root in ["D:\\Orb\\", "D:\\orb-desktop\\", "D:/Orb/", "D:/orb-desktop/"]:
                                            if _dep_file.startswith(_root):
                                                _rel_path = _dep_file[len(_root):]
                                                break
                                        _dep_file_contents[_rel_path] = _dep_content
                                    except Exception as _read_err:
                                        logger.warning(
                                            "[SEGMENT_LOOP] v5.26 Failed to read dep file %s: %s",
                                            _dep_file, _read_err,
                                        )
                        if _dep_file_contents:
                            # Merge into source_file_evidence so the architecture
                            # model sees both the original monolith AND the new modules
                            _existing = segment_context.get("source_file_evidence", {})
                            _existing.update(_dep_file_contents)
                            segment_context["source_file_evidence"] = _existing
                            _emit(f"  📚 Injected {len(_dep_file_contents)} dependency file(s) as evidence")
                            for _dfp in sorted(_dep_file_contents.keys()):
                                _emit(f"    → {_dfp} ({len(_dep_file_contents[_dfp]):,} chars)")
                            logger.info(
                                "[SEGMENT_LOOP] v5.26 Facade evidence: %d dep files injected for %s",
                                len(_dep_file_contents), seg_id,
                            )

                        # Run full pipeline: architecture generation → implementation
                        try:
                            pipeline_result = await run_segment_through_pipeline(
                                segment=seg_spec,
                                segment_context=segment_context,
                                job_id=job_id,
                                db=db,
                                project_id=project_id,
                                on_progress=on_progress,
                                contract_set=_contract_set,
                                job_dir_path=job_dir_path,
                                manifest=manifest,
                                parent_spec=parent_spec,
                                quarantine_result=_quarantine_result,
                            )
                        except Exception as e:
                            pipeline_result = {"success": False, "error": str(e), "output_files": []}
                            logger.exception("[SEGMENT_LOOP] v5.26 Facade pipeline error for %s", seg_id)

                        # Handle result
                        if pipeline_result.get("success"):
                            if pipeline_result.get("awaiting_approval"):
                                update_segment_status(state, seg_id, SegmentStatus.APPROVED, job_dir_path)
                                _emit(f"  ✅ {seg_id}: APPROVED (facade architecture ready)")
                            else:
                                output_files = pipeline_result.get("output_files", [])
                                update_segment_status(
                                    state, seg_id, SegmentStatus.COMPLETE, job_dir_path,
                                    output_files=output_files,
                                )
                                _emit(f"  ✅ {seg_id}: COMPLETE ({len(output_files)} output file(s))")
                                _progress_this_pass += 1
                        else:
                            error_msg = pipeline_result.get("error", "Unknown")
                            update_segment_status(
                                state, seg_id, SegmentStatus.FAILED, job_dir_path,
                                error=error_msg,
                            )
                            _emit(f"  ❌ {seg_id}: FAILED — {error_msg}")
                        continue
                    else:
                        _emit(f"⏳ [{idx}/{total}] {seg_id}: FACADE — waiting for all dependencies to be COMPLETE")
                        continue
                else:
                    _emit(f"⏭️ [{idx}/{total}] {seg_id}: PENDING — needs architecture first (run 'run segments')")
                    continue

            # --- Check if segment should be blocked ---
            if is_segment_blocked(seg_spec, state):
                update_segment_status(
                    state, seg_id, SegmentStatus.BLOCKED, job_dir_path,
                    error="Dependency failed or blocked",
                )
                _emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED by failed dependency")
                continue

            # --- Check dependencies ---
            # v5.26: Facade segments must wait for deps to be COMPLETE (files on disk),
            # not just APPROVED. This ensures the architecture generator has access to
            # actual exported interfaces, not just spec promises.
            _facade = _is_facade_segment(seg_spec, manifest)
            if not can_execute_segment(seg_spec, state, require_complete=_facade):
                if _facade:
                    _emit(f"⏳ [{idx}/{total}] {seg_id}: FACADE — waiting for all dependencies to be COMPLETE")
                else:
                    _emit(f"⏳ [{idx}/{total}] {seg_id}: waiting on dependencies (skipping)")
                continue

            # --- Execute segment ---
            _emit(f"\n⚙️ [{idx}/{total}] {seg_id}: {seg_spec.title}")
            _emit(f"  Files: {', '.join(seg_spec.file_scope[:5])}"
                   f"{'...' if len(seg_spec.file_scope) > 5 else ''}")
            _emit(f"  Dependencies: {seg_spec.dependencies or 'none'}")

            # Mark IN_PROGRESS
            update_segment_status(state, seg_id, SegmentStatus.IN_PROGRESS, job_dir_path)

            # Build execution context with upstream evidence + interface contracts
            segment_context = build_segment_context(
                seg_spec, state, parent_spec, job_dir_path,
                contract_set=_contract_set,
                source_file_evidence=_source_evidence,
                enrichment=_enrichment_data.get(seg_spec.segment_id),  # v5.17
            )

            # v2.3 FIX #2: Inject cohesion feedback for targeted regen
            # If this segment was reset due to cohesion failure, inject the feedback
            # so the architecture generator knows what to fix.
            _seg_state = state.segments.get(seg_id)
            if _seg_state and _seg_state.error and _seg_state.error.startswith("Cohesion regen:"):
                segment_context["cohesion_feedback"] = _seg_state.error
                logger.info("[SEGMENT_LOOP] v2.3 Injected cohesion feedback for %s regen", seg_id)
                _emit(f"  🔄 Re-generating with cohesion feedback: {_seg_state.error[:120]}")

            # Run through pipeline
            try:
                pipeline_result = await run_segment_through_pipeline(
                    segment=seg_spec,
                    segment_context=segment_context,
                    job_id=job_id,
                    db=db,
                    project_id=project_id,
                    on_progress=on_progress,
                    contract_set=_contract_set,
                    job_dir_path=job_dir_path,
                    manifest=manifest,
                    parent_spec=parent_spec,
                    quarantine_result=_quarantine_result,
                )
            except Exception as e:
                pipeline_result = {
                    "success": False,
                    "output_files": [],
                    "error": str(e),
                    "critique_warnings": [],
                }
                logger.exception("[SEGMENT_LOOP] Unexpected error processing %s", seg_id)

            # --- Handle result ---
            if pipeline_result["success"]:
                # v3.0: Check if segment is awaiting approval (architecture generated but not executed)
                if pipeline_result.get("awaiting_approval", False):
                    update_segment_status(
                        state, seg_id, SegmentStatus.APPROVED, job_dir_path,
                    )
                    _emit(f"  ✅ {seg_id}: APPROVED — architecture ready for review")
                    _progress_this_pass += 1
                else:
                    # Collect output files
                    output_files = pipeline_result.get("output_files", [])
                    if not output_files:
                        output_files = collect_segment_outputs(seg_id, job_dir_path)

                    # Mark COMPLETE
                    update_segment_status(
                        state, seg_id, SegmentStatus.COMPLETE, job_dir_path,
                        output_files=output_files,
                    )

                    # Verify interface contracts
                    contract_warnings = verify_contracts_fulfilled(seg_id, state, manifest)
                    if contract_warnings:
                        _emit(f"  ⚠️ Contract warnings: {len(contract_warnings)}")

                    _emit(f"  ✅ {seg_id}: COMPLETE ({len(output_files)} output file(s))")
                    _progress_this_pass += 1

            else:
                error_msg = pipeline_result.get("error", "Unknown error")

                # Mark FAILED
                update_segment_status(
                    state, seg_id, SegmentStatus.FAILED, job_dir_path,
                    error=error_msg,
                )
                _emit(f"  ❌ {seg_id}: FAILED — {error_msg}")

                # Block dependents
                blocked = mark_dependents_blocked(state, seg_id, manifest, job_dir_path)
                if blocked:
                    _emit(f"  🚫 Blocked {len(blocked)} dependent segment(s): {blocked}")

        # v5.11: Check if any progress was made this pass
        if _progress_this_pass == 0:
            logger.info("[SEGMENT_LOOP] v5.11 Pass %d: no progress — stopping", _pass_number)
            break
        else:
            _remaining = sum(
                1 for ss in state.segments.values()
                if ss.status not in (SegmentStatus.COMPLETE.value, SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value)
            )
            logger.info(
                "[SEGMENT_LOOP] v5.11 Pass %d: %d segment(s) progressed, %d remaining",
                _pass_number, _progress_this_pass, _remaining,
            )
            if _remaining == 0:
                break
            _emit(f"\n🔄 Pass {_pass_number} complete ({_progress_this_pass} progressed, {_remaining} remaining) — continuing...\n")

    # --- v5.12 POST-EXECUTION RECONCILIATION (Option B fallback) ---
    # After execution completes, scan all implemented files on the sandbox
    # for import mismatches and surgically fix them. This catches anything
    # that Option A (pre-execution interface injection) missed.
    _any_complete = any(
        ss.status == SegmentStatus.COMPLETE.value
        for ss in state.segments.values()
    )
    _any_failed = any(
        ss.status == SegmentStatus.FAILED.value
        for ss in state.segments.values()
    )
    if _any_complete and implement_only:
        try:
            from app.orchestrator.post_execution_reconciliation import (
                run_post_execution_reconciliation,
            )
            _emit(f"\n{'='*50}")
            _recon_result = run_post_execution_reconciliation(
                manifest=manifest,
                state=state,
                on_progress=_emit,
            )
            if _recon_result.fixes_applied:
                logger.info(
                    "[SEGMENT_LOOP] v5.12 Post-execution reconciliation: %d fix(es) in %d file(s)",
                    len(_recon_result.fixes_applied), _recon_result.files_fixed,
                )
                # If fixes were applied to a FAILED segment's files, consider
                # re-checking if the segment might now succeed
                if _any_failed:
                    _emit("  \U0001f4a1 Fixes applied to files from failed segment(s) — "
                          "these may resolve the failure on retry")
        except ImportError:
            logger.debug("[SEGMENT_LOOP] Post-execution reconciliation not available")
        except Exception as _recon_err:
            logger.warning("[SEGMENT_LOOP] v5.12 Post-execution reconciliation error (non-fatal): %s", _recon_err)
            _emit(f"\u26a0\ufe0f Post-execution reconciliation error (non-fatal): {_recon_err}")

    # --- v5.18 DEFERRED CONSUMER RECONCILIATION ---
    # After post-recon, check deferred consumer files for missing re-exports.
    # These are external files (e.g. cohesion_check.py, phase_loop.py) that
    # were excluded from segment scope but import from the refactored package.
    _deferred = getattr(manifest, 'deferred_consumer_files', []) or []
    if _deferred and _any_complete and implement_only:
        try:
            from app.orchestrator.post_execution_reconciliation import reconcile_deferred_consumers
            _consumer_result = reconcile_deferred_consumers(
                manifest=manifest,
                on_progress=_emit,
            )
            if _consumer_result.errors:
                logger.warning(
                    "[SEGMENT_LOOP] v5.18 Deferred consumer issues: %s",
                    _consumer_result.errors,
                )
        except ImportError:
            logger.debug("[SEGMENT_LOOP] Deferred consumer recon not available")
        except Exception as _dc_err:
            logger.warning(
                "[SEGMENT_LOOP] v5.18 Deferred consumer recon error (non-fatal): %s",
                _dc_err,
            )

    # --- v5.16 PHASE 2C: Cohesion Check + Automated Regen Loop ---
    # After architecture generation, run cohesion check. If blocking issues
    # remain after auto-fix (Tier 1/2), automatically re-generate the flagged
    # segments through Critical Pipeline with cohesion feedback, then re-check.
    # Loop until cohesion passes or retries exhausted.
    MAX_COHESION_RETRIES = 3
    _cohesion_retry = 0
    _cohesion_passed = False

    while _cohesion_retry < MAX_COHESION_RETRIES and not _cohesion_passed:
        _approved_seg_ids = [
            sid for sid, ss in state.segments.items()
            if ss.status in (SegmentStatus.APPROVED.value, SegmentStatus.COMPLETE.value)
        ]

        if len(_approved_seg_ids) < 2:
            break

        _cohesion_retry += 1
        _emit(f"\n{'='*50}")
        if _cohesion_retry == 1:
            _emit("🔍 Running cross-segment cohesion check...")
        else:
            _emit(f"🔍 Cohesion re-check (attempt {_cohesion_retry}/{MAX_COHESION_RETRIES})...")

        try:
            from app.orchestrator.cohesion_check import (
                run_cohesion_check,
                save_cohesion_result,
            )

            _cohesion_contract_json = None
            if _contract_set:
                _cohesion_contract_json = _contract_set.to_json()

            _cohesion_result = await run_cohesion_check(
                job_id=job_id,
                job_dir=job_dir_path,
                segment_ids=_approved_seg_ids,
                contract_json=_cohesion_contract_json,
                source_file_evidence=_source_evidence,
            )
            save_cohesion_result(_cohesion_result, job_dir_path)

            # v5.29: Emit cohesion issues to journal for experience distillation
            try:
                from app.experience.journal_writer import emit_journal_entry
                from app.experience.schemas import JournalEventType
                for _ci in _cohesion_result.issues:
                    # Map category to event type
                    _evt_map = {
                        "import_mismatch": JournalEventType.COHESION_MISMATCH,
                        "missing_export": JournalEventType.COHESION_MISMATCH,
                        "naming_mismatch": JournalEventType.COHESION_NAMING_DRIFT,
                        "shape_mismatch": JournalEventType.COHESION_INTERFACE_BREAK,
                        "contract_violation": JournalEventType.COHESION_INTERFACE_BREAK,
                        "scope_violation": JournalEventType.COHESION_MISMATCH,
                        "phantom_segment": JournalEventType.COHESION_MISMATCH,
                        "endpoint_mismatch": JournalEventType.COHESION_INTERFACE_BREAK,
                    }
                    _evt = _evt_map.get(_ci.category, JournalEventType.COHESION_MISMATCH)
                    emit_journal_entry(
                        job_id,
                        job_dir_path,
                        stage="cohesion_check",
                        event_type=_evt.value,
                        severity="blocking" if _ci.severity == "blocking" else "warning",
                        description=_ci.description[:300],
                        root_cause=_ci.category,
                        resolution=_ci.auto_fix_note if _ci.auto_fixed else _ci.suggested_fix,
                        file_scope=_ci.file_path,
                        segment_id=_ci.source_segment,
                        details={
                            "issue_id": _ci.issue_id,
                            "expected": _ci.expected[:200] if _ci.expected else "",
                            "actual": _ci.actual[:200] if _ci.actual else "",
                            "related_segment": _ci.related_segment,
                            "auto_fixed": _ci.auto_fixed,
                            "auto_fix_tier": _ci.auto_fix_tier,
                        },
                    )
            except Exception as _jrn_err:
                logger.debug("[SEGMENT_LOOP] v5.29 cohesion journal emit failed: %s", _jrn_err)

            # Show auto-fixed issues
            _auto_fixed = [ci for ci in _cohesion_result.issues if ci.auto_fixed or ci.severity == "resolved"]
            if _auto_fixed:
                _emit(f"🔧 Auto-fixed {len(_auto_fixed)} issue(s):")
                for _ci in _auto_fixed:
                    _tier_label = f"T{_ci.auto_fix_tier}" if _ci.auto_fix_tier else "?"
                    _emit(f"  ✅ {_ci.issue_id} [{_tier_label}] {_ci.auto_fix_note or _ci.description[:100]}")

            if _cohesion_result.status == "pass":
                _cohesion_passed = True
                if _auto_fixed:
                    _emit("✅ Cohesion check PASSED — all issues resolved by auto-fix!")
                else:
                    _emit("✅ Cohesion check PASSED — all segments are compatible")

            elif _cohesion_result.status == "fail":
                _n_blocking = len(_cohesion_result.blocking_issues)
                _n_warning = len(_cohesion_result.warning_issues)

                if _cohesion_retry >= MAX_COHESION_RETRIES:
                    # Exhausted retries — report to user
                    _emit(f"❌ Cohesion check FAILED after {MAX_COHESION_RETRIES} attempts — {_n_blocking} blocking, {_n_warning} warning(s)")
                    for _ci in _cohesion_result.blocking_issues:
                        _tier_label = f"T{_ci.auto_fix_tier}" if _ci.auto_fix_tier else "?"
                        _emit(f"  🚫 {_ci.issue_id} [{_ci.category}/{_tier_label}] {_ci.source_segment} ↔ {_ci.related_segment}")
                        _emit(f"     {_ci.description}")
                        if _ci.suggested_fix:
                            _emit(f"     Fix: {_ci.suggested_fix}")
                    for _ci in _cohesion_result.warning_issues:
                        _emit(f"  ⚠️ {_ci.issue_id} [{_ci.category}] {_ci.description}")

                    _regen_segs = _cohesion_result.segments_needing_regen
                    if _regen_segs:
                        for _regen_seg_id in _regen_segs:
                            if _regen_seg_id in state.segments:
                                # v5.33: Structured feedback (same as retry path)
                                _fb_parts = []
                                for ci in _cohesion_result.blocking_issues:
                                    if ci.source_segment != _regen_seg_id and ci.related_segment != _regen_seg_id:
                                        continue
                                    _part = f"[{ci.issue_id}] {ci.category}: {ci.description}"
                                    if ci.expected:
                                        _part += f" | Expected: {ci.expected[:200]}"
                                    if ci.actual:
                                        _part += f" | Actual: {ci.actual[:200]}"
                                    if ci.suggested_fix:
                                        _part += f" | Fix: {ci.suggested_fix[:200]}"
                                    _fb_parts.append(_part)
                                _feedback = "Cohesion regen:\n" + "\n".join(_fb_parts) if _fb_parts else f"Cohesion regen: blocking issues for {_regen_seg_id}"
                                state.segments[_regen_seg_id].status = SegmentStatus.PENDING.value
                                state.segments[_regen_seg_id].error = _feedback
                        _emit(f"  🔄 Marked {len(_regen_segs)} segment(s) for manual re-generation")
                        _emit(f"  💡 Say 'Astra, command: run segments' to retry architecture generation")
                    try:
                        save_state(state, get_job_dir(job_id))
                    except Exception as _save_err:
                        logger.warning("[SEGMENT_LOOP] Failed to save regen state: %s", _save_err)
                else:
                    # Still have retries — auto-regen the failing segments
                    _regen_segs = _cohesion_result.segments_needing_regen
                    if not _regen_segs:
                        _emit(f"❌ Cohesion FAILED but no segments flagged for regen — cannot auto-fix")
                        break

                    _emit(f"🔄 Cohesion found {_n_blocking} blocking issue(s) — auto-regenerating {len(_regen_segs)} segment(s)...")

                    # Mark flagged segments PENDING with cohesion feedback
                    # v5.33: Structured feedback — include issue ID, category,
                    # expected/actual values, suggested fix, and autofix failure
                    # notes so the regen prompt has full context.
                    for _regen_seg_id in _regen_segs:
                        if _regen_seg_id in state.segments:
                            _fb_parts = []
                            for ci in _cohesion_result.blocking_issues:
                                if ci.source_segment != _regen_seg_id and ci.related_segment != _regen_seg_id:
                                    continue
                                _part = f"[{ci.issue_id}] {ci.category}: {ci.description}"
                                if ci.expected:
                                    _part += f" | Expected: {ci.expected[:200]}"
                                if ci.actual:
                                    _part += f" | Actual: {ci.actual[:200]}"
                                if ci.suggested_fix:
                                    _part += f" | Fix: {ci.suggested_fix[:200]}"
                                if ci.auto_fix_note and "FAILED" in ci.auto_fix_note:
                                    _part += f" | Autofix FAILED: {ci.auto_fix_note}"
                                _fb_parts.append(_part)
                            _feedback = "Cohesion regen:\n" + "\n".join(_fb_parts) if _fb_parts else f"Cohesion regen: blocking issues for {_regen_seg_id}"
                            state.segments[_regen_seg_id].status = SegmentStatus.PENDING.value
                            state.segments[_regen_seg_id].error = _feedback
                            logger.info("[SEGMENT_LOOP] v5.33 Cohesion regen: marked %s PENDING with %d issue detail(s)", _regen_seg_id, len(_fb_parts))
                    save_state(state, get_job_dir(job_id))

                    # Re-run flagged segments through Critical Pipeline
                    for _regen_seg_id in _regen_segs:
                        seg_spec = manifest.get_segment(_regen_seg_id)
                        if seg_spec is None:
                            continue

                        if not can_execute_segment(seg_spec, state):
                            _emit(f"  ⏳ {_regen_seg_id}: waiting on dependencies (skipping regen)")
                            continue

                        _emit(f"  🔄 Re-generating architecture for {_regen_seg_id}...")
                        update_segment_status(state, _regen_seg_id, SegmentStatus.IN_PROGRESS, job_dir_path)

                        segment_context = build_segment_context(
                            seg_spec, state, parent_spec, job_dir_path,
                            contract_set=_contract_set,
                            source_file_evidence=_source_evidence,
                            enrichment=_enrichment_data.get(seg_spec.segment_id),  # v5.17
                        )

                        # v5.16: Inject cohesion issues as architecture-only feedback.
                        # Use "cohesion_issues" key (NOT "cohesion_feedback") because
                        # "cohesion_feedback" triggers the approval gate bypass (line 816),
                        # which would send it to the Overwatcher/Implementer. We only want
                        # architecture regeneration here — approval gate must hold.
                        _seg_state = state.segments.get(_regen_seg_id)
                        if _seg_state and _seg_state.error and _seg_state.error.startswith("Cohesion regen:"):
                            segment_context["cohesion_issues"] = _seg_state.error
                            _emit(f"  🧩 Injected cohesion issues for {_regen_seg_id} (arch-only, no approval bypass)")

                        try:
                            pipeline_result = await run_segment_through_pipeline(
                                segment=seg_spec,
                                segment_context=segment_context,
                                job_id=job_id,
                                db=db,
                                project_id=project_id,
                                on_progress=on_progress,
                                contract_set=_contract_set,
                                job_dir_path=job_dir_path,
                                manifest=manifest,
                                parent_spec=parent_spec,
                            )

                            if pipeline_result.get("success"):
                                if pipeline_result.get("awaiting_approval"):
                                    update_segment_status(state, _regen_seg_id, SegmentStatus.APPROVED, job_dir_path)
                                _emit(f"  ✅ {_regen_seg_id}: architecture re-generated")
                            else:
                                _emit(f"  ❌ {_regen_seg_id}: regen failed — {pipeline_result.get('error', 'unknown')}")

                        except Exception as _regen_err:
                            logger.exception("[SEGMENT_LOOP] v5.16 Regen failed for %s: %s", _regen_seg_id, _regen_err)
                            _emit(f"  ❌ {_regen_seg_id}: regen error — {_regen_err}")

                    save_state(state, get_job_dir(job_id))
                    _emit(f"  🔄 Re-generation complete — re-running cohesion check...")
                    # Loop continues → cohesion re-check at top of while

            else:
                _emit(f"⚠️ Cohesion check error: {_cohesion_result.notes or 'unknown'}")
                break

        except ImportError:
            logger.debug("[SEGMENT_LOOP] Cohesion check module not available")
            break
        except Exception as _coh_err:
            logger.warning("[SEGMENT_LOOP] Cohesion check failed (non-fatal): %s", _coh_err)
            _emit(f"⚠️ Cohesion check error (non-fatal): {_coh_err}")
            break

    # Log final cohesion status
    if _cohesion_passed:
        logger.info("[SEGMENT_LOOP] v5.16 Cohesion passed after %d attempt(s)", _cohesion_retry)
    elif _cohesion_retry > 0:
        logger.warning("[SEGMENT_LOOP] v5.16 Cohesion not resolved after %d attempt(s)", _cohesion_retry)

    # --- v5.34 COHESION HALT GATE ---
    # If cohesion ran and FAILED with unresolved blocking issues, HALT the
    # pipeline. Do NOT proceed to integration check, quarantine, Phase Checkout,
    # or Final Checkout. The implementations on disk may have stale architectures
    # and spending tokens on boot tests is wasteful when cohesion says the
    # segments don't fit together.
    #
    # The user must resolve cohesion issues (via 'run segments' to regen
    # architectures, then 'implement segments' to re-implement) before the
    # pipeline will proceed past this point.
    _cohesion_halted = False
    if _cohesion_retry > 0 and not _cohesion_passed:
        _cohesion_halted = True
        _emit(f"\n{'='*50}")
        _emit("🛑 PIPELINE HALTED: Cohesion check has unresolved blocking issues.")
        _emit("   Phase Checkout, boot test, and Final Checkout are SKIPPED.")
        _emit("   Resolve cohesion issues first, then re-run.")
        _emit(f"{'='*50}")
        logger.warning(
            "[SEGMENT_LOOP] v5.34 COHESION HALT GATE — skipping all downstream stages "
            "(%d retry attempt(s) exhausted without resolution)",
            _cohesion_retry,
        )
        # Save state so the cohesion failure is recorded
        state.overall_status = "cohesion_failed"
        state.phase_checkout_boot = "skipped"
        save_state(state, job_dir_path)

    # --- Cross-segment integration check (Phase 3) ---
    any_segments_complete = any(
        s.status == SegmentStatus.COMPLETE.value
        for s in state.segments.values()
    )
    if any_segments_complete and not _cohesion_halted:
        _emit(f"\n{'='*50}")
        _emit("🔗 Running cross-segment integration check...")

        try:
            from app.orchestrator.integration_check import run_integration_check

            # Load manifest for integration check
            integration_result = run_integration_check(
                manifest=manifest,
                state=state,
                job_dir=job_dir_path,
                on_progress=on_progress,
            )

            # Store result in state
            state.integration_check = integration_result.to_dict()
            save_state(state, job_dir_path)

            # Report
            if integration_result.status == "fail":
                _emit(
                    f"[SEGMENT_LOOP] Integration check FAILED "
                    f"-- {integration_result.error_count} error(s), "
                    f"{integration_result.warning_count} warning(s)"
                )
            elif integration_result.status == "warn":
                _emit(
                    f"[SEGMENT_LOOP] Integration check passed with "
                    f"{integration_result.warning_count} warning(s)"
                )
            elif integration_result.status == "error":
                _emit(
                    f"[SEGMENT_LOOP] Integration check encountered an error: "
                    f"{integration_result.error_message}"
                )
            elif integration_result.status == "skipped":
                _emit("[SEGMENT_LOOP] Integration check skipped (no complete segments)")
            else:
                _emit("[SEGMENT_LOOP] Integration check PASSED")

        except Exception as e:
            logger.exception("[SEGMENT_LOOP] Integration check failed to run: %s", e)
            _emit(f"[SEGMENT_LOOP] Integration check error: {e}")
            # Do NOT crash the segment loop — segments already completed

    # --- v5.31 DEFERRED QUARANTINE — Just Before Phase Checkout ---
    # Moves monolith out of the way so the boot test imports from the
    # new subpackage. Deferred from pre-execution to here so that
    # strike-loop retries can still read the monolith as source evidence.
    # v5.34: Skip if cohesion halted — no point quarantining for a boot
    # test that won't run.
    if implement_only and _quarantine_result is None and not _cohesion_halted:
        _all_impl_done = all(
            s.status in (SegmentStatus.COMPLETE.value, SegmentStatus.FAILED.value,
                         SegmentStatus.BLOCKED.value)
            for s in state.segments.values()
        )
        if _all_impl_done:
            try:
                from app.orchestrator.package_quarantine import (
                    run_quarantine,
                    QuarantineResult,
                )
                from app.overwatcher.sandbox_client import get_sandbox_client

                _q_client = get_sandbox_client()
                _q_sandbox_base = os.getenv("ORB_SANDBOX_BASE", "D:\\Orb")

                _quarantine_result = run_quarantine(
                    manifest_dict=manifest.to_dict(),
                    sandbox_base=_q_sandbox_base,
                    client=_q_client,
                    on_progress=_emit,
                )
                if _quarantine_result.has_quarantined:
                    logger.info(
                        "[SEGMENT_LOOP] v5.31 Deferred quarantine: %d file(s), %d dir(s)",
                        len([e for e in _quarantine_result.entries if e.status == 'quarantined']),
                        len(_quarantine_result.directories_created),
                    )
                    _emit(f"📦 Quarantine: monolith moved aside for boot test")
                if not _quarantine_result.all_ok:
                    for _q_err in _quarantine_result.errors:
                        _emit(f"  ⚠️ Quarantine warning: {_q_err}")
            except ImportError:
                logger.debug("[SEGMENT_LOOP] Package quarantine not available")
            except Exception as _q_err:
                logger.warning("[SEGMENT_LOOP] v5.31 Deferred quarantine failed (non-fatal): %s", _q_err)
                _emit(f"⚠️ Quarantine check failed (non-fatal): {_q_err}")

    # --- v5.0 PHASE CHECKOUT — Stage 9 Full Verification ---
    # Replaces the v4.0 boot check stub with comprehensive verification:
    # size validation + skeleton contract check + boot test + failure routing.
    all_segments_complete = all(
        s.status == SegmentStatus.COMPLETE.value
        for s in state.segments.values()
    )
    # v5.19: Also trigger Phase Checkout when implementation pass has finished
    # (at least 1 COMPLETE) even if some segments are still PENDING/BLOCKED.
    # This ensures boot check + state save happen for partial implementations.
    _any_complete = any(
        s.status == SegmentStatus.COMPLETE.value
        for s in state.segments.values()
    )
    _no_in_progress = not any(
        s.status == SegmentStatus.IN_PROGRESS.value
        for s in state.segments.values()
    )
    _implementation_pass_done = _any_complete and _no_in_progress and total > 0
    _incomplete_segments = [
        sid for sid, s in state.segments.items()
        if s.status != SegmentStatus.COMPLETE.value
    ]
    if _implementation_pass_done and _incomplete_segments and not all_segments_complete and not _cohesion_halted:
        logger.info(
            "[SEGMENT_LOOP] v5.19 Partial completion: %d/%d complete, %d incomplete — "
            "running Phase Checkout anyway for boot verification",
            total - len(_incomplete_segments), total, len(_incomplete_segments),
        )
        _emit(
            f"\n⚠️ {len(_incomplete_segments)} segment(s) incomplete "
            f"({', '.join(_incomplete_segments[:3])}{'...' if len(_incomplete_segments) > 3 else ''}) "
            f"— running Phase Checkout on completed segments"
        )
    if _implementation_pass_done and not _cohesion_halted:
        try:
            from app.orchestrator.phase_checkout import run_phase_checkout
            from app.orchestrator.skeleton_contracts import load_skeleton_contract

            _skeleton = load_skeleton_contract(job_dir_path)
            _checkout_result = await run_phase_checkout(
                job_id=job_id,
                job_dir=job_dir_path,
                state=state,
                manifest=manifest,
                skeleton=_skeleton,
                attempt=1,
                emit=_emit,
            )

            # Map Phase Checkout result to JobState fields
            if _checkout_result.boot_test:
                state.phase_checkout_boot = _checkout_result.boot_test.status
                if _checkout_result.boot_test.error_summary:
                    state.phase_checkout_error = _checkout_result.boot_test.error_summary[:500]
            
            # Store full checkout result for downstream inspection
            state.integration_check = state.integration_check or {}
            state.integration_check["phase_checkout"] = _checkout_result.to_dict()

            if _checkout_result.passed:
                logger.info("[SEGMENT_LOOP] v5.0 Phase Checkout PASSED")
            elif _checkout_result.routing:
                logger.warning(
                    "[SEGMENT_LOOP] v5.0 Phase Checkout FAILED → route to %s (seg=%s)",
                    _checkout_result.routing.target_stage,
                    _checkout_result.routing.target_segment or "all",
                )
                # NOTE: Retry routing is logged but not yet auto-executed.
                # When the phase loop orchestrator is built (Stage 3),
                # it will consume this routing to re-run the right stage.
                # For now, the failure info is saved in state for manual review.

        except (ImportError, Exception) as _pc_err:
            logger.warning("[SEGMENT_LOOP] v5.0 Phase Checkout error: %s", _pc_err)
            _emit(f"⚠️ Phase Checkout could not run: {_pc_err}")
            state.phase_checkout_boot = "error"

        save_state(state, job_dir_path)

    # --- v5.14 FINAL CHECKOUT — Stage 10 (Autonomous Closer + Learning Report) ---
    # Runs after Phase Checkout passes. Performs its own boot test, spec coverage
    # check, AI review, and compiles the Pipeline Learning Report for RAG.
    if all_segments_complete and total > 0 and state.phase_checkout_boot == "pass" and not _cohesion_halted:
        _emit(f"\n{'='*50}")
        _emit("🏁 Running Final Checkout (Stage 10)...")
        try:
            from app.orchestrator.final_checkout import run_final_checkout

            # Try to load original spec for AI review
            _original_spec = None
            if isinstance(parent_spec, dict):
                _original_spec = parent_spec.get("spec_markdown") or parent_spec.get("content", "")
                if not _original_spec:
                    try:
                        _original_spec = json.dumps(parent_spec)[:8000]
                    except Exception:
                        pass
            elif isinstance(parent_spec, str):
                _original_spec = parent_spec

            _final_result = await run_final_checkout(
                job_id=job_id,
                job_dir=job_dir_path,
                sandbox_base=os.getenv("ORB_SANDBOX_BASE", r"D:\Orb"),
                original_spec=_original_spec,
                state=state,
                manifest=manifest,
                emit=_emit,
            )

            state.integration_check = state.integration_check or {}
            state.integration_check["final_checkout"] = _final_result.to_dict()
            save_state(state, job_dir_path)

            if _final_result.status == "pass":
                _emit("🏁 Final Checkout PASSED")
            else:
                _emit(f"🏁 Final Checkout FAILED — see final_checkout_result.json")

        except ImportError:
            logger.debug("[SEGMENT_LOOP] Final Checkout module not available")
        except Exception as _fc_err:
            logger.warning("[SEGMENT_LOOP] v5.14 Final Checkout error: %s", _fc_err)
            _emit(f"⚠️ Final Checkout could not run: {_fc_err}")

    # --- v5.20: ALWAYS distill journal — no matter how the job ends ---
    # Even if the job failed, crashed mid-segment, or only got through
    # architecture generation, any data in the journal is worth ingesting.
    # The distill function handles empty journals gracefully.
    if total > 0:
        try:
            from app.experience.distillation import distill_job
            from app.db import get_db_session
            _distill_db = get_db_session()
            _patterns = distill_job(_distill_db, job_id, job_dir_path)
            if _patterns:
                _emit(f"🧠 Distilled {len(_patterns)} experience pattern(s) from journal")
                logger.info("[SEGMENT_LOOP] Distilled %d patterns for job %s", len(_patterns), job_id)
            _distill_db.close()
        except Exception as _distill_err:
            logger.debug("[SEGMENT_LOOP] Distillation skipped: %s", _distill_err)

    # --- v5.7 / v5.26 QUARANTINE STATUS REPORT (NO AUTO-DELETE) ---
    # v5.26: NEVER auto-delete or auto-rollback quarantine backups.
    # All file deletion/restoration must be human-instigated.
    # The system reports status but does not act.
    if _quarantine_result and _quarantine_result.has_quarantined:
        _final_status = state.compute_overall_status()
        if _final_status == "complete":
            _emit("\n📦 Quarantine: All segments COMPLETE.")
            _emit("  Original files preserved in .quarantined/ folders.")
            _emit("  To clean up: manually delete .quarantined/ dirs when satisfied.")
            _emit("  To rollback: 'Astra, command: rollback quarantine'")
            logger.info("[SEGMENT_LOOP] v5.26 Quarantine preserved (human cleanup required)")
        elif _final_status == "failed":
            _emit("\n📦 Quarantine: Job FAILED — original files safe in .quarantined/ folders.")
            _emit("  To rollback: 'Astra, command: rollback quarantine'")
            logger.info("[SEGMENT_LOOP] v5.26 Quarantine preserved after failure (human rollback required)")
        # else: partial/running — leave quarantine in place for resume

    # --- Final summary ---
    state.overall_status = state.compute_overall_status()
    save_state(state, job_dir_path)

    counts = state.count_by_status()
    # v3.0: Count segments awaiting execution (APPROVED status)
    approved_count = sum(
        1 for seg in state.segments.values()
        if seg.status == SegmentStatus.APPROVED.value
    )
    
    _emit(f"\n{'='*50}")
    _emit(f"📊 SEGMENTED EXECUTION COMPLETE")
    _emit(f"   Status: {state.overall_status.upper()}")
    _emit(f"   Complete: {counts.get('complete', 0)}/{total}")
    if approved_count:
        _emit(f"   ⏸️ Approved (awaiting execution): {approved_count} segment(s)")
        _emit(f"   Say 'Astra, command: implement segments' to execute approved segments")
    if counts.get("failed", 0):
        _emit(f"   Failed: {counts.get('failed', 0)}")
    if counts.get("blocked", 0):
        _emit(f"   Blocked: {counts.get('blocked', 0)}")
    if state.phase_checkout_boot == "pass":
        _emit(f"   🏁 Boot check: PASSED")
    elif state.phase_checkout_boot == "fail":
        _emit(f"   🏁 Boot check: FAILED")
    elif state.phase_checkout_boot == "skipped":
        _emit(f"   🏁 Boot check: SKIPPED (cohesion unresolved)")
    elif state.phase_checkout_boot == "error":
        _emit(f"   🏁 Boot check: ERROR (could not run)")
    _emit(f"{'='*50}")

    logger.info("[SEGMENT_LOOP] Job %s finished: %s", job_id, state.summary())
    print(f"[SEGMENT_LOOP] DONE: {state.summary()}")

    # v5.16: Clear journal context
    try:
        from app.experience.context import clear_job_context
        clear_job_context()
    except Exception:
        pass

    return state
