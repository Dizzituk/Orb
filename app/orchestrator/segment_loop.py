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

SEGMENT_LOOP_BUILD_ID = "2026-02-10-v1.5-needle-model-grounding"
print(f"[SEGMENT_LOOP_LOADED] BUILD_ID={SEGMENT_LOOP_BUILD_ID}")

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
    print(f"[SEGMENT_LOOP] ⚠️ Architecture executor import failed: {_ae}")


# Type alias for progress callback
ProgressCallback = Optional[Callable[[str], None]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================


def can_execute_segment(segment: SegmentSpec, state: JobState) -> bool:
    """
    Check if all dependencies of a segment are COMPLETE.

    Returns True if the segment has no dependencies, or all dependencies
    have status COMPLETE. Returns False if any dependency is PENDING,
    IN_PROGRESS, FAILED, or BLOCKED.
    """
    if not segment.dependencies:
        return True

    for dep_id in segment.dependencies:
        dep_state = state.segments.get(dep_id)
        if dep_state is None:
            logger.warning(
                "[SEGMENT_LOOP] Segment %s depends on unknown segment %s",
                segment.segment_id, dep_id,
            )
            return False
        # v3.0: APPROVED counts as dependency-met for architecture generation,
        # and COMPLETE counts for execution. Both allow the next segment to proceed.
        if dep_state.status not in (SegmentStatus.COMPLETE.value, SegmentStatus.APPROVED.value):
            return False

    return True


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

    # --- Save architecture per-segment on disk ---
    seg_arch_dir = os.path.join(
        get_job_dir(job_id), "segments", seg_id, "arch",
    )
    os.makedirs(seg_arch_dir, exist_ok=True)
    seg_arch_path = os.path.join(seg_arch_dir, "arch_v1.md")
    try:
        with open(seg_arch_path, "w", encoding="utf-8") as f:
            f.write(arch_text)
        _emit(f"  💾 Architecture saved: segments/{seg_id}/arch/arch_v1.md")
    except Exception as e:
        logger.warning("[SEGMENT_LOOP] Failed to save segment arch: %s", e)

    # --- v3.0: Show File Inventory from architecture for transparency ---
    try:
        import re as _re
        _file_lines = []
        for _m in _re.finditer(r'\|\s*`([^`]+)`\s*\|\s*([^|]+)', arch_text):
            _fp = _m.group(1).strip()
            _desc = _m.group(2).strip()
            if _fp and not _fp.startswith('---') and _fp.lower() != 'file':
                _file_lines.append(f"    {'CREATE' if 'new' in _desc.lower() or 'create' in _desc.lower() or 'package' in _desc.lower() else 'MODIFY'}: `{_fp}` — {_desc[:80]}")
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
    # =====================================================================
    auto_execute = os.getenv("ASTRA_SEGMENT_AUTO_EXECUTE", "0").strip()
    if auto_execute != "1":
        _emit(f"  ⏸️ AWAITING APPROVAL: Architecture ready for {seg_id}")
        _emit(f"  📄 Review: jobs/{os.path.basename(get_job_dir(job_id))}/segments/{seg_id}/arch/arch_v1.md")
        _emit(f"  💡 To execute: say 'Astra, command: execute segment {seg_id}'")
        _emit(f"  💡 To execute all: set env ASTRA_SEGMENT_AUTO_EXECUTE=1 and re-run")
        result["success"] = True
        result["awaiting_approval"] = True
        result["architecture_path"] = seg_arch_path
        return result

    # =====================================================================
    # Step 3: Overwatcher + Architecture Executor
    # Only reached if ASTRA_SEGMENT_AUTO_EXECUTE=1 or explicit approval
    # =====================================================================
    _emit(f"  🔧 Running Overwatcher for {seg_id}...")

    if not _ARCH_EXECUTOR_AVAILABLE:
        _emit(f"  ⚠️ Architecture executor not available — architecture generated only")
        result["success"] = True
        return result

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
        # v5.5 PHASE 4A: Pass interface contract for Job Checker
        _seg_contract = segment_context.get("interface_contract", "")
        arch_result = await run_architecture_execution(
            spec=spec,
            architecture_content=arch_text,
            architecture_path=seg_arch_path,
            job_id=seg_job_id,
            llm_call_fn=llm_call_fn,
            artifact_root=os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs"),
            interface_contract=_seg_contract,
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

    # --- v5.4 PHASE 2A: Critical Supervisor — Interface Contract Generation ---
    # Before generating any architectures, call the Critical Supervisor to
    # produce interface contracts. These contracts ensure independently
    # generated segment architectures remain compatible.
    _contract_set = None
    try:
        from app.orchestrator.critical_supervisor import (
            generate_interface_contracts,
            save_contracts,
            load_contracts,
            SupervisorContractSet,
        )
        _SUPERVISOR_AVAILABLE = True
    except ImportError:
        _SUPERVISOR_AVAILABLE = False
        logger.debug("[SEGMENT_LOOP] Critical Supervisor not available")

    if _SUPERVISOR_AVAILABLE:
        # Check if contracts already exist (crash recovery)
        _contract_set = load_contracts(job_dir_path)
        if _contract_set and _contract_set.boundaries:
            _emit(f"📋 Loaded existing contracts: {len(_contract_set.boundaries)} boundary(ies)")
        else:
            _emit("🔗 Running Critical Supervisor — generating interface contracts...")
            try:
                _contract_set = await generate_interface_contracts(
                    job_id=job_id,
                    spec_markdown=parent_spec.get("spot_markdown", "") if parent_spec else "",
                    manifest_dict=manifest.to_dict(),
                )
                if _contract_set.boundaries:
                    save_contracts(_contract_set, job_dir_path)
                    _emit(f"✅ Generated {len(_contract_set.boundaries)} interface boundary(ies)")
                    for _b in _contract_set.boundaries:
                        _emit(f"  🔗 {_b.exposing_segment} → {', '.join(_b.consuming_segments)}: {_b.name} ({_b.kind})")
                else:
                    _emit("ℹ️ No cross-segment boundaries detected (segments may be independent)")
                    if _contract_set.generation_notes:
                        logger.info("[SEGMENT_LOOP] Supervisor notes: %s", _contract_set.generation_notes)
            except Exception as sup_err:
                logger.warning("[SEGMENT_LOOP] Critical Supervisor failed (non-fatal): %s", sup_err)
                _emit(f"⚠️ Critical Supervisor failed (non-fatal): {sup_err}")
                _contract_set = None

    # --- Initialise or resume state ---
    state = load_or_init_state(job_id, manifest)
    _emit(f"📊 State: {state.summary()}")

    # --- Process segments in dependency order ---
    execution_order = manifest.get_execution_order()
    total = len(execution_order)

    _emit(f"🔄 Processing {total} segment(s) in dependency order...\n")

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

        # --- Skip BLOCKED segments ---
        if seg_state.status == SegmentStatus.BLOCKED.value:
            _emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED — {seg_state.error or 'dependency failed'}")
            continue

        # --- v3.0: APPROVED segments — skip architecture, go straight to execution ---
        if seg_state.status == SegmentStatus.APPROVED.value:
            # v3.1: Check if dependencies failed/blocked BEFORE executing
            if is_segment_blocked(seg_spec, state):
                update_segment_status(
                    state, seg_id, SegmentStatus.BLOCKED, job_dir_path,
                    error="Dependency failed or blocked",
                )
                _emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED by failed dependency (was APPROVED)")
                continue
            if not can_execute_segment(seg_spec, state):
                _emit(f"⏳ [{idx}/{total}] {seg_id}: APPROVED but waiting on dependencies (skipping)")
                continue

            _emit(f"\n✅ [{idx}/{total}] {seg_id}: APPROVED — executing...")
            _emit(f"  Files: {', '.join(seg_spec.file_scope[:5])}"
                   f"{'...' if len(seg_spec.file_scope) > 5 else ''}")
            update_segment_status(state, seg_id, SegmentStatus.IN_PROGRESS, job_dir_path)

            # Load the saved architecture and execute directly
            seg_dir = os.path.join(job_dir_path, "segments", seg_id)
            arch_path = os.path.join(seg_dir, "arch", "arch_v1.md")
            # Also try arch_v2 if revision happened
            arch_v2_path = os.path.join(seg_dir, "arch", "arch_v2.md")
            if os.path.isfile(arch_v2_path):
                arch_path = arch_v2_path

            if not os.path.isfile(arch_path):
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
                        arch_result = await run_architecture_execution(
                            spec=spec,
                            architecture_content=arch_text,
                            architecture_path=arch_path,
                            job_id=seg_job_id,
                            llm_call_fn=llm_call_fn,
                            artifact_root=os.getenv("ORB_JOB_ARTIFACT_ROOT", "D:/Orb/jobs"),
                            interface_contract=_seg_contract_md,
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
        # --- Check if segment should be blocked ---
        if is_segment_blocked(seg_spec, state):
            update_segment_status(
                state, seg_id, SegmentStatus.BLOCKED, job_dir_path,
                error="Dependency failed or blocked",
            )
            _emit(f"🚫 [{idx}/{total}] {seg_id}: BLOCKED by failed dependency")
            continue

        # --- Check dependencies ---
        if not can_execute_segment(seg_spec, state):
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
        )

        # Run through pipeline
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
            logger.exception("[SEGMENT_LOOP] Unexpected error processing %s", seg_id)

        # --- Handle result ---
        if pipeline_result["success"]:
            # v3.0: Check if segment is awaiting approval (architecture generated but not executed)
            if pipeline_result.get("awaiting_approval", False):
                update_segment_status(
                    state, seg_id, SegmentStatus.APPROVED, job_dir_path,
                )
                _emit(f"  ✅ {seg_id}: APPROVED — architecture ready for review")
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

    # --- v5.4 PHASE 2C: Cross-Segment Cohesion Check ---
    # After architecture generation, before execution. Runs when 2+ segments
    # have architectures (APPROVED or COMPLETE). Calls Opus 4.6 to verify
    # all imports resolve, names match, data shapes are compatible.
    _approved_seg_ids = [
        sid for sid, ss in state.segments.items()
        if ss.status in (SegmentStatus.APPROVED.value, SegmentStatus.COMPLETE.value)
    ]
    if len(_approved_seg_ids) >= 2:
        _emit(f"\n{'='*50}")
        _emit("🔍 Running cross-segment cohesion check...")
        try:
            from app.orchestrator.cohesion_check import (
                run_cohesion_check,
                save_cohesion_result,
            )

            # Load contract JSON if available
            _cohesion_contract_json = None
            if _contract_set and _contract_set.boundaries:
                _cohesion_contract_json = _contract_set.to_json()

            _cohesion_result = await run_cohesion_check(
                job_id=job_id,
                job_dir=job_dir_path,
                segment_ids=_approved_seg_ids,
                contract_json=_cohesion_contract_json,
            )
            save_cohesion_result(_cohesion_result, job_dir_path)

            if _cohesion_result.status == "pass":
                _emit("✅ Cohesion check PASSED — all segments are compatible")
            elif _cohesion_result.status == "fail":
                _n_blocking = len(_cohesion_result.blocking_issues)
                _n_warning = len(_cohesion_result.warning_issues)
                _emit(f"❌ Cohesion check FAILED — {_n_blocking} blocking, {_n_warning} warning(s)")
                for _ci in _cohesion_result.blocking_issues:
                    _emit(f"  🚫 {_ci.issue_id} [{_ci.category}] {_ci.source_segment} ↔ {_ci.related_segment}")
                    _emit(f"     {_ci.description}")
                    if _ci.suggested_fix:
                        _emit(f"     Fix: {_ci.suggested_fix}")
                for _ci in _cohesion_result.warning_issues:
                    _emit(f"  ⚠️ {_ci.issue_id} [{_ci.category}] {_ci.description}")
                _regen_segs = _cohesion_result.segments_needing_regen
                if _regen_segs:
                    _emit(f"\n  💡 Segment(s) needing re-generation: {', '.join(_regen_segs)}")
            else:
                _emit(f"⚠️ Cohesion check error: {_cohesion_result.notes or 'unknown'}")

        except ImportError:
            logger.debug("[SEGMENT_LOOP] Cohesion check module not available")
        except Exception as _coh_err:
            logger.warning("[SEGMENT_LOOP] Cohesion check failed (non-fatal): %s", _coh_err)
            _emit(f"⚠️ Cohesion check error (non-fatal): {_coh_err}")

    # --- Cross-segment integration check (Phase 3) ---
    any_segments_complete = any(
        s.status == SegmentStatus.COMPLETE.value
        for s in state.segments.values()
    )
    if any_segments_complete:
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
        _emit(f"   Say 'Astra, command: run segments' to execute approved segments")
    if counts.get("failed", 0):
        _emit(f"   Failed: {counts.get('failed', 0)}")
    if counts.get("blocked", 0):
        _emit(f"   Blocked: {counts.get('blocked', 0)}")
    _emit(f"{'='*50}")

    logger.info("[SEGMENT_LOOP] Job %s finished: %s", job_id, state.summary())
    print(f"[SEGMENT_LOOP] DONE: {state.summary()}")

    return state
