import json
import logging
import os
from app.orchestrator.__segment_loop_utils_4_utils import _ARCH_EXECUTOR_AVAILABLE, _RECONCILIATION_AVAILABLE
from app.orchestrator._segment_loop_utils import _find_latest_arch, _load_source_file_evidence, collect_segment_outputs, is_segment_blocked
from app.orchestrator._segment_loop_utils import _is_facade_segment, _save_execution_trace, build_segment_context, can_execute_segment, mark_dependents_blocked, unblock_recovered_segments, verify_contracts_fulfilled
from app.orchestrator._segment_loop_utils import run_segment_through_pipeline
from app.orchestrator._segment_loop_utils import update_segment_status
from app.orchestrator.segment_state import JobState, SegmentState, get_job_dir, load_or_init_state, save_state
from app.pot_spec.grounded.segment_schemas import SegmentManifest, SegmentStatus
from typing import Any, Dict, List
from typing import Callable, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
ProgressCallback = Optional[Callable[[str], None]]


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

    # v6.1 FIX 9 + FIX 13: For deterministic refactor jobs, quarantine immediately
    # regardless of implement_only flag. Deterministic architectures are
    # pre-generated so there's no "design only" phase that needs the monolith.
    # The monolith must be gone before the Implementer writes the facade.
    # FIX 13: Support multi-file — check deterministic_sources (list).
    print(f"[QUARANTINE_DEBUG] det_sources={manifest.deterministic_sources}, bool={bool(manifest.deterministic_sources)}, implement_only={implement_only}")
    logger.info(
        "[SEGMENT_LOOP] v6.1 FIX 16 quarantine gate: det_sources=%s, implement_only=%s",
        manifest.deterministic_sources, implement_only,
    )
    if manifest.deterministic_sources:
        logger.info(
            "[SEGMENT_LOOP] v6.1 Deterministic job — quarantine before execution: %s",
            manifest.deterministic_sources,
        )
        try:
            from app.orchestrator.package_quarantine import run_quarantine
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
                    "[SEGMENT_LOOP] v6.1 Quarantine complete: %d file(s), %d dir(s)",
                    len([e for e in _quarantine_result.entries if e.status == 'quarantined']),
                    len(_quarantine_result.directories_created),
                )
                _emit("📦 v6.1 Quarantine: monolith moved before execution")
            else:
                logger.info("[SEGMENT_LOOP] v6.1 Quarantine ran but nothing to move")
        except ImportError:
            logger.debug("[SEGMENT_LOOP] Package quarantine not available")
        except Exception as _q_err:
            logger.warning("[SEGMENT_LOOP] v6.1 Quarantine failed (non-fatal): %s", _q_err)
            _emit(f"⚠️ Quarantine failed (non-fatal): {_q_err}")
    elif not implement_only:
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

        # v5.31: Quarantine DEFERRED to just before Phase Checkout (Stage 9)
        # for greenfield jobs — the monolith is only needed gone for the boot test.
        # v6.1 FIX 9: Deterministic jobs are handled above (before this block).
        logger.info("[SEGMENT_LOOP] v5.31 Quarantine deferred to Phase Checkout")

    # --- Process segments in dependency order (multi-pass) ---
    # v5.11: The loop repeats until no further progress is made.
    # This handles segments that are skipped on early passes because
    # their dependencies aren't COMPLETE yet (e.g. seg-01 depends on seg-02..seg-09).
    # Also handles PENDING segments that get architectures generated and need
    # a second pass to execute once approved.
    _raw_order = manifest.get_execution_order()

    # v5.40 (Fix 21): Complexity-aware ordering within dependency tiers.
    # Segments with fewer dependencies are simpler and should run first
    # within their tier. This means when a complex segment runs, all
    # simpler siblings already have architectures/enrichment available,
    # giving the LLM maximum data and minimum need to hallucinate.
    def _complexity_sort(order: List[str]) -> List[str]:
        """Re-sort execution order: within each dependency tier, simple first."""
        # Build tiers: group by depth (number of transitive deps resolved)
        completed: set = set()
        tiers: List[List[str]] = []
        remaining = list(order)

        while remaining:
            # Segments whose deps are all in 'completed'
            ready = [
                sid for sid in remaining
                if all(
                    d in completed
                    for d in (manifest.get_segment(sid).dependencies or [])
                )
            ]
            if not ready:
                # Shouldn't happen — safety fallback, keep original order
                tiers.append(remaining)
                break
            # Sort this tier by dependency count ascending (simple first)
            ready.sort(key=lambda sid: len(manifest.get_segment(sid).dependencies or []))
            tiers.append(ready)
            completed.update(ready)
            remaining = [sid for sid in remaining if sid not in completed]

        flat = []
        for tier in tiers:
            flat.extend(tier)
        return flat

    execution_order = _complexity_sort(_raw_order)
    total = len(execution_order)
    _pass_number = 0
    MAX_PASSES = 5  # Safety limit to prevent infinite loops

    # Log if order changed
    if execution_order != _raw_order:
        _emit(f"\u2699\ufe0f v5.40 Complexity-sorted order: {' \u2192 '.join(s.split('-', 2)[-1][:20] for s in execution_order)}")
        logger.info("[SEGMENT_LOOP] v5.40 Complexity-sorted: %s", execution_order)

    _emit(f"\ud83d\udd04 Processing {total} segment(s) in dependency order...\n")

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
                # v6.1 FIX 15: Deterministic refactor jobs skip the two-phase flow.
                # Their architectures are pre-generated, so APPROVED = ready to execute.
                _is_det = bool(manifest.deterministic_sources)
                if not implement_only and not _is_det:
                    _emit(f"⏸️ [{idx}/{total}] {seg_id}: APPROVED — awaiting 'implement segments' command")
                    continue
                if _is_det and not implement_only:
                    _emit(f"⚡ [{idx}/{total}] {seg_id}: Deterministic — auto-executing (skip two-phase)")
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
                            # v6.0: Implementation Compiler (call site 2 — implement_only path)
                            _recon_arch_text = arch_text
                            try:
                                from app.orchestrator.implementation_compiler import (
                                    compile_implementation_briefs,
                                    save_compilation_result,
                                )
                                from app.orchestrator.brief_validator import (
                                    validate_and_fix_briefs,
                                    save_fix_log,
                                )

                                _compiler_enrichment2 = segment_context.get("enrichment") if segment_context else None
                                if not _compiler_enrichment2:
                                    try:
                                        from app.orchestrator.extraction_binding import load_segment_enrichment
                                        _compiler_enrichment2 = load_segment_enrichment(job_dir_path, seg_id)
                                    except Exception:
                                        pass

                                _sibling_enrichments2 = {}
                                try:
                                    from app.orchestrator.extraction_binding import load_segment_enrichment as _lse2
                                    for _sib2 in manifest.segments:
                                        if _sib2.segment_id != seg_id:
                                            _se2 = _lse2(job_dir_path, _sib2.segment_id)
                                            if _se2:
                                                _sibling_enrichments2[_sib2.segment_id] = _se2
                                except Exception:
                                    pass

                                _comp2 = compile_implementation_briefs(
                                    architecture_text=arch_text,
                                    enrichment=_compiler_enrichment2,
                                    segment_id=seg_id,
                                    source_file_evidence=segment_context.get("source_file_evidence") if segment_context else None,
                                    interface_contract=segment_context.get("interface_contract", "") if segment_context else "",
                                    sibling_interfaces=segment_context.get("sibling_interfaces", "") if segment_context else "",
                                    sibling_enrichments=_sibling_enrichments2,
                                )

                                _fixed2, _flog2 = validate_and_fix_briefs(
                                    briefs=_comp2.briefs,
                                    enrichment=_compiler_enrichment2,
                                    sibling_enrichments=_sibling_enrichments2,
                                )
                                _comp2.briefs = _fixed2

                                _emit(
                                    f"  📦 Compiler: {len(_comp2.briefs)} brief(s), "
                                    f"{_comp2.total_functions} func(s) (profile={_comp2.profile.value})"
                                )
                                if _flog2.had_fixes:
                                    _emit(f"  🔧 Validator: {_flog2.issues_found} issue(s) auto-fixed")

                                save_compilation_result(_comp2, job_dir_path, seg_id)
                                save_fix_log(_flog2, job_dir_path, seg_id)

                                if _comp2.briefs:
                                    _bs2 = ["\n\n---\n", "## IMPLEMENTATION BRIEFS (v6.0)\n"]
                                    _bs2.append(
                                        "Per-file briefs compiled from enrichment and contracts. "
                                        "**Follow each brief's directive.**\n"
                                    )
                                    for _b2 in _comp2.briefs:
                                        _bs2.append(_b2.to_markdown())
                                        _bs2.append("\n---\n")
                                    _bt2 = "\n".join(_bs2)
                                    _recon_arch_text = _bt2 + "\n\n" + arch_text
                                    _emit(f"  📄 Injected {len(_comp2.briefs)} brief(s) ({len(_bt2)} chars)")

                            except ImportError:
                                logger.info("[SEGMENT_LOOP] v6.0 Compiler not available — legacy fallback")
                                # Legacy v5.12 + v5.26 fallback
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
                                    except Exception:
                                        pass
                                try:
                                    from app.orchestrator.extraction_binding import (
                                        load_segment_enrichment, build_extraction_block,
                                        build_facade_export_map, inject_extraction_into_architecture,
                                    )
                                    _eb_enrichment = load_segment_enrichment(job_dir_path, seg_id)
                                    if _eb_enrichment:
                                        _is_facade = _is_facade_segment(seg_spec, manifest) if manifest else False
                                        if _is_facade and manifest:
                                            _eb_block = build_facade_export_map(job_dir_path, manifest.segments, seg_id)
                                        else:
                                            _eb_block = build_extraction_block(_eb_enrichment, seg_id)
                                        if _eb_block:
                                            _recon_arch_text = inject_extraction_into_architecture(_recon_arch_text, _eb_block)
                                except Exception:
                                    pass
                            except Exception as _comp2_err:
                                logger.warning("[SEGMENT_LOOP] v6.0 Compiler error (implement_only): %s", _comp2_err)
                                _recon_arch_text = arch_text
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

            # v5.39 (Fix 13): Defer facade entirely during design phase.
            # The facade can't have complete data until all siblings are
            # implemented. Skip it now — it gets built during implement_only
            # via the v5.26 auto-generate path (line ~1904).
            if _facade and not implement_only:
                _emit(f"⏭️ [{idx}/{total}] {seg_id}: FACADE — deferred to implementation phase")
                logger.info("[SEGMENT_LOOP] v5.39 Facade %s deferred to implementation phase", seg_id)
                continue

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

            # v6.1: Detect if this is a deterministic refactor job
            _is_deterministic_job = False
            try:
                # v6.1 FIX 13: Check both list and single forms
                _is_deterministic_job = bool(
                    manifest_data.get("deterministic_sources")
                    or manifest_data.get("deterministic_source")
                )
            except NameError:
                pass  # manifest_data not available in this code path

            _cohesion_result = await run_cohesion_check(
                job_id=job_id,
                job_dir=job_dir_path,
                segment_ids=_approved_seg_ids,
                contract_json=_cohesion_contract_json,
                source_file_evidence=_source_evidence,
                skip_llm_layer=_is_deterministic_job,
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
                    # v5.35 FILE PROTECTION: Before regen, snapshot files from 
                    # completed segments that are NOT being regenerated.
                    # These files are UNTOUCHABLE during the regen cycle.
                    _protected_files: set = set()
                    for _ps_id, _ps_state in state.segments.items():
                        if _ps_id not in _regen_segs and _ps_state.status == SegmentStatus.COMPLETE.value:
                            for _pf in (_ps_state.output_files or []):
                                _protected_files.add(_pf.replace("\\", "/"))
                    if _protected_files:
                        logger.info(
                            "[SEGMENT_LOOP] v5.35 FILE PROTECTION: %d files from %d completed segments protected during regen",
                            len(_protected_files),
                            sum(1 for s in state.segments.values() if s.status == SegmentStatus.COMPLETE.value and s.segment_id not in _regen_segs if hasattr(s, 'segment_id')) or len([sid for sid, ss in state.segments.items() if ss.status == SegmentStatus.COMPLETE.value and sid not in _regen_segs]),
                        )
                        _emit(f"  🛡️ {len(_protected_files)} files from completed segments are protected during regen")
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

                    # v5.35 POST-REGEN FILE PROTECTION CHECK
                    # Verify that no protected files were wiped during regen
                    if _protected_files:
                        _missing_protected = []
                        for _pf in _protected_files:
                            _abs = os.path.join(job_dir_path.rsplit("jobs", 1)[0].rstrip("/\\").rsplit("jobs", 1)[0].rstrip("/\\"), _pf) if not os.path.isabs(_pf) else _pf
                            # Try common resolutions
                            _candidates = [_pf, os.path.join("D:/Orb", _pf), _pf.replace("/", os.sep)]
                            _found = any(os.path.isfile(c) for c in _candidates)
                            if not _found:
                                _missing_protected.append(_pf)
                        if _missing_protected:
                            logger.error(
                                "[SEGMENT_LOOP] v5.35 PROTECTION VIOLATION: %d protected files missing after regen: %s",
                                len(_missing_protected), _missing_protected[:5],
                            )
                            _emit(f"  ⚠️ PROTECTION VIOLATION: {len(_missing_protected)} completed segment files missing after regen!")
                            _emit(f"     Missing: {', '.join(os.path.basename(f) for f in _missing_protected[:5])}")
                        else:
                            logger.info("[SEGMENT_LOOP] v5.35 All %d protected files intact after regen", len(_protected_files))

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

    # --- v5.35 DEFERRED QUARANTINE — Just Before Phase Checkout ---
    # Moves monolith out of the way so the boot test imports from the
    # new subpackage. Deferred from pre-execution to here so that
    # strike-loop retries can still read the monolith as source evidence.
    # v5.34: Skip if cohesion halted — no point quarantining for a boot
    # test that won't run.
    # v5.35: ONLY quarantine when ALL segments are COMPLETE. If any
    # segment FAILED or is BLOCKED, the package is incomplete and
    # quarantining the monolith would leave the codebase broken.
    # Phase Checkout will still run (for partial status reporting) but
    # without quarantine, it boots against the original monolith.
    if implement_only and _quarantine_result is None and not _cohesion_halted:
        _all_segments_fully_complete = all(
            s.status == SegmentStatus.COMPLETE.value
            for s in state.segments.values()
        )
        _any_failed_or_blocked = any(
            s.status in (SegmentStatus.FAILED.value, SegmentStatus.BLOCKED.value)
            for s in state.segments.values()
        )
        if _any_failed_or_blocked:
            logger.info(
                "[SEGMENT_LOOP] v5.35 Quarantine SKIPPED — incomplete job "
                "(failed/blocked segments). Monolith stays in place."
            )
            _emit("📦 Quarantine: SKIPPED — not all segments complete, monolith preserved")
        if _all_segments_fully_complete:
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
