import json
import logging
import os
from app.orchestrator.__segment_loop_utils_3_utils import _ARCH_EXECUTOR_AVAILABLE, _CRITICAL_PIPELINE_AVAILABLE, _RECONCILIATION_AVAILABLE
from app.orchestrator._segment_loop_utils import _clear_stale_arch_versions
from app.orchestrator._segment_loop_utils import _is_facade_segment, _save_execution_trace
from app.orchestrator.segment_state import get_job_dir, load_or_init_state
from app.pot_spec.grounded.segment_schemas import SegmentSpec
from typing import Any, Dict, List
from typing import Callable, Optional
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
ProgressCallback = Optional[Callable[[str], None]]


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

    # v6.1: DETERMINISTIC REFACTOR BYPASS
    # If this segment was produced by the deterministic refactor pipeline,
    # the architecture is already pre-generated and saved to disk.
    # Skip the LLM Critical Pipeline entirely — zero cost.
    _is_deterministic = segment_context.get("segment_spec", {}).get("deterministic_refactor", False)

    if _is_deterministic:
        _emit(f"  ⚡ Deterministic refactor path for {seg_id} — skipping LLM architecture")
        logger.info("[SEGMENT_LOOP] v6.1 Deterministic refactor bypass for %s", seg_id)

        # Load pre-generated architecture from disk
        _pre_arch_path = os.path.join(
            get_job_dir(job_id), "segments", seg_id, "arch", "arch_v1.md",
        )
        if os.path.isfile(_pre_arch_path):
            with open(_pre_arch_path, "r", encoding="utf-8") as _af:
                arch_text = _af.read()
            critique_passed = True  # Deterministic architecture needs no LLM critique
            _emit(f"  ✅ Pre-generated architecture loaded ({len(arch_text)} chars)")
            logger.info(
                "[SEGMENT_LOOP] v6.1 Loaded deterministic arch for %s: %d chars",
                seg_id, len(arch_text),
            )
        else:
            # Fallback: architecture wasn't pre-generated — run LLM path
            logger.warning(
                "[SEGMENT_LOOP] v6.1 Deterministic flag set but no pre-generated arch at %s — falling back to LLM",
                _pre_arch_path,
            )
            _is_deterministic = False  # Fall through to LLM path below

    if not _is_deterministic:
        # --- Original LLM architecture generation path ---
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
    # Step 1b: Deterministic Import Validator — HARD GATE (Fix 15, v5.38)
    # Zero-LLM-cost check: every cross-segment import must reference a
    # symbol that actually exists in a sibling segment's enrichment.
    # If violations found, inject feedback and regenerate (max 1 retry).
    # v6.1: Skip for deterministic refactor segments — imports are computed
    # from the codebase scan, not hallucinated by an LLM.
    # =====================================================================
    _MAX_IMPORT_REGEN = 1  # One retry attempt with feedback
    _import_regen_count = 0

    if _is_deterministic:
        _emit(f"  ⚡ Skipping import validation (deterministic imports)")
        logger.info("[SEGMENT_LOOP] v6.1 Skipping import validator for deterministic segment %s", seg_id)

    try:
        if _is_deterministic:
            raise ImportError("v6.1: Deterministic segment — skip import validation")

        from app.orchestrator.import_validator import validate_architecture_imports

        # Derive parent job dir from job_id (strip __seg-* suffix)
        _parent_jid = job_id.split("__")[0] if "__" in job_id else job_id
        _artifact_root = os.getenv("ORB_JOB_ARTIFACT_ROOT", "jobs")
        _parent_job_dir = os.path.join(_artifact_root, "jobs", _parent_jid)

        _import_result = validate_architecture_imports(
            arch_text=arch_text,
            segment_id=seg_id,
            parent_job_dir=_parent_job_dir,
        )

        if _import_result.passed:
            _emit(f"  ✅ Import validation: {_import_result.symbols_checked} cross-segment import(s) verified")
        else:
            _emit(f"  ❌ Import validation: {len(_import_result.violations)} violation(s) found")
            for _v in _import_result.violations:
                _emit(f"    ⚠️ {_v.symbol_name}: {_v.message}")

            # v5.38 HARD GATE: Regenerate architecture with violation feedback
            if _import_regen_count < _MAX_IMPORT_REGEN:
                _import_regen_count += 1
                _emit(f"  🔄 Import validation regen {_import_regen_count}/{_MAX_IMPORT_REGEN} — regenerating architecture...")
                logger.info(
                    "[SEGMENT_LOOP] v5.38 Import validation regen %d/%d for %s: %d violation(s)",
                    _import_regen_count, _MAX_IMPORT_REGEN, seg_id, len(_import_result.violations),
                )

                # Inject feedback into segment_context for the LLM
                segment_context["import_validation_feedback"] = _import_result.format_feedback()

                # Re-run Critical Pipeline with feedback
                arch_content_parts_regen: List[str] = []
                done_metadata_regen: Dict[str, Any] = {}
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
                        for line in event.split("\n"):
                            if not line.startswith("data: "):
                                continue
                            try:
                                payload = json.loads(line[6:])
                            except (json.JSONDecodeError, ValueError):
                                continue
                            evt_type = payload.get("type")
                            if evt_type == "token":
                                arch_content_parts_regen.append(payload.get("content", ""))
                            elif evt_type == "done":
                                done_metadata_regen = payload

                    if arch_content_parts_regen:
                        arch_text = "".join(arch_content_parts_regen)
                        _emit(f"  ✅ Architecture regenerated ({len(arch_text)} chars)")

                        # Re-validate
                        _import_result_2 = validate_architecture_imports(
                            arch_text=arch_text,
                            segment_id=seg_id,
                            parent_job_dir=_parent_job_dir,
                        )
                        if _import_result_2.passed:
                            _emit(f"  ✅ Import validation (regen): {_import_result_2.symbols_checked} import(s) verified")
                        else:
                            _emit(f"  ⚠️ Import validation (regen): {len(_import_result_2.violations)} violation(s) remain")
                            for _v2 in _import_result_2.violations:
                                _emit(f"    ⚠️ {_v2.symbol_name}: {_v2.message}")
                            logger.warning(
                                "[SEGMENT_LOOP] v5.38 Import regen still has %d violation(s) for %s — proceeding",
                                len(_import_result_2.violations), seg_id,
                            )

                        # Re-save the regenerated architecture
                        try:
                            with open(seg_arch_path, "w", encoding="utf-8") as f:
                                f.write(arch_text)
                            _emit(f"  💾 Regenerated architecture saved")
                        except Exception as _save_err:
                            logger.warning("[SEGMENT_LOOP] v5.38 Failed to save regen arch: %s", _save_err)
                    else:
                        _emit(f"  ⚠️ Regen produced no output — keeping original")
                except Exception as _regen_err:
                    _emit(f"  ⚠️ Regen failed: {_regen_err} — keeping original")
                    logger.warning("[SEGMENT_LOOP] v5.38 Import regen failed for %s: %s", seg_id, _regen_err)

                # Clear feedback so it doesn't persist to next stages
                segment_context.pop("import_validation_feedback", None)
            else:
                logger.warning(
                    "[SEGMENT_LOOP] v5.38 Import validation failed for %s (no regen left): %d violation(s)",
                    seg_id, len(_import_result.violations),
                )
    except ImportError:
        logger.debug("[SEGMENT_LOOP] import_validator not available — skipping")
    except Exception as _iv_err:
        logger.warning("[SEGMENT_LOOP] v5.38 Import validator error (non-fatal): %s", _iv_err)

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

        # =================================================================
        # v6.0: Implementation Compiler + Brief Validator
        # Replaces the patchwork of v5.12 reconciliation injection and
        # v5.26 extraction binding injection with a unified compilation
        # step that produces per-file briefs.
        #
        # The old injections bolted data onto the architecture text as
        # separate appendices. The compiler produces structured briefs
        # where source code leads, not trails.
        # =================================================================
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

            # Gather enrichment data (same source as old v5.26 extraction binding)
            _parent_job_id = job_id.split('__')[0]
            _compiler_job_dir = get_job_dir(_parent_job_id)
            _compiler_enrichment = segment_context.get("enrichment")

            if not _compiler_enrichment:
                try:
                    from app.orchestrator.extraction_binding import load_segment_enrichment
                    _compiler_enrichment = load_segment_enrichment(_compiler_job_dir, seg_id)
                except Exception:
                    pass

            # Gather reconciliation data (same source as old v5.12)
            _compiler_recon = ""
            if _RECONCILIATION_AVAILABLE and segment.dependencies:
                try:
                    _recon_state = load_or_init_state(job_id.split('__')[0], manifest) if manifest else None
                    if _recon_state:
                        _recon_block = read_dependency_interfaces_from_sandbox(
                            segment=segment,
                            completed_segments=_recon_state.segments,
                            manifest=manifest,
                        )
                        if _recon_block:
                            _compiler_recon = _recon_block
                except Exception as _cr_err:
                    logger.warning("[SEGMENT_LOOP] v6.0 Reconciliation gather failed (non-fatal): %s", _cr_err)

            # Gather sibling enrichments for cross-segment validation
            _sibling_enrichments = {}
            try:
                from app.orchestrator.extraction_binding import load_segment_enrichment as _load_sib_enrich
                if manifest:
                    for _sib_seg in manifest.segments:
                        if _sib_seg.segment_id != seg_id:
                            _sib_e = _load_sib_enrich(_compiler_job_dir, _sib_seg.segment_id)
                            if _sib_e:
                                _sibling_enrichments[_sib_seg.segment_id] = _sib_e
            except Exception:
                pass

            # Compile briefs
            _compilation = compile_implementation_briefs(
                architecture_text=arch_text,
                enrichment=_compiler_enrichment,
                segment_id=seg_id,
                source_file_evidence=segment_context.get("source_file_evidence"),
                interface_contract=segment_context.get("interface_contract", ""),
                sibling_interfaces=segment_context.get("sibling_interfaces", ""),
                cohesion_feedback=segment_context.get("cohesion_feedback", ""),
                implementation_feedback=segment_context.get("implementation_feedback", ""),
                import_validation_feedback=segment_context.get("import_validation_feedback", ""),
                sibling_enrichments=_sibling_enrichments,
            )

            _emit(
                f"  📦 Implementation compiler: {len(_compilation.briefs)} brief(s), "
                f"{_compilation.total_functions} function(s), "
                f"~{_compilation.total_estimated_lines} lines "
                f"(profile={_compilation.profile.value})"
            )

            # Validate and auto-fix briefs
            _fixed_briefs, _fix_log = validate_and_fix_briefs(
                briefs=_compilation.briefs,
                enrichment=_compiler_enrichment,
                sibling_enrichments=_sibling_enrichments,
            )
            _compilation.briefs = _fixed_briefs

            if _fix_log.had_fixes:
                _emit(f"  🔧 Brief validator: {_fix_log.issues_found} issue(s) found and auto-fixed")
                for _fix in _fix_log.fixes:
                    _emit(f"    [{_fix.check}] {_fix.description}")
            else:
                _emit(f"  ✅ Brief validator: all checks passed")

            # Persist compilation artifacts
            save_compilation_result(_compilation, _compiler_job_dir, seg_id)
            save_fix_log(_fix_log, _compiler_job_dir, seg_id)

            # Inject compiled briefs into architecture text
            # Each brief becomes a structured section that the Implementer
            # sees BEFORE the old-style architecture prose.
            if _compilation.briefs:
                _brief_sections = []
                _brief_sections.append("\n\n---\n")
                _brief_sections.append("## IMPLEMENTATION BRIEFS (v6.0 — Compiler Output)\n")
                _brief_sections.append(
                    "The following per-file briefs were compiled from enrichment data, "
                    "source extractions, and interface contracts. **These briefs are the "
                    "primary instruction for implementation.** Follow the directive in each "
                    "brief. If a brief says TRANSPLANT VERBATIM, copy the provided source "
                    "code exactly — do not rewrite, simplify, or reimagine.\n"
                )
                for _brief in _compilation.briefs:
                    _brief_sections.append(_brief.to_markdown())
                    _brief_sections.append("\n---\n")

                _briefs_text = "\n".join(_brief_sections)
                # Prepend briefs BEFORE the architecture so they're seen first
                _recon_arch_text = _briefs_text + "\n\n" + arch_text
                _emit(f"  📄 Injected {len(_compilation.briefs)} compiled brief(s) ({len(_briefs_text)} chars)")
                logger.info(
                    "[SEGMENT_LOOP] v6.0 Implementation compiler: %d brief(s), %d chars for %s",
                    len(_compilation.briefs), len(_briefs_text), seg_id,
                )

            # Also inject reconciliation if available (for non-refactor jobs
            # where the compiler may not have all the data)
            if _compiler_recon and _compiler_recon not in _recon_arch_text:
                _recon_arch_text = inject_reconciliation_into_architecture(
                    _recon_arch_text, _compiler_recon,
                )
                _emit(f"  🧩 Interface reconciliation: supplementary injection ({len(_compiler_recon)} chars)")

        except ImportError as _comp_imp_err:
            logger.info("[SEGMENT_LOOP] v6.0 Implementation compiler not available: %s — falling back to legacy injections", _comp_imp_err)
            _emit(f"  ⚠️ Compiler not available — using legacy injection path")

            # ===== LEGACY FALLBACK: v5.12 + v5.26 injection path =====
            if _RECONCILIATION_AVAILABLE and segment.dependencies:
                try:
                    _job_dir = get_job_dir(job_id.split('__')[0])
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
                except Exception as _recon_err:
                    logger.warning("[SEGMENT_LOOP] v5.12 Legacy reconciliation failed: %s", _recon_err)

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
            except Exception as _eb_err:
                logger.warning("[SEGMENT_LOOP] v5.26 Legacy extraction binding failed: %s", _eb_err)

        except Exception as _comp_err:
            logger.warning("[SEGMENT_LOOP] v6.0 Implementation compiler error (non-fatal): %s", _comp_err)
            _emit(f"  ⚠️ Compiler error (non-fatal): {_comp_err} — using raw architecture")
            _recon_arch_text = arch_text

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
